#!/usr/bin/env python3
"""
Scope Manager for web-based oscilloscope control
Contains SocketScope class (tested logic from usethis.py) and ScopeManager wrapper for multi-device support
"""

import socket
import time
import struct
from datetime import datetime
import traceback
import os
import threading

class SocketScope:
    """Socket-based communication class for Siglent SDS5104X oscilloscope"""

    # TDIV enum from manual (time per division values) - Ensure this matches your scope model
    TDIV_ENUM = [100e-12, 200e-12, 500e-12,
                 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
                 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
                 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
                 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    HORI_NUM = 10  # Number of horizontal divisions (SDS5000X)

    def __init__(self, ip_address="192.168.1.10", port=5025, timeout=15.0):
        """
        Initialize socket connection to oscilloscope

        Args:
            ip_address (str): IP address of the oscilloscope (default: 192.168.1.10)
            port (int): Port number (default: 5025 for SCPI socket)
            timeout (float): Socket timeout in seconds (increased default)
        """
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.query_lock = threading.Lock()  # Lock to prevent query mixing

    def connect(self):
        """Establish TCP socket connection to the oscilloscope"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            print(f"Connecting to oscilloscope at {self.ip_address}:{self.port}...")
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            print("✓ Connected successfully!")
            
            # Clear any stale data from buffer immediately after connection
            try:
                self.socket.setblocking(0)
                flushed = 0
                while True:
                    try:
                        chunk = self.socket.recv(4096)
                        if not chunk:
                            break
                        flushed += len(chunk)
                    except:
                        break
                self.socket.setblocking(1)
                if flushed > 0:
                    print(f"DEBUG: Flushed {flushed} bytes of stale data on connect")
            except:
                pass
            
            return True
        except (socket.error, socket.timeout) as e:
            print(f"✗ Failed to connect: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Close socket connection"""
        if self.socket:
            try:
                self.socket.close()
            except socket.error as e:
                print(f"Error closing socket: {e}")
            finally:
                self.socket = None
                self.connected = False
                print("Disconnected from oscilloscope")

    def send_command(self, command, delay_after=0.1):
        """
        Send SCPI command to oscilloscope

        Args:
            command (str): SCPI command string
            delay_after (float): Optional small delay after sending
        """
        if not self.connected or self.socket is None:
            raise ConnectionError("Not connected to oscilloscope")

        try:
            cmd_bytes = (command + '\n').encode('utf-8')
            self.socket.sendall(cmd_bytes)
            if delay_after > 0:
                time.sleep(delay_after)

        except (socket.error, socket.timeout) as e:
            print(f"Error sending command '{command}': {e}")
            self.connected = False
            raise ConnectionError(f"Connection lost while sending: {e}") from e

    def read_response(self, buffer_size=4096, expect_binary=False):
        """
        Read response from the socket. Handles text and simple binary blocks.

        Args:
            buffer_size (int): Size of chunks to read.
            expect_binary (bool): If True, attempts to parse binary header.

        Returns:
            bytes: Raw response data.
        """
        if not self.connected or self.socket is None:
            raise ConnectionError("Not connected to oscilloscope")

        response = b""
        data_len = -1
        header_parsed = False
        data_start = 0

        try:
            while True:
                chunk = self.socket.recv(buffer_size)
                if not chunk:
                    self.connected = False
                    raise ConnectionAbortedError("Socket connection closed by oscilloscope during read")

                response += chunk

                if expect_binary and not header_parsed and b'#' in response:
                    try:
                        _, data_len, data_start = self.parse_binary_header(response)
                        header_parsed = True
                    except ValueError:
                        pass

                if expect_binary and header_parsed:
                    if len(response) >= data_start + data_len:
                        break
                elif not expect_binary:
                    if b'\n' in chunk:
                        break

        except socket.timeout:
            print("Error reading response: Socket timed out.")
            if expect_binary and data_len > 0 and len(response) > data_start:
                print(f"Warning: Timeout during binary read. Expected {data_len}, got {len(response)-data_start}.")
                pass
            else:
                raise TimeoutError("Socket timed out while waiting for response.") from None

        except socket.error as e:
            print(f"Error reading response: {e}")
            self.connected = False
            raise ConnectionError(f"Connection lost while reading: {e}") from e

        return response

    def query(self, query_string):
        """
        Send SCPI query and return decoded string response.
        Uses lock to prevent queries from getting mixed up.

        Args:
            query_string (str): SCPI query string

        Returns:
            str: Response from oscilloscope (decoded and stripped)
        """
        with self.query_lock:
            self.send_command(query_string)
            response_bytes = self.read_response(expect_binary=False)
            response_str = response_bytes.decode('utf-8', errors='ignore').strip()
            return response_str

    def get_waveform_preamble(self, channel):
        """
        Get waveform preamble data (binary block) for proper scaling

        Args:
            channel (int): Channel number (1-4)

        Returns:
            dict: Parsed preamble data, or {} on error.
        """
        PREAMBLE_DATA_LENGTH = 346

        try:
            # Verify connection before attempting preamble read
            if not self.connected or self.socket is None:
                raise ConnectionError("Not connected to oscilloscope")
            
            # If the channel display is OFF, avoid selecting it as WAV:SOUR
            # Some scope models implicitly enable a channel when WAV:SOUR is used.
            try:
                ch_state = self.query(f":CHANnel{channel}:SWITch?")
                if ch_state and ch_state.strip().upper() == "OFF":
                    print(f"DEBUG: Channel {channel} display is OFF; skipping waveform request.")
                    return {}
            except Exception as e:
                # If we cannot query the channel state, continue but log a warning
                print(f"Warning: Could not query channel {channel} state: {e} (continuing to request waveform)")

            self.send_command(f"WAV:SOUR C{channel}", delay_after=0.05)
            self.send_command("WAV:PRE?")

            preamble_response = self.read_response(expect_binary=True)

            header_len, data_len, preamble_start = self.parse_binary_header(preamble_response)

            if data_len == 0:
                raise ValueError("Preamble header parsing failed or reported zero length.")

            if data_len != PREAMBLE_DATA_LENGTH:
                print(f"Warning: Preamble header reported {data_len} bytes, expected {PREAMBLE_DATA_LENGTH}. Check manual/model.")

            if len(preamble_response) < preamble_start + PREAMBLE_DATA_LENGTH:
                raise ValueError(f"Incomplete preamble data received. Got {len(preamble_response)-preamble_start}, expected {PREAMBLE_DATA_LENGTH}")

            preamble_data = preamble_response[preamble_start : preamble_start + PREAMBLE_DATA_LENGTH]

            preamble = {}
            preamble['comm_type'] = struct.unpack('<h', preamble_data[0x20:0x22])[0]
            preamble['wave_array_count'] = struct.unpack('<l', preamble_data[0x74:0x78])[0]
            preamble['vdiv'] = struct.unpack('<f', preamble_data[0x9c:0xa0])[0]
            preamble['voffset'] = struct.unpack('<f', preamble_data[0xa0:0xa4])[0]
            preamble['code_per_div'] = struct.unpack('<f', preamble_data[0xa4:0xa8])[0]
            preamble['adc_bit'] = struct.unpack('<h', preamble_data[0xac:0xae])[0]
            preamble['interval'] = struct.unpack('<f', preamble_data[0xb0:0xb4])[0]
            preamble['delay'] = struct.unpack('<d', preamble_data[0xb4:0xbc])[0]
            preamble['tdiv_index'] = struct.unpack('<h', preamble_data[0x144:0x146])[0]
            preamble['probe_atten'] = struct.unpack('<f', preamble_data[0x148:0x14c])[0]

            if preamble['code_per_div'] == 0:
                print("Warning: Parsed 'code_per_div' is zero, scaling will fail.")
                return {}

            return preamble

        except (socket.timeout, ConnectionError, ValueError, struct.error, IndexError) as e:
            print(f"Error getting waveform preamble for C{channel}: {e}")
            traceback.print_exc()
            return {}

    def parse_binary_header(self, data):
        """
        Parse SCPI binary data header #N<Digits><Data>

        Args:
            data (bytes): Raw data starting with '#'

        Returns:
            tuple: (total_header_length, data_length, data_start_index in original data)
                   Returns (0, 0, 0) if header is invalid or incomplete.
        """
        try:
            hash_pos = data.find(b'#')
            if hash_pos == -1:
                raise ValueError("Binary header marker '#' not found")

            if len(data) < hash_pos + 2:
                raise ValueError("Incomplete header: Missing N digit")
            n_digits = int(data[hash_pos + 1:hash_pos + 2])
            if n_digits == 0:
                raise ValueError("Invalid header: N cannot be 0")

            digits_start = hash_pos + 2
            digits_end = digits_start + n_digits
            if len(data) < digits_end:
                raise ValueError("Incomplete header: Missing <Digits>")

            data_length = int(data[digits_start:digits_end])
            total_header_length = digits_end - hash_pos
            data_start_index = digits_end

            return total_header_length, data_length, data_start_index

        except (ValueError, IndexError) as e:
            return 0, 0, 0

    def get_waveform_data(self, channel):
        """
        Get waveform data from specified channel with proper scaling using preamble.
        Uses formulas from Siglent programming manual example.

        Args:
            channel (int): Channel number (1-4)

        Returns:
            tuple: (time_data, voltage_data) arrays, or (None, None) on error.
        """
        try:
            preamble = self.get_waveform_preamble(channel)
            if not preamble:
                print(f"Failed to get preamble for channel C{channel}")
                return None, None

            if preamble.get('code_per_div') == 0.0 or preamble.get('wave_array_count', 0) == 0:
                print(f"Warning: Preamble indicates C{channel} might be off or no data acquired (code_per_div={preamble.get('code_per_div')}, points={preamble.get('wave_array_count')}).")
                return None, None

            # Verify connection is still active
            if not self.connected or self.socket is None:
                raise ConnectionError("Connection lost before waveform data request")
            
            # WAV:SOUR is already set by get_waveform_preamble, no need to set again
            # self.send_command(f"WAV:SOUR C{channel}", delay_after=0.05)

            adc_bit = preamble.get('adc_bit', 8)
            comm_type = preamble.get('comm_type', 0)
            expected_bytes_per_point = 1
            if adc_bit > 8:
                if comm_type != 1:
                    print(f"Warning: ADC bits ({adc_bit}) > 8 but preamble comm_type is BYTE. Setting to WORD.")
                    self.send_command("WAV:WIDT WORD")
                    time.sleep(0.1)
                expected_bytes_per_point = 2
            else:
                if comm_type != 0:
                    print(f"Warning: ADC bits ({adc_bit}) <= 8 but preamble comm_type is WORD. Setting to BYTE.")
                    self.send_command("WAV:WIDT BYTE")
                    time.sleep(0.1)
                expected_bytes_per_point = 1

            self.send_command("WAV:DATA?")
            # Increase timeout for large waveform data transfers
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(30.0)  # 30 seconds for large waveform
            response_data = self.read_response(expect_binary=True)
            self.socket.settimeout(original_timeout)

            header_len, data_len, data_start = self.parse_binary_header(response_data)

            if data_len == 0 or len(response_data) < data_start + data_len:
                raise ValueError(f"Incomplete waveform data received. Header: len={data_len}, start={data_start}. Received total: {len(response_data)}")

            raw_data = response_data[data_start : data_start + data_len]

            vdiv = preamble.get('vdiv', 1.0)
            voffset = preamble.get('voffset', 0.0)
            code_per_div = preamble.get('code_per_div', 30.0)
            interval = preamble.get('interval', 1e-6)
            delay = preamble.get('delay', 0.0)
            tdiv_index = preamble.get('tdiv_index', 12)
            points_in_preamble = preamble.get('wave_array_count', 0)

            if 8 < adc_bit <= 12:
                base_code_per_div_8bit = 30.0
                code_per_div = base_code_per_div_8bit * pow(2, adc_bit - 8)
            elif adc_bit == 8:
                code_per_div = 30.0

            if code_per_div <= 0:
                raise ValueError(f"Invalid code_per_div ({code_per_div}) calculated or read from preamble.")

            if 0 <= tdiv_index < len(self.TDIV_ENUM):
                tdiv = self.TDIV_ENUM[tdiv_index]
            else:
                print(f"Warning: Invalid tdiv_index {tdiv_index} from preamble. Using 1us/div.")
                tdiv = 1e-6

            convert_data = []
            actual_points = 0
            if expected_bytes_per_point == 1:
                convert_data = list(raw_data)
                actual_points = len(convert_data)
            elif expected_bytes_per_point == 2:
                num_words = data_len // 2
                for i in range(num_words):
                    data_16bit = struct.unpack('<h', raw_data[i*2:(i*2)+2])[0]
                    convert_data.append(data_16bit)
                actual_points = len(convert_data)
            else:
                raise ValueError(f"Unsupported expected_bytes_per_point: {expected_bytes_per_point}")

            voltage_data = []
            time_data = []

            print(f"Scaling {actual_points} points. Vdiv={vdiv}, Voffs={voffset}, CodePerDiv={code_per_div}, Interval={interval}, Delay={delay}")

            for idx, code_val in enumerate(convert_data):
                signed_code = code_val
                if expected_bytes_per_point == 1:
                    if signed_code > pow(2, 7) - 1:
                        signed_code = signed_code - pow(2, 8)

                voltage = (signed_code / code_per_div * vdiv) - voffset
                voltage_data.append(voltage)

                time_point = -(tdiv * self.HORI_NUM / 2) + idx * interval + delay
                time_data.append(time_point)

            if actual_points != points_in_preamble:
                print(f"Warning: Number of points processed ({actual_points}) differs from preamble count ({points_in_preamble}). Data length: {data_len}")

            return time_data, voltage_data

        except (socket.timeout, ConnectionError, ConnectionAbortedError, ValueError, struct.error, IndexError, TimeoutError) as e:
            print(f"Error getting waveform data for C{channel}: {e}")
            traceback.print_exc()
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred in get_waveform_data for C{channel}: {e}")
            traceback.print_exc()
            return None, None

    def save_waveform_csv(self, channel, filename=None):
        """
        Grab waveform data and save to CSV file

        Args:
            channel (int): Channel number (1-4)
            filename (str): Output filename (optional, auto-generated if not provided)

        Returns:
            str: Filename of saved CSV file, or None on failure.
        """
        try:
            print(f"Grabbing waveform data from channel {channel}...")
            time_data, voltage_data = self.get_waveform_data(channel)

            if time_data is None or voltage_data is None or not time_data:
                print(f"✗ Failed to get waveform data for channel {channel}")
                return None

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                waveform_dir = os.path.join(script_dir, "waveform_data")
                os.makedirs(waveform_dir, exist_ok=True)
                filename = os.path.join(waveform_dir, f"waveform_ch{channel}_{timestamp}.csv")

            with open(filename, 'w', newline='') as f:
                f.write("Time (s),Voltage (V)\n")
                for t, v in zip(time_data, voltage_data):
                    f.write(f"{t:.12e},{v:.6f}\n")

            print(f"✓ Waveform data saved to {filename}")
            print(f"  Data points: {len(time_data)}")
            if time_data:
                print(f"  Time range: {time_data[0]*1e6:.3f} to {time_data[-1]*1e6:.3f} μs")
                print(f"  Voltage range: {min(voltage_data):.4f} to {max(voltage_data):.4f} V")

            return filename

        except Exception as e:
            print(f"Error saving waveform data: {e}")
            traceback.print_exc()
            return None

    def reset(self):
        """Reset oscilloscope to default settings"""
        self.send_command("*RST", delay_after=2.0)
        print("Oscilloscope reset requested.")

    def set_channel_state(self, channel, state):
        """
        Enable/disable channel display
        Manual Command: :CHANnel<n>:SWITch ON|OFF (Page 51)
        """
        state_str = "ON" if state else "OFF"
        self.send_command(f":CHANnel{channel}:SWITch {state_str}")
        # Print a short stack trace so we can see who is toggling channels (helps debug unexpected changes)
        stack = ''.join(traceback.format_stack(limit=6))
        print(f"Channel {channel} display set to {state_str}\nCall stack:\n{stack}")

    def set_voltage_scale(self, channel, volts_per_div):
        """
        Set voltage scale for channel
        Manual Command: :CHANnel<n>:SCALe <scale> (Page 49)
        """
        self.send_command(f":CHANnel{channel}:SCALe {volts_per_div:.4E}")
        stack = ''.join(traceback.format_stack(limit=6))
        print(f"Channel {channel} voltage scale set to {volts_per_div} V/div\nCall stack:\n{stack}")

    def set_channel_coupling(self, channel, coupling):
        """
        Set channel coupling mode
        Manual Command: :CHANnel<n>:COUPling <coupling>
        coupling options: DC|AC|GND
        """
        coupling_upper = coupling.upper()
        if coupling_upper not in ["DC", "AC", "GND"]:
            print(f"Warning: Invalid coupling '{coupling}'. Using DC.")
            coupling_upper = "DC"
        self.send_command(f":CHANnel{channel}:COUPling {coupling_upper}")
        print(f"Channel {channel} coupling set to {coupling_upper}")

    def set_time_scale(self, time_per_div):
        """
        Set time scale
        Manual Command: :TIMebase:SCALe <value> (Page 394)
        """
        self.send_command(f":TIMebase:SCALe {time_per_div:.4E}")
        print(f"Time scale set to {time_per_div} s/div")

    def set_trigger_mode(self, mode="SINGle"):
        """
        Set trigger mode
        Manual Command: :TRIGger:MODE <mode> (Page 400)
        Mode options: SINGle|NORMal|AUTO|FTRIG
        """
        mode_upper = mode.upper()
        if mode_upper not in ["SINGLE", "NORMAL", "AUTO", "FTRIG"]:
            print(f"Warning: Invalid trigger mode '{mode}'. Using SINGLE.")
            mode_upper = "SINGLE"
        self.send_command(f":TRIGger:MODE {mode_upper}")
        print(f"Trigger mode set to {mode_upper}")

    def save_screenshot(self, filename):
        """
        Save oscilloscope screenshot using PRINT? command (page 22)

        Args:
            filename (str): Output filename (should end in .bmp or .png)
        """
        try:
            file_format = "BMP"
            if filename.lower().endswith(".png"):
                file_format = "PNG"

            print(f"Requesting {file_format} screenshot...")
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(30.0)

            self.send_command(f":PRINT? {file_format}")
            screenshot_data = self.read_response(expect_binary=True)

            self.socket.settimeout(original_timeout)

            if not screenshot_data or not screenshot_data.startswith(b'#'):
                print("✗ No valid screenshot data received.")
                return

            header_len, data_len, data_start = self.parse_binary_header(screenshot_data)
            if data_len == 0:
                print("✗ Screenshot data header invalid or zero length.")
                return

            image_bytes = screenshot_data[data_start:]
            if len(image_bytes) < data_len:
                print(f"Warning: Screenshot data incomplete. Expected {data_len} bytes, got {len(image_bytes)}.")
            elif len(image_bytes) > data_len:
                print(f"Warning: Received more screenshot data than expected ({len(image_bytes)} vs {data_len}). Saving truncated data.")
                image_bytes = image_bytes[:data_len]

            with open(filename, 'wb') as f:
                f.write(image_bytes)
            print(f"✓ Screenshot saved to {filename} ({len(image_bytes)} bytes)")

        except (socket.timeout, ConnectionError, ConnectionAbortedError, ValueError, TimeoutError) as e:
            print(f"✗ Error saving screenshot: {e}")
            traceback.print_exc()
            if self.socket:
                try:
                    self.socket.settimeout(original_timeout)
                except:
                    pass


class ScopeManager:
    """
    Manager class for handling multiple oscilloscope connections
    Thread-safe for web application use
    """
    
    def __init__(self):
        self.scopes = {}  # {ip: SocketScope_instance}
        self.active_ip = None
        self.lock = threading.Lock()
    
    def get_scope(self, ip):
        """
        Get existing scope instance or return None
        
        Args:
            ip (str): IP address of scope
            
        Returns:
            SocketScope or None
        """
        with self.lock:
            return self.scopes.get(ip)
    
    def connect_scope(self, ip, port=5025, timeout=15.0):
        """
        Connect to a scope (creates new instance or reuses existing)
        
        Args:
            ip (str): IP address
            port (int): Port number
            timeout (float): Socket timeout
            
        Returns:
            tuple: (success: bool, idn: str or None, error: str or None)
        """
        with self.lock:
            # Check if already connected
            if ip in self.scopes and self.scopes[ip].connected:
                try:
                    idn = self.scopes[ip].query("*IDN?")
                    self.active_ip = ip
                    return True, idn, None
                except:
                    # Connection might be stale, remove it
                    del self.scopes[ip]
            
            # Create new scope instance
            scope = SocketScope(ip_address=ip, port=port, timeout=timeout)
            
            try:
                if scope.connect():
                    # Attempt IDN query with a shorter timeout, but don't fail connection if it times out
                    idn = None
                    original_timeout = None
                    try:
                        original_timeout = scope.socket.gettimeout()
                        scope.socket.settimeout(3.0)
                        idn = scope.query("*IDN?")
                        scope.socket.settimeout(original_timeout)
                        print(f"Device ID: {idn}")
                    except Exception as idn_err:
                        print(f"Warning: IDN query failed (continuing): {idn_err}")
                        idn = "Unknown (IDN timeout)"
                        if original_timeout is not None:
                            try:
                                scope.socket.settimeout(original_timeout)
                            except:
                                pass
                    self.scopes[ip] = scope
                    self.active_ip = ip
                    return True, idn, None
                else:
                    return False, None, "Failed to connect to oscilloscope"
            except Exception as e:
                return False, None, str(e)
    
    def disconnect_scope(self, ip):
        """
        Disconnect from a scope
        
        Args:
            ip (str): IP address
            
        Returns:
            tuple: (success: bool, error: str or None)
        """
        with self.lock:
            if ip in self.scopes:
                try:
                    self.scopes[ip].disconnect()
                    del self.scopes[ip]
                    if self.active_ip == ip:
                        self.active_ip = None
                    return True, None
                except Exception as e:
                    return False, str(e)
            return True, None  # Already disconnected
    
    def get_active_scope(self):
        """Get the currently active scope instance"""
        with self.lock:
            if self.active_ip and self.active_ip in self.scopes:
                return self.scopes[self.active_ip]
            return None

