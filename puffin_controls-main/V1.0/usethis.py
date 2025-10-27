#!/usr/bin/env python3
"""
Socket-based communication test script for Siglent SDS5104X oscilloscope
Includes corrected waveform data acquisition based on programming manual, set trigger on external and rising edge, set trigger level to 0.5V, set trigger mode to single, set waveform source to channel 1, and set voltage scale to 1V/div.
"""

import socket
import sys
import time
import struct
from datetime import datetime
import traceback # Added for detailed error printing

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: matplotlib/pandas not available. Install with: pip install matplotlib pandas")

class SocketScope:
    """Socket-based communication class for Siglent SDS5104X oscilloscope"""

    # TDIV enum from manual (time per division values) - Ensure this matches your scope model
    TDIV_ENUM = [100e-12, 200e-12, 500e-12,
                 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
                 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
                 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
                 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    HORI_NUM = 10  # Number of horizontal divisions (SDS5000X)

    def __init__(self, ip_address="192.168.1.10", port=5025, timeout=15.0): # Increased default timeout
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

    def connect(self):
        """Establish TCP socket connection to the oscilloscope"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            print(f"Connecting to oscilloscope at {self.ip_address}:{self.port}...")
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            print("✓ Connected successfully!")
            response = self.query("*IDN?")
            print(f"Device ID: {response}")
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
                print(f"Error closing socket: {e}") # Handle potential errors on close
            finally:
                self.socket = None # Ensure socket is None after closing
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
            # print(f"Sending: {command}") # Debug print
            self.socket.sendall(cmd_bytes)
            if delay_after > 0:
                time.sleep(delay_after) # Small delay needed for some commands

        except (socket.error, socket.timeout) as e:
            print(f"Error sending command '{command}': {e}")
            self.connected = False # Assume connection lost on send error
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
                    # Connection closed by peer
                    self.connected = False
                    raise ConnectionAbortedError("Socket connection closed by oscilloscope during read")

                response += chunk

                if expect_binary and not header_parsed and b'#' in response:
                    try:
                        _, data_len, data_start = self.parse_binary_header(response)
                        header_parsed = True
                        # print(f"Parsed header: data_len={data_len}, data_start={data_start}, current_len={len(response)}") # Debug print
                    except ValueError:
                        # Header might be incomplete, continue reading
                        pass

                if expect_binary and header_parsed:
                    # Check if we have received all expected binary data
                    if len(response) >= data_start + data_len:
                        # print("Binary data complete.") # Debug print
                        break
                elif not expect_binary:
                     # For text commands, look for newline
                     # Be careful: binary data might contain \n
                    if b'\n' in chunk: # Check in chunk to avoid re-scanning full buffer
                        # print("Text data complete (newline found).") # Debug print
                        break

        except socket.timeout:
            print("Error reading response: Socket timed out.")
            # If we expected binary data and got some, return what we have? Or raise?
            if expect_binary and data_len > 0 and len(response) > data_start :
                 print(f"Warning: Timeout during binary read. Expected {data_len}, got {len(response)-data_start}.")
                 # Let's return what we have, might be usable or indicate error upstream
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

        Args:
            query_string (str): SCPI query string

        Returns:
            str: Response from oscilloscope (decoded and stripped)
        """
        # Send query first, then read response
        self.send_command(query_string)
        response_bytes = self.read_response(expect_binary=False) # Standard query expects text
        response_str = response_bytes.decode('utf-8', errors='ignore').strip()
        # print(f"Query '{query_string}' -> Response: '{response_str}'") # Debug print
        return response_str

    def get_waveform_preamble(self, channel):
        """
        Get waveform preamble data (binary block) for proper scaling

        Args:
            channel (int): Channel number (1-4)

        Returns:
            dict: Parsed preamble data, or {} on error.
        """
        PREAMBLE_DATA_LENGTH = 346 # Fixed length specified in manual (page 670)

        try:
            self.send_command(f"WAV:SOUR C{channel}", delay_after=0.05) # Need source set first
            self.send_command("WAV:PRE?") # Send the query

            # Read the binary response specifically
            preamble_response = self.read_response(expect_binary=True)

            # Parse binary header (#9<9-Digits>...)
            header_len, data_len, preamble_start = self.parse_binary_header(preamble_response)

            if data_len == 0:
                 raise ValueError("Preamble header parsing failed or reported zero length.")

            if data_len != PREAMBLE_DATA_LENGTH:
                print(f"Warning: Preamble header reported {data_len} bytes, expected {PREAMBLE_DATA_LENGTH}. Check manual/model.")
                # Adjust expected length if needed, but proceed cautiously
                # PREAMBLE_DATA_LENGTH = data_len # Uncomment if necessary, but risky

            if len(preamble_response) < preamble_start + PREAMBLE_DATA_LENGTH:
                 raise ValueError(f"Incomplete preamble data received. Got {len(preamble_response)-preamble_start}, expected {PREAMBLE_DATA_LENGTH}")

            preamble_data = preamble_response[preamble_start : preamble_start + PREAMBLE_DATA_LENGTH]

            # Parse preamble binary structure using '<' for little-endian
            # Offsets are from the start of the binary *data*, not the '#'
            preamble = {}
            preamble['comm_type'] = struct.unpack('<h', preamble_data[0x20:0x22])[0] # short (h) not H
            preamble['wave_array_count'] = struct.unpack('<l', preamble_data[0x74:0x78])[0] # long (l) not I
            preamble['vdiv'] = struct.unpack('<f', preamble_data[0x9c:0xa0])[0]
            preamble['voffset'] = struct.unpack('<f', preamble_data[0xa0:0xa4])[0]
            preamble['code_per_div'] = struct.unpack('<f', preamble_data[0xa4:0xa8])[0]
            preamble['adc_bit'] = struct.unpack('<h', preamble_data[0xac:0xae])[0] # short (h) not H
            preamble['interval'] = struct.unpack('<f', preamble_data[0xb0:0xb4])[0]
            preamble['delay'] = struct.unpack('<d', preamble_data[0xb4:0xbc])[0]
            preamble['tdiv_index'] = struct.unpack('<h', preamble_data[0x144:0x146])[0] # short (h) not H
            preamble['probe_atten'] = struct.unpack('<f', preamble_data[0x148:0x14c])[0]
            # Add other fields if needed using manual offsets and types

            # Simple sanity check
            if preamble['code_per_div'] == 0:
                 print("Warning: Parsed 'code_per_div' is zero, scaling will fail.")
                 # This often indicates channel is off or preamble parsing error
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

            # Check if N digit exists
            if len(data) < hash_pos + 2:
                 raise ValueError("Incomplete header: Missing N digit")
            n_digits = int(data[hash_pos + 1:hash_pos + 2])
            if n_digits == 0:
                 raise ValueError("Invalid header: N cannot be 0")

            # Check if <Digits> exist
            digits_start = hash_pos + 2
            digits_end = digits_start + n_digits
            if len(data) < digits_end:
                 raise ValueError("Incomplete header: Missing <Digits>")

            data_length = int(data[digits_start:digits_end])
            total_header_length = digits_end - hash_pos # Length of #N<Digits> part
            data_start_index = digits_end

            return total_header_length, data_length, data_start_index

        except (ValueError, IndexError) as e:
            # Don't print error here, might just be incomplete read buffer
            # print(f"Could not parse binary header: {e} - Data: {data[:20]}") # Debug
            return 0, 0, 0 # Indicate failure

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
            # --- 1. Get Preamble ---
            preamble = self.get_waveform_preamble(channel)
            if not preamble:
                print(f"Failed to get preamble for channel C{channel}")
                return None, None

            # Check if channel appears disabled based on typical preamble values
            if preamble.get('code_per_div') == 0.0 or preamble.get('wave_array_count', 0) == 0:
                 print(f"Warning: Preamble indicates C{channel} might be off or no data acquired (code_per_div={preamble.get('code_per_div')}, points={preamble.get('wave_array_count')}).")
                 # Check explicitly if needed: status = self.query(f"C{channel}:TRA?") -> returns ON/OFF
                 return None, None

            # --- 2. Query Waveform Data ---
            self.send_command(f"WAV:SOUR C{channel}", delay_after=0.05) # Redundant? Preamble already set source

            # Set data width based on ADC bits if > 8
            adc_bit = preamble.get('adc_bit', 8)
            comm_type = preamble.get('comm_type', 0) # Use comm_type from preamble
            expected_bytes_per_point = 1
            if adc_bit > 8:
                if comm_type != 1: # If preamble says BYTE but ADC>8, force WORD
                    print(f"Warning: ADC bits ({adc_bit}) > 8 but preamble comm_type is BYTE. Setting to WORD.")
                    self.send_command("WAV:WIDT WORD")
                    time.sleep(0.1)
                expected_bytes_per_point = 2
            else:
                if comm_type != 0: # If preamble says WORD but ADC<=8, force BYTE
                    print(f"Warning: ADC bits ({adc_bit}) <= 8 but preamble comm_type is WORD. Setting to BYTE.")
                    self.send_command("WAV:WIDT BYTE")
                    time.sleep(0.1)
                expected_bytes_per_point = 1


            # --- 3. Read Waveform Data Binary Block ---
            self.send_command("WAV:DATA?")
            # Read response specifically expecting binary
            response_data = self.read_response(expect_binary=True)

            # Parse binary header to get actual data
            header_len, data_len, data_start = self.parse_binary_header(response_data)

            if data_len == 0 or len(response_data) < data_start + data_len:
                raise ValueError(f"Incomplete waveform data received. Header: len={data_len}, start={data_start}. Received total: {len(response_data)}")

            raw_data = response_data[data_start : data_start + data_len]

            # --- 4. Process Raw Data ---
            vdiv = preamble.get('vdiv', 1.0)
            voffset = preamble.get('voffset', 0.0)
            code_per_div = preamble.get('code_per_div', 30.0) # Using 30 default based on manual examples if parse failed
            interval = preamble.get('interval', 1e-6)
            delay = preamble.get('delay', 0.0)
            tdiv_index = preamble.get('tdiv_index', 12) # Default to 1us/div index
            points_in_preamble = preamble.get('wave_array_count', 0)

            # Recalculate code_per_div based on adc_bit for SDS5000X/SDS2000X+/HD (Manual page 744, Table 1 values imply this relationship)
            # This seems more reliable than the direct preamble value sometimes.
            if 8 < adc_bit <= 12:
                 # Manual table shows code_per_div ~25 for 8bit, ~400 for 12bit etc. It scales approx with 2^(adc_bit - 8) * (25 or 30)
                 # Let's try the common 30 base for 5000X/2000X+
                 base_code_per_div_8bit = 30.0 # From manual example page 744/Table 1 for these models
                 code_per_div = base_code_per_div_8bit * pow(2, adc_bit - 8)
                 # print(f"Recalculated code_per_div for {adc_bit}-bit: {code_per_div}") # Debug print
            elif adc_bit == 8:
                 code_per_div = 30.0 # Ensure consistency for 8-bit


            # Check if code_per_div is valid
            if code_per_div <= 0:
                 raise ValueError(f"Invalid code_per_div ({code_per_div}) calculated or read from preamble.")


            # Get tdiv from enum using index from preamble
            if 0 <= tdiv_index < len(self.TDIV_ENUM):
                tdiv = self.TDIV_ENUM[tdiv_index]
            else:
                print(f"Warning: Invalid tdiv_index {tdiv_index} from preamble. Using 1us/div.")
                tdiv = 1e-6

            convert_data = []
            actual_points = 0
            if expected_bytes_per_point == 1: # BYTE
                convert_data = list(raw_data)
                actual_points = len(convert_data)
            elif expected_bytes_per_point == 2: # WORD
                num_words = data_len // 2
                for i in range(num_words):
                    # LSB first (little-endian) based on manual examples
                    data_16bit = struct.unpack('<h', raw_data[i*2:(i*2)+2])[0]
                    # Data seems to be already signed in WORD format based on testing/examples
                    # No need to shift if adc_bit=16? Check scope behavior.
                    # If ADC bit is less than 16, it might be left-justified. Need confirmation.
                    # Assuming direct signed 16-bit for now if WORD is used.
                    # Revisit if scaling is off for >8bit ADC modes.
                    # Let's trust adc_bit from preamble for sign conversion boundary
                    # Manual's python example implies signed conversion needed even for WORD? Re-implementing that.
                    # data_16bit = raw_data[i] + raw_data[i + 1] * 256
                    # if data_16bit > pow(2, 15) - 1: # Assuming 16 bit signed max
                    #     data_16bit = data_16bit - pow(2, 16)
                    # Shift might be needed if it's truly left-aligned and unsigned/offset binary
                    # data = data_16bit >> (16 - adc_bit) # If left-aligned
                    # convert_data.append(data) # Use shifted value if needed

                    convert_data.append(data_16bit) # Use direct signed unpack for now

                actual_points = len(convert_data)
            else:
                raise ValueError(f"Unsupported expected_bytes_per_point: {expected_bytes_per_point}")


            # --- 5. Scale Data ---
            voltage_data = []
            time_data = []

            # Use manual's formula structure
            # Voltage = Code * (Vdiv / CodePerDiv) - Voffset
            # Time = -(Tdiv * Grid/2) + Index * Interval - Delay
            # Note: Manual example's Time calculation seems off. Standard SCPI is usually:
            # Time = XOrigin + (Index - XRef) * XInc
            # Let's use the standard SCPI interpretation based on extracted preamble values:
            # XOrigin = delay, XInc = interval, XRef = 0 (usually)

            print(f"Scaling {actual_points} points. Vdiv={vdiv}, Voffs={voffset}, CodePerDiv={code_per_div}, Interval={interval}, Delay={delay}") # Debug print

            for idx, code_val in enumerate(convert_data):
                # Apply scaling formula from manual (page 744, Step 3 adjusted)
                # Ensure code_val is treated as signed based on ADC bits if using BYTE
                signed_code = code_val
                if expected_bytes_per_point == 1: # BYTE
                     # Convert 8-bit unsigned byte to signed value centered around 0
                     if signed_code > pow(2, 7) - 1:
                          signed_code = signed_code - pow(2, 8)

                # Voltage calculation
                voltage = (signed_code / code_per_div * vdiv) - voffset
                voltage_data.append(voltage)

                # Time calculation (Standard SCPI way using preamble fields)
                # time_point = delay + idx * interval # Assuming XRef is 0
                # Using manual's example formula structure:
                time_point = -(tdiv * self.HORI_NUM / 2) + idx * interval + delay # Corrected delay sign based on manual text
                time_data.append(time_point)


            # Sanity check points count
            if actual_points != points_in_preamble:
                print(f"Warning: Number of points processed ({actual_points}) differs from preamble count ({points_in_preamble}). Data length: {data_len}")

            return time_data, voltage_data

        except (socket.timeout, ConnectionError, ConnectionAbortedError, ValueError, struct.error, IndexError, TimeoutError) as e:
            print(f"Error getting waveform data for C{channel}: {e}")
            traceback.print_exc()
            return None, None
        except Exception as e: # Catch any other unexpected errors
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
                filename = f"waveform_ch{channel}_{timestamp}.csv"

            with open(filename, 'w', newline='') as f: # Added newline='' for csv standard
                f.write("Time (s),Voltage (V)\n")
                for t, v in zip(time_data, voltage_data):
                    f.write(f"{t:.12e},{v:.6f}\n") # Use scientific notation for time

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

    # --- Other methods (reset, set_channel_state etc.) remain largely the same ---
    # --- but need updates for correct SCPI commands ---

    def reset(self):
        """Reset oscilloscope to default settings"""
        self.send_command("*RST", delay_after=2.0) # Use longer delay for reset
        print("Oscilloscope reset requested.")

    def set_channel_state(self, channel, state):
        """
        Enable/disable channel display
        Manual Command: :CHANnel<n>:SWITch ON|OFF (Page 51)
        """
        state_str = "ON" if state else "OFF"
        # Correct command from manual: :CHANnel<n>:SWITch
        self.send_command(f":CHANnel{channel}:SWITch {state_str}")
        print(f"Channel {channel} display set to {state_str}")

    def set_voltage_scale(self, channel, volts_per_div):
        """
        Set voltage scale for channel
        Manual Command: :CHANnel<n>:SCALe <scale> (Page 49)
        """
        # Correct command from manual: :CHANnel<n>:SCALe
        self.send_command(f":CHANnel{channel}:SCALe {volts_per_div:.4E}") # Use scientific notation
        print(f"Channel {channel} voltage scale set to {volts_per_div} V/div")

    def set_time_scale(self, time_per_div):
        """
        Set time scale
        Manual Command: :TIMebase:SCALe <value> (Page 394)
        """
        # Correct command from manual: :TIMebase:SCALe
        self.send_command(f":TIMebase:SCALe {time_per_div:.4E}") # Use scientific notation
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
        # Correct command from manual: :TRIGger:MODE
        self.send_command(f":TRIGger:MODE {mode_upper}")
        print(f"Trigger mode set to {mode_upper}")

    def save_screenshot(self, filename):
        """
        Save oscilloscope screenshot using PRINT? command (page 22)

        Args:
            filename (str): Output filename (should end in .bmp or .png)
        """
        try:
            file_format = "BMP" # Default to BMP
            if filename.lower().endswith(".png"):
                 file_format = "PNG"

            print(f"Requesting {file_format} screenshot...")
            # Use longer timeout specifically for this potentially long transfer
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(30.0)

            # Send command and read binary response
            self.send_command(f":PRINT? {file_format}")
            screenshot_data = self.read_response(expect_binary=True)

            # Restore original timeout
            self.socket.settimeout(original_timeout)

            # Check if we got data (read_response might return partial on timeout)
            if not screenshot_data or not screenshot_data.startswith(b'#'):
                 print("✗ No valid screenshot data received.")
                 return

            # Parse header to verify length (optional but good practice)
            header_len, data_len, data_start = self.parse_binary_header(screenshot_data)
            if data_len == 0:
                 print("✗ Screenshot data header invalid or zero length.")
                 return

            image_bytes = screenshot_data[data_start:]
            if len(image_bytes) < data_len:
                 print(f"Warning: Screenshot data incomplete. Expected {data_len} bytes, got {len(image_bytes)}.")
            elif len(image_bytes) > data_len:
                 print(f"Warning: Received more screenshot data than expected ({len(image_bytes)} vs {data_len}). Saving truncated data.")
                 image_bytes = image_bytes[:data_len] # Truncate

            with open(filename, 'wb') as f:
                f.write(image_bytes)
            print(f"✓ Screenshot saved to {filename} ({len(image_bytes)} bytes)")

        except (socket.timeout, ConnectionError, ConnectionAbortedError, ValueError, TimeoutError) as e:
            print(f"✗ Error saving screenshot: {e}")
            traceback.print_exc()
            # Restore original timeout in case of error during read
            try:
                if self.socket: self.socket.settimeout(original_timeout)
            except: pass
        except Exception as e:
            print(f"✗ Unexpected error saving screenshot: {e}")
            traceback.print_exc()
            try:
                 if self.socket: self.socket.settimeout(original_timeout)
            except: pass


# --- Plotting function (from output.txt suggestions) ---

def plot_csv(filename):
    """
    Reads a waveform CSV file and plots the data.
    
    Args:
        filename (str): The path to the CSV file.
    """
    if not PLOTTING_AVAILABLE:
        print("Cannot plot: matplotlib and pandas are not installed.")
        print("Install with: pip install matplotlib pandas")
        return
    
    if not filename:
        print("No filename provided for plotting.")
        return
    
    try:
        print(f"\nPlotting data from {filename}...")
        # Read the CSV file using pandas
        # Assumes header is 'Time (s),Voltage (V)'
        df = pd.read_csv(filename)
        
        # Check if columns exist
        if "Time (s)" not in df.columns or "Voltage (V)" not in df.columns:
            print(f"Error: CSV file '{filename}' does not contain expected 'Time (s)' or 'Voltage (V)' columns.")
            return
        
        # Extract time and voltage data
        time_data = df["Time (s)"]
        voltage_data = df["Voltage (V)"]
        
        # Create the plot
        plt.figure(figsize=(12, 6)) # Adjust figure size as needed
        plt.plot(time_data * 1e6, voltage_data) # Plot time in microseconds
        plt.title(f"Waveform Data from {filename}")
        plt.xlabel("Time (μs)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.margins(x=0.01, y=0.1) # Add some margin
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # Use scientific notation if needed
        plt.tight_layout() # Adjust layout
        plt.show() # Display the plot
        
    except FileNotFoundError:
        print(f"Error: Could not find file {filename} to plot.")
    except Exception as e:
        print(f"Error plotting CSV file {filename}: {e}")
        traceback.print_exc()


# --- Test functions remain largely the same, but call get_waveform_data ---

def test_waveform_data():
    """Test waveform data acquisition and display"""
    print("\n" + "=" * 60)
    print("WAVEFORM DATA ACQUISITION TEST")
    print("=" * 60)

    # Use the specific IP address of your scope
    scope = SocketScope(ip_address="192.168.1.10", port=5025, timeout=15.0)

    try:
        if not scope.connect():
            return False

        # Configure scope for measurement (Example settings)
        scope.set_channel_state(1, True)
        scope.set_voltage_scale(2, 1.0)     # 1 V/div
        scope.set_time_scale(1e-6)         # 10 μs/div
        # Example trigger setup (Edge trigger on C1, level 0.5V, Rising slope)
        scope.send_command(":TRIGger:TYPE EDGE")
        scope.send_command(":TRIGger:EDGE:SOURce EX")
        scope.send_command(":TRIGger:EDGE:SLOPe RISing")
        scope.send_command(":TRIGger:EDGE:LEVel 0.5") # Assuming 0.5V trigger level

        scope.set_trigger_mode("SINGle")    # Arm for single capture
        print("Waiting for trigger...")
        time.sleep(2) # Give scope time to potentially trigger and stop

        # Check trigger status (Optional but helpful)
        trig_status = scope.query(":TRIGger:STATus?")
        print(f"Trigger status after arming: {trig_status}")
        if trig_status != "Stop":
            print("Warning: Scope did not trigger or stop as expected. Attempting data grab anyway.")
            # Forcing a trigger might be needed if signal isn't present: scope.send_command("TRIG:MODE FTRIG")


        # Acquire data using the corrected method
        print("Acquiring waveform data...")
        # Replaced get_channel_data_simple with get_waveform_data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"waveform_data_ch1_{timestamp}.csv"
        saved_file = scope.save_waveform_csv(1, filename=csv_file) # Channel 1

        if saved_file:
            print(f"✓ Waveform test potentially successful. Data saved to {saved_file}")
            # Automatically plot the waveform
            plot_csv(saved_file)
            return True
        else:
            print("✗ Waveform data acquisition or save failed.")
            return False

    except Exception as e:
        print(f"Waveform test failed: {e}")
        traceback.print_exc()
        return False

    finally:
        scope.disconnect()


def main():
    """Main test function"""
    print("Siglent SDS5104X Socket Communication Test")
    print("Using TCP sockets instead of NI-VISA")
    print(f"Test started at: {datetime.now()}")

    test_waveform_data()

    print(f"\nTest completed at: {datetime.now()}")
    print("\nNote: Make sure the oscilloscope IP address is correct")
    print("Configured IP: 192.168.1.10") # Make sure this matches your scope
    print("Default Port: 5025")


if __name__ == "__main__":
    main()