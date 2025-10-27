#!/usr/bin/env python3
"""
Simplified waveform data grabber for Siglent SDS5104X oscilloscope
Only grabs waveform data - no configuration or setup
"""

import socket
import sys
import time
import struct
from datetime import datetime

class WaveformGrabber:
    """Simplified class for grabbing waveform data from Siglent SDS5104X oscilloscope"""
    
    # TDIV enum from manual (time per division values)
    TDIV_ENUM = [100e-12, 200e-12, 500e-12,
                 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
                 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
                 1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
                 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    HORI_NUM = 10  # Number of horizontal divisions (SDS5000X)
    
    def __init__(self, ip_address="192.168.1.10", port=5025, timeout=10.0):
        """
        Initialize socket connection to oscilloscope
        
        Args:
            ip_address (str): IP address of the oscilloscope (default: 192.168.1.10)
            port (int): Port number (default: 5025 for SCPI socket)
            timeout (float): Socket timeout in seconds
        """
        self.ip_address = ip_address
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Establish TCP socket connection to the oscilloscope"""
        try:
            # Create TCP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            
            # Connect to oscilloscope
            print(f"Connecting to oscilloscope at {self.ip_address}:{self.port}...")
            self.socket.connect((self.ip_address, self.port))
            self.connected = True
            print("✓ Connected successfully!")
            
            # Test connection with *IDN? query
            response = self.query("*IDN?")
            print(f"Device ID: {response}")
            return True
            
        except socket.error as e:
            print(f"✗ Failed to connect: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close socket connection"""
        if self.socket:
            self.socket.close()
            self.connected = False
            print("Disconnected from oscilloscope")
    
    def send_command(self, command):
        """
        Send SCPI command to oscilloscope
        
        Args:
            command (str): SCPI command string
        """
        if not self.connected:
            raise Exception("Not connected to oscilloscope")
        
        try:
            # Send command with newline terminator
            cmd_bytes = (command + '\n').encode('utf-8')
            self.socket.sendall(cmd_bytes)
            time.sleep(2.0)  # 2 second delay for command processing (same as original)
            
        except socket.error as e:
            print(f"Error sending command '{command}': {e}")
            raise
    
    def query(self, query_string):
        """
        Send SCPI query and return response
        
        Args:
            query_string (str): SCPI query string
            
        Returns:
            str: Response from oscilloscope
        """
        if not self.connected:
            raise Exception("Not connected to oscilloscope")
        
        try:
            # Send query
            self.send_command(query_string)
            
            # Receive response
            response = b""
            while True:
                data = self.socket.recv(4096)
                if not data:
                    break
                response += data
                if b'\n' in response:
                    break
            
            # Decode and clean response
            response_str = response.decode('utf-8').strip()
            return response_str
            
        except socket.error as e:
            print(f"Error querying '{query_string}': {e}")
            raise
    
    def get_waveform_preamble(self, channel):
        """
        Get waveform preamble data (binary block) for proper scaling
        
        Args:
            channel (int): Channel number (1-4)
            
        Returns:
            dict: Preamble data with all scaling parameters
        """
        try:
            # Set waveform source
            self.send_command(f"WAV:SOUR C{channel}")
            
            # Query preamble data (returns binary block)
            cmd_bytes = ("WAV:PRE?" + '\n').encode('utf-8')
            self.socket.sendall(cmd_bytes)
            time.sleep(2.0)  # 2 second delay for scope to prepare preamble (same as original)
            
            # Receive binary preamble data
            data = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Check if we have enough data (preamble is 346 bytes + header)
                if b'#' in data and len(data) > 400:
                    break
            
            # Parse binary header (#9000000346)
            hash_pos = data.find(b'#')
            if hash_pos == -1:
                raise ValueError("No binary header found in preamble")
            
            # Skip the header and get preamble binary data
            n_digits = int(data[hash_pos + 1:hash_pos + 2])
            preamble_start = hash_pos + 2 + n_digits
            preamble_data = data[preamble_start:]
            
            # Parse preamble binary structure according to manual (pages 670-672)
            # Address 32-33: COMM_TYPE (0=BYTE, 1=WORD)
            comm_type = struct.unpack('<H', preamble_data[0x20:0x22])[0]
            
            # Address 116-119: wave_array_count (number of points)
            wave_array_count = struct.unpack('<I', preamble_data[0x74:0x78])[0]
            
            # Address 156-159: vdiv (vertical scale)
            vdiv = struct.unpack('<f', preamble_data[0x9c:0xa0])[0]
            
            # Address 160-163: voffset (vertical offset)
            voffset = struct.unpack('<f', preamble_data[0xa0:0xa4])[0]
            
            # Address 164-167: code_per_div
            code_per_div = struct.unpack('<f', preamble_data[0xa4:0xa8])[0]
            
            # Address 172-173: adc_bit
            adc_bit = struct.unpack('<H', preamble_data[0xac:0xae])[0]
            
            # Address 176-179: interval (sampling interval)
            interval = struct.unpack('<f', preamble_data[0xb0:0xb4])[0]
            
            # Address 180-187: delay (horizontal offset/trigger delay)
            delay = struct.unpack('<d', preamble_data[0xb4:0xbc])[0]
            
            # Address 324-325: tdiv_index (time division)
            tdiv_index = struct.unpack('<H', preamble_data[0x144:0x146])[0]
            
            # Address 328-331: probe attenuation
            probe_atten = struct.unpack('<f', preamble_data[0x148:0x14c])[0]
            
            # Create preamble dictionary
            preamble = {
                'comm_type': comm_type,           # 0=BYTE, 1=WORD
                'wave_array_count': wave_array_count,  # Number of points
                'vdiv': vdiv,                     # Vertical scale (V/div)
                'voffset': voffset,               # Vertical offset
                'code_per_div': code_per_div,     # Code value per division
                'adc_bit': adc_bit,               # ADC bit depth
                'interval': interval,             # Sampling interval (s)
                'delay': delay,                   # Trigger delay (s)
                'tdiv_index': tdiv_index,         # Time division index
                'probe_atten': probe_atten        # Probe attenuation
            }
            
            return preamble
            
        except Exception as e:
            print(f"Error getting waveform preamble: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def parse_binary_header(self, data):
        """
        Parse binary data header in format #N<Digits><Data>
        
        Args:
            data (bytes): Raw data starting with header
            
        Returns:
            tuple: (header_length, data_length, data_start_index)
        """
        try:
            # Find the '#' character
            hash_pos = data.find(b'#')
            if hash_pos == -1:
                raise ValueError("No binary header found")
            
            # Read the number of digits (N)
            n_digits = int(data[hash_pos + 1:hash_pos + 2])
            
            # Read the actual number of data bytes
            digits_start = hash_pos + 2
            digits_end = digits_start + n_digits
            data_length = int(data[digits_start:digits_end])
            
            # Calculate positions
            header_length = digits_end - hash_pos
            data_start = digits_end
            
            return header_length, data_length, data_start
            
        except Exception as e:
            print(f"Error parsing binary header: {e}")
            return 0, 0, 0
    
    def get_waveform_data(self, channel):
        """
        Get waveform data from specified channel with proper scaling using preamble
        Follows exact formulas from programming manual
        
        Args:
            channel (int): Channel number (1-4)
            
        Returns:
            tuple: (time_data, voltage_data) arrays with proper scaling
        """
        try:
            # Get waveform preamble first
            preamble = self.get_waveform_preamble(channel)
            if not preamble:
                print(f"Failed to get preamble for channel {channel}")
                return None, None
            
            # Check if we got valid preamble data
            if not preamble.get('vdiv'):
                print(f"Channel {channel} is not enabled or no data available")
                return None, None
            
            # Set waveform source
            self.send_command(f"WAV:SOUR C{channel}")
            
            # Get waveform data
            cmd_bytes = ("WAV:DATA?" + '\n').encode('utf-8')
            self.socket.sendall(cmd_bytes)
            time.sleep(2.0)  # 2 second delay for scope to prepare data (same as original)
            
            # Receive data with proper binary header parsing
            data = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                # For binary data, we need to parse the header to know when to stop
                if b'#' in data:
                    # Parse header to get exact data length
                    header_len, data_len, data_start = self.parse_binary_header(data)
                    if data_len > 0 and len(data) >= data_start + data_len:
                        break
            
            # Parse binary header to get exact data
            header_len, data_len, data_start = self.parse_binary_header(data)
            if data_len == 0:
                print("No waveform data received")
                return None, None
            
            # Extract only the waveform data (skip header and terminator)
            raw_data = data[data_start:data_start + data_len]
            
            # Get preamble parameters
            comm_type = preamble.get('comm_type', 0)  # 0=BYTE, 1=WORD
            adc_bit = preamble.get('adc_bit', 8)
            vdiv = preamble.get('vdiv', 1.0)
            voffset = preamble.get('voffset', 0.0)
            code_per_div = preamble.get('code_per_div', 25.0)
            interval = preamble.get('interval', 1e-6)
            delay = preamble.get('delay', 0.0)
            tdiv_index = preamble.get('tdiv_index', 0)
            
            # Get tdiv from enum
            if tdiv_index < len(self.TDIV_ENUM):
                tdiv = self.TDIV_ENUM[tdiv_index]
            else:
                tdiv = 1e-6  # Default fallback
            
            # Convert raw data based on comm_type (BYTE vs WORD)
            convert_data = []
            if comm_type == 0:  # BYTE format (8-bit)
                convert_data = list(raw_data)
            elif comm_type == 1:  # WORD format (16-bit)
                # Unpack 16-bit little-endian data and shift based on ADC bits
                for i in range(0, len(raw_data), 2):
                    if i + 1 < len(raw_data):
                        # LSB first (little-endian)
                        data_16bit = raw_data[i] + raw_data[i+1] * 256
                        # Shift right to get actual ADC value
                        data = data_16bit >> (16 - adc_bit)
                        convert_data.append(data)
            else:
                print(f"Unsupported comm_type: {comm_type}")
                return None, None
            
            # Convert to signed values and calculate voltage
            # Formula from manual: volt_value[idx] = volt_value[idx] / code * float(vdiv) - float(ofst)
            voltage_data = []
            for data in convert_data:
                # Convert to signed value if needed
                if data > pow(2, adc_bit - 1) - 1:
                    data = data - pow(2, adc_bit)
                
                # Calculate voltage using manual formula
                voltage = data / code_per_div * vdiv - voffset
                voltage_data.append(voltage)
            
            # Generate time array using manual formula
            # Formula: time_data = -(float(tdiv) * HORI_NUM / 2) + idx * interval - delay
            time_data = []
            for idx in range(len(voltage_data)):
                time_point = -(tdiv * self.HORI_NUM / 2) + idx * interval - delay
                time_data.append(time_point)
            
            return time_data, voltage_data
            
        except Exception as e:
            print(f"Error getting channel {channel} data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_waveform_csv(self, channel, filename=None):
        """
        Grab waveform data and save to CSV file
        
        Args:
            channel (int): Channel number (1-4)
            filename (str): Output filename (optional, auto-generated if not provided)
            
        Returns:
            str: Filename of saved CSV file
        """
        try:
            # Get waveform data
            print(f"Grabbing waveform data from channel {channel}...")
            time_data, voltage_data = self.get_waveform_data(channel)
            
            if time_data is None or voltage_data is None:
                print("Failed to get waveform data")
                return None
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"waveform_ch{channel}_{timestamp}.csv"
            
            # Save to CSV
            with open(filename, 'w') as f:
                f.write("Time (s),Voltage (V)\n")
                for t, v in zip(time_data, voltage_data):
                    f.write(f"{t},{v}\n")
            
            print(f"✓ Waveform data saved to {filename}")
            print(f"  Data points: {len(time_data)}")
            print(f"  Time range: {time_data[0]*1e6:.2f} to {time_data[-1]*1e6:.2f} μs")
            print(f"  Voltage range: {min(voltage_data):.3f} to {max(voltage_data):.3f} V")
            
            return filename
            
        except Exception as e:
            print(f"Error saving waveform data: {e}")
            return None


def grab_waveform(channel=1, ip_address="192.168.1.10", filename=None):
    """
    Simple function to grab waveform data from oscilloscope
    
    Args:
        channel (int): Channel number (1-4)
        ip_address (str): Oscilloscope IP address
        filename (str): Output CSV filename (optional)
        
    Returns:
        str: Filename of saved CSV file, or None if failed
    """
    grabber = WaveformGrabber(ip_address=ip_address)
    
    try:
        # Connect to oscilloscope
        if not grabber.connect():
            print("Failed to connect to oscilloscope")
            return None
        
        # Grab and save waveform data
        result = grabber.save_waveform_csv(channel, filename)
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    finally:
        grabber.disconnect()


def main():
    """Main function for command line usage"""
    print("Siglent SDS5104X Waveform Grabber")
    print("=" * 40)
    
    # Default settings
    channel = 1
    ip_address = "192.168.1.10"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            channel = int(sys.argv[1])
        except ValueError:
            print("Invalid channel number. Using channel 1.")
    
    if len(sys.argv) > 2:
        ip_address = sys.argv[2]
    
    print(f"Grabbing waveform from channel {channel}")
    print(f"Oscilloscope IP: {ip_address}")
    print()
    
    # Grab waveform data
    result = grab_waveform(channel=channel, ip_address=ip_address)
    
    if result:
        print(f"\n✓ Success! Waveform saved to: {result}")
    else:
        print("\n✗ Failed to grab waveform data")
        print("\nTroubleshooting:")
        print("1. Check oscilloscope is powered on")
        print("2. Verify network connection")
        print("3. Confirm IP address is correct")
        print("4. Ensure channel is enabled on oscilloscope")


if __name__ == "__main__":
    main()
