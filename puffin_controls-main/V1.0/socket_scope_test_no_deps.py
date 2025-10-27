#!/usr/bin/env python3
"""
Socket-based communication test script for Siglent SDS5104X oscilloscope
No external dependencies - uses only standard Python libraries
"""

import socket
import sys
import time
import struct
from datetime import datetime

class SocketScope:
    """Socket-based communication class for Siglent SDS5104X oscilloscope"""
    
    def __init__(self, ip_address="192.168.1.10", port=5025, timeout=5.0):
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
            time.sleep(2.0)  # 2 second delay for command processing
            
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
    
    def get_device_info(self):
        """Get device identification information"""
        return self.query("*IDN?")
    
    def reset(self):
        """Reset oscilloscope to default settings"""
        self.send_command("*RST")
        time.sleep(2)  # Wait for reset to complete
        print("Oscilloscope reset to default settings")
    
    def set_channel_state(self, channel, state):
        """
        Enable/disable channel
        
        Args:
            channel (int): Channel number (1-4)
            state (bool): True to enable, False to disable
        """
        state_str = "ON" if state else "OFF"
        self.send_command(f"CHAN{channel}:SWIT {state_str}")
        print(f"Channel {channel} {'enabled' if state else 'disabled'}")
    
    def set_voltage_scale(self, channel, volts_per_div):
        """
        Set voltage scale for channel
        
        Args:
            channel (int): Channel number (1-4)
            volts_per_div (float): Volts per division
        """
        self.send_command(f"CHAN{channel}:SCAL {volts_per_div}V")
        print(f"Channel {channel} voltage scale set to {volts_per_div} V/div")
    
    def set_time_scale(self, time_per_div):
        """
        Set time scale
        
        Args:
            time_per_div (float): Time per division in seconds
        """
        self.send_command(f"TDIV {time_per_div}S")
        print(f"Time scale set to {time_per_div} s/div")
    
    def set_trigger_mode(self, mode="SINGLE"):
        """
        Set trigger mode
        
        Args:
            mode (str): Trigger mode (SINGLE, AUTO, NORM)
        """
        self.send_command(f"TRIG_MODE {mode}")
        print(f"Trigger mode set to {mode}")
    
    def get_channel_data_simple(self, channel):
        """
        Get basic waveform data from specified channel (simplified version)
        
        Args:
            channel (int): Channel number (1-4)
            
        Returns:
            tuple: (time_data, voltage_data) arrays
        """
        try:
            # Set waveform source
            self.send_command(f"WAV:SOUR C{channel}")
            
            # Get waveform data (simplified)
            self.send_command("WAV:DATA?")
            
            # Receive data
            data = b""
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b'\n' in chunk:
                    break
            
            # Remove header and convert to list
            data = data[16:-2]  # Remove SCPI header and terminator
            voltage_raw = list(data)
            
            # Simple conversion (this is a basic version)
            voltage_data = [(v - 128) / 30.0 for v in voltage_raw]  # Basic conversion
            
            # Generate time array (simplified)
            time_data = [i * 1e-6 for i in range(len(voltage_data))]  # 1μs per sample
            
            return time_data, voltage_data
            
        except Exception as e:
            print(f"Error getting channel {channel} data: {e}")
            return None, None
    
    def save_screenshot(self, filename):
        """
        Save oscilloscope screenshot
        
        Args:
            filename (str): Output filename
        """
        try:
            # Increase timeout for large data transfer
            original_timeout = self.socket.gettimeout()
            self.socket.settimeout(30.0)  # 30 second timeout for screenshot
            
            self.send_command("SCDP")
            data = b""
            start_time = time.time()
            
            while True:
                try:
                    chunk = self.socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    # Check if we've been receiving data for too long
                    if time.time() - start_time > 25:  # 25 second limit
                        print("Screenshot transfer taking too long, stopping...")
                        break
                except socket.timeout:
                    print("Screenshot transfer timeout")
                    break
            
            # Restore original timeout
            self.socket.settimeout(original_timeout)
            
            if data:
                with open(filename, 'wb') as f:
                    f.write(data)
                print(f"Screenshot saved to {filename} ({len(data)} bytes)")
            else:
                print("No screenshot data received")
            
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            # Restore original timeout in case of error
            try:
                self.socket.settimeout(original_timeout)
            except:
                pass


def test_basic_communication():
    """Test basic socket communication with oscilloscope"""
    print("=" * 60)
    print("SOCKET-BASED OSCILLOSCOPE COMMUNICATION TEST")
    print("=" * 60)
    
    # Initialize scope connection
    scope = SocketScope(ip_address="192.168.1.10", port=5025)
    
    try:
        # Connect to oscilloscope
        if not scope.connect():
            print("Failed to connect. Please check:")
            print("1. Oscilloscope is powered on")
            print("2. Network connection is established")
            print("3. IP address is correct (configured: 192.168.1.10)")
            print("4. Port 5025 is accessible")
            return False
        
        # Test basic queries
        print("\n--- Testing Basic Queries ---")
        print(f"Device ID: {scope.get_device_info()}")
        
        # Test channel control
        print("\n--- Testing Channel Control ---")
        scope.set_channel_state(1, True)
        scope.set_voltage_scale(1, 1.0)  # 1 V/div
        scope.set_time_scale(1e-6)  # 1 μs/div
        
        # Test trigger
        print("\n--- Testing Trigger ---")
        scope.set_trigger_mode("SINGLE")
        
        # Test data acquisition
        print("\n--- Testing Data Acquisition ---")
        time_data, voltage_data = scope.get_channel_data_simple(1)
        if time_data is not None:
            print(f"Acquired {len(time_data)} data points")
            print(f"Time range: {time_data[0]*1e6:.2f} to {time_data[-1]*1e6:.2f} μs")
            print(f"Voltage range: {min(voltage_data):.3f} to {max(voltage_data):.3f} V")
        
        # Test screenshot
        print("\n--- Testing Screenshot ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = f"scope_screenshot_{timestamp}.bmp"
        scope.save_screenshot(screenshot_file)
        
        print("\n✓ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        scope.disconnect()


def test_waveform_data():
    """Test waveform data acquisition and display"""
    print("\n" + "=" * 60)
    print("WAVEFORM DATA ACQUISITION TEST")
    print("=" * 60)
    
    scope = SocketScope(ip_address="192.168.1.10", port=5025)
    
    try:
        if not scope.connect():
            return False
        
        # Configure scope for measurement
        scope.set_channel_state(1, True)
        scope.set_voltage_scale(1, 1.0)
        scope.set_time_scale(10e-6)  # 10 μs/div
        scope.set_trigger_mode("SINGLE")
        
        # Acquire data
        print("Acquiring waveform data...")
        time_data, voltage_data = scope.get_channel_data_simple(1)
        
        if time_data is not None and len(time_data) > 0:
            print(f"✓ Acquired {len(time_data)} data points")
            print(f"Time range: {time_data[0]*1e6:.2f} to {time_data[-1]*1e6:.2f} μs")
            print(f"Voltage range: {min(voltage_data):.3f} to {max(voltage_data):.3f} V")
            
            # Save data to CSV file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = f"waveform_data_{timestamp}.csv"
            with open(csv_file, 'w') as f:
                f.write("Time (s),Voltage (V)\n")
                for t, v in zip(time_data, voltage_data):
                    f.write(f"{t},{v}\n")
            print(f"Data saved to {csv_file}")
            
            return True
        else:
            print("No data acquired")
            return False
            
    except Exception as e:
        print(f"Waveform test failed: {e}")
        return False
        
    finally:
        scope.disconnect()


def main():
    """Main test function"""
    print("Siglent SDS5104X Socket Communication Test")
    print("Using TCP sockets instead of NI-VISA")
    print(f"Test started at: {datetime.now()}")
    
    # Run basic communication test
    # success = test_basic_communication()
    
    # if success:
        # Run waveform data test
    test_waveform_data()
    
    print(f"\nTest completed at: {datetime.now()}")
    print("\nNote: Make sure the oscilloscope IP address is correct")
    print("Configured IP: 192.168.1.10")
    print("Default Port: 5025")


if __name__ == "__main__":
    main()
