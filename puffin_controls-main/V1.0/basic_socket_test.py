#!/usr/bin/env python3
"""
Basic socket test for Siglent SDS5104X oscilloscope
No external dependencies - uses only standard Python libraries
"""

import socket
import time

def basic_socket_test():
    """Basic socket communication test with detailed output"""
    
    # Your oscilloscope configuration
    IP_ADDRESS = "192.168.1.10"
    PORT = 5025
    
    print("=" * 60)
    print("BASIC SOCKET TEST FOR SIGLENT SDS5104X")
    print("=" * 60)
    print(f"Target: {IP_ADDRESS}:{PORT}")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Create socket
        print("1. Creating TCP socket...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10.0)  # 10 second timeout
        print("   ✓ Socket created successfully")
        
        # Connect
        print(f"2. Connecting to {IP_ADDRESS}:{PORT}...")
        s.connect((IP_ADDRESS, PORT))
        print("   ✓ Connected successfully!")
        print()
        
        # Test device identification
        print("3. Testing device identification...")
        s.sendall(b"*IDN?\n")
        time.sleep(2.0)
        
        response = s.recv(1024)
        device_info = response.decode('utf-8').strip()
        print(f"   Device ID: {device_info}")
        print()
        
        # Test basic commands
        print("4. Testing basic SCPI commands...")
        commands = [
            ("*RST", "Reset oscilloscope"),
            ("CHAN1:SWIT ON", "Enable channel 1"),
            ("CHAN1:SCAL 1V", "Set 1V/div scale"),
            ("TDIV 1E-6S", "Set 1μs/div time scale"),
            ("TRIG_MODE SINGLE", "Set single trigger mode")
        ]
        
        for cmd, description in commands:
            print(f"   Sending: {cmd} ({description})")
            s.sendall((cmd + "\n").encode('utf-8'))
            time.sleep(2.0)
            print(f"   ✓ Command sent successfully")
        
        print()
        
        # Test query commands
        print("5. Testing query commands...")
        queries = [
            ("CHAN1:SCAL?", "Get voltage scale"),
            ("TDIV?", "Get time scale"),
            ("CHAN1:SWIT?", "Get channel state")
        ]
        
        for query, description in queries:
            print(f"   Querying: {query} ({description})")
            s.sendall((query + "\n").encode('utf-8'))
            time.sleep(2.0)
            
            # Try to receive response
            try:
                response = s.recv(1024)
                if response:
                    result = response.decode('utf-8').strip()
                    print(f"   Response: {result}")
                else:
                    print(f"   No response received")
            except Exception as e:
                print(f"   Error receiving response: {e}")
        
        print()
        print("✓ All tests completed successfully!")
        print(f"Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except socket.timeout:
        print("✗ Connection timeout!")
        print("   Possible causes:")
        print("   - Oscilloscope is not powered on")
        print("   - Wrong IP address")
        print("   - Network connectivity issues")
        print("   - Firewall blocking port 5025")
        
    except socket.error as e:
        print(f"✗ Connection failed: {e}")
        print()
        print("Troubleshooting steps:")
        print("1. Verify oscilloscope is powered on")
        print("2. Check IP address: 192.168.1.10")
        print("3. Ensure PC is on same network (192.168.1.x)")
        print("4. Test network connectivity:")
        print(f"   ping {IP_ADDRESS}")
        print("5. Check if port 5025 is accessible:")
        print(f"   telnet {IP_ADDRESS} 5025")
        print("6. Verify SCPI socket is enabled on oscilloscope")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        
    finally:
        if 's' in locals():
            s.close()
            print("Socket closed")

if __name__ == "__main__":
    basic_socket_test()

