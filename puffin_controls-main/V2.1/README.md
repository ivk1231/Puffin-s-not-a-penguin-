# Flask Web Interface for Oscilloscope Control

A web-based interface for controlling the Siglent SDS5104X oscilloscope, providing the same functionality as `usethis.py` but with a browser-based UI.

## Features

- **Web-based Control**: Control the oscilloscope through a modern web interface
- **Real-time Status**: Live connection status and acquisition progress
- **Waveform Acquisition**: Trigger-based waveform capture with proper scaling
- **Data Visualization**: Automatic plot generation using matplotlib
- **File Management**: Download CSV files and view plots in the browser
- **Threaded Operations**: Non-blocking operations for responsive UI
- **Multi-device Ready**: Architecture supports multiple oscilloscopes (future)

## Quick Start

1. **Install Dependencies**:
   ```bash
   cd V1.1
   python3 -m pip install -r requirements.txt
   ```

2. **Start the Application**:
   ```bash
   python3 app.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5001`

## Server Management

### Starting the Server
```bash
cd V1.1
python3 app.py
```

### Stopping the Server
- **If running in foreground**: Press `Ctrl + C`
- **If running in background**: 
  ```bash
  pkill -f "python3 app.py"
  # Or kill by port:
  kill -9 $(lsof -t -i:5001)
  ```

### Check Server Status
```bash
lsof -i :5001
```

## Usage

### Connection
1. Enter the oscilloscope IP address (default: 192.168.1.10)
2. Click "Connect" to establish connection
3. Status indicator shows connection state and device ID

### Configuration
1. **Channel Settings**: Select target channel, enable/disable, set voltage scale
2. **Timebase**: Set time scale (s/div)
3. **Trigger**: Configure trigger type, source, slope, level, and mode
4. Click "Apply Settings" to send configuration to scope

### Acquisition
1. Click "Acquire Waveform" to start acquisition
2. Monitor progress in the status area
3. **If trigger doesn't fire**: Use "Reset Trigger" button to re-arm the scope
4. View generated plot when complete
5. Download CSV file using "Download CSV" button

### Troubleshooting
- **First acquisition fails**: This is normal - the scope may trigger before the acquisition process starts
- **Use "Reset Trigger"**: Click this button to re-arm the scope for another attempt
- **Check trigger status**: The terminal shows real-time trigger status updates
- **External trigger**: Ensure your external trigger signal is connected and configured

## API Endpoints

- `POST /api/connect` - Connect to oscilloscope
- `POST /api/disconnect` - Disconnect from oscilloscope
- `POST /api/config/channel` - Configure channel settings
- `POST /api/config/timebase` - Configure timebase settings
- `POST /api/config/trigger` - Configure trigger settings
- `POST /api/acquire` - Start waveform acquisition
- `GET /api/acquire/status/<task_id>` - Get acquisition status
- `POST /api/reset-trigger` - Reset trigger state and re-arm scope
- `GET /api/download/<filename>` - Download CSV file
- `GET /api/plot/<filename>` - Serve plot image

## File Structure

```
V1.1/
├── app.py                 # Flask application
├── scope_manager.py       # SocketScope and ScopeManager classes
├── test_basic.py         # Basic functionality tests
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Custom styling
│   └── js/
│       └── app.js        # Frontend JavaScript
└── waveform_data/        # Generated CSV and plot files (gitignored)
```

## Testing

Run the basic test suite:
```bash
python3 test_basic.py
```

This tests:
- ScopeManager functionality
- Flask app creation and routes
- Plot generation with dummy data

## Dependencies

- Flask 3.0.0 - Web framework
- matplotlib 3.8.0 - Plotting
- pandas 2.1.0 - Data handling
- Python 3.12+

## Notes

- The application uses the same tested SocketScope logic from `usethis.py`
- All oscilloscope operations run in background threads to keep the UI responsive
- Waveform data is saved to `waveform_data/` directory
- The interface is designed for lab use with clean, professional styling
- **Event-driven trigger detection**: Uses polling instead of fixed delays for reliable triggering
- **Multiple trigger states supported**: Recognizes "Stop", "FStop", and "Trig'd" statuses
- **Real-time status updates**: Terminal shows live trigger status during acquisition
