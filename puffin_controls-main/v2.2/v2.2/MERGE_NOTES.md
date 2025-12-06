# V2.2 - Merged Oscilloscope Control Application

## Overview
This is the merged version combining features from V2.1 and v2.1 copy:
- **Shot Directory System** (from V2.1)
- **Multi-Scope Support** (from v2.1 copy)
- **Scope-Labeled CSV Files** (new feature)

## Key Features

### 1. Shot Directory System
- Waveform data is saved to organized shot directories
- Directory format: `pYYMMDD_NN` (e.g., `p241206_01` for December 6, 2024, shot 1)
- Located in: `../shot directory/` (one level up from v2.2 folder)
- Device settings (JSON) are saved per shot
- Automatic progression to next shot after acquisition

### 2. Multi-Scope Support
- Connect to multiple oscilloscopes simultaneously
- Each scope is identified by its IP address
- API endpoints accept `ip` parameter to target specific scopes
- Concurrent monitoring and data acquisition from multiple scopes

### 3. Scope-Labeled File Naming
**CSV Files:**
- Format: `scope1_waveform_YYMMDD_NN.csv`
- Examples:
  - `scope1_waveform_241206_01.csv` - First scope, shot 1
  - `scope2_waveform_241206_01.csv` - Second scope, shot 1
  - `scope3_waveform_241206_01.csv` - Third scope, shot 1

**PDF Plots:**
- Format: `scope1_waveform_YYMMDD_NN.pdf`
- High-quality plots (300 DPI) with individual subplots per channel
- Saved alongside CSV files in shot directory

### 4. Scope Labeling Logic
- Scopes are labeled based on connection order
- Sorted alphabetically by IP address for consistent labeling
- `get_scope_label(ip)` function determines scope number (scope1, scope2, scope3, etc.)
- First connected scope (alphabetically by IP) = scope1, etc.

## File Structure

```
v2.2/
├── app.py                    # Main Flask application (merged)
├── scope_manager.py          # Multi-scope manager
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── static/
│   ├── css/
│   │   └── style.css        # Frontend styles
│   └── js/
│       └── app.js           # Frontend JavaScript (multi-scope UI)
├── templates/
│   └── index.html           # Web interface
└── waveform_data/           # Fallback directory (for backward compatibility)

../shot directory/           # Shot folders (one level up)
├── p241206_01/             # Shot folder example
│   ├── device settings.json
│   ├── scope1_waveform_241206_01.csv
│   ├── scope1_waveform_241206_01.pdf
│   ├── scope2_waveform_241206_01.csv
│   └── scope2_waveform_241206_01.pdf
└── p241206_02/             # Next shot folder
    └── ...
```

## Usage

### Running the Application
```bash
cd v2.2
python app.py
```
Access at: `http://localhost:5001`

### Connecting Multiple Scopes
1. In the web UI, connect to first scope (e.g., 192.168.1.10) - becomes **scope1**
2. Connect to second scope (e.g., 192.168.1.11) - becomes **scope2**
3. Connect to third scope (e.g., 192.168.1.12) - becomes **scope3**

### Acquiring Data
1. Configure each scope's settings independently
2. Arm each scope separately
3. When triggered, data is saved with appropriate scope labels
4. Each scope's data goes to the same shot folder with different labels

### Shot Management
- New shot folder created automatically after each acquisition
- Previous shot's device settings are copied to new shot
- Manual shot progression available through API

## API Changes

### Multi-Scope Endpoints
All configuration and control endpoints now accept an `ip` parameter:
- `/api/config/channel` - Configure specific scope's channel
- `/api/config/trigger` - Configure specific scope's trigger
- `/api/arm` - Arm specific scope
- `/api/disconnect` - Disconnect specific scope

### Shot Directory Endpoints
- `/api/upload-json` - Saves to current shot folder
- `/api/write-scope-to-json` - Writes to current shot folder (preserves multi-device array)
- `/api/set-scope-from-json` - Reads from current shot folder
- `/api/download/<filename>` - Downloads from current shot folder (or fallback)

## Migration Notes

### From V2.1
- Existing shot directory structure is preserved
- Old CSV naming (`scope1_waveform_YYMMDD_NN.csv`) remains compatible
- Add multi-scope support to expand capabilities

### From v2.1 copy
- Waveforms now saved to shot directories instead of `waveform_data/`
- CSV naming updated to include shot information
- Device settings organized by shot

## Technical Implementation

### Key Functions

**`get_current_shot_folder()`**
- Returns path to current day's active shot folder
- Creates folder if it doesn't exist

**`prepare_next_shot_folder()`**
- Creates next shot folder
- Copies device settings from current shot
- Called automatically after each acquisition

**`get_scope_label(scope_ip)`**
- Returns scope label (scope1, scope2, scope3)
- Based on alphabetically sorted list of connected IPs

**`get_target_scope(request_data)`**
- Helper to extract scope from API request
- Uses IP address to identify scope
- Returns scope instance or error response

### Acquisition Worker
- Modified to use shot directory system
- Generates scope-specific filenames
- Creates multi-channel PDF plots
- Automatically prepares next shot folder

## Dependencies
See `requirements.txt`:
- Flask
- matplotlib
- pandas
- (Other dependencies from original versions)

## Future Enhancements
- Web UI for shot browser
- Multi-scope synchronized triggering
- Comparative waveform analysis across scopes
- Shot annotation and metadata
