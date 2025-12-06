# Quick Reference: V2.2 Key Features

## Multi-Scope CSV File Naming

### Single Scope (Old V2.1)
```
scope1_waveform_241206_01.csv
```

### Multiple Scopes (New V2.2)
If you connect 3 scopes (e.g., 192.168.1.10, 192.168.1.11, 192.168.1.12):

```
scope1_waveform_241206_01.csv  (from 192.168.1.10)
scope2_waveform_241206_01.csv  (from 192.168.1.11)
scope3_waveform_241206_01.csv  (from 192.168.1.12)
```

All in the same shot folder: `../shot directory/p241206_01/`

## Scope Label Assignment

Scopes are labeled based on **alphabetical order of IP addresses**:

| IP Address     | Scope Label |
|----------------|-------------|
| 192.168.1.10   | scope1      |
| 192.168.1.11   | scope2      |
| 192.168.1.12   | scope3      |

**Note:** The label is determined when the scope connects and remains consistent as long as the same IPs are used.

## Shot Directory Structure

```
shot directory/
├── p241206_01/                    # Shot 1 of Dec 6, 2024
│   ├── device settings.json       # Multi-device settings array
│   ├── scope1_waveform_241206_01.csv
│   ├── scope1_waveform_241206_01.pdf
│   ├── scope2_waveform_241206_01.csv
│   ├── scope2_waveform_241206_01.pdf
│   ├── scope3_waveform_241206_01.csv
│   └── scope3_waveform_241206_01.pdf
│
├── p241206_02/                    # Shot 2 (auto-created after shot 1)
│   ├── device settings.json       # Copied from shot 1
│   └── ...
│
└── p241207_01/                    # First shot of next day
    └── ...
```

## Example Workflow

### 1. Connect Three Scopes
```javascript
// Scope 1
POST /api/connect
{
  "ip": "192.168.1.10",
  "port": 5025
}

// Scope 2
POST /api/connect
{
  "ip": "192.168.1.11",
  "port": 5025
}

// Scope 3  
POST /api/connect
{
  "ip": "192.168.1.12",
  "port": 5025
}
```

### 2. Configure Each Scope
```javascript
// Configure channel 1 on scope2
POST /api/config/channel
{
  "ip": "192.168.1.11",
  "channel": 1,
  "state": true,
  "voltage_scale": 2.0
}
```

### 3. Arm Scopes for Trigger
```javascript
// Arm scope1
POST /api/arm
{
  "ip": "192.168.1.10"
}

// Arm scope2
POST /api/arm
{
  "ip": "192.168.1.11"
}

// Arm scope3
POST /api/arm
{
  "ip": "192.168.1.12"
}
```

### 4. After Trigger Event
All three scopes save to the same shot folder:
```
p241206_01/
├── scope1_waveform_241206_01.csv  
├── scope2_waveform_241206_01.csv  
├── scope3_waveform_241206_01.csv  
└── ...
```

Next shot folder automatically created: `p241206_02/`

## Important Notes

### Device Settings JSON Format
Supports multiple devices in an array:
```json
[
  {
    "instrument": "Siglent SDS5104X",
    "ip": "192.168.1.10",
    "settings": { ... }
  },
  {
    "instrument": "Siglent SDS5104X", 
    "ip": "192.168.1.11",
    "settings": { ... }
  },
  {
    "instrument": "SRS DG645",
    "ip": "192.168.1.20",
    "settings": { ... }
  }
]
```

### Backward Compatibility
- Old waveform_data/ directory still exists for fallback
- Single-scope operation works exactly as before
- Existing shot directories are preserved

### PDF Plots
- Multi-channel plots (one subplot per channel)
- 300 DPI for publication quality
- Saved alongside CSV files
- Named consistently: `scope1_waveform_241206_01.pdf`

## Common Tasks

### Download Waveform Data
```
GET /api/download/scope1_waveform_241206_01.csv
GET /api/download/scope2_waveform_241206_01.csv
GET /api/download/scope1_waveform_241206_01.pdf
```

### Save Scope Settings to JSON
```javascript
POST /api/write-scope-to-json
{
  "ip": "192.168.1.10"
}
```
Saves to: `shot directory/p241206_01/device settings.json`

### Load Settings from JSON
```javascript
POST /api/set-scope-from-json
{
  "ip": "192.168.1.10"
}
```
Reads from: `shot directory/p241206_01/device settings.json`
