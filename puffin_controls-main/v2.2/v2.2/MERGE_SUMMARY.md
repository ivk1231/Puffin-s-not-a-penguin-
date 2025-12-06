# V2.2 Merge Summary

## ✅ Merge Completed Successfully

Created a new `v2.2` folder that combines the best features from both V2.1 and v2.1 copy.

## What Was Merged

### From V2.1 (Shot Directory System)
✓ `get_current_shot_folder()` - Returns current day's shot folder
✓ `prepare_next_shot_folder()` - Creates next shot and copies settings
✓ Shot directory structure: `pYYMMDD_NN` format
✓ Automatic shot progression after acquisition
✓ Device settings per shot folder
✓ PDF plot generation (300 DPI, multi-channel subplots)

### From v2.1 copy (Multi-Scope Support)
✓ `ScopeManager` class - Manages multiple scope connections
✓ `get_target_scope()` - Helper to extract scope from API requests
✓ IP-based scope identification
✓ Multi-scope frontend UI (alerts, scope cards)
✓ Independent scope configuration and control

### New Features in V2.2
✓ `get_scope_label()` - Assigns scope1, scope2, scope3 labels
✓ Scope-labeled CSV files: `scope1_waveform_241206_01.csv`
✓ Scope-labeled PDF files: `scope1_waveform_241206_01.pdf`
✓ Multi-device JSON array support in device settings
✓ Concurrent data acquisition from multiple scopes to same shot folder

## File Structure

```
v2.2/
├── app.py                    # Merged Flask app (1362 lines)
├── scope_manager.py          # Multi-scope manager (662 lines)
├── requirements.txt          # Dependencies
├── README.md                 # Original documentation
├── MERGE_NOTES.md           # Detailed merge documentation
├── QUICK_REFERENCE.md       # Quick usage guide
├── static/
│   ├── css/
│   │   └── style.css        # Frontend styles (4.6K)
│   └── js/
│       └── app.js           # Multi-scope UI (42K)
├── templates/
│   └── index.html           # Web interface (29K)
└── waveform_data/           # Fallback directory

../shot directory/           # Shot folders (parent directory)
└── pYYMMDD_NN/             # Shot folders with all scope data
```

## Key Changes in app.py

### Added Functions
1. `get_current_shot_folder()` - Shot directory management
2. `prepare_next_shot_folder()` - Auto-create next shot
3. `get_scope_label(ip)` - Generate scope labels

### Modified Functions
1. `acquire_all_channels_worker()` - Save to shot directory with scope labels
2. API endpoints - Updated to use shot directories
3. File download - Check shot folder first, fallback to waveform_data

### API Endpoints Updated
- `/api/upload-json` - Saves to current shot folder
- `/api/write-scope-to-json` - Multi-device array support
- `/api/set-scope-from-json` - Reads from shot folder
- `/api/get-ddg-config` - Reads from shot folder
- `/api/write-ddg-to-json` - Writes to shot folder
- `/api/download/<filename>` - Downloads from shot folder

## CSV File Naming

### Format
`{scope_label}_waveform_{YYMMDD}_{NN}.csv`

### Examples
- `scope1_waveform_241206_01.csv` - Scope 1, December 6, 2024, Shot 1
- `scope2_waveform_241206_01.csv` - Scope 2, December 6, 2024, Shot 1
- `scope3_waveform_241206_02.csv` - Scope 3, December 6, 2024, Shot 2

### Scope Label Assignment
Based on alphabetical order of connected scope IP addresses:
- 192.168.1.10 → scope1
- 192.168.1.11 → scope2
- 192.168.1.12 → scope3

## Testing Checklist

### Before First Use
- [ ] Verify Python dependencies: `pip install -r requirements.txt`
- [ ] Check that shot directory exists: `../shot directory/`
- [ ] Verify scopes are accessible on network

### Single Scope Test
- [ ] Connect to one scope
- [ ] Configure settings
- [ ] Arm and trigger
- [ ] Verify CSV saved as `scope1_waveform_YYMMDD_01.csv`
- [ ] Verify PDF saved as `scope1_waveform_YYMMDD_01.pdf`
- [ ] Check next shot folder created

### Multi-Scope Test
- [ ] Connect to 2-3 scopes
- [ ] Verify each gets scope1, scope2, scope3 labels
- [ ] Arm all scopes
- [ ] Trigger all scopes
- [ ] Verify all CSVs in same shot folder
- [ ] Verify each has correct scope label

### Device Settings Test
- [ ] Upload JSON config
- [ ] Verify saved to shot folder
- [ ] Write scope settings to JSON
- [ ] Verify multi-device array format
- [ ] Load settings from JSON

## Migration Path

### From V2.1
1. Copy your existing `shot directory/` folder to parent of v2.2
2. Run v2.2 application
3. Connect scopes - will use existing shot folders
4. Previous shots remain accessible

### From v2.1 copy
1. Move your `waveform_data/` files if needed
2. Run v2.2 application
3. New acquisitions will use shot directory system
4. Old files remain in waveform_data/ as backup

## Known Limitations
- Scope labels are assigned alphabetically by IP
- Changing IP addresses may change scope labels
- Maximum tested with 3 scopes (should work with more)
- Shot folder created at parent directory level

## Next Steps
1. Test with your oscilloscopes
2. Verify network connectivity
3. Check shot directory creation
4. Validate CSV file naming
5. Test multi-scope acquisition

## Support Files
- `MERGE_NOTES.md` - Detailed technical documentation
- `QUICK_REFERENCE.md` - API usage examples
- `README.md` - Original project documentation

---

**Merge Date:** December 6, 2024
**Status:** ✅ Complete and ready for testing
**Python Version:** 3.x
**Dependencies:** See requirements.txt
