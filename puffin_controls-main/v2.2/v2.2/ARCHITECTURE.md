# V2.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            V2.2 ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐
│   Web Browser     │
│   (Frontend UI)   │
│                   │
│ - Multi-scope UI  │
│ - Alerts system   │
│ - Scope cards     │
└─────────┬─────────┘
          │ HTTP Requests
          │ (API calls with IP param)
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Flask App (app.py)                                   │
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐ │
│  │  API Endpoints       │  │  Helper Functions    │  │  Shot Directory  │ │
│  │                      │  │                      │  │  Management      │ │
│  │ • /api/connect       │  │ • get_target_scope() │  │                  │ │
│  │ • /api/arm           │  │ • get_scope_label()  │  │ • get_current_   │ │
│  │ • /api/config/*      │  │                      │  │   shot_folder()  │ │
│  │ • /api/download/*    │  │                      │  │ • prepare_next_  │ │
│  │                      │  │                      │  │   shot_folder()  │ │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘ │
│                                                                              │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ScopeManager (scope_manager.py)                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ scopes = {                                                          │   │
│  │   "192.168.1.10": SocketScope(...),  ◄─── scope1                   │   │
│  │   "192.168.1.11": SocketScope(...),  ◄─── scope2                   │   │
│  │   "192.168.1.12": SocketScope(...)   ◄─── scope3                   │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Methods:                                                                    │
│  • connect_scope(ip, port)                                                   │
│  • disconnect_scope(ip)                                                      │
│  • get_scope(ip)                                                             │
│  • get_active_scope()                                                        │
└────────┬────────────┬────────────┬────────────────────────────────────────┘
         │            │            │
         ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ SocketScope  │ │ SocketScope  │ │ SocketScope  │
│ (Scope 1)    │ │ (Scope 2)    │ │ (Scope 3)    │
│              │ │              │ │              │
│ IP: .10      │ │ IP: .11      │ │ IP: .12      │
│ Port: 5025   │ │ Port: 5025   │ │ Port: 5025   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       │ TCP Socket     │ TCP Socket     │ TCP Socket
       │ (SCPI)         │ (SCPI)         │ (SCPI)
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Oscilloscope │ │ Oscilloscope │ │ Oscilloscope │
│ #1           │ │ #2           │ │ #3           │
│              │ │              │ │              │
│ SDS5104X     │ │ SDS5104X     │ │ SDS5104X     │
│ 192.168.1.10 │ │ 192.168.1.11 │ │ 192.168.1.12 │
└──────────────┘ └──────────────┘ └──────────────┘


DATA FLOW AFTER TRIGGER:
═════════════════════════

┌──────────────┐
│ Scope 1      │ ──► Waveform Data ──┐
│ Triggers     │                      │
└──────────────┘                      │
                                      │
┌──────────────┐                      │
│ Scope 2      │ ──► Waveform Data ──┼──► Shot Folder
│ Triggers     │                      │    p241206_01/
└──────────────┘                      │
                                      │    ├── scope1_waveform_241206_01.csv
┌──────────────┐                      │    ├── scope1_waveform_241206_01.pdf
│ Scope 3      │ ──► Waveform Data ──┘    ├── scope2_waveform_241206_01.csv
│ Triggers     │                           ├── scope2_waveform_241206_01.pdf
└──────────────┘                           ├── scope3_waveform_241206_01.csv
                                           ├── scope3_waveform_241206_01.pdf
                                           └── device settings.json

SHOT DIRECTORY HIERARCHY:
═════════════════════════

shot directory/  (parent folder)
│
├── p241206_01/  ◄── Current shot (Dec 6, 2024, Shot 1)
│   ├── device settings.json
│   ├── scope1_waveform_241206_01.csv
│   ├── scope1_waveform_241206_01.pdf
│   ├── scope2_waveform_241206_01.csv
│   ├── scope2_waveform_241206_01.pdf
│   ├── scope3_waveform_241206_01.csv
│   └── scope3_waveform_241206_01.pdf
│
├── p241206_02/  ◄── Auto-created after shot 1 completes
│   └── device settings.json (copied from shot 1)
│
└── p241207_01/  ◄── First shot of next day
    └── ...


SCOPE LABEL ASSIGNMENT:
═══════════════════════

Connected Scopes (sorted by IP):
┌────────────────┬─────────────┐
│ IP Address     │ Label       │
├────────────────┼─────────────┤
│ 192.168.1.10   │ scope1      │
│ 192.168.1.11   │ scope2      │
│ 192.168.1.12   │ scope3      │
└────────────────┴─────────────┘

Label determined by: get_scope_label(ip)
- Sorts all connected IPs alphabetically
- Assigns labels in order: scope1, scope2, scope3, ...
- Consistent as long as same IPs are used


CONCURRENT ACQUISITION:
═══════════════════════

Timeline:
─────────────────────────────────────────────────────►

t=0s    Scope 1 armed  │
        Scope 2 armed  │  All scopes waiting
        Scope 3 armed  │  for trigger event

t=10s   External trigger event

t=10.1s Scope 1 acquires data
        Scope 2 acquires data  ◄── Concurrent
        Scope 3 acquires data

t=10.5s Scope 1 saves: scope1_waveform_241206_01.csv
        Scope 2 saves: scope2_waveform_241206_01.csv
        Scope 3 saves: scope3_waveform_241206_01.csv
                       └─► All to same shot folder

t=11s   Shot p241206_02/ auto-created for next acquisition


KEY FEATURES:
═════════════

✓ Multi-scope support (3+ scopes simultaneously)
✓ Shot directory organization (pYYMMDD_NN format)
✓ Scope-labeled filenames (scope1, scope2, scope3)
✓ Automatic shot progression
✓ Device settings per shot
✓ PDF plot generation (300 DPI)
✓ Backward compatible with single scope
✓ Thread-safe concurrent operations
```
