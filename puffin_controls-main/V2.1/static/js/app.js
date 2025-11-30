/**
 * Frontend JavaScript for Oscilloscope Control
 * Handles all API communication and UI updates
 */

class OscilloscopeController {
    constructor() {
        this.connected = false;
        this.currentTaskId = null;
        this.pollingInterval = null;
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Connection controls
        document.getElementById('connectBtn').addEventListener('click', () => this.connect());
        document.getElementById('disconnectBtn').addEventListener('click', () => this.disconnect());
        
        // Configuration controls
        document.getElementById('applySettingsBtn').addEventListener('click', () => this.applySettings());
        
        // Arm controls
        document.getElementById('armBtn').addEventListener('click', () => this.armScope());
    }
    
    async connect() {
        const ip = document.getElementById('ipAddress').value;
        const port = parseInt(document.getElementById('port').value);
        
        this.setConnectionState('connecting', 'Connecting...');
        
        try {
            const response = await fetch('/api/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ip, port })
            });
            
            const data = await response.json();
            
            console.log('Connection response:', data);
            
            if (data.status === 'connected') {
                this.connected = true;
                this.setConnectionState('connected', `Connected to ${data.idn}`);
                this.updateUIForConnection(true);
                this.showToast('Successfully connected to oscilloscope', 'success');
                
                // Populate GUI with current scope settings
                if (data.settings) {
                    this.populateGUIFromSettings(data.settings);
                }
                
                // Check if scope is already armed
                console.log('Armed status:', data.armed_status);
                if (data.armed_status && data.armed_status.armed && data.armed_status.task_id) {
                    console.log('Scope is armed! Starting polling with task_id:', data.armed_status.task_id);
                    this.currentTaskId = data.armed_status.task_id;
                    this.startArmPolling();
                    this.updateArmUI('armed', 'Scope was already armed, monitoring for trigger...');
                    this.showToast('Scope is armed and waiting for trigger', 'info');
                } else {
                    console.log('Scope is not armed');
                }
            } else {
                this.setConnectionState('error', `Error: ${data.error}`);
                this.showToast(`Connection failed: ${data.error}`, 'error');
            }
        } catch (error) {
            this.setConnectionState('error', `Error: ${error.message}`);
            this.showToast(`Connection error: ${error.message}`, 'error');
        }
    }
    
    populateGUIFromSettings(settings) {
        // Populate channel settings
        if (settings.channels) {
            for (let ch = 1; ch <= 4; ch++) {
                const chKey = `CH${ch}`;
                if (settings.channels[chKey]) {
                    const chSettings = settings.channels[chKey];
                    
                    if (chSettings.vdiv !== undefined) {
                        document.getElementById(`ch${ch}_vdiv`).value = chSettings.vdiv;
                    }
                    if (chSettings.offset !== undefined) {
                        document.getElementById(`ch${ch}_offset`).value = chSettings.offset;
                    }
                    if (chSettings.display !== undefined) {
                        document.getElementById(`ch${ch}_state`).checked = chSettings.display;
                    }
                }
            }
        }
        
        // Populate trigger settings
        if (settings.trigger) {
            if (settings.trigger.source) {
                document.getElementById('triggerSource').value = settings.trigger.source;
            }
            if (settings.trigger.mode) {
                document.getElementById('triggerMode').value = settings.trigger.mode;
            }
            if (settings.trigger.slope) {
                const slopeValue = settings.trigger.slope.toUpperCase();
                if (slopeValue.includes('RIS')) {
                    document.getElementById('slopeRising').checked = true;
                } else {
                    document.getElementById('slopeFalling').checked = true;
                }
            }
        }
        
        // Populate timebase settings
        if (settings.timebase && settings.timebase.horizontal_div) {
            document.getElementById('timeScale').value = parseFloat(settings.timebase.horizontal_div);
        }
        
        console.log('GUI populated with scope settings:', settings);
    }
    
    async disconnect() {
        this.setConnectionState('disconnecting', 'Disconnecting...');
        
        try {
            const response = await fetch('/api/disconnect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            this.connected = false;
            this.setConnectionState('disconnected', 'Disconnected');
            this.updateUIForConnection(false);
            this.showToast('Disconnected from oscilloscope', 'info');
            
        } catch (error) {
            this.setConnectionState('error', `Error: ${error.message}`);
            this.showToast(`Disconnect error: ${error.message}`, 'error');
        }
    }
    
    async applySettings() {
        if (!this.connected) {
            this.showToast('Not connected to oscilloscope', 'error');
            return;
        }
        
        // Gather configuration data
        const timeScale = parseFloat(document.getElementById('timeScale').value);
        const triggerMode = document.getElementById('triggerMode').value;
        const triggerSource = document.getElementById('triggerSource').value;
        const triggerSlope = document.querySelector('input[name="triggerSlope"]:checked').value;
        const triggerLevel = parseFloat(document.getElementById('triggerLevel').value);
        
        try {
            // Apply settings for all channels
            for (let ch = 1; ch <= 4; ch++) {
                const state = document.getElementById(`ch${ch}_state`).checked;
                const voltageScale = parseFloat(document.getElementById(`ch${ch}_vdiv`).value);
                const offset = parseFloat(document.getElementById(`ch${ch}_offset`).value);
                
                await fetch('/api/config/channel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        channel: ch, 
                        state, 
                        voltage_scale: voltageScale,
                        offset: offset
                    })
                });
            }
            
            // Apply timebase settings
            await fetch('/api/config/timebase', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ time_scale: timeScale })
            });
            
            // Apply trigger settings
            const triggerResponse = await fetch('/api/config/trigger', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    mode: triggerMode, 
                    source: triggerSource, 
                    slope: triggerSlope, 
                    level: triggerLevel 
                })
            });
            
            const triggerData = await triggerResponse.json();
            
            // Check if trigger settings were skipped due to active monitoring
            if (triggerData.status === 'skipped') {
                this.showToast('Channel and timebase settings applied. Trigger settings skipped while monitoring.', 'warning');
            } else {
                this.showToast('Settings applied successfully', 'success');
            }
            
        } catch (error) {
            this.showToast(`Failed to apply settings: ${error.message}`, 'error');
        }
    }
    
    async armScope() {
        if (!this.connected) {
            this.showToast('Not connected to oscilloscope', 'error');
            return;
        }
        
        try {
            // Arm the scope
            const response = await fetch('/api/arm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.task_id) {
                this.currentTaskId = data.task_id;
                this.startArmPolling();
                this.updateArmUI('armed', data.message || 'Scope armed, waiting for trigger...');
                
                // Show different toast based on whether scope was already armed
                if (data.already_armed) {
                    this.showToast('Scope was already armed - monitoring for trigger', 'info');
                } else {
                    this.showToast('Scope armed - waiting for trigger', 'success');
                }
            } else {
                this.showToast(`Failed to arm scope: ${data.error}`, 'error');
            }
            
        } catch (error) {
            this.showToast(`Arm error: ${error.message}`, 'error');
        }
    }
    
    startArmPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/arm/status/${this.currentTaskId}`);
                const data = await response.json();
                
                this.updateArmStatus(data);
                
                if (data.status === 'complete') {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                    this.currentTaskId = null;
                    
                    // Auto-download CSV
                    if (data.result && data.result.csv_file) {
                        this.downloadFile(data.result.csv_file);
                    }
                    
                } else if (data.status === 'error' || data.status === 'timeout') {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                    this.currentTaskId = null;
                }
                
            } catch (error) {
                console.error('Polling error:', error);
                clearInterval(this.pollingInterval);
                this.pollingInterval = null;
                this.currentTaskId = null;
            }
        }, 500); // Poll every 500ms
    }
    
    updateArmStatus(data) {
        if (data.status === 'armed') {
            this.updateArmUI('armed', data.progress || 'Scope armed, waiting for trigger...');
            
        } else if (data.status === 'triggered') {
            this.updateArmUI('triggered', data.progress || 'Trigger detected! Acquiring data...');
            
        } else if (data.status === 'complete') {
            this.updateArmUI('complete', data.progress || 'Acquisition complete!');
            
            if (data.result) {
                this.displayArmResults(data.result);
            }
            
            this.showToast('Acquisition complete! Downloading CSV...', 'success');
            
        } else if (data.status === 'timeout') {
            this.updateArmUI('idle', 'Idle');
            this.showToast(data.error || 'Trigger timeout', 'warning');
            
        } else if (data.status === 'error') {
            this.updateArmUI('idle', 'Idle');
            this.showToast(`Error: ${data.error}`, 'error');
        }
    }
    
    displayArmResults(result) {
        const armInfo = document.getElementById('armInfo');
        const channelsAcquired = document.getElementById('channelsAcquired');
        const csvFileName = document.getElementById('csvFileName');
        
        // Display acquisition information
        channelsAcquired.textContent = result.channels ? result.channels.join(', ') : 'N/A';
        csvFileName.textContent = result.csv_file || 'N/A';
        armInfo.style.display = 'block';
    }
    
    downloadFile(filename) {
        // Create a temporary anchor element to trigger download
        const a = document.createElement('a');
        a.href = `/api/download/${filename}`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    updateArmUI(state, message) {
        const armBtn = document.getElementById('armBtn');
        const armStatus = document.getElementById('armStatus');
        const armStatusText = document.getElementById('armStatusText');
        
        armStatusText.textContent = message;
        
        switch (state) {
            case 'armed':
                armBtn.disabled = true;
                armBtn.textContent = 'Armed...';
                armStatus.className = 'alert alert-warning';
                break;
                
            case 'triggered':
                armBtn.disabled = true;
                armBtn.textContent = 'Acquiring...';
                armStatus.className = 'alert alert-info';
                break;
                
            case 'complete':
                armBtn.disabled = false;
                armBtn.innerHTML = '<i class="bi bi-crosshair"></i> Arm Scope';
                armStatus.className = 'alert alert-success';
                // Reset to idle after 3 seconds
                setTimeout(() => {
                    this.updateArmUI('idle', 'Idle');
                    document.getElementById('armInfo').style.display = 'none';
                }, 3000);
                break;
                
            case 'idle':
            default:
                armBtn.disabled = this.connected ? false : true;
                armBtn.innerHTML = '<i class="bi bi-crosshair"></i> Arm Scope';
                armStatus.className = 'alert alert-secondary';
                break;
        }
    }
    
    setConnectionState(state, message) {
        const statusElement = document.getElementById('connectionStatus');
        const deviceIdElement = document.getElementById('deviceId');
        
        statusElement.className = 'badge';
        
        switch (state) {
            case 'connecting':
                statusElement.className += ' bg-warning';
                statusElement.textContent = 'Connecting...';
                break;
            case 'connected':
                statusElement.className += ' bg-success';
                statusElement.textContent = 'Connected';
                deviceIdElement.textContent = message;
                break;
            case 'disconnecting':
                statusElement.className += ' bg-warning';
                statusElement.textContent = 'Disconnecting...';
                break;
            case 'disconnected':
                statusElement.className += ' bg-secondary';
                statusElement.textContent = 'Disconnected';
                deviceIdElement.textContent = '';
                break;
            case 'error':
                statusElement.className += ' bg-danger';
                statusElement.textContent = 'Error';
                deviceIdElement.textContent = message;
                break;
        }
    }
    
    updateUIForConnection(connected) {
        document.getElementById('connectBtn').disabled = connected;
        document.getElementById('disconnectBtn').disabled = !connected;
        document.getElementById('applySettingsBtn').disabled = !connected;
        document.getElementById('armBtn').disabled = !connected;
    }
    
    showToast(message, type = 'info') {
        const toastElement = document.getElementById('toast');
        const toastBody = document.getElementById('toastBody');
        
        toastBody.textContent = message;
        
        // Set toast color based on type
        toastElement.className = 'toast';
        if (type === 'success') {
            toastElement.classList.add('text-bg-success');
        } else if (type === 'error') {
            toastElement.classList.add('text-bg-danger');
        } else if (type === 'warning') {
            toastElement.classList.add('text-bg-warning');
        } else {
            toastElement.classList.add('text-bg-info');
        }
        
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new OscilloscopeController();

    // Function to update UI fields from config data
    function updateUIFromConfig(config) {
        // Update connection fields
        if (config.ip) {
            document.getElementById('ipAddress').value = config.ip;
        }
        if (config.port) {
            document.getElementById('port').value = config.port;
        }
        
        // Update channel settings - populate all channels in the table
        const channels = config.settings?.channels || {};
        for (let ch = 1; ch <= 4; ch++) {
            const chKey = `CH${ch}`;
            if (channels[chKey]) {
                const chSettings = channels[chKey];
                
                // Set voltage scale
                if (chSettings.vdiv !== undefined) {
                    document.getElementById(`ch${ch}_vdiv`).value = chSettings.vdiv;
                }
                
                // Set offset (default to 0 if not specified)
                const offset = chSettings.offset || 0;
                document.getElementById(`ch${ch}_offset`).value = offset;
                
                // Set channel state
                const channelState = chSettings.display === 'on';
                document.getElementById(`ch${ch}_state`).checked = channelState;
            }
        }
        
        // Update trigger settings
        const trigger = config.settings?.trigger || {};
        if (trigger.source) {
            const sourceMap = {'EXT': 'EX', 'EX': 'EX'};
            document.getElementById('triggerSource').value = sourceMap[trigger.source] || trigger.source;
        }
        if (trigger.mode) {
            document.getElementById('triggerMode').value = trigger.mode;
        }
        if (trigger.slope) {
            const slopeRadio = trigger.slope === 'RISE' ? 'slopeRising' : 'slopeFalling';
            document.getElementById(slopeRadio).checked = true;
        }
        
        // Update timebase settings
        const timescale = config.settings?.timescale || {};
        if (timescale.horizontal_div) {
            document.getElementById('timeScale').value = parseFloat(timescale.horizontal_div);
        }
    }

    // Function to update DDG panel from config data
    function updateDDGFromConfig(ddgConfig) {
        if (!ddgConfig || !ddgConfig.settings) return;
        
        const settings = ddgConfig.settings;
        
        // Update mode and trigger settings
        if (settings.mode) {
            document.getElementById('ddg_mode').value = settings.mode;
        }
        
        if (settings.trigger) {
            if (settings.trigger.status) {
                document.getElementById('ddg_trigger_status').value = settings.trigger.status;
            }
            if (settings.trigger.level !== undefined) {
                document.getElementById('ddg_trigger_level').value = settings.trigger.level;
            }
            if (settings.trigger.edge) {
                document.getElementById('ddg_trigger_edge').value = settings.trigger.edge;
            }
        }
        
        if (settings.gate) {
            if (settings.gate.mode) {
                document.getElementById('ddg_gate_mode').value = settings.gate.mode;
            }
            if (settings.gate.level) {
                document.getElementById('ddg_gate_level').value = settings.gate.level;
            }
        }
        
        // Update channel settings
        if (settings.channels) {
            const channelLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
            channelLetters.forEach(letter => {
                const chKey = `ch${letter}`;
                if (settings.channels[chKey]) {
                    const chSettings = settings.channels[chKey];
                    
                    // Update status badge
                    const statusBadge = document.getElementById(`ddg_ch${letter}_status`);
                    if (statusBadge && chSettings.status) {
                        statusBadge.textContent = chSettings.status;
                        statusBadge.className = chSettings.status === 'enabled' ? 'badge bg-success' : 'badge bg-secondary';
                    }
                    
                    // Update amplitude
                    if (chSettings.amplitude !== undefined) {
                        const ampInput = document.getElementById(`ddg_ch${letter}_amplitude`);
                        if (ampInput) ampInput.value = chSettings.amplitude;
                    }
                    
                    // Update delay
                    if (chSettings.delay !== undefined) {
                        const delayInput = document.getElementById(`ddg_ch${letter}_delay`);
                        if (delayInput) delayInput.value = chSettings.delay;
                    }
                    
                    // Update width
                    if (chSettings.width !== undefined) {
                        const widthInput = document.getElementById(`ddg_ch${letter}_width`);
                        if (widthInput) widthInput.value = chSettings.width;
                    }
                    
                    // Update active
                    if (chSettings.active !== undefined) {
                        const activeBadge = document.getElementById(`ddg_ch${letter}_active`);
                        if (activeBadge) {
                            activeBadge.textContent = chSettings.active;
                            activeBadge.className = chSettings.active === 'high' ? 
                                'badge bg-info ddg-active-toggle' : 
                                'badge bg-warning ddg-active-toggle';
                            activeBadge.style.cursor = 'pointer';
                        }
                    }
                }
            });
        }
    }

    // Function to load DDG config from JSON file
    async function loadDDGConfigFromFile() {
        try {
            const response = await fetch('/api/get-ddg-config', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success' && data.config) {
                    updateDDGFromConfig(data.config);
                }
            }
        } catch (error) {
            console.log('Could not load DDG config:', error.message);
        }
    }
    
    // Load DDG config on page load
    loadDDGConfigFromFile();

    // Add click handlers for DDG status toggles
    document.querySelectorAll('.ddg-status-toggle').forEach(badge => {
        badge.addEventListener('click', function() {
            const currentStatus = this.textContent.trim();
            if (currentStatus === 'enabled') {
                this.textContent = 'disabled';
                this.className = 'badge bg-secondary ddg-status-toggle';
            } else {
                this.textContent = 'enabled';
                this.className = 'badge bg-success ddg-status-toggle';
            }
            this.style.cursor = 'pointer';
        });
    });

    // Add click handlers for DDG active toggles
    document.querySelectorAll('.ddg-active-toggle').forEach(badge => {
        badge.addEventListener('click', function() {
            const currentActive = this.textContent.trim();
            if (currentActive === 'high') {
                this.textContent = 'low';
                this.className = 'badge bg-warning ddg-active-toggle';
            } else {
                this.textContent = 'high';
                this.className = 'badge bg-info ddg-active-toggle';
            }
            this.style.cursor = 'pointer';
        });
    });


    // Write DDG Settings to JSON Handler
    const writeDDGToJsonBtn = document.getElementById('writeDDGToJsonBtn');
    if (writeDDGToJsonBtn) {
        writeDDGToJsonBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('importDDGStatus');
            statusDiv.textContent = 'Saving DDG settings to JSON...';
            statusDiv.className = 'small text-info mt-2';
            
            try {
                // Gather DDG settings from the UI
                const ddgConfig = {
                    instrument: "BNC Model 577 Pulse Generator",
                    settings: {
                        mode: document.getElementById('ddg_mode').value,
                        trigger: {
                            status: document.getElementById('ddg_trigger_status').value,
                            level: parseFloat(document.getElementById('ddg_trigger_level').value),
                            edge: document.getElementById('ddg_trigger_edge').value
                        },
                        gate: {
                            mode: document.getElementById('ddg_gate_mode').value,
                            level: document.getElementById('ddg_gate_level').value
                        },
                        channels: {}
                    }
                };
                
                // Gather channel settings
                const channelLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
                channelLetters.forEach(letter => {
                    const statusBadge = document.getElementById(`ddg_ch${letter}_status`);
                    const ampInput = document.getElementById(`ddg_ch${letter}_amplitude`);
                    const delayInput = document.getElementById(`ddg_ch${letter}_delay`);
                    const widthInput = document.getElementById(`ddg_ch${letter}_width`);
                    const activeBadge = document.getElementById(`ddg_ch${letter}_active`);
                    
                    ddgConfig.settings.channels[`ch${letter}`] = {
                        status: statusBadge ? statusBadge.textContent.trim() : 'disabled',
                        amplitude: ampInput ? parseFloat(ampInput.value) : 5.0,
                        delay: delayInput ? delayInput.value : '0.00E+00',
                        width: widthInput ? widthInput.value : '1.00E-02',
                        active: activeBadge ? activeBadge.textContent.trim() : 'high'
                    };
                });
                
                const response = await fetch('/api/write-ddg-to-json', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(ddgConfig)
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    statusDiv.textContent = data.message;
                    statusDiv.className = 'small text-success mt-2';
                } else {
                    statusDiv.textContent = 'Write error: ' + (data.error || 'Unknown error');
                    statusDiv.className = 'small text-danger mt-2';
                }
            } catch (err) {
                statusDiv.textContent = 'Write failed: ' + err.message;
                statusDiv.className = 'small text-danger mt-2';
            }
        });
    }

    // Load DDG Settings from JSON Handler
    const setDDGFromJsonBtn = document.getElementById('setDDGFromJsonBtn');
    if (setDDGFromJsonBtn) {
        setDDGFromJsonBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('importDDGStatus');
            const fileInput = document.getElementById('importConfigFile'); // Use the same file input
            
            statusDiv.textContent = 'Loading DDG settings...';
            statusDiv.className = 'small text-info mt-2';
            
            try {
                let ddgConfig = null;
                
                // Check if a file is selected
                if (fileInput && fileInput.files.length > 0) {
                    // Parse the uploaded file
                    const file = fileInput.files[0];
                    const text = await file.text();
                    const jsonData = JSON.parse(text);
                    
                    if (Array.isArray(jsonData)) {
                        // Find DDG config in array
                        ddgConfig = jsonData.find(device => 
                            device.instrument && device.instrument.includes('577')
                        );
                    } else if (jsonData.instrument && jsonData.instrument.includes('577')) {
                        ddgConfig = jsonData;
                    }
                } else {
                    // Use the existing device settings.json file
                    const response = await fetch('/api/get-ddg-config', {
                        method: 'GET',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        ddgConfig = data.config;
                    } else {
                        throw new Error(data.error || 'Failed to load DDG config');
                    }
                }
                
                if (ddgConfig) {
                    updateDDGFromConfig(ddgConfig);
                    statusDiv.textContent = 'DDG settings loaded successfully';
                    statusDiv.className = 'small text-success mt-2';
                    
                    // Clear file selection after successful import
                    if (fileInput && fileInput.files.length > 0) {
                        fileInput.value = '';
                    }
                } else {
                    statusDiv.textContent = 'No DDG configuration found in file';
                    statusDiv.className = 'small text-warning mt-2';
                }
                
            } catch (err) {
                statusDiv.textContent = 'Failed: ' + err.message;
                statusDiv.className = 'small text-danger mt-2';
            }
        });
    }

    // Write Scope Settings to JSON Handler
    const writeScopeToJsonBtn = document.getElementById('writeScopeToJsonBtn');
    if (writeScopeToJsonBtn) {
        writeScopeToJsonBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('importConfigStatus');
            statusDiv.textContent = 'Writing scope settings to JSON...';
            statusDiv.className = 'small text-info mt-2';
            
            try {
                const response = await fetch('/api/write-scope-to-json', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    statusDiv.textContent = data.message;
                    statusDiv.className = 'small text-success mt-2';
                } else {
                    statusDiv.textContent = 'Write error: ' + (data.error || 'Unknown error');
                    statusDiv.className = 'small text-danger mt-2';
                }
            } catch (err) {
                statusDiv.textContent = 'Write failed: ' + err.message;
                statusDiv.className = 'small text-danger mt-2';
            }
        });
    }

    // Set Scope Settings from JSON Handler
    const setScopeFromJsonBtn = document.getElementById('setScopeFromJsonBtn');
    if (setScopeFromJsonBtn) {
        setScopeFromJsonBtn.addEventListener('click', async () => {
            const statusDiv = document.getElementById('importConfigStatus');
            const fileInput = document.getElementById('importConfigFile');
            
            statusDiv.textContent = 'Loading settings...';
            statusDiv.className = 'small text-info mt-2';
            
            try {
                let response;
                
                // Check if a file is selected
                if (fileInput && fileInput.files.length > 0) {
                    // Use the selected file
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    response = await fetch('/api/import-config', {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    // Use the existing device settings.json file
                    response = await fetch('/api/set-scope-from-json', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                }
                
                const data = await response.json();
                if (data.status === 'success') {
                    // Show appropriate message based on connection status
                    if (data.connection_error) {
                        statusDiv.textContent = `Settings loaded for UI (scope not connected: ${data.connection_error})`;
                        statusDiv.className = 'small text-warning mt-2';
                    } else {
                        statusDiv.textContent = data.message || 'Settings loaded and applied successfully';
                        statusDiv.className = 'small text-success mt-2';
                    }
                    
                    // Update GUI fields with config data (works regardless of connection)
                    updateUIFromConfig(data.config);
                    
                    // If file was uploaded, also try to load DDG config
                    if (fileInput && fileInput.files.length > 0) {
                        // Parse the file to check if it contains DDG config
                        const file = fileInput.files[0];
                        const text = await file.text();
                        try {
                            const jsonData = JSON.parse(text);
                            if (Array.isArray(jsonData)) {
                                // Find DDG config in array
                                const ddgConfig = jsonData.find(device => 
                                    device.instrument && device.instrument.includes('577')
                                );
                                if (ddgConfig) {
                                    updateDDGFromConfig(ddgConfig);
                                }
                            }
                        } catch (e) {
                            console.log('Could not parse DDG config from file:', e);
                        }
                    }
                    
                    // Clear file selection after successful import
                    if (fileInput && fileInput.files.length > 0) {
                        fileInput.value = '';
                    }
                    
                } else {
                    statusDiv.textContent = 'Error: ' + (data.error || 'Unknown error');
                    statusDiv.className = 'small text-danger mt-2';
                }
            } catch (err) {
                statusDiv.textContent = 'Failed: ' + err.message;
                statusDiv.className = 'small text-danger mt-2';
            }
        });
    }
});
