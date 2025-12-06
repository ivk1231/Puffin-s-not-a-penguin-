/**
 * Frontend JavaScript for Oscilloscope Control
 * Handles all API communication and UI updates for multiple scopes
 */

// Global Alert Manager
class AlertManager {
    constructor(maxAlerts = 30) {
        this.maxAlerts = maxAlerts;
        this.alerts = [];
        this.container = null;
        this.badge = null;
    }
    
    initialize() {
        this.container = document.getElementById('alertsContainer');
        this.badge = document.getElementById('alertBadge');
        
        // Clear alerts button
        const clearBtn = document.getElementById('clearAlertsBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearAll());
        }
    }
    
    addAlert(message, type = 'info', source = 'System') {
        const timestamp = new Date();
        const alert = {
            id: Date.now() + Math.random(),
            timestamp,
            message,
            type,
            source
        };
        
        // Add to beginning of array
        this.alerts.unshift(alert);
        
        // Keep only last maxAlerts
        if (this.alerts.length > this.maxAlerts) {
            this.alerts = this.alerts.slice(0, this.maxAlerts);
        }
        
        this.render();
        this.updateBadge();
    }
    
    clearAll() {
        this.alerts = [];
        this.render();
        this.updateBadge();
    }
    
    updateBadge() {
        if (this.badge) {
            this.badge.textContent = this.alerts.length;
            
            // Update badge color based on count
            this.badge.className = 'badge ms-1';
            if (this.alerts.length === 0) {
                this.badge.classList.add('bg-secondary');
            } else if (this.alerts.length < 10) {
                this.badge.classList.add('bg-primary');
            } else if (this.alerts.length < 20) {
                this.badge.classList.add('bg-warning');
            } else {
                this.badge.classList.add('bg-danger');
            }
        }
    }
    
    render() {
        if (!this.container) return;
        
        if (this.alerts.length === 0) {
            this.container.innerHTML = `
                <div class="text-muted text-center py-4">
                    <i class="bi bi-info-circle" style="font-size: 2rem;"></i>
                    <p class="mt-2">No alerts yet. Activity will appear here.</p>
                </div>
            `;
            return;
        }
        
        const html = this.alerts.map(alert => {
            const timeStr = alert.timestamp.toLocaleTimeString();
            const icon = this.getIcon(alert.type);
            
            return `
                <div class="alert-log-entry alert-${alert.type}">
                    <span class="alert-log-icon">${icon}</span>
                    <span class="alert-log-timestamp">${timeStr}</span>
                    <span class="alert-log-message"><strong>${alert.source}:</strong> ${alert.message}</span>
                </div>
            `;
        }).join('');
        
        this.container.innerHTML = html;
    }
    
    getIcon(type) {
        const icons = {
            'success': '✓',
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ'
        };
        return icons[type] || 'ℹ';
    }
}

// Global alert manager instance
const alertManager = new AlertManager(30);

// Global DDG Controller (Pulse Generator - remains global)
class DDGController {
    constructor() {
        this.initializeEventListeners();
        this.loadDDGConfigFromFile();
    }
    
    initializeEventListeners() {
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
            writeDDGToJsonBtn.addEventListener('click', () => this.writeDDGToJson());
        }

        // Load DDG Settings from JSON Handler
        const setDDGFromJsonBtn = document.getElementById('setDDGFromJsonBtn');
        if (setDDGFromJsonBtn) {
            setDDGFromJsonBtn.addEventListener('click', () => this.setDDGFromJson());
        }
    }
    
    async loadDDGConfigFromFile() {
        try {
            const response = await fetch('/api/get-ddg-config', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success' && data.config) {
                    this.updateDDGFromConfig(data.config);
                }
            }
        } catch (error) {
            console.log('Could not load DDG config:', error.message);
        }
    }
    
    updateDDGFromConfig(ddgConfig) {
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
                    
                    const statusBadge = document.getElementById(`ddg_ch${letter}_status`);
                    if (statusBadge && chSettings.status) {
                        statusBadge.textContent = chSettings.status;
                        statusBadge.className = chSettings.status === 'enabled' ? 'badge bg-success ddg-status-toggle' : 'badge bg-secondary ddg-status-toggle';
                        statusBadge.style.cursor = 'pointer';
                    }
                    
                    if (chSettings.amplitude !== undefined) {
                        const ampInput = document.getElementById(`ddg_ch${letter}_amplitude`);
                        if (ampInput) ampInput.value = chSettings.amplitude;
                    }
                    
                    if (chSettings.delay !== undefined) {
                        const delayInput = document.getElementById(`ddg_ch${letter}_delay`);
                        if (delayInput) delayInput.value = chSettings.delay;
                    }
                    
                    if (chSettings.width !== undefined) {
                        const widthInput = document.getElementById(`ddg_ch${letter}_width`);
                        if (widthInput) widthInput.value = chSettings.width;
                    }
                    
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
    
    async writeDDGToJson() {
        const statusDiv = document.getElementById('importDDGStatus');
        statusDiv.textContent = 'Saving DDG settings to JSON...';
        statusDiv.className = 'small text-info mt-2';
        
        try {
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
    }
    
    async setDDGFromJson() {
        const statusDiv = document.getElementById('importDDGStatus');
        const fileInput = document.getElementById('importDDGFile');
        
        statusDiv.textContent = 'Loading DDG settings...';
        statusDiv.className = 'small text-info mt-2';
        
        try {
            let ddgConfig = null;
            
            if (fileInput && fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const text = await file.text();
                const jsonData = JSON.parse(text);
                
                if (Array.isArray(jsonData)) {
                    ddgConfig = jsonData.find(device => 
                        device.instrument && device.instrument.includes('577')
                    );
                } else if (jsonData.instrument && jsonData.instrument.includes('577')) {
                    ddgConfig = jsonData;
                }
            } else {
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
                this.updateDDGFromConfig(ddgConfig);
                statusDiv.textContent = 'DDG settings loaded successfully';
                statusDiv.className = 'small text-success mt-2';
                
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
    }
}

// Scope Instance Controller (one per tab)
class ScopeInstance {
    constructor(index) {
        this.index = index;
        this.connected = false;
        this.currentIp = null;
        this.currentTaskId = null;
        this.pollingInterval = null;
        this.initializeEventListeners();
    }
    
    getElement(suffix) {
        return document.getElementById(`${suffix}_${this.index}`);
    }
    
    initializeEventListeners() {
        this.getElement('connectBtn').addEventListener('click', () => this.connect());
        this.getElement('disconnectBtn').addEventListener('click', () => this.disconnect());
        this.getElement('applySettingsBtn').addEventListener('click', () => this.applySettings());
        this.getElement('armBtn').addEventListener('click', () => this.armScope());
        
        // JSON Import/Export handlers
        const writeScopeBtn = this.getElement('writeScopeToJsonBtn');
        if (writeScopeBtn) {
            writeScopeBtn.addEventListener('click', () => this.writeScopeToJson());
        }
        
        const setScopeBtn = this.getElement('setScopeFromJsonBtn');
        if (setScopeBtn) {
            setScopeBtn.addEventListener('click', () => this.setScopeFromJson());
        }
    }
    
    async connect() {
        const ip = this.getElement('ipAddress').value;
        const port = parseInt(this.getElement('port').value);
        
        this.currentIp = ip;
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
            
            console.log(`Scope ${this.index} connection response:`, data);
            
            if (data.status === 'connected') {
                this.connected = true;
                this.currentIp = data.ip || ip;
                this.setConnectionState('connected', `Connected to ${data.idn}`);
                this.updateUIForConnection(true);
                this.showToast(`Scope ${this.index}: Successfully connected`, 'success');
                
                if (data.settings) {
                    this.populateGUIFromSettings(data.settings);
                }
                
                if (data.armed_status && data.armed_status.armed && data.armed_status.task_id) {
                    console.log(`Scope ${this.index} is armed! Starting polling with task_id:`, data.armed_status.task_id);
                    this.currentTaskId = data.armed_status.task_id;
                    this.startArmPolling();
                    this.updateArmUI('armed', 'Scope was already armed, monitoring for trigger...');
                    this.showToast(`Scope ${this.index}: Armed and waiting for trigger`, 'info');
                } else {
                    console.log(`Scope ${this.index}: Not armed on connection`);
                }
            } else {
                this.setConnectionState('error', `Error: ${data.error}`);
                this.showToast(`Scope ${this.index}: Connection failed - ${data.error}`, 'error');
            }
        } catch (error) {
            this.setConnectionState('error', `Error: ${error.message}`);
            this.showToast(`Scope ${this.index}: Connection error - ${error.message}`, 'error');
        }
    }
    
    populateGUIFromSettings(settings) {
        if (settings.channels) {
            for (let ch = 1; ch <= 4; ch++) {
                const chKey = `CH${ch}`;
                if (settings.channels[chKey]) {
                    const chSettings = settings.channels[chKey];
                    
                    if (chSettings.vdiv !== undefined) {
                        this.getElement(`ch${ch}_vdiv`).value = chSettings.vdiv;
                    }
                    if (chSettings.offset !== undefined) {
                        this.getElement(`ch${ch}_offset`).value = chSettings.offset;
                    }
                    if (chSettings.display !== undefined) {
                        this.getElement(`ch${ch}_state`).checked = chSettings.display;
                    }
                }
            }
        }
        
        if (settings.trigger) {
            if (settings.trigger.source) {
                this.getElement('triggerSource').value = settings.trigger.source;
            }
            if (settings.trigger.mode) {
                this.getElement('triggerMode').value = settings.trigger.mode;
            }
            if (settings.trigger.slope) {
                const slopeValue = settings.trigger.slope.toUpperCase();
                if (slopeValue.includes('RIS')) {
                    this.getElement('slopeRising').checked = true;
                } else {
                    this.getElement('slopeFalling').checked = true;
                }
            }
        }
        
        if (settings.timebase && settings.timebase.horizontal_div) {
            this.getElement('timeScale').value = parseFloat(settings.timebase.horizontal_div);
        }
        
        console.log(`Scope ${this.index} GUI populated with settings:`, settings);
    }
    
    async disconnect() {
        if (!this.currentIp) {
            this.showToast(`Scope ${this.index}: No IP address set`, 'error');
            return;
        }
        
        // CRITICAL: Stop any active polling before disconnecting
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
            console.log(`Scope ${this.index}: Cleared polling interval on disconnect`);
        }
        
        // Reset task ID
        this.currentTaskId = null;
        
        this.setConnectionState('disconnecting', 'Disconnecting...');
        
        try {
            const response = await fetch('/api/disconnect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ip: this.currentIp })
            });
            
            const data = await response.json();
            
            this.connected = false;
            this.currentIp = null;
            this.setConnectionState('disconnected', 'Disconnected');
            this.updateUIForConnection(false);
            
            // Reset arm UI to idle state
            this.updateArmUI('idle', 'Idle');
            
            this.showToast(`Scope ${this.index}: Disconnected`, 'info');
            
        } catch (error) {
            this.setConnectionState('error', `Error: ${error.message}`);
            this.showToast(`Scope ${this.index}: Disconnect error - ${error.message}`, 'error');
        }
    }
    
    async applySettings() {
        if (!this.connected || !this.currentIp) {
            this.showToast(`Scope ${this.index}: Not connected`, 'error');
            return;
        }
        
        const timeScale = parseFloat(this.getElement('timeScale').value);
        const triggerMode = this.getElement('triggerMode').value;
        const triggerSource = this.getElement('triggerSource').value;
        const triggerSlope = document.querySelector(`input[name="triggerSlope_${this.index}"]:checked`).value;
        const triggerLevel = parseFloat(this.getElement('triggerLevel').value);
        
        try {
            for (let ch = 1; ch <= 4; ch++) {
                const state = this.getElement(`ch${ch}_state`).checked;
                const voltageScale = parseFloat(this.getElement(`ch${ch}_vdiv`).value);
                const offset = parseFloat(this.getElement(`ch${ch}_offset`).value);
                
                await fetch('/api/config/channel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        ip: this.currentIp,
                        channel: ch, 
                        state, 
                        voltage_scale: voltageScale,
                        offset: offset
                    })
                });
            }
            
            await fetch('/api/config/timebase', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    ip: this.currentIp,
                    time_scale: timeScale 
                })
            });
            
            const triggerResponse = await fetch('/api/config/trigger', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    ip: this.currentIp,
                    mode: triggerMode, 
                    source: triggerSource, 
                    slope: triggerSlope, 
                    level: triggerLevel 
                })
            });
            
            const triggerData = await triggerResponse.json();
            
            if (triggerData.status === 'skipped') {
                this.showToast(`Scope ${this.index}: Channel/timebase applied. Trigger skipped (monitoring active).`, 'warning');
            } else {
                this.showToast(`Scope ${this.index}: Settings applied successfully`, 'success');
            }
            
        } catch (error) {
            this.showToast(`Scope ${this.index}: Failed to apply settings - ${error.message}`, 'error');
        }
    }
    
    async armScope() {
        if (!this.connected || !this.currentIp) {
            this.showToast(`Scope ${this.index}: Not connected`, 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/arm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: this.currentIp })
            });
            
            const data = await response.json();
            
            if (data.task_id) {
                this.currentTaskId = data.task_id;
                this.startArmPolling();
                this.updateArmUI('armed', data.message || 'Scope armed, waiting for trigger...');
                
                if (data.already_armed) {
                    this.showToast(`Scope ${this.index}: Was already armed - monitoring for trigger`, 'info');
                } else {
                    this.showToast(`Scope ${this.index}: Armed - waiting for trigger`, 'success');
                }
            } else {
                this.showToast(`Scope ${this.index}: Failed to arm - ${data.error}`, 'error');
            }
            
        } catch (error) {
            this.showToast(`Scope ${this.index}: Arm error - ${error.message}`, 'error');
        }
    }
    
    startArmPolling() {
        // Clear any existing polling first
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
            console.log(`Scope ${this.index}: Cleared existing polling interval`);
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
                    console.log(`Scope ${this.index}: Polling stopped - acquisition complete`);
                    
                    if (data.result && data.result.csv_file) {
                        this.downloadFile(data.result.csv_file);
                    }
                    
                    if (data.result && data.result.plot_file) {
                        this.displayPlot(data.result.plot_file);
                    }
                    
                } else if (data.status === 'error' || data.status === 'timeout') {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                    this.currentTaskId = null;
                    console.log(`Scope ${this.index}: Polling stopped - ${data.status}`);
                }
                
            } catch (error) {
                console.error(`Scope ${this.index} polling error:`, error);
                clearInterval(this.pollingInterval);
                this.pollingInterval = null;
                this.currentTaskId = null;
            }
        }, 500);
        
        console.log(`Scope ${this.index}: Started polling for task ${this.currentTaskId}`);
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
            
            this.showToast(`Scope ${this.index}: Acquisition complete! Downloading CSV...`, 'success');
            
        } else if (data.status === 'timeout') {
            this.updateArmUI('idle', 'Idle');
            this.showToast(`Scope ${this.index}: ${data.error || 'Trigger timeout'}`, 'warning');
            
        } else if (data.status === 'error') {
            this.updateArmUI('idle', 'Idle');
            this.showToast(`Scope ${this.index}: Error - ${data.error}`, 'error');
        }
    }
    
    displayArmResults(result) {
        const armInfo = this.getElement('armInfo');
        const channelsAcquired = this.getElement('channelsAcquired');
        const csvFileName = this.getElement('csvFileName');
        
        channelsAcquired.textContent = result.channels ? result.channels.join(', ') : 'N/A';
        csvFileName.textContent = result.csv_file || 'N/A';
        armInfo.style.display = 'block';
    }
    
    displayPlot(plotFilename) {
        const plotContainer = this.getElement('plotContainer');
        const plotImage = this.getElement('plotImage');
        
        if (plotFilename) {
            plotImage.src = `/api/plot/${plotFilename}?t=${Date.now()}`;
            plotContainer.style.display = 'block';
        }
    }
    
    downloadFile(filename) {
        const a = document.createElement('a');
        a.href = `/api/download/${filename}`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    updateArmUI(state, message) {
        const armBtn = this.getElement('armBtn');
        const armStatus = this.getElement('armStatus');
        const armStatusText = this.getElement('armStatusText');
        
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
                setTimeout(() => {
                    this.updateArmUI('idle', 'Idle');
                    this.getElement('armInfo').style.display = 'none';
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
        const statusElement = this.getElement('connectionStatus');
        const deviceIdElement = this.getElement('deviceId');
        
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
        this.getElement('connectBtn').disabled = connected;
        this.getElement('disconnectBtn').disabled = !connected;
        this.getElement('applySettingsBtn').disabled = !connected;
        this.getElement('armBtn').disabled = !connected;
    }
    
    showToast(message, type = 'info') {
        const toastElement = document.getElementById('toast');
        const toastBody = document.getElementById('toastBody');
        
        toastBody.textContent = message;
        
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
        
        // Also add to alert log
        alertManager.addAlert(message, type, `Scope ${this.index}`);
    }
    
    async writeScopeToJson() {
        if (!this.connected || !this.currentIp) {
            this.showToast(`Scope ${this.index}: Not connected`, 'error');
            return;
        }
        
        const statusDiv = this.getElement('importConfigStatus');
        statusDiv.textContent = 'Writing scope settings to JSON...';
        statusDiv.className = 'small text-info mt-2';
        
        try {
            const response = await fetch('/api/write-scope-to-json', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ip: this.currentIp })
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
    }
    
    async setScopeFromJson() {
        const statusDiv = this.getElement('importConfigStatus');
        const fileInput = this.getElement('importConfigFile');
        
        statusDiv.textContent = 'Loading settings...';
        statusDiv.className = 'small text-info mt-2';
        
        try {
            let response;
            
            if (fileInput && fileInput.files.length > 0) {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                if (this.currentIp) {
                    formData.append('target_ip', this.currentIp);
                }
                
                response = await fetch('/api/import-config', {
                    method: 'POST',
                    body: formData
                });
            } else {
                const payload = this.currentIp ? { ip: this.currentIp } : {};
                response = await fetch('/api/set-scope-from-json', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }
            
            const data = await response.json();
            if (data.status === 'success') {
                if (data.connection_error) {
                    statusDiv.textContent = `Settings loaded for UI (scope not connected: ${data.connection_error})`;
                    statusDiv.className = 'small text-warning mt-2';
                } else {
                    statusDiv.textContent = data.message || 'Settings loaded and applied successfully';
                    statusDiv.className = 'small text-success mt-2';
                }
                
                if (data.config) {
                    this.updateUIFromConfig(data.config);
                }
                
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
    }
    
    updateUIFromConfig(config) {
        if (config.ip) {
            this.getElement('ipAddress').value = config.ip;
        }
        if (config.port) {
            this.getElement('port').value = config.port;
        }
        
        const channels = config.settings?.channels || {};
        for (let ch = 1; ch <= 4; ch++) {
            const chKey = `CH${ch}`;
            if (channels[chKey]) {
                const chSettings = channels[chKey];
                
                if (chSettings.vdiv !== undefined) {
                    this.getElement(`ch${ch}_vdiv`).value = chSettings.vdiv;
                }
                
                const offset = chSettings.offset || 0;
                this.getElement(`ch${ch}_offset`).value = offset;
                
                const channelState = chSettings.display === 'on';
                this.getElement(`ch${ch}_state`).checked = channelState;
            }
        }
        
        const trigger = config.settings?.trigger || {};
        if (trigger.source) {
            const sourceMap = {'EXT': 'EX', 'EX': 'EX'};
            this.getElement('triggerSource').value = sourceMap[trigger.source] || trigger.source;
        }
        if (trigger.mode) {
            this.getElement('triggerMode').value = trigger.mode;
        }
        if (trigger.slope) {
            const slopeRadio = trigger.slope === 'RISE' ? 'slopeRising' : 'slopeFalling';
            this.getElement(slopeRadio).checked = true;
        }
        
        const timescale = config.settings?.timescale || {};
        if (timescale.horizontal_div) {
            this.getElement('timeScale').value = parseFloat(timescale.horizontal_div);
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Alert Manager
    alertManager.initialize();
    alertManager.addAlert('System initialized', 'success', 'System');
    
    // Initialize DDG Controller (global)
    const ddgController = new DDGController();
    
    // Initialize Scope Instances (one per tab)
    const scopes = [
        new ScopeInstance(1),
        new ScopeInstance(2),
        new ScopeInstance(3)
    ];
    
    // Connect All button handler
    const connectAllBtn = document.getElementById('connectAllBtn');
    if (connectAllBtn) {
        connectAllBtn.addEventListener('click', async () => {
            console.log('Connect All clicked');
            
            // Use Promise.allSettled to handle partial success
            const connectionPromises = scopes.map(scope => scope.connect());
            
            const results = await Promise.allSettled(connectionPromises);
            
            // Count successes and failures
            let successCount = 0;
            let failCount = 0;
            
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    successCount++;
                } else {
                    failCount++;
                    console.error(`Scope ${index + 1} connection failed:`, result.reason);
                }
            });
            
            // Show summary toast and log to alerts
            if (successCount === 3) {
                const msg = 'All scopes connected successfully!';
                scopes[0].showToast(msg, 'success');
                alertManager.addAlert(msg, 'success', 'Connect All');
            } else if (successCount > 0) {
                const msg = `${successCount} scope(s) connected, ${failCount} failed`;
                scopes[0].showToast(msg, 'warning');
                alertManager.addAlert(msg, 'warning', 'Connect All');
            } else {
                const msg = 'All scope connections failed';
                scopes[0].showToast(msg, 'error');
                alertManager.addAlert(msg, 'error', 'Connect All');
            }
        });
    }
    
    // Disconnect All button handler
    const disconnectAllBtn = document.getElementById('disconnectAllBtn');
    if (disconnectAllBtn) {
        disconnectAllBtn.addEventListener('click', async () => {
            console.log('Disconnect All clicked');
            
            // Use Promise.allSettled to handle partial success
            const disconnectionPromises = scopes.map(scope => scope.disconnect());
            
            const results = await Promise.allSettled(disconnectionPromises);
            
            // Count successes and failures
            let successCount = 0;
            let failCount = 0;
            
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    successCount++;
                } else {
                    failCount++;
                    console.error(`Scope ${index + 1} disconnection failed:`, result.reason);
                }
            });
            
            // Show summary toast and log to alerts
            if (successCount === 3) {
                const msg = 'All scopes disconnected successfully!';
                scopes[0].showToast(msg, 'info');
                alertManager.addAlert(msg, 'info', 'Disconnect All');
            } else if (successCount > 0) {
                const msg = `${successCount} scope(s) disconnected, ${failCount} failed`;
                scopes[0].showToast(msg, 'warning');
                alertManager.addAlert(msg, 'warning', 'Disconnect All');
            } else {
                const msg = 'All scope disconnections failed';
                scopes[0].showToast(msg, 'error');
                alertManager.addAlert(msg, 'error', 'Disconnect All');
            }
        });
    }
});
