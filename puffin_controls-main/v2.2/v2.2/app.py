#!/usr/bin/env python3
"""
Flask web application for oscilloscope control
Provides web UI for controlling Siglent SDS5104X oscilloscope
"""

from flask import Flask, render_template, request, jsonify, send_file, abort
import json
from concurrent.futures import ThreadPoolExecutor, Future
import uuid
import os
import time
import traceback
from datetime import datetime
import threading
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

from scope_manager import ScopeManager

app = Flask(__name__)

# Global scope manager and task executor
scope_manager = ScopeManager()
executor = ThreadPoolExecutor(max_workers=4)
tasks = {}  # {task_id: {"future": Future, "status": dict}}

# Lock for matplotlib plot creation (not thread-safe)
plot_lock = threading.Lock()

# Shot directory configuration
SHOT_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shot directory")


def get_current_shot_folder():
    """
    Get or create the current shot folder for today.
    If the current shot folder already has waveform data, create the next shot.
    Format: pYYMMDD_NN where NN is the shot number starting at 01
    
    Returns:
        str: Path to current shot folder
    """
    today = datetime.now().strftime("%y%m%d")
    prefix = f"p{today}_"
    
    # Ensure base shot directory exists
    os.makedirs(SHOT_BASE_DIR, exist_ok=True)
    
    # Find existing shot folders for today
    existing_shots = []
    for item in os.listdir(SHOT_BASE_DIR):
        if item.startswith(prefix) and os.path.isdir(os.path.join(SHOT_BASE_DIR, item)):
            try:
                # Extract shot number from folder name
                shot_num = int(item.split('_')[1])
                existing_shots.append(shot_num)
            except (ValueError, IndexError):
                pass
    
    # Determine current shot number
    if existing_shots:
        current_shot = max(existing_shots)
        current_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{current_shot:02d}")
        
        # Check if this shot folder already has waveform data (CSV files)
        # If it does, we need to create the next shot folder
        has_waveform_data = False
        if os.path.exists(current_shot_folder):
            for file in os.listdir(current_shot_folder):
                if file.endswith('_waveform_' + today + f'_{current_shot:02d}.csv'):
                    has_waveform_data = True
                    break
        
        if has_waveform_data:
            # Current shot already has data, create next shot
            current_shot += 1
            current_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{current_shot:02d}")
            os.makedirs(current_shot_folder, exist_ok=True)
            
            # Copy device settings from previous shot
            prev_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{current_shot-1:02d}")
            prev_settings = os.path.join(prev_shot_folder, "device settings.json")
            current_settings = os.path.join(current_shot_folder, "device settings.json")
            
            if os.path.exists(prev_settings) and not os.path.exists(current_settings):
                import shutil
                shutil.copy2(prev_settings, current_settings)
                print(f"Created shot {current_shot:02d} and copied settings from shot {current_shot-1:02d}")
            else:
                print(f"Created new shot folder: {current_shot_folder}")
    else:
        # No existing shots for today, create shot 01
        current_shot = 1
        current_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{current_shot:02d}")
        os.makedirs(current_shot_folder, exist_ok=True)
        print(f"Created initial shot folder for today: {current_shot_folder}")
    
    return current_shot_folder

def prepare_next_shot_folder():
    """
    Create the next shot folder and copy device settings from current shot.
    
    Returns:
        str: Path to new shot folder
    """
    today = datetime.now().strftime("%y%m%d")
    prefix = f"p{today}_"
    
    # Find current highest shot number
    existing_shots = []
    for item in os.listdir(SHOT_BASE_DIR):
        if item.startswith(prefix) and os.path.isdir(os.path.join(SHOT_BASE_DIR, item)):
            try:
                shot_num = int(item.split('_')[1])
                existing_shots.append(shot_num)
            except (ValueError, IndexError):
                pass
    
    # Create next shot folder
    next_shot_num = max(existing_shots) + 1 if existing_shots else 1
    next_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{next_shot_num:02d}")
    os.makedirs(next_shot_folder, exist_ok=True)
    
    # Copy device settings from previous shot if it exists
    if existing_shots:
        current_shot_num = max(existing_shots)
        current_shot_folder = os.path.join(SHOT_BASE_DIR, f"{prefix}{current_shot_num:02d}")
        current_settings = os.path.join(current_shot_folder, "device settings.json")
        next_settings = os.path.join(next_shot_folder, "device settings.json")
        
        if os.path.exists(current_settings):
            import shutil
            shutil.copy2(current_settings, next_settings)
            print(f"Copied device settings from shot {current_shot_num:02d} to {next_shot_num:02d}")
    
    print(f"Prepared next shot folder: {next_shot_folder}")
    return next_shot_folder

def get_scope_label(scope_ip):
    """
    Get label for scope based on its position in connected scopes list.
    Returns 'scope1', 'scope2', or 'scope3' etc.
    
    Args:
        scope_ip (str): IP address of the scope
        
    Returns:
        str: Scope label like 'scope1', 'scope2', etc.
    """
    # Get all connected scope IPs and sort them for consistent labeling
    connected_ips = sorted(scope_manager.scopes.keys())
    
    try:
        index = connected_ips.index(scope_ip) + 1  # 1-indexed
        return f"scope{index}"
    except ValueError:
        # If not found, default to scope1
        return "scope1"

def get_target_scope(request_data):
    """
    Helper function to get scope instance from request data
    
    Args:
        request_data (dict): Request JSON data containing 'ip' field
        
    Returns:
        tuple: (scope_instance, error_response)
               If successful: (scope, None)
               If failed: (None, (error_dict, status_code))
    """
    if not request_data:
        return None, (jsonify({"error": "No request data provided"}), 400)
    
    ip = request_data.get('ip')
    if not ip:
        return None, (jsonify({"error": "Missing 'ip' field in request"}), 400)
    
    scope = scope_manager.get_scope(ip)
    if not scope:
        return None, (jsonify({"error": f"Scope with IP {ip} not found or not connected"}), 400)
    
    return scope, None


# --- JSON Import Endpoint ---
@app.route('/api/import-config', methods=['POST'])
def api_import_config():
    """Import oscilloscope configuration from a JSON script"""
    try:
        # First, extract the IP from the request to target the correct scope
        target_ip = None
        
        if request.content_type.startswith('application/json'):
            config = request.get_json()
            target_ip = config.get('target_ip')  # Optional IP to target specific scope
        elif request.content_type.startswith('multipart/form-data'):
            if 'file' not in request.files:
                return jsonify({"error": "No file part in request"}), 400
            file = request.files['file']
            target_ip = request.form.get('target_ip')  # Optional IP from form data
            # Read file content and parse JSON
            file_content = file.read()
            config = json.loads(file_content)
        else:
            return jsonify({"error": "Unsupported content type"}), 400

        # Handle both single config object and array of configs
        if isinstance(config, list):
            # If it's a list, try to find the oscilloscope configuration
            scope_config = None
            for device in config:
                if isinstance(device, dict) and 'SDS' in device.get('instrument', ''):
                    scope_config = device
                    break
            
            if not scope_config:
                return jsonify({"error": "No oscilloscope configuration found in the JSON array. Looking for instrument containing 'SDS'."}), 400
            
            config = scope_config
        elif not isinstance(config, dict):
            return jsonify({"error": f"Invalid JSON format: expected object/dict or array, got {type(config).__name__}"}), 400

        # Validate required fields
        instrument = config.get('instrument')
        ip = config.get('ip')
        port = config.get('port', 5025)
        settings = config.get('settings', {})
        if not instrument or not ip:
            return jsonify({"error": "Missing required fields: instrument, ip"}), 400

        # Try to get existing scope connection or use target_ip if provided
        scope = None
        idn = None
        connection_error = None
        
        # If target_ip is provided, use that scope; otherwise try to connect to the IP in config
        lookup_ip = target_ip if target_ip else ip
        
        try:
            scope = scope_manager.get_scope(lookup_ip)
            if not scope:
                # Try to connect if not already connected
                future = executor.submit(scope_manager.connect_scope, ip, port, 15.0)
                success, idn, error = future.result(timeout=20)
                if success:
                    scope = scope_manager.get_scope(ip)
                else:
                    connection_error = error
        except Exception as e:
            connection_error = str(e)

        # Apply settings to scope if connected
        applied_settings = {}
        if scope:
            # Apply channel settings
            channels = settings.get('channels', {})
            for ch_name, ch_settings in channels.items():
                if ch_name.startswith('CH'):
                    ch_num = int(ch_name[2:])  # Extract channel number from CH1, CH2, etc.
                    vdiv = ch_settings.get('vdiv')
                    display = ch_settings.get('display', 'on')
                    
                    if vdiv is not None:
                        try:
                            scope.set_voltage_scale(ch_num, vdiv)
                            scope.set_channel_state(ch_num, display.lower() == 'on')
                            applied_settings[f'channel_{ch_num}'] = {'vdiv': vdiv, 'display': display}
                        except Exception as e:
                            return jsonify({"error": f"Failed to set channel {ch_num}: {e}"}), 400
            
            # Apply trigger settings (skip if scope is armed/monitoring)
            trigger = settings.get('trigger', {})
            if trigger:
                # Check if there's an active monitoring task
                active_task = None
                for task_id, task_data in tasks.items():
                    if task_data["status"]["status"] in ["armed", "triggered"]:
                        active_task = task_id
                        break
                
                if active_task:
                    print(f"DEBUG: Skipping trigger configuration in import-config - active monitoring task {active_task}")
                    applied_settings['trigger'] = "skipped (monitoring active)"
                else:
                    try:
                        trigger_source = trigger.get('source', 'EX')
                        trigger_slope = trigger.get('slope', 'RISing')
                        trigger_mode = trigger.get('mode', 'SINGLE')
                        
                        scope.send_command(f":TRIGger:EDGE:SOURce {trigger_source}")
                        scope.send_command(f":TRIGger:EDGE:SLOPe {trigger_slope}")
                        scope.set_trigger_mode(trigger_mode)
                        applied_settings['trigger'] = trigger
                    except Exception as e:
                        return jsonify({"error": f"Failed to set trigger: {e}"}), 400
            
            # Apply timebase settings
            timescale = settings.get('timescale', {})
            horizontal_div = timescale.get('horizontal_div')
            if horizontal_div:
                try:
                    time_scale = float(horizontal_div)
                    scope.set_time_scale(time_scale)
                    applied_settings['timescale'] = timescale
                except Exception as e:
                    return jsonify({"error": f"Failed to set timebase: {e}"}), 400

        # Return success with config data for UI update (regardless of connection status)
        result = {
            "status": "success", 
            "config": config,  # Return the full config for UI update
            "applied_settings": applied_settings
        }
        
        if scope:
            result["idn"] = idn
            result["message"] = "Config imported and applied to oscilloscope successfully"
        else:
            result["message"] = f"Config imported for UI update only (connection failed: {connection_error})"
            result["connection_error"] = connection_error
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload-json', methods=['POST'])
def api_upload_json():
    """Upload a JSON file and save it as device settings.json"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Parse and validate JSON
        try:
            file_content = file.read()
            config = json.loads(file_content)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON format: {e}"}), 400
        
        # Validate that config is a dictionary
        if isinstance(config, list):
            # If it's a list, try to find the oscilloscope configuration
            scope_config = None
            for device in config:
                if isinstance(device, dict) and 'SDS' in device.get('instrument', ''):
                    scope_config = device
                    break
            
            if not scope_config:
                return jsonify({"error": "No oscilloscope configuration found in the JSON array. Looking for instrument containing 'SDS'."}), 400
            
            config = scope_config
        elif not isinstance(config, dict):
            return jsonify({"error": f"Invalid JSON format: expected object/dict or array, got {type(config).__name__}"}), 400
        
        # Basic validation
        if not config.get('instrument') or not config.get('ip'):
            return jsonify({"error": "Missing required fields: instrument, ip"}), 400
        
        # Save to device settings.json in current shot folder
        shot_folder = get_current_shot_folder()
        json_file_path = os.path.join(shot_folder, "device settings.json")
        with open(json_file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": f"File uploaded and saved to {os.path.basename(shot_folder)}/device settings.json"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/write-scope-to-json', methods=['POST'])
def api_write_scope_to_json():
    """Write current scope settings to JSON file"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            return error

        # Get current scope settings
        def read_scope_settings():
            settings = {
                "instrument": "Siglent SDS5104X",
                "serial number": scope.query("*IDN?").split(',')[2] if scope.query("*IDN?") else "Unknown",
                "ip": scope.ip_address,
                "port": scope.port,
                "settings": {
                    "channels": {},
                    "trigger": {},
                    "timescale": {}
                }
            }
            
            # Read channel settings
            for ch in range(1, 5):
                try:
                    vdiv_resp = scope.query(f":CHANnel{ch}:SCALe?")
                    offset_resp = scope.query(f":CHANnel{ch}:OFFSet?")
                    state_resp = scope.query(f":CHANnel{ch}:SWITch?")
                    
                    # Parse responses, removing units
                    vdiv = float(vdiv_resp.strip().rstrip('V')) if vdiv_resp else 1.0
                    offset = float(offset_resp.strip().rstrip('V')) if offset_resp else 0.0
                    display = "on" if state_resp and state_resp.strip().upper() == "ON" else "off"
                    
                    settings["settings"]["channels"][f"CH{ch}"] = {
                        "vdiv": vdiv,
                        "offset": offset,
                        "display": display
                    }
                except Exception as e:
                    print(f"Warning: Could not read channel {ch} settings: {e}")
                    settings["settings"]["channels"][f"CH{ch}"] = {
                        "vdiv": 1.0,
                        "offset": 0.0,
                        "display": "on"
                    }
            
            # Read trigger settings
            try:
                trigger_source = scope.query(":TRIGger:EDGE:SOURce?")
                trigger_slope = scope.query(":TRIGger:EDGE:SLOPe?")
                trigger_mode = scope.query(":TRIGger:MODE?")
                
                settings["settings"]["trigger"] = {
                    "source": trigger_source.strip() if trigger_source else "EXT",
                    "slope": trigger_slope.strip() if trigger_slope else "RISE",
                    "mode": trigger_mode.strip() if trigger_mode else "SINGLE"
                }
            except Exception as e:
                print(f"Warning: Could not read trigger settings: {e}")
                settings["settings"]["trigger"] = {
                    "source": "EXT",
                    "slope": "RISE",
                    "mode": "SINGLE"
                }
            
            # Read timebase settings
            try:
                time_scale_resp = scope.query(":TIMebase:SCALe?")
                # Parse and normalize to string format
                time_val = time_scale_resp.strip().rstrip('S') if time_scale_resp else "1E-3"
                settings["settings"]["timescale"] = {
                    "horizontal_div": time_val
                }
            except Exception as e:
                print(f"Warning: Could not read timebase settings: {e}")
                settings["settings"]["timescale"] = {
                    "horizontal_div": "1E-3"
                }
            
            return settings

        # Run in thread pool
        future = executor.submit(read_scope_settings)
        scope_settings = future.result(timeout=15)
        
        # Read existing JSON file and preserve structure (from shot folder)
        shot_folder = get_current_shot_folder()
        json_file_path = os.path.join(shot_folder, "device settings.json")
        
        existing_data = []
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    existing_data = json.load(f)
                    # Ensure it's a list
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
            except Exception as e:
                print(f"Warning: Could not read existing JSON: {e}")
                existing_data = []
        
        # Find and update oscilloscope settings in the array, preserve other devices
        scope_index = None
        for i, device in enumerate(existing_data):
            if isinstance(device, dict) and device.get('ip') == scope.ip_address:
                scope_index = i
                break
        
        if scope_index is not None:
            # Update existing scope entry
            existing_data[scope_index] = scope_settings
        else:
            # Add new scope entry
            existing_data.append(scope_settings)
        
        # Write back to file
        with open(json_file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": f"Scope settings written to {os.path.basename(shot_folder)}/device settings.json",
            "settings": scope_settings
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/set-scope-from-json', methods=['POST'])
def api_set_scope_from_json():
    """Set scope settings from the device settings.json file in current shot folder"""
    try:
        # Read the JSON file first from current shot folder
        shot_folder = get_current_shot_folder()
        json_file_path = os.path.join(shot_folder, "device settings.json")
        if not os.path.exists(json_file_path):
            return jsonify({"error": f"device settings.json file not found in {os.path.basename(shot_folder)}"}), 400
        
        with open(json_file_path, 'r') as f:
            config = json.load(f)
        
        data = request.get_json() or {}
        scope, error = get_target_scope(data)
        if error:
            # If no scope found, still return config for UI update
            scope = None
        
        settings = config.get('settings', {})
        applied_settings = {}
        connection_error = None if scope else "Not connected to oscilloscope"
        
        # Apply settings to scope if connected
        if scope:
            # Apply channel settings
            channels = settings.get('channels', {})
            for ch_name, ch_settings in channels.items():
                if ch_name.startswith('CH'):
                    ch_num = int(ch_name[2:])
                    vdiv = ch_settings.get('vdiv')
                    offset = ch_settings.get('offset', 0)
                    display = ch_settings.get('display', 'on')
                    
                    try:
                        if vdiv is not None:
                            scope.set_voltage_scale(ch_num, vdiv)
                        scope.send_command(f":C{ch_num}:OFFSet {offset}")
                        scope.set_channel_state(ch_num, display.lower() == 'on')
                        applied_settings[f'channel_{ch_num}'] = {'vdiv': vdiv, 'offset': offset, 'display': display}
                    except Exception as e:
                        return jsonify({"error": f"Failed to set channel {ch_num}: {e}"}), 400
            
            # Apply trigger settings (skip if scope is armed/monitoring)
            trigger = settings.get('trigger', {})
            if trigger:
                # Check if there's an active monitoring task
                active_task = None
                for task_id, task_data in tasks.items():
                    if task_data["status"]["status"] in ["armed", "triggered"]:
                        active_task = task_id
                        break
                
                if active_task:
                    print(f"DEBUG: Skipping trigger configuration in set-scope-from-json - active monitoring task {active_task}")
                    applied_settings['trigger'] = "skipped (monitoring active)"
                else:
                    try:
                        trigger_source = trigger.get('source', 'EX')
                        trigger_slope = trigger.get('slope', 'RISing')
                        trigger_mode = trigger.get('mode', 'SINGLE')
                        
                        scope.send_command(f":TRIGger:EDGE:SOURce {trigger_source}")
                        scope.send_command(f":TRIGger:EDGE:SLOPe {trigger_slope}")
                        scope.set_trigger_mode(trigger_mode)
                        applied_settings['trigger'] = trigger
                    except Exception as e:
                        return jsonify({"error": f"Failed to set trigger: {e}"}), 400
            
            # Apply timebase settings
            timescale = settings.get('timescale', {})
            horizontal_div = timescale.get('horizontal_div')
            if horizontal_div:
                try:
                    time_scale = float(horizontal_div)
                    scope.set_time_scale(time_scale)
                    applied_settings['timescale'] = timescale
                except Exception as e:
                    return jsonify({"error": f"Failed to set timebase: {e}"}), 400
        else:
            connection_error = "Not connected to oscilloscope"

        # Return success with config data for UI update (regardless of connection status)
        result = {
            "status": "success",
            "config": config,  # Return the full config for UI update
            "applied_settings": applied_settings
        }
        
        if scope:
            result["message"] = "Settings applied from 'device settings.json'"
        else:
            result["message"] = "Settings loaded for UI update only (scope not connected)"
            result["connection_error"] = connection_error
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-ddg-config', methods=['GET'])
def api_get_ddg_config():
    """Get DDG configuration from device settings.json file in current shot folder"""
    try:
        shot_folder = get_current_shot_folder()
        json_file_path = os.path.join(shot_folder, "device settings.json")
        if not os.path.exists(json_file_path):
            return jsonify({"status": "error", "error": f"device settings.json file not found in {os.path.basename(shot_folder)}"}), 404
        
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single config and array of configs
        ddg_config = None
        if isinstance(data, list):
            # Find DDG config in array
            for device in data:
                if isinstance(device, dict) and '577' in device.get('instrument', ''):
                    ddg_config = device
                    break
        elif isinstance(data, dict) and '577' in data.get('instrument', ''):
            ddg_config = data
        
        if ddg_config:
            return jsonify({"status": "success", "config": ddg_config})
        else:
            return jsonify({"status": "error", "error": "No DDG configuration found"}), 404
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/write-ddg-to-json', methods=['POST'])
def api_write_ddg_to_json():
    """Write DDG settings to device settings.json file in current shot folder"""
    try:
        ddg_config = request.get_json()
        
        if not ddg_config:
            return jsonify({"status": "error", "error": "No DDG configuration provided"}), 400
        
        shot_folder = get_current_shot_folder()
        json_file_path = os.path.join(shot_folder, "device settings.json")
        
        # Read existing file
        existing_data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                existing_data = json.load(f)
                # Ensure it's a list
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        
        # Find and update DDG config in the array, or add it
        ddg_index = None
        for i, device in enumerate(existing_data):
            if isinstance(device, dict) and '577' in device.get('instrument', ''):
                ddg_index = i
                break
        
        if ddg_index is not None:
            # Update existing DDG config
            existing_data[ddg_index] = ddg_config
        else:
            # Add new DDG config
            existing_data.append(ddg_config)
        
        # Write back to file
        with open(json_file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return jsonify({
            "status": "success",
            "message": "DDG settings written to 'device settings.json'"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ensure waveform_data directory exists
WAVEFORM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waveform_data")
os.makedirs(WAVEFORM_DIR, exist_ok=True)

def create_plot(csv_file, plot_file):
    """
    Create matplotlib plot from CSV file and save as PNG
    
    Args:
        csv_file (str): Path to CSV file
        plot_file (str): Path to save plot PNG
        
    Returns:
        bool: Success status
    """
    try:
        df = pd.read_csv(csv_file)
        if "Time (s)" not in df.columns or "Voltage (V)" not in df.columns:
            print(f"Error: CSV file '{csv_file}' does not contain expected columns.")
            return False
        
        time_data = df["Time (s)"]
        voltage_data = df["Voltage (V)"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_data * 1e6, voltage_data)  # Plot time in microseconds
        plt.title(f"Waveform Data from {os.path.basename(csv_file)}")
        plt.xlabel("Time (μs)")
        plt.ylabel("Voltage (V)")
        plt.grid(True)
        plt.margins(x=0.01, y=0.1)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()  # Important: close figure to free memory
        
        return True
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        traceback.print_exc()
        return False

def acquire_all_channels_worker(scope, task_id):
    """
    Background worker for monitoring trigger and acquiring from all channels
    
    Args:
        scope: SocketScope instance
        task_id (str): Task identifier
    """
    try:
        # Update status - scope is now armed
        tasks[task_id]["status"] = {
            "status": "armed",
            "progress": "Scope armed, waiting for trigger...",
            "result": None,
            "error": None
        }
        
        # Event-driven polling for trigger status
        poll_timeout = 300  # 5 minutes timeout
        poll_interval = 0.1  # 100ms polling
        start_time = time.time()
        triggered = False
        
        print(f"Scope armed. Monitoring for trigger event (timeout: {poll_timeout}s)...")
        
        while time.time() - start_time < poll_timeout:
            try:
                trig_status = scope.query(":TRIGger:STATus?").strip()
                
                if trig_status in ["Stop", "FStop", "Trig'd"]:
                    triggered = True
                    print(f"✓ Trigger detected! Scope status: {trig_status}")
                    break
                elif trig_status in ["Ready", "Arm", "Auto"]:
                    # Scope is armed and waiting - continue polling
                    time.sleep(poll_interval)
                else:
                    # Unknown status - log and continue
                    time.sleep(poll_interval)
                    
            except Exception as e:
                print(f"Error polling trigger status: {e}")
                time.sleep(poll_interval)
        
        if not triggered:
            print(f"✗ Trigger event not detected within {poll_timeout} seconds.")
            tasks[task_id]["status"] = {
                "status": "timeout",
                "progress": f"Timeout waiting for trigger after {poll_timeout}s",
                "result": None,
                "error": f"Trigger timeout after {poll_timeout} seconds. Scope disarmed."
            }
            return
        
        # Update status - triggered, now acquiring
        tasks[task_id]["status"]["status"] = "triggered"
        tasks[task_id]["status"]["progress"] = "Trigger detected! Acquiring data from all channels..."
        
        # Get current shot folder for saving data
        shot_folder = get_current_shot_folder()
        shot_folder_name = os.path.basename(shot_folder)
        # Extract date and shot number from folder name (e.g., p251123_04 -> 251123_04)
        date_shot = shot_folder_name[1:]  # Remove 'p' prefix
        
        # Get scope label (scope1, scope2, or scope3 based on connection order)
        scope_label = get_scope_label(scope.ip_address)
        
        # IMPORTANT: Acquire waveform data FIRST before screenshot
        # Screenshot command contaminates the buffer and must come last
        
        # Small delay to let scope settle after trigger
        time.sleep(0.2)
        
        # Set waveform parameters for full memory depth
        try:
            scope.send_command("WAV:POIN 0")  # Request all available points
            time.sleep(0.1)
        except:
            pass
        
        # Send synchronization command
        try:
            scope.send_command("*CLS")
            time.sleep(0.1)
        except:
            pass
        
        # Acquire data from all 4 channels
        # Try to acquire from all channels - let get_waveform_data handle failures
        all_channel_data = {}
        
        for ch in range(1, 5):
            try:
                print(f"Acquiring data from channel {ch}...")
                time_data, voltage_data = scope.get_waveform_data(ch)

                if time_data is not None and voltage_data is not None and len(time_data) > 0:
                    all_channel_data[ch] = {
                        'time': time_data,
                        'voltage': voltage_data
                    }
                    print(f"✓ Channel {ch}: {len(time_data)} points acquired")
                else:
                    print(f"⚠ Channel {ch}: No data received (channel may be OFF or preamble invalid)")

            except Exception as e:
                print(f"⚠ Error acquiring channel {ch}: {e}")
            finally:
                # Small gap between channel acquisitions to let instrument settle
                time.sleep(0.05)
        
        if not all_channel_data:
            tasks[task_id]["status"] = {
                "status": "error",
                "progress": "Failed to acquire waveform data from any channel",
                "result": None,
                "error": "No waveform data acquired. Check that channels are enabled."
            }
            return
        
        # Update status - saving
        tasks[task_id]["status"]["progress"] = "Saving data to CSV..."
        
        # Create combined CSV file with all channels in shot folder
        # Format: scope1_waveform_YYMMDD_NN.csv (e.g., scope1_waveform_251123_04.csv)
        csv_file = os.path.join(shot_folder, f"{scope_label}_waveform_{date_shot}.csv")
        
        try:
            # Find the maximum number of data points
            max_points = max(len(data['time']) for data in all_channel_data.values())
            
            with open(csv_file, 'w', newline='') as f:
                # Write header
                header = "Time (s)"
                for ch in sorted(all_channel_data.keys()):
                    header += f",CH{ch} Voltage (V)"
                f.write(header + "\n")
                
                # Write data rows
                for i in range(max_points):
                    row = ""
                    
                    # Use time from first available channel
                    first_ch = sorted(all_channel_data.keys())[0]
                    if i < len(all_channel_data[first_ch]['time']):
                        row = f"{all_channel_data[first_ch]['time'][i]:.12e}"
                    else:
                        row = ""
                    
                    # Add voltage data for each channel
                    for ch in sorted(all_channel_data.keys()):
                        if i < len(all_channel_data[ch]['voltage']):
                            row += f",{all_channel_data[ch]['voltage'][i]:.6f}"
                        else:
                            row += ","
                    
                    f.write(row + "\n")
            
            print(f"✓ Combined waveform data saved to {csv_file}")
            
        except Exception as e:
            print(f"Error saving combined CSV: {e}")
            traceback.print_exc()
            tasks[task_id]["status"] = {
                "status": "error",
                "progress": "Failed to save CSV file",
                "result": None,
                "error": str(e)
            }
            return
        
        # Create plot of waveform data and save as PDF in shot folder
        tasks[task_id]["status"]["progress"] = "Creating waveform plot..."
        pdf_file = os.path.join(shot_folder, f"{scope_label}_waveform_{date_shot}.pdf")
        plot_success = False
        
        # Use lock to prevent matplotlib threading issues when multiple scopes plot simultaneously
        with plot_lock:
            try:
                # Create figure with subplots for each channel
                num_channels = len(all_channel_data)
                fig, axes = plt.subplots(num_channels, 1, figsize=(12, 3*num_channels), sharex=True)
                
                # Handle single channel case (axes is not a list)
                if num_channels == 1:
                    axes = [axes]
                
                # Plot each channel
                for idx, (ch, data) in enumerate(sorted(all_channel_data.items())):
                    time_us = [t * 1e6 for t in data['time']]  # Convert to microseconds
                    axes[idx].plot(time_us, data['voltage'], linewidth=0.5)
                    axes[idx].set_ylabel(f'CH{ch} Voltage (V)', fontsize=10)
                    axes[idx].grid(True, alpha=0.3)
                    axes[idx].set_title(f'Channel {ch}', fontsize=11, loc='left')
                    
                    # Set y-axis limits with minimum 5V range
                    v_min = min(data['voltage'])
                    v_max = max(data['voltage'])
                    v_range = v_max - v_min
                    
                    if v_range < 5.0:
                        # Expand range to 5V centered on the data
                        v_center = (v_max + v_min) / 2
                        axes[idx].set_ylim(v_center - 2.5, v_center + 2.5)
                    else:
                        # Add 5% padding to existing range
                        padding = v_range * 0.05
                        axes[idx].set_ylim(v_min - padding, v_max + padding)
                
                # Set x-label only on bottom plot
                axes[-1].set_xlabel('Time (μs)', fontsize=10)
                
                # Add overall title
                fig.suptitle(f'Oscilloscope Waveform - {scope_label.capitalize()} - Shot {date_shot}', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
                plt.close(fig)
                plot_success = True
                print(f"✓ Waveform plot saved to {pdf_file} for {scope_label}")
            except Exception as e:
                print(f"ERROR: Could not create waveform plot for {scope_label}: {e}")
                traceback.print_exc()
                plot_success = False
        
        # Update status to complete
        result_data = {
            "csv_file": os.path.basename(csv_file),
            "pdf_file": os.path.basename(pdf_file) if plot_success else None,
            "shot_folder": os.path.basename(shot_folder),
            "date_shot": date_shot,
            "scope_label": scope_label,
            "channels": list(all_channel_data.keys())
        }
        
        tasks[task_id]["status"] = {
            "status": "complete",
            "progress": "Acquisition complete! Downloading...",
            "result": result_data,
            "error": None
        }
        
        print(f"✓ Acquisition complete! File: {csv_file}")
        print(f"✓ Data saved to shot folder: {shot_folder}")
        print(f"✓ Scope label: {scope_label}")
        
    except Exception as e:
        print(f"Error in acquisition worker: {e}")
        traceback.print_exc()
        tasks[task_id]["status"] = {
            "status": "error",
            "progress": "Acquisition failed",
            "result": None,
            "error": str(e)
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

# Connection Management API
@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to oscilloscope"""
    try:
        data = request.get_json()
        ip = data.get('ip', '192.168.1.10')
        port = data.get('port', 5025)
        timeout = data.get('timeout', 15.0)
        
        # Run connection in thread pool
        future = executor.submit(scope_manager.connect_scope, ip, port, timeout)
        success, idn, error = future.result(timeout=20)  # 20s timeout for connection
        
        if success:
            # Read current scope settings
            scope = scope_manager.get_active_scope()
            current_settings = {}
            
            if scope:
                # Wrap all settings reading in a try-catch with short timeout
                # to prevent connection failure if scope is slow to respond
                try:
                    # Read all current settings from the scope
                    def read_current_settings():
                        settings = {}
                        
                        # Read channel settings
                        settings['channels'] = {}
                        for ch in range(1, 5):
                            try:
                                vdiv = scope.query(f":CHANnel{ch}:SCALe?")
                                offset = scope.query(f":CHANnel{ch}:OFFSet?")
                                state = scope.query(f":CHANnel{ch}:SWITch?")
                                
                                settings['channels'][f'CH{ch}'] = {
                                    'vdiv': float(vdiv.strip().rstrip('V')) if vdiv else 1.0,
                                    'offset': float(offset.strip().rstrip('V')) if offset else 0.0,
                                    'display': state.strip().upper() == 'ON' if state else True
                                }
                            except Exception as e:
                                print(f"Warning: Could not read channel {ch}: {e}")
                                settings['channels'][f'CH{ch}'] = {
                                    'vdiv': 1.0,
                                    'offset': 0.0,
                                    'display': True
                                }
                        
                        # Read trigger settings
                        try:
                            trig_src = scope.query(":TRIGger:EDGE:SOURce?")
                            trig_slope = scope.query(":TRIGger:EDGE:SLOPe?")
                            trig_mode = scope.query(":TRIGger:MODE?")
                            
                            settings['trigger'] = {
                                'source': trig_src.strip() if trig_src else 'EX',
                                'slope': trig_slope.strip() if trig_slope else 'RISing',
                                'mode': trig_mode.strip() if trig_mode else 'SINGLE'
                            }
                        except Exception as e:
                            print(f"Warning: Could not read trigger settings: {e}")
                            settings['trigger'] = {
                                'source': 'EX',
                                'slope': 'RISing',
                                'mode': 'SINGLE'
                            }
                        
                        # Read timebase
                        try:
                            timebase = scope.query(":TIMebase:SCALe?")
                            settings['timebase'] = {
                                'horizontal_div': timebase.strip().rstrip('S') if timebase else '1E-3'
                            }
                        except Exception as e:
                            print(f"Warning: Could not read timebase: {e}")
                            settings['timebase'] = {
                                'horizontal_div': '1E-3'
                            }
                        
                        return settings
                    
                    future_settings = executor.submit(read_current_settings)
                    current_settings = future_settings.result(timeout=5)
                    
                except Exception as e:
                    print(f"Warning: Could not read scope settings: {e}")
                    # Use default settings to allow connection to proceed
                    current_settings = {
                        'channels': {
                            'CH1': {'vdiv': 1.0, 'offset': 0.0, 'display': True},
                            'CH2': {'vdiv': 1.0, 'offset': 0.0, 'display': True},
                            'CH3': {'vdiv': 1.0, 'offset': 0.0, 'display': True},
                            'CH4': {'vdiv': 1.0, 'offset': 0.0, 'display': True}
                        },
                        'trigger': {'source': 'EX', 'slope': 'RISing', 'mode': 'SINGLE'},
                        'timebase': {'horizontal_div': '1E-3'}
                    }
            
            # Check if scope is already armed on SINGLE trigger
            armed_status = {"armed": False, "task_id": None}
            if scope:
                try:
                    # Query trigger mode and status directly with short timeout
                    original_timeout = scope.socket.gettimeout()
                    scope.socket.settimeout(2.0)
                    
                    trig_mode = scope.query(":TRIGger:MODE?").strip()
                    trig_status = scope.query(":TRIGger:STATus?").strip()
                    
                    scope.socket.settimeout(original_timeout)
                    
                    print(f"DEBUG: Trigger mode = '{trig_mode}', Trigger status = '{trig_status}'")
                    
                    # Check if in SINGLE mode and armed/ready (case-insensitive comparison)
                    if trig_mode.upper() == "SINGLE" and trig_status in ["Ready", "Arm", "Auto"]:
                        # Create a background task to monitor this armed state
                        task_id = str(uuid.uuid4())
                        
                        # Set up the task before submitting
                        tasks[task_id] = {
                            "future": None,
                            "status": {
                                "status": "armed",
                                "progress": "Scope was already armed, monitoring for trigger...",
                                "result": None,
                                "error": None
                            }
                        }
                        
                        # Now submit the worker
                        future = executor.submit(acquire_all_channels_worker, scope, task_id)
                        tasks[task_id]["future"] = future
                        
                        armed_status = {"armed": True, "task_id": task_id}
                        print(f"DEBUG: Scope is armed! Created task {task_id}")
                    else:
                        print(f"DEBUG: Scope is not armed (mode={trig_mode}, status={trig_status})")
                    
                except Exception as e:
                    print(f"Warning: Could not check armed status: {e}")
                    traceback.print_exc()
                    armed_status = {"armed": False, "task_id": None}
            
            return jsonify({
                "status": "connected",
                "ip": ip,
                "idn": idn,
                "error": None,
                "settings": current_settings,
                "armed_status": armed_status
            })
        else:
            return jsonify({
                "status": "error",
                "idn": None,
                "error": error,
                "settings": {}
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "idn": None,
            "error": str(e),
            "settings": {}
        }), 500

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """Disconnect from oscilloscope"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            # If scope not found, consider it already disconnected
            return jsonify({
                "status": "disconnected",
                "error": None
            })
        
        # Try to stop any active trigger mode before disconnecting
        try:
            print(f"DEBUG: Stopping trigger mode for scope at {scope.ip_address}")
            scope.send_command(":STOP")  # Stop acquisition
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not stop trigger mode: {e}")
        
        # Run disconnection in thread pool
        future = executor.submit(scope_manager.disconnect_scope, scope.ip_address)
        success, error = future.result(timeout=10)
        
        return jsonify({
            "status": "disconnected",
            "error": error
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Configuration API
@app.route('/api/config/channel', methods=['POST'])
def api_config_channel():
    """Configure channel settings"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            return error
        channel = data.get('channel', 1)
        state = data.get('state', True)
        voltage_scale = data.get('voltage_scale', 1.0)
        offset = data.get('offset', 0.0)
        
        print(f"DEBUG: Configuring CH{channel}: state={state} (type={type(state)}), vdiv={voltage_scale}, offset={offset}")
        
        # Run configuration in thread pool
        def config_worker():
            scope.set_channel_state(channel, state)
            scope.set_voltage_scale(channel, voltage_scale)
            # Add offset setting if the scope supports it
            try:
                scope.send_command(f":CHANnel{channel}:OFFSet {offset}")
            except Exception as e:
                print(f"Warning: Could not set offset for channel {channel}: {e}")
        
        future = executor.submit(config_worker)
        future.result(timeout=10)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/timebase', methods=['POST'])
def api_config_timebase():
    """Configure timebase settings"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            return error
        time_scale = data.get('time_scale', 1e-6)
        
        # Run configuration in thread pool
        future = executor.submit(scope.set_time_scale, time_scale)
        future.result(timeout=10)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/trigger', methods=['POST'])
def api_config_trigger():
    """Configure trigger settings"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            return error
        mode = data.get('mode', 'SINGLE')
        source = data.get('source', 'EX')
        slope = data.get('slope', 'RISing')
        level = data.get('level', 0.5)
        
        # Check if there's an active monitoring task
        active_task = None
        for task_id, task_data in tasks.items():
            if task_data["status"]["status"] in ["armed", "triggered"]:
                active_task = task_id
                break
        
        if active_task:
            print(f"DEBUG: Skipping trigger configuration - active monitoring task {active_task}")
            return jsonify({
                "status": "skipped",
                "message": "Trigger settings not applied while monitoring for trigger",
                "active_task": active_task
            })
        
        # Run configuration in thread pool
        def trigger_worker():
            scope.send_command(":TRIGger:TYPE EDGE")
            scope.send_command(f":TRIGger:EDGE:SOURce {source}")
            scope.send_command(f":TRIGger:EDGE:SLOPe {slope}")
            scope.send_command(f":TRIGger:EDGE:LEVel {level}")
            scope.set_trigger_mode(mode)
        
        future = executor.submit(trigger_worker)
        future.result(timeout=10)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Acquisition API
@app.route('/api/arm', methods=['POST'])
def api_arm():
    """Arm scope and monitor for trigger"""
    try:
        data = request.get_json()
        scope, error = get_target_scope(data)
        if error:
            return error
        
        # Check if scope is already armed
        trig_mode = scope.query(":TRIGger:MODE?").strip()
        trig_status = scope.query(":TRIGger:STATus?").strip()
        
        already_armed = False
        if trig_mode.upper() == "SINGLE" and trig_status in ["Ready", "Arm", "Auto"]:
            already_armed = True
            print(f"DEBUG: Scope was already armed (mode={trig_mode}, status={trig_status})")
        else:
            # Set trigger mode to SINGLE
            scope.set_trigger_mode("SINGLE")
            print(f"DEBUG: Armed scope (was mode={trig_mode}, status={trig_status})")
        
        # Create acquisition task
        task_id = str(uuid.uuid4())
        future = executor.submit(acquire_all_channels_worker, scope, task_id)
        
        progress_msg = "Scope was already armed, monitoring for trigger..." if already_armed else "Scope armed, waiting for trigger..."
        
        tasks[task_id] = {
            "future": future,
            "status": {
                "status": "armed",
                "progress": progress_msg,
                "result": None,
                "error": None
            }
        }
        
        return jsonify({
            "task_id": task_id,
            "already_armed": already_armed,
            "message": progress_msg
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/arm/status/<task_id>', methods=['GET'])
def api_arm_status(task_id):
    """Get arm/acquisition status"""
    try:
        if task_id not in tasks:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify(tasks[task_id]["status"])
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# File access API
@app.route('/api/download/<filename>')
def api_download(filename):
    """Download CSV or PDF file from current shot folder"""
    try:
        # Look in current shot folder first
        shot_folder = get_current_shot_folder()
        file_path = os.path.join(shot_folder, filename)
        
        # Fallback to old waveform_data directory for backward compatibility
        if not os.path.exists(file_path):
            file_path = os.path.join(WAVEFORM_DIR, filename)
        
        if not os.path.exists(file_path):
            abort(404)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        abort(500)

@app.route('/api/plot/<filename>')
def api_plot(filename):
    """Serve plot image"""
    try:
        file_path = os.path.join(WAVEFORM_DIR, filename)
        if not os.path.exists(file_path):
            abort(404)
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        abort(500)

if __name__ == '__main__':
    print("Starting Flask oscilloscope control application...")
    print(f"Waveform data directory: {WAVEFORM_DIR}")
    app.run(host='0.0.0.0', port=5001, debug=True)
