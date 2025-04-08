import os
import glob
import datetime
import time
import threading
import json
import re
import logging
from collections import Counter

import pandas as pd
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS

from typing import Optional

# --- Configuration ---
LOG_MONITOR_PORT: int = 8777
MONITORED_LOG_DIR: str = "valuation_v3_web_log" # Directory where the matchmaking logs are
FILE_SCAN_INTERVAL_SECONDS: int = 30 # How often to check for new log file / reread
DATA_RETENTION_MINUTES: int = 30 # How far back to keep/display logs
FRONTEND_UPDATE_INTERVAL_MS: int = 5000 # How often the frontend JS polls for data (in milliseconds)

LOG_LEVEL = logging.INFO # Logging for the dashboard itself
DASHBOARD_LOG_FILE = os.path.join(MONITORED_LOG_DIR, "dashboard_monitor.log")

# --- Dashboard Logging Setup ---
# Avoid conflict with the monitored app's logging if run in same env
dashboard_logger = logging.getLogger("dashboard_logger")
dashboard_logger.setLevel(LOG_LEVEL)
# Log to file
fh = logging.FileHandler(DASHBOARD_LOG_FILE, encoding='utf-8')
fh.setLevel(LOG_LEVEL)
# Log to console
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Dashboard] - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# Add handlers
if not dashboard_logger.handlers: # Avoid adding handlers multiple times if script reloads
    dashboard_logger.addHandler(fh)
    dashboard_logger.addHandler(ch)

# --- Global Data Store ---
log_data_lock = threading.Lock()
# Initialize with empty DataFrame with correct columns and types
log_data_df = pd.DataFrame(columns=[
    'timestamp', 'level', 'thread', 'func', 'message'
])
log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp']) # Ensure correct dtype

last_file_processed: Optional[str] = None
last_log_count: int = 0
last_error_count: int = 0
last_warning_count: int = 0
last_update_time: Optional[datetime.datetime] = None
log_parsing_errors: int = 0

# Regex to parse the specific log format
# Example: 2023-11-20 15:30:00,123 - INFO - [MainThread] - my_function - This is a log message
# Improved regex to handle potential variations, especially in thread name
LOG_LINE_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})" # Timestamp
    r"\s+-\s+" # Separator
    r"(?P<level>\w+)" # Log Level (INFO, WARNING, ERROR, etc.)
    r"\s+-\s+" # Separator
    r"\[(?P<thread>.*?)\]" # Thread Name (non-greedy)
    r"\s+-\s+" # Separator
    r"(?P<func>\w+)" # Function Name
    r"\s+-\s+" # Separator
    r"(?P<message>.*)$" # The rest is the message
)

# --- Helper Functions ---
def find_latest_log_file(directory: str) -> Optional[str]:
    """Finds the most recently modified .log file in a directory."""
    try:
        log_files = glob.glob(os.path.join(directory, "*.log"))
        # Exclude our own log file
        log_files = [f for f in log_files if os.path.basename(f) != os.path.basename(DASHBOARD_LOG_FILE)]
        if not log_files:
            return None
        latest_file = max(log_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        dashboard_logger.error(f"Error finding latest log file in '{directory}': {e}", exc_info=True)
        return None

def parse_log_file(file_path: str) -> pd.DataFrame:
    """Parses a log file into a Pandas DataFrame."""
    global log_parsing_errors
    parsed_lines = []
    errors_in_parse = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                match = LOG_LINE_REGEX.match(line)
                if match:
                    log_entry = match.groupdict()
                    try:
                         # Convert timestamp string to datetime object
                         log_entry['timestamp'] = datetime.datetime.strptime(log_entry['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                         parsed_lines.append(log_entry)
                    except ValueError as ve:
                         dashboard_logger.warning(f"File '{os.path.basename(file_path)}', Line {i+1}: Timestamp parse error: {ve} - Line: '{line[:100]}...'")
                         errors_in_parse += 1
                else:
                    # Maybe log the unparsed line if debugging needed
                    # dashboard_logger.debug(f"Line {i+1} did not match regex: {line}")
                    errors_in_parse += 1 # Count lines that don't match format

    except FileNotFoundError:
        dashboard_logger.error(f"Log file not found during parsing: {file_path}")
        return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'func', 'message']) # Return empty
    except Exception as e:
        dashboard_logger.error(f"Error reading/parsing log file '{file_path}': {e}", exc_info=True)
        return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'func', 'message']) # Return empty

    with log_data_lock: # Update global parse error count
        log_parsing_errors += errors_in_parse

    if not parsed_lines:
         return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'func', 'message'])

    df = pd.DataFrame(parsed_lines)
    # Ensure correct dtypes after creation
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['level'] = df['level'].astype(str)
    df['thread'] = df['thread'].astype(str)
    df['func'] = df['func'].astype(str)
    df['message'] = df['message'].astype(str)
    return df

def update_log_data():
    """Background task to find, parse, and update log data."""
    global log_data_df, last_file_processed, last_log_count, last_error_count, \
           last_warning_count, last_update_time, log_parsing_errors

    dashboard_logger.info("Background update task started.")
    while True:
        try:
            latest_file = find_latest_log_file(MONITORED_LOG_DIR)
            current_time = datetime.datetime.now()

            if latest_file:
                # Read the entire latest file for simplicity
                dashboard_logger.info(f"Processing latest log file: {os.path.basename(latest_file)}")
                new_df = parse_log_file(latest_file)

                if not new_df.empty:
                    # Filter out very old data immediately after parsing if needed,
                    # but filtering on request might be better for stats.
                    # cutoff_time = current_time - datetime.timedelta(minutes=DATA_RETENTION_MINUTES * 2) # Keep a bit longer internally
                    # new_df = new_df[new_df['timestamp'] >= cutoff_time]

                    with log_data_lock:
                        # Replace the entire DataFrame (simplest approach)
                        log_data_df = new_df
                        last_file_processed = os.path.basename(latest_file)
                        last_update_time = current_time

                        # Calculate stats on the full new DF
                        last_log_count = len(log_data_df)
                        level_counts = log_data_df['level'].value_counts()
                        last_error_count = level_counts.get('ERROR', 0)
                        last_warning_count = level_counts.get('WARNING', 0)

                        dashboard_logger.info(f"Data updated. Total lines: {last_log_count}, Errors: {last_error_count}, Warnings: {last_warning_count}. Parse Errors: {log_parsing_errors}")
                else:
                     dashboard_logger.info(f"Log file '{os.path.basename(latest_file)}' parsed but resulted in empty DataFrame.")
                     # Optionally clear old stats if file is empty? Or keep last known stats? Keeping seems better.

            else:
                dashboard_logger.warning(f"No log files found in '{MONITORED_LOG_DIR}'.")
                # Clear data if no logs found for a while?
                with log_data_lock:
                    if not log_data_df.empty: # Clear only if it had data before
                        dashboard_logger.info("Clearing stale data as no log file was found.")
                        log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'func', 'message'])
                        log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                        last_log_count = 0
                        last_error_count = 0
                        last_warning_count = 0
                        last_file_processed = None
                        last_update_time = current_time

        except Exception as e:
            dashboard_logger.error(f"Error in background update loop: {e}", exc_info=True)

        # Wait before the next scan
        time.sleep(FILE_SCAN_INTERVAL_SECONDS)


# --- Flask App ---
app = Flask(__name__)
CORS(app) # Allow all origins

# --- HTML Template ---
# (Includes Tailwind CDN, FontAwesome, Chart.js CDN, and JS for polling/updating)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Monitor Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 50%, #1a202c 100%); /* Dark gradient */
            color: #e2e8f0; /* Light gray text */
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }

        /* Glassmorphism Card Base */
        .glass-card {
            background: rgba(45, 55, 72, 0.6); /* bg-gray-700 with opacity */
            backdrop-filter: blur(12px) saturate(150%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.75rem; /* rounded-xl */
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.2 ); /* Subtle shadow */
        }
         .glass-card:hover {
             background: rgba(45, 55, 72, 0.7);
             transform: translateY(-3px);
             box-shadow: 0 12px 40px 0 rgba( 31, 38, 135, 0.25 );
         }

        /* Specific Level Colors */
        .level-DEBUG { color: #a0aec0; } /* gray-500 */
        .level-INFO { color: #63b3ed; } /* blue-400 */
        .level-WARNING { color: #f6e05e; } /* yellow-400 */
        .level-ERROR { color: #fc8181; } /* red-400 */
        .level-CRITICAL { color: #f56565; background-color: rgba(255,0,0,0.1); padding: 0 2px; border-radius: 3px;} /* red-500 */

        .icon-DEBUG { color: #a0aec0; }
        .icon-INFO { color: #63b3ed; }
        .icon-WARNING { color: #f6e05e; }
        .icon-ERROR { color: #fc8181; }
        .icon-CRITICAL { color: #f56565; }

        /* Log Entry Styling */
        .log-entry {
            font-family: 'Menlo', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
            font-size: 0.8rem;
            line-height: 1.4;
            padding: 0.3rem 0.5rem;
            border-radius: 0.25rem;
            margin-bottom: 0.25rem;
            background-color: rgba(26, 32, 44, 0.4); /* gray-800 slightly transparent */
             display: flex;
             flex-wrap: nowrap; /* Prevent wrapping */
             gap: 0.5rem;
        }
         .log-entry:hover {
             background-color: rgba(26, 32, 44, 0.7);
         }
         .log-time { color: #718096; flex-shrink: 0;} /* gray-600 */
         .log-level { font-weight: bold; width: 70px; flex-shrink: 0; text-align: right;}
         .log-func { color: #9f7aea; width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } /* purple-500 */
         .log-message { color: #e2e8f0; flex-grow: 1; word-break: break-word; white-space: pre-wrap;} /* Allow message wrap */


        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(74, 85, 104, 0.3); border-radius: 3px;} /* gray-600ish transparent */
        ::-webkit-scrollbar-thumb { background: rgba(160, 174, 192, 0.5); border-radius: 3px; } /* gray-500ish */
        ::-webkit-scrollbar-thumb:hover { background: rgba(160, 174, 192, 0.7); }

        /* Fade-in animation */
         @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.5s ease-out forwards; }

         #logContainer .log-entry { opacity: 0; } /* Start hidden for animation */
    </style>
</head>
<body class="p-4 md:p-8 min-h-screen">

    <!-- Header -->
    <header class="mb-8 flex justify-between items-center">
        <h1 class="text-3xl md:text-4xl font-orbitron font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
           <i class="fas fa-wave-square mr-2"></i>Log Monitor
        </h1>
        <div class="text-xs text-gray-400 text-right">
             <p>Watching: <code id="monitoringFile" class="text-gray-300 bg-gray-700 px-1 rounded">N/A</code></p>
             <p>Last Update: <span id="lastUpdateTime" class="text-gray-300">Never</span></p>
             <p><i class="fas fa-bug mr-1 text-red-500"></i> Parse Errors: <span id="parseErrorCount" class="text-red-400 font-semibold">0</span></p>
         </div>
    </header>

    <!-- Stats Cards -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-8">
        <!-- Total Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in" style="animation-delay: 0.1s;">
            <div class="text-3xl text-blue-400"><i class="fas fa-file-alt fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-sm uppercase tracking-wider">Total Logs (30min)</div>
                <div id="totalLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
        <!-- Info Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in" style="animation-delay: 0.2s;">
             <div class="text-3xl text-blue-400"><i class="fas fa-info-circle fa-fw"></i></div>
             <div>
                 <div class="text-gray-400 text-sm uppercase tracking-wider">Info</div>
                 <div id="infoLogs" class="text-2xl font-bold text-gray-100">0</div>
             </div>
         </div>
        <!-- Warning Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in" style="animation-delay: 0.3s;">
            <div class="text-3xl text-yellow-400"><i class="fas fa-exclamation-triangle fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-sm uppercase tracking-wider">Warnings</div>
                <div id="warningLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
        <!-- Error Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in" style="animation-delay: 0.4s;">
            <div class="text-3xl text-red-400"><i class="fas fa-times-circle fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-sm uppercase tracking-wider">Errors</div>
                <div id="errorLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
    </section>

    <!-- Chart and Recent Logs -->
    <section class="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8">
        <!-- Chart Column -->
        <div class="lg:col-span-2">
             <div class="glass-card p-4 h-full fade-in" style="animation-delay: 0.5s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300"><i class="fas fa-chart-bar mr-2 text-cyan-400"></i>Log Activity (Last 30 Minutes)</h2>
                <div class="relative h-64 md:h-80">
                   <canvas id="logChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Recent Logs Column -->
        <div class="lg:col-span-1">
             <div class="glass-card p-4 h-full flex flex-col fade-in" style="animation-delay: 0.6s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300"><i class="fas fa-stream mr-2 text-cyan-400"></i>Recent Log Entries</h2>
                <div id="logContainer" class="flex-grow overflow-y-auto pr-2 space-y-1" style="max-height: 400px;">
                    <!-- Log entries will be injected here -->
                     <p id="noLogsMessage" class="text-center text-gray-500 italic mt-4">Waiting for log data...</p>
                 </div>
            </div>
        </div>
    </section>

     <!-- Footer -->
     <footer class="text-center text-xs text-gray-500 mt-10 pt-4 border-t border-gray-700/50">
         Log Monitor Dashboard | Powered by Flask & Chart.js | <span id="currentYear"></span>
     </footer>


    <script>
        const API_ENDPOINT = '/api/data';
        const UPDATE_INTERVAL = {{ FRONTEND_UPDATE_INTERVAL_MS }}; // Get from Flask config
        let logChart = null; // Chart instance holder

        // --- DOM Elements ---
        const totalLogsEl = document.getElementById('totalLogs');
        const infoLogsEl = document.getElementById('infoLogs');
        const warningLogsEl = document.getElementById('warningLogs');
        const errorLogsEl = document.getElementById('errorLogs');
        const logContainerEl = document.getElementById('logContainer');
        const noLogsMessageEl = document.getElementById('noLogsMessage');
        const monitoringFileEl = document.getElementById('monitoringFile');
        const lastUpdateTimeEl = document.getElementById('lastUpdateTime');
        const parseErrorCountEl = document.getElementById('parseErrorCount');
        const currentYearEl = document.getElementById('currentYear');

        // --- Chart Setup ---
        function initializeChart() {
            const ctx = document.getElementById('logChart').getContext('2d');
            logChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [], // Timestamps
                    datasets: [
                        {
                            label: 'Info',
                            data: [],
                            backgroundColor: 'rgba(99, 179, 237, 0.6)', // blue-400
                            borderColor: 'rgba(99, 179, 237, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Warning',
                            data: [],
                             backgroundColor: 'rgba(246, 224, 94, 0.6)', // yellow-400
                             borderColor: 'rgba(246, 224, 94, 1)',
                             borderWidth: 1
                         },
                         {
                             label: 'Error',
                             data: [],
                             backgroundColor: 'rgba(252, 129, 129, 0.6)', // red-400
                             borderColor: 'rgba(252, 129, 129, 1)',
                            borderWidth: 1
                        }
                     ]
                 },
                 options: {
                     responsive: true,
                     maintainAspectRatio: false,
                     scales: {
                         x: {
                             stacked: true,
                             ticks: { color: '#a0aec0' }, // gray-500
                             grid: { color: 'rgba(255, 255, 255, 0.1)' }
                         },
                         y: {
                             stacked: true,
                             beginAtZero: true,
                             ticks: { color: '#a0aec0', stepSize: 1 }, // Ensure integer steps if counts are low
                             grid: { color: 'rgba(255, 255, 255, 0.1)' }
                         }
                     },
                     plugins: {
                         legend: {
                             position: 'bottom',
                             labels: { color: '#e2e8f0' } // gray-200
                         },
                         tooltip: {
                             mode: 'index',
                             intersect: false,
                             backgroundColor: 'rgba(26, 32, 44, 0.9)', // gray-800
                             titleColor: '#e2e8f0',
                             bodyColor: '#e2e8f0',
                         }
                     }
                 }
             });
        }

        // --- Log Level Formatting ---
        function getLevelClass(level) {
             return `level-${level.toUpperCase()}`;
         }
         function getLevelIcon(level) {
             level = level.toUpperCase();
             switch (level) {
                 case 'DEBUG': return 'fas fa-bug';
                 case 'INFO': return 'fas fa-info-circle';
                 case 'WARNING': return 'fas fa-exclamation-triangle';
                 case 'ERROR': return 'fas fa-times-circle';
                 case 'CRITICAL': return 'fas fa-skull-crossbones'; // Or fa-bomb
                 default: return 'fas fa-question-circle';
             }
         }
         function getIconClass(level) {
             return `icon-${level.toUpperCase()}`;
         }

         // --- Data Update Function ---
         async function fetchDataAndUpdate() {
             try {
                 const response = await fetch(API_ENDPOINT);
                 if (!response.ok) {
                     console.error("Failed to fetch data:", response.status, response.statusText);
                     // Optionally display an error message on the dashboard
                     noLogsMessageEl.textContent = `Error fetching data: ${response.status}`;
                     noLogsMessageEl.style.display = 'block';
                     noLogsMessageEl.style.color = '#fc8181'; // red-400
                    return;
                 }
                 const data = await response.json();

                 // Update Stats Cards
                 totalLogsEl.textContent = data.stats.total_30min || 0;
                 infoLogsEl.textContent = data.stats.info_30min || 0;
                 warningLogsEl.textContent = data.stats.warning_30min || 0;
                 errorLogsEl.textContent = data.stats.error_30min || 0;

                 // Update Header Info
                monitoringFileEl.textContent = data.status.last_file || 'N/A';
                parseErrorCountEl.textContent = data.status.parse_errors || 0;
                lastUpdateTimeEl.textContent = data.status.last_update ? new Date(data.status.last_update).toLocaleString() : 'Never';


                 // Update Recent Logs
                 logContainerEl.innerHTML = ''; // Clear previous logs
                 if (data.recent_logs && data.recent_logs.length > 0) {
                     noLogsMessageEl.style.display = 'none';
                     data.recent_logs.forEach((log, index) => {
                         const logDiv = document.createElement('div');
                         logDiv.className = 'log-entry fade-in';
                         // Stagger the animation slightly
                         logDiv.style.animationDelay = `${index * 0.02}s`;

                         // Format timestamp (e.g., HH:MM:SS)
                         const timeStr = new Date(log.timestamp).toLocaleTimeString();

                         logDiv.innerHTML = `
                             <span class="log-time">${timeStr}</span>
                            <span class="log-level ${getLevelClass(log.level)}">
                                 <i class="${getLevelIcon(log.level)} ${getIconClass(log.level)} mr-1" title="${log.level}"></i>${log.level}
                             </span>
                             <span class="log-func" title="${log.func}">${log.func}</span>
                             <span class="log-message">${escapeHtml(log.message)}</span>
                        `;
                         logContainerEl.appendChild(logDiv);
                     });
                      // Ensure fade-in starts after element is added
                     setTimeout(() => {
                         document.querySelectorAll('#logContainer .log-entry').forEach(el => el.style.opacity = 1);
                     }, 10);

                 } else {
                     noLogsMessageEl.textContent = 'No recent log entries found.';
                     noLogsMessageEl.style.display = 'block';
                 }


                // Update Chart
                if (logChart && data.chart_data) {
                    logChart.data.labels = data.chart_data.labels; // Time labels
                    logChart.data.datasets[0].data = data.chart_data.info; // Info counts
                    logChart.data.datasets[1].data = data.chart_data.warning; // Warning counts
                    logChart.data.datasets[2].data = data.chart_data.error; // Error counts
                    logChart.update();
                }

            } catch (error) {
                console.error("Error fetching or processing data:", error);
                 noLogsMessageEl.textContent = 'Error loading dashboard data. Check console.';
                 noLogsMessageEl.style.display = 'block';
                 noLogsMessageEl.style.color = '#fc8181'; // red-400
            }
        }

         // Simple HTML escape function
         function escapeHtml(unsafe) {
             if (!unsafe) return '';
             return unsafe
                  .replace(/&/g, "&")
                  .replace(/</g, "<")
                  .replace(/>/g, ">")
                  .replace(/'/g, "'");
          }

        // --- Initialization ---
        document.addEventListener('DOMContentLoaded', () => {
            initializeChart();
            fetchDataAndUpdate(); // Initial fetch
            setInterval(fetchDataAndUpdate, UPDATE_INTERVAL); // Periodic fetch

            // Set current year in footer
             if (currentYearEl) currentYearEl.textContent = new Date().getFullYear();
        });

    </script>

</body>
</html>
"""

# --- API Endpoint ---
@app.route('/api/data')
def get_api_data():
    """Provides log data, stats, and chart info to the frontend."""
    with log_data_lock:
        # Make a copy to avoid holding lock during processing
        current_df = log_data_df.copy()
        file_processed = last_file_processed
        update_time = last_update_time
        parse_errors = log_parsing_errors

    # Filter data for the last N minutes for display/charting
    now = datetime.datetime.now()
    cutoff_time = now - datetime.timedelta(minutes=DATA_RETENTION_MINUTES)

    recent_df = current_df[current_df['timestamp'] >= cutoff_time].copy() # Use copy

    # --- Prepare Stats ---
    stats = {
        'total_30min': len(recent_df),
        'info_30min': int(recent_df[recent_df['level'] == 'INFO'].shape[0]),
        'warning_30min': int(recent_df[recent_df['level'] == 'WARNING'].shape[0]),
        'error_30min': int(recent_df[recent_df['level'] == 'ERROR'].shape[0]),
        'critical_30min': int(recent_df[recent_df['level'] == 'CRITICAL'].shape[0]),
    }

    # --- Prepare Recent Logs (limit to ~50 for display) ---
    # Sort by timestamp descending to get the latest first
    recent_logs_display = recent_df.sort_values(by='timestamp', ascending=False).head(50)
    # Convert timestamp back to ISO format string for JSON
    recent_logs_display['timestamp'] = recent_logs_display['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' # ISO 8601 format
    recent_logs = recent_logs_display.to_dict(orient='records')

    # --- Prepare Chart Data (Group by minute) ---
    chart_data = {'labels': [], 'info': [], 'warning': [], 'error': []}
    if not recent_df.empty:
        # Set timestamp as index for resampling
        recent_df.set_index('timestamp', inplace=True)

        # Resample by minute and count levels. Fill NaNs with 0.
        # Using 'T' for minute frequency.
        level_counts_per_minute = recent_df.groupby(pd.Grouper(freq='T'))['level'].value_counts().unstack(fill_value=0)

        # Ensure all key levels exist as columns, add if missing
        for level in ['INFO', 'WARNING', 'ERROR']:
             if level not in level_counts_per_minute.columns:
                 level_counts_per_minute[level] = 0

        # Prepare data for Chart.js
        # Format timestamp labels (e.g., HH:MM)
        chart_data['labels'] = level_counts_per_minute.index.strftime('%H:%M').tolist()
        chart_data['info'] = level_counts_per_minute['INFO'].tolist()
        chart_data['warning'] = level_counts_per_minute['WARNING'].tolist()
        chart_data['error'] = level_counts_per_minute['ERROR'].tolist()


    # --- Status Info ---
    status = {
         'last_file': file_processed,
         'last_update': update_time.isoformat() if update_time else None,
         'parse_errors': parse_errors,
         'monitoring_dir': MONITORED_LOG_DIR,
         'retention_minutes': DATA_RETENTION_MINUTES
     }

    # --- Combine and Return JSON ---
    response_data = {
        'status': status,
        'stats': stats,
        'recent_logs': recent_logs,
        'chart_data': chart_data
    }
    return jsonify(response_data)


# --- Main Route ---
@app.route('/')
def index():
    """Serves the main dashboard HTML page."""
    # Pass config to JS via the template
    return render_template_string(
        html_template,
        FRONTEND_UPDATE_INTERVAL_MS=FRONTEND_UPDATE_INTERVAL_MS
    )

# --- Main Execution ---
if __name__ == '__main__':
    dashboard_logger.info(f"--- Log Dashboard Starting ---")
    if not os.path.exists(MONITORED_LOG_DIR):
        dashboard_logger.warning(f"Monitored directory '{MONITORED_LOG_DIR}' does not exist. Please create it or ensure the other application writes logs there.")
        # Create it maybe? Depends on desired behavior.
        # os.makedirs(MONITORED_LOG_DIR, exist_ok=True)

    # Start the background thread
    dashboard_logger.info("Starting background log update thread...")
    update_thread = threading.Thread(target=update_log_data, name="LogUpdateThread", daemon=True)
    update_thread.start()

    dashboard_logger.info(f"Starting Flask server on http://127.0.0.1:{LOG_MONITOR_PORT}")
    print(f"\n🚀 Log Dashboard Available at: http://127.0.0.1:{LOG_MONITOR_PORT}")
    print(f"   Monitoring logs in directory: '{MONITORED_LOG_DIR}'")
    print(f"   Dashboard logs are written to: '{DASHBOARD_LOG_FILE}'")
    print(f"   Press CTRL+C to stop.")

    try:
        from waitress import serve
        print("   Using Waitress server.")
        serve(app, host='0.0.0.0', port=LOG_MONITOR_PORT, threads=4) # Use 0.0.0.0 to be accessible on network if needed
    except ImportError:
        print("   Waitress not found, using Flask development server (WARNING: Not for production).")
        app.run(host='0.0.0.0', port=LOG_MONITOR_PORT, debug=False) # Debug=False is important