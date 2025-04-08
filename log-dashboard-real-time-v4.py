# -*- coding: utf-8 -*-
import os
import glob
import datetime
import time
import threading
import re
import logging
from pathlib import Path
from collections import Counter
import html # Para escapar HTML no modal - Embora não seja usado pelo JS escapeHtml agora

import pandas as pd
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from waitress import serve # Use waitress para um servidor mais robusto

from typing import Optional, List, Dict, Any

# --- Configuração (Mantida) ---
DASHBOARD_PORT: int = 8444 # <--- Porta pode ser alterada
MONITORED_LOG_DIR: str = "valuation_v3_web_log" # <--- Diretório monitorado pode ser alterado
FILENAME_TIMESTAMP_REGEX = re.compile(r'_(\d{8})_(\d{6})\.log$') # Para encontrar o mais recente pelo nome
FILE_SCAN_INTERVAL_SECONDS: int = 1 # Intervalo curto para verificação de arquivos
FRONTEND_UPDATE_INTERVAL_MS: int = 2000 # Intervalo de atualização do frontend (ligeiramente maior)
DATA_RETENTION_MINUTES: int = 30 # Quanto tempo para trás manter/exibir logs na UI
MAX_RECENT_LOGS_DISPLAY: int = 150 # Limitar linhas exibidas em "Logs Recentes"
MAX_ERROR_LOGS_DISPLAY: int = 100 # Limitar linhas exibidas em "Logs de Erro/Críticos"

LOG_LEVEL = logging.INFO # Logging para o próprio dashboard
DASHBOARD_LOG_DIR = "dashboard_logs"
os.makedirs(DASHBOARD_LOG_DIR, exist_ok=True)
DASHBOARD_LOG_FILE = os.path.join(DASHBOARD_LOG_DIR, f"dashboard_monitor_{DASHBOARD_PORT}.log")

# --- Configuração do Logging do Dashboard (Mantida) ---
dashboard_logger = logging.getLogger(f"dashboard_logger_{DASHBOARD_PORT}")
dashboard_logger.setLevel(LOG_LEVEL)
fh = logging.FileHandler(DASHBOARD_LOG_FILE, encoding='utf-8')
fh.setLevel(LOG_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Dashboard] - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
if not dashboard_logger.handlers:
    dashboard_logger.addHandler(fh)
    dashboard_logger.addHandler(ch)

# --- Armazenamento Global de Dados (Mantido) ---
log_data_lock = threading.Lock()
log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])

last_file_processed: Optional[str] = None
last_file_mod_time: Optional[float] = None
log_parsing_errors: int = 0
last_update_time: Optional[datetime.datetime] = None

# Regex para analisar o formato de log VALUATION (Mantido)
LOG_LINE_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})"
    r"\s+-\s+"
    r"(?P<level>\w+)"
    r"\s+-\s+"
    r"(?:\[(?P<thread>.*?)\])?"
    r"\s+-\s+"
    r"(?:<?(?P<module>[\w.-]+)>?)?"
    r"\s+-\s+"
    r"(?P<message>.*)$"
)

# --- Funções Auxiliares (parse_filename_timestamp, find_latest_log_file, parse_log_file - Mantidas Idênticas) ---
def parse_filename_timestamp(filename: str) -> Optional[datetime.datetime]:
    """Extrai timestamp do nome do arquivo de log (e.g., _YYYYMMDD_HHMMSS.log)."""
    match = FILENAME_TIMESTAMP_REGEX.search(filename)
    if match:
        try:
            timestamp_str = match.group(1) + match.group(2)
            return datetime.datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except ValueError:
            return None
    return None

def find_latest_log_file(directory: str) -> Optional[Path]:
    """Encontra o arquivo .log com o timestamp mais recente no nome ou hora de modificação."""
    latest_file_path: Optional[Path] = None
    latest_effective_ts: Optional[datetime.datetime] = None
    dir_path = Path(directory)

    if not dir_path.is_dir():
        dashboard_logger.warning(f"Diretório de logs não encontrado: {directory}")
        return None
    try:
        log_files = list(dir_path.glob('*.log'))
        if not log_files:
            return None

        for file_path in log_files:
            if not file_path.is_file(): continue
            current_effective_ts = None
            try:
                filename_ts = parse_filename_timestamp(file_path.name)
                mod_time_ts = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                current_effective_ts = max(filter(None, [filename_ts, mod_time_ts]), default=mod_time_ts)

                if latest_effective_ts is None or (current_effective_ts and current_effective_ts > latest_effective_ts):
                    latest_effective_ts = current_effective_ts
                    latest_file_path = file_path
            except OSError as e:
                dashboard_logger.warning(f"Não foi possível obter atributos para {file_path.name}: {e}")
            except Exception as e:
                dashboard_logger.warning(f"Erro ao processar o arquivo {file_path.name}: {e}")

        if latest_file_path:
             dashboard_logger.debug(f"Arquivo de log mais recente identificado: {latest_file_path.name} (Timestamp Efetivo: {latest_effective_ts})")
        elif log_files:
             dashboard_logger.warning(f"Não foi possível determinar o arquivo de log mais recente em {directory}.")

        return latest_file_path
    except Exception as e:
        dashboard_logger.error(f"Erro ao encontrar o arquivo de log mais recente em '{directory}': {e}", exc_info=True)
        return None

def parse_log_file(file_path: Path) -> pd.DataFrame:
    """Analisa o arquivo de log inteiro em um DataFrame Pandas, capturando raw_line."""
    global log_parsing_errors
    parsed_lines_data = []
    errors_in_parse_session = 0
    lines_read = 0
    start_time = time.monotonic()

    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                lines_read += 1
                raw_line = line # Guarda a linha bruta original
                line = line.strip()
                if not line:
                    continue

                match = LOG_LINE_REGEX.match(line)
                if match:
                    log_entry = match.groupdict()
                    log_entry['thread'] = log_entry.get('thread', '') or ''
                    log_entry['module'] = log_entry.get('module', '') or ''
                    log_entry['raw_line'] = raw_line.strip() # Armazena linha bruta limpa

                    try:
                        log_entry['timestamp'] = pd.to_datetime(log_entry['timestamp'], format='%Y-%m-%d %H:%M:%S,%f', errors='coerce')
                        if pd.isna(log_entry['timestamp']):
                            dashboard_logger.debug(f"File '{file_path.name}', Line ~{i+1}: Timestamp NaT - Line: '{line[:100]}...'")
                        parsed_lines_data.append(log_entry)
                    except ValueError as ve:
                        dashboard_logger.warning(f"File '{file_path.name}', Line ~{i+1}: Erro de timestamp: {ve} - Line: '{line[:100]}...'")
                        errors_in_parse_session += 1
                else:
                    dashboard_logger.debug(f"Linha {i+1} não correspondeu ao regex em {file_path.name}: {line[:100]}...")
                    errors_in_parse_session += 1
                    # Opção: Adicionar linhas não parseadas
                    # parsed_lines_data.append({
                    #     'timestamp': pd.NaT, 'level': 'UNPARSED', 'thread': '', 'module': '',
                    #     'message': line, 'raw_line': raw_line.strip()
                    # })


        duration = time.monotonic() - start_time
        dashboard_logger.debug(f"Analisou {len(parsed_lines_data)}/{lines_read} linhas de {file_path.name}. {errors_in_parse_session} problemas. Duração: {duration:.3f}s")

    except Exception as e:
        dashboard_logger.error(f"Erro ao ler/analisar o arquivo de log '{file_path}': {e}", exc_info=True)
        return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])

    log_parsing_errors += errors_in_parse_session

    if not parsed_lines_data:
        df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    df = pd.DataFrame(parsed_lines_data)
    initial_rows = len(df)
    df.dropna(subset=['timestamp'], inplace=True) # Remove linhas onde o timestamp não pôde ser parseado
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        dashboard_logger.warning(f"Removidas {dropped_rows} linhas com timestamp inválido (NaT) de '{file_path.name}'.")

    if df.empty:
         return df

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['level', 'thread', 'module', 'message', 'raw_line']:
         if col in df.columns:
              df[col] = df[col].astype(str).fillna('')

    return df.sort_values(by='timestamp')

# --- Tarefa de Background (update_log_data_periodically - Mantida Idêntica) ---
def update_log_data_periodically():
    """Tarefa de fundo para encontrar o log mais recente, analisá-lo e atualizar dados globais."""
    global log_data_df, last_file_processed, last_file_mod_time, last_update_time, log_parsing_errors

    dashboard_logger.info("Thread de atualização de log em background iniciada.")
    while True:
        try:
            latest_file = find_latest_log_file(MONITORED_LOG_DIR)
            current_time = datetime.datetime.now()
            needs_processing = False
            current_mod_time = None

            if latest_file:
                try:
                    current_mod_time = latest_file.stat().st_mtime
                except Exception as e:
                    dashboard_logger.warning(f"Não foi possível obter tempo de modificação para {latest_file.name}: {e}.")

                is_new_file = (last_file_processed is None) or (latest_file.name != last_file_processed)
                is_modified = (current_mod_time is not None) and (last_file_mod_time is None or current_mod_time > last_file_mod_time)

                if is_new_file:
                    dashboard_logger.info(f"Novo arquivo de log detectado ou primeira execução: {latest_file.name}")
                    needs_processing = True
                elif is_modified:
                    dashboard_logger.debug(f"Arquivo {latest_file.name} modificado desde a última leitura.")
                    needs_processing = True

                if needs_processing:
                    dashboard_logger.info(f"Processando arquivo de log: {latest_file.name}")
                    new_df = parse_log_file(latest_file)
                    parse_session_errors = log_parsing_errors # Snapshot dos erros

                    with log_data_lock:
                        if not new_df.empty:
                            log_data_df = new_df
                            last_file_processed = latest_file.name
                            last_file_mod_time = current_mod_time
                            last_update_time = current_time
                            level_counts = log_data_df['level'].value_counts()
                            dashboard_logger.info(f"Dados atualizados de '{latest_file.name}'. {len(log_data_df)} linhas. Erros/Críticos: {level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)}, Avisos: {level_counts.get('WARNING', 0)}. Erros de Parse Totais: {parse_session_errors}")
                        elif latest_file.name == last_file_processed and current_mod_time == last_file_mod_time:
                             dashboard_logger.debug(f"Parse de '{latest_file.name}' retornou vazio, arquivo não modificado. Sem alterações.")
                        else:
                            dashboard_logger.warning(f"Análise de '{latest_file.name}' resultou em DataFrame vazio. Limpando dados anteriores.")
                            log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                            log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                            last_file_processed = latest_file.name
                            last_file_mod_time = current_mod_time
                            last_update_time = current_time
            else:
                if last_file_processed is not None:
                    dashboard_logger.warning(f"Nenhum arquivo de log encontrado em '{MONITORED_LOG_DIR}'. Limpando dados anteriores.")
                    with log_data_lock:
                        log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                        log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                        last_file_processed = None
                        last_file_mod_time = None
                        last_update_time = current_time

        except Exception as e:
            dashboard_logger.error(f"Erro crítico no loop de atualização em background: {e}", exc_info=True)
            time.sleep(FILE_SCAN_INTERVAL_SECONDS * 5) # Wait longer after a critical error

        time.sleep(FILE_SCAN_INTERVAL_SECONDS)

# --- Flask App ---
app = Flask(__name__)
CORS(app) # Permitir todas as origens

# --- Template HTML/CSS/JS (Grafana Inspired) ---
# noqa
grafana_html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Grafana Style - {MONITORED_LOG_DIR}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --grafana-bg: #161719;
            --grafana-bg-secondary: #111214;
            --grafana-panel-bg: #1f2124; /* Slightly lighter panel */
            --grafana-panel-border: #323233;
            --grafana-text-primary: #d8d9da;
            --grafana-text-secondary: #a0a3a8; /* Softer secondary */
            --grafana-text-muted: #717579;
            --grafana-scrollbar-track: rgba(74, 85, 104, 0.1);
            --grafana-scrollbar-thumb: rgba(160, 174, 192, 0.3);
            --grafana-scrollbar-thumb-hover: rgba(160, 174, 192, 0.5);
            --grafana-cyan: #00bfff; /* Brighter cyan */
            --grafana-blue: #33a2e5; /* Slightly softer blue */
            --grafana-yellow: #fadc4d; /* Grafana yellow */
            --grafana-red: #f2495c;   /* Grafana red */
            --grafana-red-dark: #e02f44;
            --grafana-green: #73bf69;  /* Grafana green */
            --grafana-orange: #ff780a; /* Grafana orange */
        }
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--grafana-bg);
            color: var(--grafana-text-primary);
            overscroll-behavior-y: none;
        }
        /* Panel Styling */
        .grafana-panel {
            background-color: var(--grafana-panel-bg);
            border: 1px solid var(--grafana-panel-border);
            border-radius: 4px;
            padding: 1rem 1.25rem; /* 16px 20px */
            display: flex;
            flex-direction: column;
            transition: border-color 0.2s ease-in-out;
            height: 100%; /* Ensure panels fill their grid cells */
        }
        .grafana-panel:hover {
            border-color: #4a4d52;
        }
        .panel-title {
            font-size: 0.9rem; /* 14.4px */
            font-weight: 500;
            color: var(--grafana-text-primary);
            margin-bottom: 0.75rem; /* 12px */
            padding-bottom: 0.5rem; /* 8px */
            border-bottom: 1px solid var(--grafana-panel-border);
            display: flex;
            align-items: center;
            gap: 0.5rem; /* 8px */
        }
        .panel-content {
            flex-grow: 1;
            position: relative; /* For chart canvas absolute positioning */
        }

        /* Card Specific Styles (Stat Panels) */
        .stat-card {
            display: flex;
            align-items: center;
            gap: 1rem; /* 16px */
        }
        .stat-card .icon {
            font-size: 1.75rem; /* 28px */
            opacity: 0.8;
            width: 30px; /* Fixed width for alignment */
            text-align: center;
        }
        .stat-card .text .title {
            font-size: 0.75rem; /* 12px */
            color: var(--grafana-text-secondary);
            text-transform: uppercase;
            margin-bottom: 0.1rem; /* 2px */
        }
        .stat-card .text .value {
            font-size: 1.5rem; /* 24px */
            font-weight: 500;
            color: var(--grafana-text-primary);
            line-height: 1.2;
        }

        /* Log Entry Styling */
        .log-block {
            overflow-y: auto;
            height: 100%; /* Needs a defined height or max-height */
            max-height: 400px; /* Example max height */
            padding-right: 5px; /* Space for scrollbar */
        }
        .log-entry {
            font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
            font-size: 0.78rem; /* Slightly smaller */
            line-height: 1.6;
            padding: 0.3rem 0.5rem;
            border-radius: 3px;
            margin-bottom: 0.15rem;
            background-color: rgba(0, 0, 0, 0.1);
            display: flex;
            flex-wrap: nowrap;
            gap: 0.6rem;
            align-items: baseline;
            cursor: pointer;
            transition: background-color 0.2s ease;
            opacity: 0; /* Start hidden for fade-in */
            animation: fadeInLog 0.5s ease-out forwards;
        }
        @keyframes fadeInLog { from { opacity: 0; } to { opacity: 1; } }
        .log-entry:hover { background-color: rgba(255, 255, 255, 0.05); }

        .log-time { color: var(--grafana-text-muted); width: 65px; flex-shrink: 0; }
        .log-level-cont { width: 80px; flex-shrink: 0; text-align: left; }
        .log-level { font-weight: 500;}
        .log-thread { color: var(--grafana-text-secondary); width: 100px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .log-module { color: var(--grafana-cyan); width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .log-message { color: var(--grafana-text-primary); flex-grow: 1; word-break: break-word; white-space: pre-wrap; }

        /* Log Level Colors (Grafana Style) */
        .level-DEBUG { color: var(--grafana-text-muted); }
        .level-INFO { color: var(--grafana-blue); }
        .level-WARNING { color: var(--grafana-yellow); }
        .level-ERROR { color: var(--grafana-red); }
        .level-CRITICAL {
            color: #ffffff; background-color: var(--grafana-red-dark);
            padding: 0px 4px; border-radius: 3px; font-weight: bold;
        }
        .level-UNPARSED { color: var(--grafana-orange); font-style: italic;}

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--grafana-scrollbar-track); border-radius: 3px;}
        ::-webkit-scrollbar-thumb { background: var(--grafana-scrollbar-thumb); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--grafana-scrollbar-thumb-hover); }

        /* Modal Styling (Adapted from previous) */
        .modal-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0, 0, 0, 0.75); display: flex; align-items: center; justify-content: center;
            opacity: 0; visibility: hidden; transition: opacity 0.3s ease, visibility 0.3s ease; z-index: 1000;
        }
        .modal-overlay.active { opacity: 1; visibility: visible; }
        .modal-content {
            background: var(--grafana-panel-bg); color: var(--grafana-text-primary); padding: 1.5rem 2rem; border-radius: 4px;
            border: 1px solid var(--grafana-panel-border); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            width: 85%; max-width: 1000px; max-height: 85vh; display: flex; flex-direction: column; transform: scale(0.95); transition: transform 0.3s ease;
        }
        .modal-overlay.active .modal-content { transform: scale(1); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--grafana-panel-border); padding-bottom: 0.75rem;}
        .modal-title { font-size: 1.1rem; font-weight: 500; color: var(--grafana-text-primary); }
        .modal-close-btn { background: none; border: none; color: var(--grafana-text-secondary); font-size: 1.6rem; cursor: pointer; transition: color 0.2s ease; padding: 0; line-height: 1;}
        .modal-close-btn:hover { color: var(--grafana-text-primary); }
        .modal-body { overflow-y: auto; flex-grow: 1; }
        .modal-body pre {
            white-space: pre-wrap; word-wrap: break-word; font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem; line-height: 1.6; background-color: var(--grafana-bg-secondary);
            padding: 1rem; border-radius: 3px; border: 1px solid var(--grafana-panel-border);
        }

        /* Chart container */
        .chart-container {
            position: relative;
            width: 100%;
            height: 250px; /* Default height, adjust as needed */
        }
         /* Utility to hide element visually but keep for screen readers */
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border-width: 0; }

    </style>
</head>
<body class="p-4 md:p-6 bg-grafana-bg">

    <!-- Header Row -->
    <header class="mb-4 flex flex-col sm:flex-row justify-between items-center gap-2">
        <h1 class="text-xl md:text-2xl font-medium text-grafana-text-primary flex items-center">
           <i class="fas fa-chart-line mr-2 text-grafana-cyan"></i>Dashboard: <span class="text-grafana-text-secondary ml-1">{MONITORED_LOG_DIR}</span>
        </h1>
        <div class="text-xs text-grafana-text-muted flex flex-wrap gap-x-4 gap-y-1 justify-center sm:justify-end">
             <span><i class="fas fa-file-alt mr-1"></i>Arquivo: <code id="monitoringFile" class="text-grafana-text-secondary">N/D</code></span>
             <span><i class="fas fa-sync-alt mr-1"></i>Atualizado: <span id="lastUpdateTime" class="text-grafana-text-secondary">Nunca</span></span>
             <span><i class="fas fa-bug mr-1 text-grafana-red"></i>Erros Parse: <span id="parseErrorCount" class="text-grafana-red font-medium">0</span></span>
         </div>
    </header>

    <!-- Main Content Grid -->
    <main class="grid grid-cols-12 gap-4">

        <!-- Cards Row (Span Full Width) -->
        <section class="col-span-12 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-10 gap-3 mb-4">
            <!-- Create 20 Card Placeholders -->
            <div id="card-total" class="grafana-panel !p-3 xl:col-span-2"></div>
            <div id="card-info" class="grafana-panel !p-3"></div>
            <div id="card-warning" class="grafana-panel !p-3"></div>
            <div id="card-error" class="grafana-panel !p-3"></div>
            <div id="card-critical" class="grafana-panel !p-3"></div>
            <div id="card-debug" class="grafana-panel !p-3"></div>
            <div id="card-unparsed" class="grafana-panel !p-3"></div>
            <div id="card-error-rate" class="grafana-panel !p-3"></div> <!-- Example Derived -->
            <div id="card-log-rate" class="grafana-panel !p-3"></div> <!-- Example Derived -->
            <!-- Placeholder Cards -->
            <div id="card-placeholder-10" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-11" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-12" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-13" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-14" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-15" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-16" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-17" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-18" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-19" class="grafana-panel !p-3"></div>
            <div id="card-placeholder-20" class="grafana-panel !p-3"></div>
        </section>

        <!-- Charts Row (Example: 2x2 layout on larger screens) -->
        <section class="col-span-12 grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-chart-bar text-grafana-cyan"></i>Atividade por Minuto</h2>
                <div class="panel-content chart-container" style="height: 280px;">
                   <canvas id="chart-activity-bar"></canvas>
                </div>
            </div>
            <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-chart-pie text-grafana-cyan"></i>Distribuição por Nível</h2>
                 <div class="panel-content chart-container" style="height: 280px;">
                   <canvas id="chart-level-doughnut"></canvas>
                </div>
            </div>
            <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-chart-line text-grafana-cyan"></i>Erros e Avisos por Minuto</h2>
                 <div class="panel-content chart-container" style="height: 280px;">
                   <canvas id="chart-error-warn-line"></canvas>
                </div>
            </div>
            <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-chart-area text-grafana-cyan"></i>Total de Logs por Minuto</h2>
                 <div class="panel-content chart-container" style="height: 280px;">
                   <canvas id="chart-total-line"></canvas>
                </div>
            </div>
        </section>

        <!-- Log Blocks Row (Example: Side-by-side on larger screens) -->
        <section class="col-span-12 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-stream text-grafana-cyan"></i>Logs Recentes (Max {MAX_RECENT_LOGS_DISPLAY})</h2>
                <div class="panel-content">
                    <div id="recentLogContainer" class="log-block">
                        <p id="noRecentLogsMessage" class="text-center text-grafana-text-muted italic mt-8">Aguardando dados...</p>
                    </div>
                </div>
            </div>
             <div class="grafana-panel">
                <h2 class="panel-title"><i class="fas fa-exclamation-triangle text-grafana-red"></i>Logs de Erro/Críticos (Max {MAX_ERROR_LOGS_DISPLAY})</h2>
                <div class="panel-content">
                    <div id="errorLogContainer" class="log-block">
                        <p id="noErrorLogsMessage" class="text-center text-grafana-text-muted italic mt-8">Aguardando dados...</p>
                    </div>
                </div>
            </div>
        </section>

    </main>

     <!-- Footer -->
     <footer class="text-center text-xs text-grafana-text-muted mt-8 pt-4 border-t border-grafana-panel-border">
         Dashboard v2.0 | Monitorando: {MONITORED_LOG_DIR} | <span id="currentYear"></span>
     </footer>

     <!-- Modal (Same Logic) -->
     <div id="logModal" class="modal-overlay">
         <div class="modal-content">
             <div class="modal-header">
                 <h3 class="modal-title">Detalhes do Log</h3>
                 <button id="modalCloseBtn" class="modal-close-btn" aria-label="Fechar modal">×</button>
             </div>
             <div class="modal-body">
                 <pre id="modalLogContent">...</pre>
             </div>
         </div>
     </div>

    <script>
        const API_ENDPOINT = '/api/data';
        const UPDATE_INTERVAL = {{ FRONTEND_UPDATE_INTERVAL_MS }};
        const MAX_RECENT_LOGS = {{ MAX_RECENT_LOGS_DISPLAY }};
        const MAX_ERROR_LOGS = {{ MAX_ERROR_LOGS_DISPLAY }};
        // Corrigido para pegar o valor da variável Python
        const DATA_RETENTION_MINUTES = {{ DATA_RETENTION_MINUTES }};


        let charts = {
            activityBar: null,
            levelDoughnut: null,
            errorWarnLine: null,
            totalLine: null
        };
        let lastUpdateTimeStr = 'Nunca'; // Cache last update time

        // --- DOM Elements ---
        const monitoringFileEl = document.getElementById('monitoringFile');
        const lastUpdateTimeEl = document.getElementById('lastUpdateTime');
        const parseErrorCountEl = document.getElementById('parseErrorCount');
        const currentYearEl = document.getElementById('currentYear');
        // Log Containers
        const recentLogContainerEl = document.getElementById('recentLogContainer');
        const errorLogContainerEl = document.getElementById('errorLogContainer');
        const noRecentLogsMessageEl = document.getElementById('noRecentLogsMessage');
        const noErrorLogsMessageEl = document.getElementById('noErrorLogsMessage');
        // Modal Elements
        const logModal = document.getElementById('logModal');
        const modalLogContent = document.getElementById('modalLogContent');
        const modalCloseBtn = document.getElementById('modalCloseBtn');
        // Card Elements (Get all placeholders)
        const cardElements = {};
        const cardIds = [
            'card-total', 'card-info', 'card-warning', 'card-error', 'card-critical',
            'card-debug', 'card-unparsed', 'card-error-rate', 'card-log-rate',
            'card-placeholder-10', 'card-placeholder-11', 'card-placeholder-12',
            'card-placeholder-13', 'card-placeholder-14', 'card-placeholder-15',
            'card-placeholder-16', 'card-placeholder-17', 'card-placeholder-18',
            'card-placeholder-19', 'card-placeholder-20'
        ];
        cardIds.forEach(id => cardElements[id] = document.getElementById(id));

        // --- Chart Config ---
        const chartColors = {
            info: 'var(--grafana-blue)',
            warning: 'var(--grafana-yellow)',
            error: 'var(--grafana-red)',
            critical: 'var(--grafana-red-dark)', // Use darker red for critical if needed
            debug: 'var(--grafana-text-muted)',
            unparsed: 'var(--grafana-orange)',
            cyan: 'var(--grafana-cyan)',
            grid: 'rgba(216, 217, 218, 0.1)', // Use primary text color with alpha for grid
            text: 'var(--grafana-text-secondary)'
        };

        const commonChartOptions = {
             responsive: true, maintainAspectRatio: false,
             interaction: { mode: 'index', intersect: false },
             scales: {
                 x: { ticks: { color: chartColors.text, maxRotation: 0, autoSkip: true, autoSkipPadding: 20 }, grid: { color: chartColors.grid } },
                 y: { beginAtZero: true, ticks: { color: chartColors.text, precision: 0 }, grid: { color: chartColors.grid } }
             },
             plugins: {
                 legend: { display: false }, // Usually hide legend for panel dashboards, title is enough
                 tooltip: {
                     backgroundColor: 'rgba(31, 33, 36, 0.9)', // panel bg darker
                     titleColor: 'var(--grafana-text-primary)', bodyColor: 'var(--grafana-text-secondary)',
                     borderColor: 'var(--grafana-panel-border)', borderWidth: 1, padding: 10, boxPadding: 5
                 }
             },
             animation: { duration: 300, easing: 'linear' }
        };

        function initializeCharts() {
            // 1. Activity Bar Chart (Stacked)
            const ctxBar = document.getElementById('chart-activity-bar')?.getContext('2d');
            if (ctxBar) {
                 if(charts.activityBar) charts.activityBar.destroy();
                 charts.activityBar = new Chart(ctxBar, {
                    type: 'bar',
                    data: { labels: [], datasets: [
                        { label: 'Info', data: [], backgroundColor: chartColors.info, stack: 'stack0' },
                        { label: 'Warning', data: [], backgroundColor: chartColors.warning, stack: 'stack0' },
                        { label: 'Error/Critical', data: [], backgroundColor: chartColors.error, stack: 'stack0' },
                    ] },
                    options: { ...commonChartOptions, scales: {...commonChartOptions.scales, x: {...commonChartOptions.scales.x, stacked: true}, y: {...commonChartOptions.scales.y, stacked: true}}, plugins: {...commonChartOptions.plugins, legend: {display: true, position: 'bottom', labels: {color: chartColors.text, boxWidth: 10, padding: 15}}} }
                 });
            }

            // 2. Level Distribution Doughnut Chart
            const ctxDoughnut = document.getElementById('chart-level-doughnut')?.getContext('2d');
            if (ctxDoughnut) {
                if(charts.levelDoughnut) charts.levelDoughnut.destroy();
                charts.levelDoughnut = new Chart(ctxDoughnut, {
                    type: 'doughnut',
                    data: { labels: ['INFO', 'WARNING', 'ERROR', 'CRITICAL', 'DEBUG', 'UNPARSED'], datasets: [{
                        label: 'Log Levels', data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: [chartColors.info, chartColors.warning, chartColors.error, chartColors.critical, chartColors.debug, chartColors.unparsed],
                        borderColor: 'var(--grafana-panel-bg)', // Match panel bg for separation
                        borderWidth: 2, hoverOffset: 8
                    }] },
                    options: { ...commonChartOptions, cutout: '65%', plugins: {...commonChartOptions.plugins, legend: {display: true, position: 'right', labels: {color: chartColors.text, boxWidth: 12, padding: 15}}} }
                });
            }

            // 3. Error/Warning Line Chart
            const ctxLineErr = document.getElementById('chart-error-warn-line')?.getContext('2d');
            if(ctxLineErr) {
                 if(charts.errorWarnLine) charts.errorWarnLine.destroy();
                 charts.errorWarnLine = new Chart(ctxLineErr, {
                     type: 'line',
                     data: { labels: [], datasets: [
                         { label: 'Warning', data: [], borderColor: chartColors.warning, backgroundColor: hexToRgba(chartColors.warning, 0.1), fill: 'start', tension: 0.3, pointRadius: 1, pointHoverRadius: 5 },
                         { label: 'Error/Critical', data: [], borderColor: chartColors.error, backgroundColor: hexToRgba(chartColors.error, 0.1), fill: 'start', tension: 0.3, pointRadius: 1, pointHoverRadius: 5 }
                     ] },
                     options: { ...commonChartOptions, plugins: {...commonChartOptions.plugins, legend: {display: true, position: 'bottom', labels: {color: chartColors.text, boxWidth: 10, padding: 15}}} }
                 });
            }

            // 4. Total Logs Line Chart (Example - can be changed)
            const ctxLineTotal = document.getElementById('chart-total-line')?.getContext('2d');
             if(ctxLineTotal) {
                 if(charts.totalLine) charts.totalLine.destroy();
                 charts.totalLine = new Chart(ctxLineTotal, {
                     type: 'line', // Or 'bar'
                     data: { labels: [], datasets: [
                         { label: 'Total Logs', data: [], borderColor: chartColors.cyan, backgroundColor: hexToRgba(chartColors.cyan, 0.1), fill: 'start', tension: 0.3, pointRadius: 1, pointHoverRadius: 5 }
                     ] },
                      options: commonChartOptions // Simplified options
                 });
             }
        }

        // --- Helper Functions ---
        function getLevelClass(level) { return `level-${level?.toUpperCase() || 'UNPARSED'}`; }

        // CORRIGIDO: Função escapeHtml com lógica correta e sem erro de sintaxe Python
        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return '';
            // Correctly escape HTML special characters
            return unsafe
                .replace(/&/g, "&")
                .replace(/</g, "<")
                .replace(/>/g, ">")
                .replace(/'/g, "'"); // Use ' for broader compatibility than '
        }

        function hexToRgba(cssVar, alpha) {
            try { // Add error handling for getComputedStyle
                const colorValue = getComputedStyle(document.documentElement).getPropertyValue(cssVar.match(/--[\w-]+/)[0]).trim();
                if (!colorValue) return `rgba(100, 100, 100, ${alpha})`;
                if (colorValue.startsWith("#")) {
                    let hex = colorValue.slice(1);
                    if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
                    const bigint = parseInt(hex, 16);
                    const r = (bigint >> 16) & 255;
                    const g = (bigint >> 8) & 255;
                    const b = bigint & 255;
                    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
                } else if (colorValue.startsWith('rgb')) { // Handle rgb() or rgba()
                    return colorValue.replace(/rgb\(/, `rgba(`).replace(/\)/, `, ${alpha})`);
                }
            } catch (e) {
                 console.warn("Could not parse color variable:", cssVar, e);
            }
            return `rgba(100, 100, 100, ${alpha})`; // Fallback
        }

        // Function to render a stat card
        function renderStatCard(element, iconClass, colorVar, title, value) {
            if (!element) return;
            const color = `var(${colorVar})`;
            element.innerHTML = `
                <div class="stat-card w-full">
                    <div class="icon" style="color: ${color};"><i class="${iconClass} fa-fw"></i></div>
                    <div class="text">
                        <div class="title">${title}</div>
                        <div class="value">${value?.toLocaleString() ?? '0'}</div>
                    </div>
                </div>`;
        }
        function renderPlaceholderCard(element, id) {
             if (!element) return;
             element.innerHTML = `
                 <div class="stat-card w-full opacity-50">
                    <div class="icon text-grafana-text-muted"><i class="fas fa-question-circle fa-fw"></i></div>
                    <div class="text">
                        <div class="title">Placeholder ${id.split('-').pop()}</div>
                        <div class="value">-</div>
                    </div>
                </div>`;
         }

        // Function to update log block (more efficient than innerHTML='')
        function updateLogBlock(containerEl, noLogsMsgEl, logs, maxLogs) {
            if (!containerEl || !noLogsMsgEl) return;

            const existingLogTimestamps = new Set(Array.from(containerEl.querySelectorAll('.log-entry')).map(el => el.dataset.timestamp));
            const fragment = document.createDocumentFragment();
            let addedCount = 0;

            // Sort logs by timestamp descending before processing, ensure API provides them sorted or sort here.
            // Assuming API already sends recent_logs sorted descending.
            logs.slice(0, maxLogs).forEach(log => {
                 const timestampKey = log.timestamp; // ISO format from backend
                 if (timestampKey && !existingLogTimestamps.has(timestampKey)) { // Add check for valid timestampKey
                    const logDiv = document.createElement('div');
                    logDiv.className = 'log-entry'; // Animation handles fade-in
                    logDiv.dataset.timestamp = timestampKey;
                    // Use escapeHtml for raw_line content displayed in modal
                    logDiv.dataset.rawlog = log.raw_line || '';

                    // Format time safely
                    let timeStr = '??:??:??';
                    try {
                        timeStr = new Date(log.timestamp).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                    } catch (e) { console.warn("Invalid log timestamp for display:", log.timestamp); }

                    const level = log.level || 'UNPARSED';
                    const thread = log.thread || '-';
                    const module = log.module || '-';
                    const message = log.message || '';

                    // Use escapeHtml for displayed content to prevent XSS if logs contain HTML/JS
                    logDiv.innerHTML = `
                        <span class="log-time">${timeStr}</span>
                        <span class="log-level-cont">
                            <span class="log-level ${getLevelClass(level)}">${escapeHtml(level)}</span>
                        </span>
                        <span class="log-thread" title="${escapeHtml(thread)}">${escapeHtml(thread)}</span>
                        <span class="log-module" title="${escapeHtml(module)}">${escapeHtml(module)}</span>
                        <span class="log-message">${escapeHtml(message)}</span>
                    `;
                    fragment.appendChild(logDiv);
                    addedCount++;
                 }
                 if (timestampKey) existingLogTimestamps.add(timestampKey); // Keep track even if not added now
            });

            // Prepend new logs
            if (addedCount > 0) {
                 containerEl.prepend(fragment);
            }

            // Remove old logs exceeding max limit more robustly
            const currentLogs = containerEl.querySelectorAll('.log-entry');
            if (currentLogs.length > maxLogs) {
                 // Convert NodeList to Array to use slice
                 const logsToRemove = Array.from(currentLogs).slice(maxLogs);
                 logsToRemove.forEach(logEl => logEl.remove());
             }

            // Toggle visibility of "no logs" message
             const hasLogEntries = containerEl.querySelector('.log-entry') !== null;
             noLogsMsgEl.style.display = hasLogEntries ? 'none' : 'block';
        }


        // --- Main Data Fetch and Update Function ---
        async function fetchDataAndUpdate() {
            try {
                const response = await fetch(API_ENDPOINT);
                if (!response.ok) {
                    console.error("API Error:", response.status, response.statusText);
                    const errorMsg = `Erro ${response.status}`;
                    if (noRecentLogsMessageEl) { noRecentLogsMessageEl.textContent = errorMsg; noRecentLogsMessageEl.style.color='var(--grafana-red)'; noRecentLogsMessageEl.style.display = 'block'; }
                    if (noErrorLogsMessageEl) { noErrorLogsMessageEl.textContent = errorMsg; noErrorLogsMessageEl.style.color='var(--grafana-red)'; noErrorLogsMessageEl.style.display = 'block'; }
                    // Clear potentially stale data from previous successful fetches
                    if (recentLogContainerEl) recentLogContainerEl.innerHTML = '';
                    if (errorLogContainerEl) errorLogContainerEl.innerHTML = '';
                    return;
                }
                const data = await response.json();

                // --- Update Header ---
                if (monitoringFileEl) monitoringFileEl.textContent = data.status?.last_file || 'N/D';
                if (parseErrorCountEl) parseErrorCountEl.textContent = data.status?.parse_errors || 0;
                if (data.status?.last_update) {
                    try {
                        lastUpdateTimeStr = new Date(data.status.last_update).toLocaleTimeString('pt-BR');
                    } catch (e) { console.warn("Invalid last_update timestamp:", data.status.last_update); lastUpdateTimeStr = "Inválido"; }
                } else {
                    lastUpdateTimeStr = "Nunca";
                }
                if (lastUpdateTimeEl) lastUpdateTimeEl.textContent = lastUpdateTimeStr;

                // --- Update Cards ---
                const stats = data.stats || {};
                const totalLogs = stats.total_filtered ?? 0;
                renderStatCard(cardElements['card-total'], 'fas fa-database', '--grafana-cyan', `Total (${DATA_RETENTION_MINUTES} min)`, totalLogs);
                renderStatCard(cardElements['card-info'], 'fas fa-info-circle', '--grafana-blue', 'Info', stats.info_filtered);
                renderStatCard(cardElements['card-warning'], 'fas fa-exclamation-triangle', '--grafana-yellow', 'Warnings', stats.warning_filtered);
                renderStatCard(cardElements['card-error'], 'fas fa-times-circle', '--grafana-red', 'Errors', stats.error_filtered);
                renderStatCard(cardElements['card-critical'], 'fas fa-skull-crossbones', '--grafana-red-dark', 'Critical', stats.critical_filtered);
                renderStatCard(cardElements['card-debug'], 'fas fa-bug', '--grafana-text-muted', 'Debug', stats.debug_filtered);
                renderStatCard(cardElements['card-unparsed'], 'fas fa-question-circle', '--grafana-orange', 'Unparsed', stats.unparsed_filtered);

                 const errorCount = (stats.error_filtered ?? 0) + (stats.critical_filtered ?? 0);
                 const errorRate = totalLogs > 0 ? ((errorCount / totalLogs) * 100).toFixed(1) + '%' : '0%';
                 renderStatCard(cardElements['card-error-rate'], 'fas fa-percentage', '--grafana-red', 'Error Rate', errorRate);
                 // Avoid division by zero if DATA_RETENTION_MINUTES is somehow 0
                 const logRate = DATA_RETENTION_MINUTES > 0 ? (totalLogs / DATA_RETENTION_MINUTES).toFixed(1) : 'N/A';
                 renderStatCard(cardElements['card-log-rate'], 'fas fa-tachometer-alt', '--grafana-green', 'Logs/Min', logRate);

                for (let i = 10; i <= 20; i++) {
                    renderPlaceholderCard(cardElements[`card-placeholder-${i}`], `card-placeholder-${i}`);
                }


                // --- Update Charts ---
                const chartData = data.chart_data || {};
                const labels = chartData.labels || [];

                if (charts.activityBar && chartData.info && chartData.warning && chartData.error_critical) {
                    charts.activityBar.data.labels = labels;
                    charts.activityBar.data.datasets[0].data = chartData.info || [];
                    charts.activityBar.data.datasets[1].data = chartData.warning || [];
                    charts.activityBar.data.datasets[2].data = chartData.error_critical || [];
                    charts.activityBar.update('none');
                }

                if (charts.levelDoughnut) {
                    charts.levelDoughnut.data.datasets[0].data = [
                        stats.info_filtered ?? 0, stats.warning_filtered ?? 0, stats.error_filtered ?? 0,
                        stats.critical_filtered ?? 0, stats.debug_filtered ?? 0, stats.unparsed_filtered ?? 0
                    ];
                    charts.levelDoughnut.update('none');
                }

                 if (charts.errorWarnLine && chartData.warning && chartData.error_critical) {
                     charts.errorWarnLine.data.labels = labels;
                     charts.errorWarnLine.data.datasets[0].data = chartData.warning || [];
                     charts.errorWarnLine.data.datasets[1].data = chartData.error_critical || [];
                     charts.errorWarnLine.update('none');
                 }

                if (charts.totalLine) {
                     charts.totalLine.data.labels = labels;
                     const totalPerMinute = labels.map((_, i) =>
                         (chartData.info?.[i] ?? 0) +
                         (chartData.warning?.[i] ?? 0) +
                         (chartData.error_critical?.[i] ?? 0) +
                         (chartData.debug?.[i] ?? 0) + // Include debug if available per minute
                         (chartData.unparsed?.[i] ?? 0) // Include unparsed if available per minute
                     );
                     charts.totalLine.data.datasets[0].data = totalPerMinute;
                     charts.totalLine.update('none');
                }

                // --- Update Log Blocks ---
                const recentLogs = data.recent_logs || [];
                updateLogBlock(recentLogContainerEl, noRecentLogsMessageEl, recentLogs, MAX_RECENT_LOGS);

                const errorLogs = recentLogs.filter(log => log.level === 'ERROR' || log.level === 'CRITICAL');
                updateLogBlock(errorLogContainerEl, noErrorLogsMessageEl, errorLogs, MAX_ERROR_LOGS);


            } catch (error) {
                console.error("JS Error processing data:", error);
                const jsErrorMsg = 'Erro no Javascript';
                 if (noRecentLogsMessageEl) { noRecentLogsMessageEl.textContent = jsErrorMsg; noRecentLogsMessageEl.style.color='var(--grafana-red)'; noRecentLogsMessageEl.style.display = 'block'; }
                 if (noErrorLogsMessageEl) { noErrorLogsMessageEl.textContent = jsErrorMsg; noErrorLogsMessageEl.style.color='var(--grafana-red)'; noErrorLogsMessageEl.style.display = 'block'; }
                 // Clear potentially stale data
                 if (recentLogContainerEl) recentLogContainerEl.innerHTML = '';
                 if (errorLogContainerEl) errorLogContainerEl.innerHTML = '';
            }
        }

        // --- Modal Logic (Identical to previous version) ---
        function openModal(rawLogContent) {
            if (!logModal || !modalLogContent) return;
            // Use textContent to prevent rendering HTML inside the <pre> tag
            modalLogContent.textContent = rawLogContent;
            logModal.classList.add('active');
        }
        function closeModal() {
            if (!logModal) return;
            logModal.classList.remove('active');
        }
        // Event listeners for modal (using event delegation on containers)
        if (recentLogContainerEl) {
            recentLogContainerEl.addEventListener('click', (event) => {
                const logEntry = event.target.closest('.log-entry');
                if (logEntry && logEntry.dataset.rawlog) openModal(logEntry.dataset.rawlog);
            });
        }
         if (errorLogContainerEl) {
            errorLogContainerEl.addEventListener('click', (event) => {
                const logEntry = event.target.closest('.log-entry');
                if (logEntry && logEntry.dataset.rawlog) openModal(logEntry.dataset.rawlog);
            });
         }
        if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
        if (logModal) logModal.addEventListener('click', (event) => { if (event.target === logModal) closeModal(); });
        document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && logModal?.classList.contains('active')) closeModal(); });


        // --- Initialization ---
        document.addEventListener('DOMContentLoaded', () => {
            initializeCharts();
            fetchDataAndUpdate(); // Initial fetch
            setInterval(fetchDataAndUpdate, UPDATE_INTERVAL); // Periodic update
            if (currentYearEl) currentYearEl.textContent = new Date().getFullYear();

            // Set initial "no logs" message state correctly
            if (noRecentLogsMessageEl) noRecentLogsMessageEl.style.display = 'block';
            if (noErrorLogsMessageEl) noErrorLogsMessageEl.style.display = 'block';
        });

    </script>

</body>
</html>
"""

# --- Endpoint da API (Revisado - Mantido Idêntico ao Original, pois já fornece dados necessários) ---
@app.route('/api/data')
def get_api_data():
    """Fornece dados de log filtrados, estatísticas e informações do gráfico para o frontend."""
    with log_data_lock:
        # Make a deep copy to avoid modifying the global df during processing
        current_df = log_data_df.copy(deep=True)
        file_processed = last_file_processed
        update_time = last_update_time
        parse_errors = log_parsing_errors

    now_naive = datetime.datetime.now()
    cutoff_time_naive = now_naive - datetime.timedelta(minutes=DATA_RETENTION_MINUTES)

    filtered_df = pd.DataFrame(columns=current_df.columns)
    if 'timestamp' in current_df.columns and not current_df.empty:
         try:
             # Ensure timestamp column is datetime type, coercing errors
             if not pd.api.types.is_datetime64_any_dtype(current_df['timestamp']):
                 current_df['timestamp'] = pd.to_datetime(current_df['timestamp'], errors='coerce')

             # Drop rows where timestamp conversion failed (NaT) *before* filtering
             current_df.dropna(subset=['timestamp'], inplace=True)

             if not current_df.empty:
                # Perform the time-based filtering
                # Assumes timestamps in the log and 'now' are comparable (e.g., both UTC or both local naive)
                 filtered_df = current_df[current_df['timestamp'] >= cutoff_time_naive].copy() # Use .copy() to avoid SettingWithCopyWarning
         except Exception as e:
             dashboard_logger.error(f"Erro durante a filtragem de dados: {e}", exc_info=True)
             # Return empty if filtering fails catastrophically
             filtered_df = pd.DataFrame(columns=current_df.columns)
             # Ensure the timestamp column exists even if empty, and has the right type
             if 'timestamp' not in filtered_df.columns:
                 filtered_df['timestamp'] = pd.Series(dtype='datetime64[ns]')
             else:
                 filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])


    # --- Estatísticas (dos dados filtrados) ---
    level_counts_filtered = filtered_df['level'].value_counts() if not filtered_df.empty else pd.Series(dtype=int)
    stats = {
        'total_filtered': len(filtered_df),
        'info_filtered': int(level_counts_filtered.get('INFO', 0)),
        'warning_filtered': int(level_counts_filtered.get('WARNING', 0)),
        'error_filtered': int(level_counts_filtered.get('ERROR', 0)),
        'critical_filtered': int(level_counts_filtered.get('CRITICAL', 0)),
        'debug_filtered': int(level_counts_filtered.get('DEBUG', 0)),
        'unparsed_filtered': int(level_counts_filtered.get('UNPARSED', 0)), # Inclui se houver
    }

    # --- Logs Recentes (dos dados filtrados, incluindo raw_line) ---
    recent_logs_list = []
    if not filtered_df.empty:
        # Sort by timestamp DESCENDING to get the most recent first
        # Use the constant for max display count
        recent_logs_display_df = filtered_df.sort_values(by='timestamp', ascending=False).head(MAX_RECENT_LOGS_DISPLAY)

        # Ensure timestamp is in ISO format string for JS consistency
        # Handle potential NaT values although they should have been dropped earlier
        recent_logs_display_df['timestamp_iso'] = recent_logs_display_df['timestamp'].apply(lambda x: x.isoformat() if pd.notna(x) else None)

        # Ensure all needed columns exist, fill with empty string if missing
        cols_to_send = ['timestamp_iso', 'level', 'thread', 'module', 'message', 'raw_line']
        for col in cols_to_send:
             if col not in recent_logs_display_df.columns and col != 'timestamp_iso':
                  recent_logs_display_df[col] = '' # Add blank column if missing

        # Select and rename 'timestamp_iso' back to 'timestamp' for the JSON output
        # Fill NaN/NaT values in non-timestamp columns before converting to dict
        recent_logs_list = recent_logs_display_df[cols_to_send].rename(
            columns={'timestamp_iso': 'timestamp'}
        ).fillna('').to_dict(orient='records')


    # --- Dados do Gráfico (Agrupado por minuto, dos dados filtrados) ---
    # Initialize with empty lists
    chart_data = {'labels': [], 'info': [], 'warning': [], 'error_critical': [], 'debug': [], 'unparsed': []}
    if not filtered_df.empty and 'timestamp' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
        try:
            # Check if DataFrame is not empty after potential NaT drops
            if not filtered_df.empty:
                # Set timestamp as index for resampling
                chart_df_indexed = filtered_df.set_index('timestamp')

                # Resample per minute and count levels within each minute bin
                chart_df_resample = chart_df_indexed.resample('min')
                level_counts_per_minute = chart_df_resample['level'].value_counts().unstack(fill_value=0)

                # Create a full minute-by-minute index covering the filtered data range
                start_time = filtered_df['timestamp'].min()
                end_time = filtered_df['timestamp'].max()

                # Ensure start/end times are valid before creating range
                if pd.notna(start_time) and pd.notna(end_time):
                     full_range_index = pd.date_range(start=start_time.floor('min'), end=end_time.ceil('min'), freq='min')

                     # Reindex to include empty minutes and fill with 0
                     level_counts_per_minute = level_counts_per_minute.reindex(full_range_index, fill_value=0)

                     # Prepare data for Chart.js, ensuring columns exist before accessing
                     chart_data['labels'] = level_counts_per_minute.index.strftime('%H:%M').tolist()

                     chart_data['info'] = level_counts_per_minute['INFO'].tolist() if 'INFO' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))
                     chart_data['warning'] = level_counts_per_minute['WARNING'].tolist() if 'WARNING' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))

                     # Combine ERROR and CRITICAL for the chart data point
                     errors = level_counts_per_minute['ERROR'].tolist() if 'ERROR' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))
                     criticals = level_counts_per_minute['CRITICAL'].tolist() if 'CRITICAL' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))
                     chart_data['error_critical'] = [e + c for e, c in zip(errors, criticals)]

                     # Also provide individual counts if needed elsewhere or for future charts
                     chart_data['debug'] = level_counts_per_minute['DEBUG'].tolist() if 'DEBUG' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))
                     chart_data['unparsed'] = level_counts_per_minute['UNPARSED'].tolist() if 'UNPARSED' in level_counts_per_minute.columns else ([0] * len(level_counts_per_minute))
                else:
                    dashboard_logger.warning("No valid start/end time for chart data range, possibly empty filtered data.")
                    # Keep chart_data as initialized empty lists

        except Exception as e:
             dashboard_logger.error(f"Erro ao preparar dados do gráfico: {e}", exc_info=True)
             # Reset chart_data to empty on error
             chart_data = {'labels': [], 'info': [], 'warning': [], 'error_critical': [], 'debug': [], 'unparsed': []}


    # --- Informações de Status ---
    status = {
         'last_file': file_processed,
         'last_update': update_time.isoformat() if update_time else None,
         'parse_errors': parse_errors,
         'monitoring_dir': MONITORED_LOG_DIR,
         'retention_minutes': DATA_RETENTION_MINUTES
     }

    # --- Combina e Retorna JSON ---
    response_data = {
        'status': status,
        'stats': stats,
        'recent_logs': recent_logs_list, # Already sorted most recent first
        'chart_data': chart_data
    }
    # Use Flask's jsonify which correctly handles content type and potential security issues
    return jsonify(response_data)


# --- Rota Principal ---
@app.route('/')
def index():
    """Serve a página HTML principal do dashboard."""
    # Injeta variáveis de configuração no NOVO template
    return render_template_string(
        grafana_html_template, # Use the new template string
        # Pass necessary variables
        FRONTEND_UPDATE_INTERVAL_MS=FRONTEND_UPDATE_INTERVAL_MS,
        MONITORED_LOG_DIR=MONITORED_LOG_DIR,
        DATA_RETENTION_MINUTES=DATA_RETENTION_MINUTES,
        MAX_RECENT_LOGS_DISPLAY=MAX_RECENT_LOGS_DISPLAY,
        MAX_ERROR_LOGS_DISPLAY=MAX_ERROR_LOGS_DISPLAY # Pass new constant
    )

# --- Execução Principal (Mantida) ---
if __name__ == '__main__':
    dashboard_logger.info(f"--- Iniciando Dashboard de Logs (Grafana Style) na Porta {DASHBOARD_PORT} ---")
    monitored_dir_path = Path(MONITORED_LOG_DIR)
    if not monitored_dir_path.exists():
        dashboard_logger.warning(f"Diretório monitorado '{MONITORED_LOG_DIR}' não existe. Tentando criar...")
        try:
            monitored_dir_path.mkdir(parents=True, exist_ok=True)
            dashboard_logger.info(f"Diretório '{MONITORED_LOG_DIR}' criado.")
        except Exception as e:
             dashboard_logger.error(f"Falha ao criar diretório monitorado '{MONITORED_LOG_DIR}': {e}")
             # Consider exiting if the directory is crucial and cannot be created

    dashboard_logger.info("Iniciando thread de atualização de log em background...")
    update_thread = threading.Thread(target=update_log_data_periodically, name="LogUpdateThread", daemon=True)
    update_thread.start()

    print("\n" + "="*60)
    print(f"🚀 Dashboard Grafana Style Iniciado!")
    print(f"   ➡️ Acesse em: http://127.0.0.1:{DASHBOARD_PORT} (ou http://<seu_ip>:{DASHBOARD_PORT})")
    print(f"   📂 Monitorando: '{MONITORED_LOG_DIR}'")
    print(f"   📄 Logs do Dashboard: '{DASHBOARD_LOG_FILE}'")
    print(f"   🕒 Retenção: {DATA_RETENTION_MINUTES} min")
    print(f"   🔄 Intervalo Update: {FRONTEND_UPDATE_INTERVAL_MS / 1000}s (UI), {FILE_SCAN_INTERVAL_SECONDS}s (Scan)")
    print(f"   Pressione CTRL+C para parar.")
    print("="*60 + "\n")

    dashboard_logger.info(f"Servidor Waitress escutando em http://0.0.0.0:{DASHBOARD_PORT}")
    # Use Waitress for serving
    serve(app, host='0.0.0.0', port=DASHBOARD_PORT, threads=8)