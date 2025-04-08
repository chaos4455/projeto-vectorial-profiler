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
import html # Para escapar HTML no modal

import pandas as pd
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from waitress import serve # Use waitress para um servidor mais robusto

from typing import Optional, List, Dict, Any

# --- Configuração ---
DASHBOARD_PORT: int = 8444 # <--- Porta pode ser alterada
MONITORED_LOG_DIR: str = "valuation_v3_web_log" # <--- Diretório monitorado pode ser alterado
FILENAME_TIMESTAMP_REGEX = re.compile(r'_(\d{8})_(\d{6})\.log$') # Para encontrar o mais recente pelo nome
FILE_SCAN_INTERVAL_SECONDS: int = 1 # Intervalo curto para verificação de arquivos
FRONTEND_UPDATE_INTERVAL_MS: int = 1500 # Intervalo de atualização do frontend (um pouco maior para suavidade)
DATA_RETENTION_MINUTES: int = 30 # Quanto tempo para trás manter/exibir logs na UI
MAX_RECENT_LOGS_DISPLAY: int = 150 # Limitar linhas exibidas em "Logs Recentes" (aumentado)

LOG_LEVEL = logging.INFO # Logging para o próprio dashboard
DASHBOARD_LOG_DIR = "dashboard_logs"
os.makedirs(DASHBOARD_LOG_DIR, exist_ok=True)
DASHBOARD_LOG_FILE = os.path.join(DASHBOARD_LOG_DIR, f"dashboard_monitor_{DASHBOARD_PORT}.log")

# --- Configuração do Logging do Dashboard ---
dashboard_logger = logging.getLogger(f"dashboard_logger_{DASHBOARD_PORT}")
dashboard_logger.setLevel(LOG_LEVEL)
# Log para arquivo
fh = logging.FileHandler(DASHBOARD_LOG_FILE, encoding='utf-8')
fh.setLevel(LOG_LEVEL)
# Log para console
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Dashboard] - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# Adicionar handlers apenas uma vez
if not dashboard_logger.handlers:
    dashboard_logger.addHandler(fh)
    dashboard_logger.addHandler(ch)

# --- Armazenamento Global de Dados ---
log_data_lock = threading.Lock()
# Inicializar com DataFrame vazio correspondendo ao formato do log alvo
log_data_df = pd.DataFrame(columns=[
    'timestamp', 'level', 'thread', 'module', 'message', 'raw_line' # Garante que raw_line está aqui
])
log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp']) # Garantir dtype correto

last_file_processed: Optional[str] = None
last_file_mod_time: Optional[float] = None
log_parsing_errors: int = 0
last_update_time: Optional[datetime.datetime] = None

# Regex para analisar o formato de log VALUATION (mantido o mesmo)
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

# --- Funções Auxiliares ---

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
                # Usa o mais recente entre o timestamp do nome e o de modificação
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
                    # Adiciona a linha não parseada com um marcador? Opcional.
                    # Pode ser útil ter a linha bruta mesmo sem parse.
                    # parsed_lines_data.append({
                    #     'timestamp': pd.NaT, # Ou usar um timestamp aproximado?
                    #     'level': 'UNPARSED',
                    #     'thread': '', 'module': '',
                    #     'message': line,
                    #     'raw_line': raw_line.strip()
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
        # Contar estas como erros de parse adicionais? Pode fazer sentido.
        # log_parsing_errors += dropped_rows

    if df.empty:
         return df # Retorna vazio se todas as linhas foram descartadas

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['level', 'thread', 'module', 'message', 'raw_line']:
         if col in df.columns:
              df[col] = df[col].astype(str).fillna('') # Garante string e trata nulos

    return df.sort_values(by='timestamp') # Ordena por timestamp


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
                    dashboard_logger.warning(f"Não foi possível obter tempo de modificação para {latest_file.name}: {e}. Verificando.")
                    # Não assume necessidade de processar ainda, apenas tenta comparar

                # Lógica de decisão para processar
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
                    parse_session_errors = log_parsing_errors # Pega o contador global atual

                    with log_data_lock:
                        if not new_df.empty:
                            # Substitui completamente os dados pelos do arquivo mais recente
                            log_data_df = new_df
                            last_file_processed = latest_file.name
                            last_file_mod_time = current_mod_time
                            last_update_time = current_time
                            level_counts = log_data_df['level'].value_counts()
                            dashboard_logger.info(f"Dados atualizados de '{latest_file.name}'. {len(log_data_df)} linhas. Erros/Críticos: {level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)}, Avisos: {level_counts.get('WARNING', 0)}. Erros de Parse Totais: {parse_session_errors}")
                        elif latest_file.name == last_file_processed and current_mod_time == last_file_mod_time:
                             # Parse retornou vazio, mas era o mesmo arquivo e não foi modificado
                             # Provavelmente não há novas linhas válidas, não faz nada
                             dashboard_logger.debug(f"Parse de '{latest_file.name}' retornou vazio, mas arquivo não modificado. Sem alterações.")
                        else:
                            # Parse retornou vazio E era um arquivo novo ou modificado
                            dashboard_logger.warning(f"Análise de '{latest_file.name}' resultou em DataFrame vazio. Limpando dados anteriores.")
                            log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                            log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                            last_file_processed = latest_file.name
                            last_file_mod_time = current_mod_time
                            last_update_time = current_time
            else:
                # Nenhum arquivo encontrado
                if last_file_processed is not None:
                    dashboard_logger.warning(f"Nenhum arquivo de log encontrado em '{MONITORED_LOG_DIR}'. Limpando dados anteriores.")
                    with log_data_lock:
                        log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                        log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                        last_file_processed = None
                        last_file_mod_time = None
                        last_update_time = current_time
                        # Opcional: resetar log_parsing_errors = 0

        except Exception as e:
            dashboard_logger.error(f"Erro crítico no loop de atualização em background: {e}", exc_info=True)
            time.sleep(FILE_SCAN_INTERVAL_SECONDS * 5) # Espera mais em caso de erro

        time.sleep(FILE_SCAN_INTERVAL_SECONDS)

# --- Flask App ---
app = Flask(__name__)
CORS(app) # Permitir todas as origens

# --- Template HTML (Grafana Dark Theme - pt-BR) ---
html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Logs - {MONITORED_LOG_DIR}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark-start: #1a202c; /* cinza-900 */
            --bg-dark-end: #2d3748;   /* cinza-800 */
            --card-bg: rgba(45, 55, 72, 0.65); /* cinza-700 com alpha */
            --card-bg-hover: rgba(45, 55, 72, 0.8);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-primary: #e2e8f0;  /* cinza-200 */
            --text-secondary: #a0aec0; /* cinza-500 */
            --text-muted: #718096;   /* cinza-600 */
            --color-cyan: #4fd1c5;    /* teal-300 */
            --color-blue: #63b3ed;    /* blue-400 */
            --color-yellow: #f6e05e;  /* yellow-400 */
            --color-red: #fc8181;     /* red-400 */
            --color-red-dark: #f56565;/* red-500 */
            --color-green: #68d391;  /* green-400 */ /* Adicionado para possível uso futuro */
            --scrollbar-track: rgba(74, 85, 104, 0.3); /* cinza-700 alpha */
            --scrollbar-thumb: rgba(160, 174, 192, 0.5); /* cinza-500 alpha */
            --scrollbar-thumb-hover: rgba(160, 174, 192, 0.7);
        }
        body {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(135deg, var(--bg-dark-start) 0%, var(--bg-dark-end) 100%);
            color: var(--text-primary);
            overscroll-behavior-y: none; /* Previne pull-to-refresh */
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }

        /* Card com Glassmorphism */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px) saturate(160%);
            -webkit-backdrop-filter: blur(10px) saturate(160%);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem; /* rounded-xl */
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        .glass-card:hover {
            background: var(--card-bg-hover);
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        }

        /* Cores dos Níveis de Log */
        .level-DEBUG { color: var(--text-muted); }
        .level-INFO { color: var(--color-blue); }
        .level-WARNING { color: var(--color-yellow); }
        .level-ERROR { color: var(--color-red); }
        .level-CRITICAL {
            color: #ffffff; /* Branco para destacar sobre fundo vermelho */
            background-color: var(--color-red-dark);
            padding: 1px 5px;
            border-radius: 4px;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .level-UNPARSED { color: var(--text-secondary); font-style: italic;}

        /* Ícones dos Níveis */
        .icon-DEBUG { color: var(--text-muted); }
        .icon-INFO { color: var(--color-blue); }
        .icon-WARNING { color: var(--color-yellow); }
        .icon-ERROR { color: var(--color-red); }
        .icon-CRITICAL { color: var(--color-red-dark); } /* Usa vermelho mais escuro pro icone */
        .icon-UNPARSED { color: var(--text-secondary); }

        /* Estilo da Entrada de Log */
        .log-entry {
            font-family: 'Menlo', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
            padding: 0.4rem 0.6rem;
            border-radius: 0.25rem;
            margin-bottom: 0.2rem;
            background-color: rgba(26, 32, 44, 0.5); /* cinza-900 com alpha */
            display: flex;
            flex-wrap: nowrap;
            gap: 0.7rem;
            align-items: baseline;
            cursor: pointer; /* Indica que é clicável */
            transition: background-color 0.2s ease, opacity 0.4s ease-out;
            opacity: 0; /* Inicia invisível para fade-in */
        }
        .log-entry.visible { opacity: 1; } /* Classe para ativar fade-in */
        .log-entry:hover { background-color: rgba(26, 32, 44, 0.8); }

        .log-time { color: var(--text-muted); flex-shrink: 0; width: 75px; text-align: left; }
        .log-level-cont { width: 90px; flex-shrink: 0; text-align: left; } /* Container para nivel e icone */
        .log-level {} /* O estilo vem da classe level-* */
        .log-thread { color: var(--text-secondary); width: 110px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .log-module { color: var(--color-cyan); width: 130px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .log-message { color: var(--text-primary); flex-grow: 1; word-break: break-word; white-space: pre-wrap; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--scrollbar-track); border-radius: 4px;}
        ::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-thumb-hover); }

        /* Animação de Fade-in para Seções */
         @keyframes fadeInSection { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in-section { animation: fadeInSection 0.6s ease-out forwards; opacity: 0; } /* Inicia oculto */

        /* Estilos do Modal */
        .modal-overlay {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex; align-items: center; justify-content: center;
            opacity: 0; visibility: hidden;
            transition: opacity 0.3s ease, visibility 0.3s ease;
            z-index: 1000;
        }
        .modal-overlay.active { opacity: 1; visibility: visible; }
        .modal-content {
            background: var(--bg-dark-end);
            color: var(--text-primary);
            padding: 1.5rem 2rem;
            border-radius: 0.5rem;
            border: 1px solid var(--card-border);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
            width: 80%; max-width: 900px;
            max-height: 80vh;
            display: flex; flex-direction: column;
            transform: scale(0.95);
            transition: transform 0.3s ease;
        }
        .modal-overlay.active .modal-content { transform: scale(1); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem;}
        .modal-title { font-family: 'Orbitron', sans-serif; font-size: 1.2rem; color: var(--color-cyan); }
        .modal-close-btn { background: none; border: none; color: var(--text-secondary); font-size: 1.5rem; cursor: pointer; transition: color 0.2s ease; }
        .modal-close-btn:hover { color: var(--text-primary); }
        .modal-body { overflow-y: auto; flex-grow: 1; }
        .modal-body pre {
            white-space: pre-wrap;       /* Quebra linhas longas */
            word-wrap: break-word;      /* Quebra palavras longas se necessário */
            font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            background-color: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 4px;
            border: 1px solid var(--card-border);
        }

    </style>
</head>
<body class="p-4 md:p-6 lg:p-8 min-h-screen">

    <!-- Header -->
    <header class="mb-6 md:mb-8 flex flex-col sm:flex-row justify-between items-center gap-3 fade-in-section">
        <h1 class="text-2xl md:text-4xl font-orbitron font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 flex items-center">
           <i class="fas fa-terminal mr-3 text-cyan-400"></i>Monitor de Logs <span class="text-lg text-gray-400 ml-2 hidden md:inline">({MONITORED_LOG_DIR})</span>
        </h1>
        <div class="text-xs text-gray-400 text-center sm:text-right space-y-1">
             <p><i class="fas fa-file-alt mr-1 text-gray-500"></i>Arquivo: <code id="monitoringFile" class="text-gray-300 bg-gray-700/50 px-2 py-0.5 rounded text-xs">N/D</code></p>
             <p><i class="fas fa-sync-alt mr-1 text-gray-500"></i>Última Atualização: <span id="lastUpdateTime" class="text-gray-300">Nunca</span></p>
             <p><i class="fas fa-bug mr-1 text-red-500"></i> Erros de Parse: <span id="parseErrorCount" class="text-red-400 font-semibold">0</span></p>
         </div>
    </header>

    <!-- Cards de Estatísticas -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-6 md:mb-8">
        <!-- Card Logs Totais -->
        <div class="glass-card p-4 md:p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.1s;">
            <div class="text-4xl text-blue-400 opacity-80"><i class="fas fa-database fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-xs uppercase tracking-wider">Total Visível ({DATA_RETENTION_MINUTES}min)</div>
                <div id="totalLogs" class="text-2xl lg:text-3xl font-bold text-gray-100 font-orbitron">0</div>
            </div>
        </div>
        <!-- Card Logs Info -->
        <div class="glass-card p-4 md:p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.2s;">
             <div class="text-4xl text-blue-400 opacity-80"><i class="fas fa-info-circle fa-fw"></i></div>
             <div>
                 <div class="text-gray-400 text-xs uppercase tracking-wider">Info</div>
                 <div id="infoLogs" class="text-2xl lg:text-3xl font-bold text-gray-100 font-orbitron">0</div>
             </div>
         </div>
        <!-- Card Logs Warning -->
        <div class="glass-card p-4 md:p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.3s;">
            <div class="text-4xl text-yellow-400 opacity-80"><i class="fas fa-exclamation-triangle fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-xs uppercase tracking-wider">Warnings</div>
                <div id="warningLogs" class="text-2xl lg:text-3xl font-bold text-gray-100 font-orbitron">0</div>
            </div>
        </div>
        <!-- Card Logs Error/Critical -->
        <div class="glass-card p-4 md:p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.4s;">
            <div class="text-4xl text-red-400 opacity-80"><i class="fas fa-shield-virus fa-fw"></i></div> <!-- Ícone diferente para erros -->
            <div>
                <div class="text-gray-400 text-xs uppercase tracking-wider">Errors & Critical</div>
                <div id="errorLogs" class="text-2xl lg:text-3xl font-bold text-gray-100 font-orbitron">0</div>
            </div>
        </div>
    </section>

    <!-- Gráfico e Logs Recentes -->
    <section class="grid grid-cols-1 lg:grid-cols-5 gap-6 md:gap-8">
        <!-- Coluna do Gráfico (Mais Larga) -->
        <div class="lg:col-span-3">
             <div class="glass-card p-4 md:p-5 h-full fade-in-section" style="animation-delay: 0.5s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300 flex items-center"><i class="fas fa-chart-bar mr-2 text-cyan-400"></i>Atividade por Minuto (Últimos {DATA_RETENTION_MINUTES} Min)</h2>
                <div class="relative h-72 md:h-96"> {/* Altura aumentada */}
                   <canvas id="logChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Coluna de Logs Recentes (Mais Estreita) -->
        <div class="lg:col-span-2">
             <div class="glass-card p-4 md:p-5 h-full flex flex-col fade-in-section" style="animation-delay: 0.6s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300 flex items-center"><i class="fas fa-stream mr-2 text-cyan-400"></i>Logs Recentes (Max {MAX_RECENT_LOGS_DISPLAY})</h2>
                <div id="logContainer" class="flex-grow overflow-y-auto pr-2 space-y-1" style="max-height: calc(var(--log-container-height, 400px) + 96px);"> {/* Altura aumentada e dinâmica */}
                    <!-- Entradas de log serão injetadas aqui -->
                     <p id="noLogsMessage" class="text-center text-gray-500 italic mt-8">Aguardando dados de log...</p>
                 </div>
            </div>
        </div>
    </section>

     <!-- Footer -->
     <footer class="text-center text-xs text-gray-500 mt-10 pt-5 border-t border-gray-700/50">
         Dashboard de Logs | Desenvolvido com Flask & Chart.js | <span id="currentYear"></span>
     </footer>

     <!-- Modal para Detalhes do Log -->
     <div id="logModal" class="modal-overlay">
         <div class="modal-content">
             <div class="modal-header">
                 <h3 class="modal-title">Detalhes do Log</h3>
                 <button id="modalCloseBtn" class="modal-close-btn">×</button>
             </div>
             <div class="modal-body">
                 <pre id="modalLogContent">Conteúdo do log aqui...</pre>
             </div>
         </div>
     </div>


    <script>
        const API_ENDPOINT = '/api/data';
        const UPDATE_INTERVAL = {{ FRONTEND_UPDATE_INTERVAL_MS }};
        const MAX_LOGS = {{ MAX_RECENT_LOGS_DISPLAY }};
        let logChart = null;

        // --- Elementos DOM ---
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
        const logChartCanvas = document.getElementById('logChart');

        // --- Elementos DOM do Modal ---
        const logModal = document.getElementById('logModal');
        const modalLogContent = document.getElementById('modalLogContent');
        const modalCloseBtn = document.getElementById('modalCloseBtn');


        // --- Configuração do Chart.js ---
        function initializeChart() {
            if (!logChartCanvas) return;
            const ctx = logChartCanvas.getContext('2d');
            if (logChart) { logChart.destroy(); }

            // Gradientes para as barras
            const infoGradient = ctx.createLinearGradient(0, 0, 0, 300);
            infoGradient.addColorStop(0, 'rgba(99, 179, 237, 0.8)');
            infoGradient.addColorStop(1, 'rgba(99, 179, 237, 0.3)');

            const warnGradient = ctx.createLinearGradient(0, 0, 0, 300);
            warnGradient.addColorStop(0, 'rgba(246, 224, 94, 0.8)');
            warnGradient.addColorStop(1, 'rgba(246, 224, 94, 0.3)');

            const errGradient = ctx.createLinearGradient(0, 0, 0, 300);
            errGradient.addColorStop(0, 'rgba(252, 129, 129, 0.8)');
            errGradient.addColorStop(1, 'rgba(252, 129, 129, 0.3)');


            logChart = new Chart(ctx, {
                type: 'bar',
                data: { labels: [], datasets: [] }, // Inicializa vazio, será preenchido
                 options: {
                     responsive: true, maintainAspectRatio: false,
                     interaction: { mode: 'index', intersect: false }, // Tooltip mostra todos no mesmo índice
                     scales: {
                         x: {
                             stacked: true, ticks: { color: 'var(--text-secondary)', maxRotation: 0, autoSkip: true, autoSkipPadding: 15 },
                             grid: { color: 'rgba(255, 255, 255, 0.05)' }
                         },
                         y: {
                             stacked: true, beginAtZero: true, ticks: { color: 'var(--text-secondary)', precision: 0 },
                             grid: { color: 'rgba(255, 255, 255, 0.07)' }
                         }
                     },
                     plugins: {
                         legend: { position: 'bottom', labels: { color: 'var(--text-primary)', usePointStyle: true, boxWidth: 8, padding: 20 } },
                         tooltip: {
                             backgroundColor: 'rgba(26, 32, 44, 0.9)', titleColor: 'var(--text-primary)', bodyColor: 'var(--text-secondary)',
                             borderColor: 'var(--card-border)', borderWidth: 1, padding: 10, boxPadding: 4, usePointStyle: true,
                         }
                     },
                     animation: { duration: 500, easing: 'easeOutCubic' } // Animação suave
                 }
             });

             // Define os datasets após criar o chart para usar os gradientes
             logChart.data.datasets = [
                { label: 'Info', data: [], backgroundColor: infoGradient, borderColor: 'rgba(99, 179, 237, 1)', borderWidth: 0, stack: 'Stack 0' },
                { label: 'Warning', data: [], backgroundColor: warnGradient, borderColor: 'rgba(246, 224, 94, 1)', borderWidth: 0, stack: 'Stack 0' },
                { label: 'Error/Critical', data: [], backgroundColor: errGradient, borderColor: 'rgba(252, 129, 129, 1)', borderWidth: 0, stack: 'Stack 0' }
             ];
             logChart.update();
        }

        // --- Funções Auxiliares ---
        function getLevelClass(level) { return `level-${level.toUpperCase()}`; }
        function getIconClass(level) { return `icon-${level.toUpperCase()}`; }
        function getLevelIcon(level) {
             level = level ? level.toUpperCase() : 'UNKNOWN';
             switch (level) {
                 case 'DEBUG': return 'fas fa-bug fa-fw';
                 case 'INFO': return 'fas fa-info-circle fa-fw';
                 case 'WARNING': return 'fas fa-exclamation-triangle fa-fw';
                 case 'ERROR': return 'fas fa-times-circle fa-fw';
                 case 'CRITICAL': return 'fas fa-skull-crossbones fa-fw'; // Ícone mais forte
                 case 'UNPARSED': return 'fas fa-question-circle fa-fw';
                 default: return 'fas fa-question-circle fa-fw';
             }
         }
         // Função segura para escapar HTML
         function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') return '';
            return unsafe
                 .replace(/&/g, "&")
                 .replace(/</g, "<")
                 .replace(/>/g, ">")
                 .replace(/'/g, "'");
         }

         // --- Atualização de Dados ---
         let lastUpdateTimeStr = 'Nunca'; // Armazena o último timestamp para evitar piscar

         async function fetchDataAndUpdate() {
             // console.debug("Buscando dados..."); // Descomentar para debug
             try {
                 const response = await fetch(API_ENDPOINT);
                 if (!response.ok) {
                     console.error("Falha ao buscar dados:", response.status, response.statusText);
                     noLogsMessageEl.textContent = `Erro ${response.status} ao buscar dados. Verifique a conexão com o backend.`;
                     noLogsMessageEl.style.display = 'block'; noLogsMessageEl.style.color = 'var(--color-red)';
                     // Limpa os contadores em caso de erro? Opcional.
                     // totalLogsEl.textContent = '-'; infoLogsEl.textContent = '-'; ...
                    return;
                 }
                 const data = await response.json();
                 // console.debug("Dados recebidos:", data); // Descomentar para debug

                 // Atualiza Cards de Estatísticas
                 totalLogsEl.textContent = data.stats.total_filtered?.toLocaleString() || 0;
                 infoLogsEl.textContent = data.stats.info_filtered?.toLocaleString() || 0;
                 warningLogsEl.textContent = data.stats.warning_filtered?.toLocaleString() || 0;
                 errorLogsEl.textContent = (data.stats.error_filtered + data.stats.critical_filtered)?.toLocaleString() || 0;

                 // Atualiza Informações do Header
                monitoringFileEl.textContent = data.status.last_file || 'N/D';
                parseErrorCountEl.textContent = data.status.parse_errors || 0;
                if (data.status.last_update) {
                    lastUpdateTimeStr = new Date(data.status.last_update).toLocaleString('pt-BR');
                }
                lastUpdateTimeEl.textContent = lastUpdateTimeStr;


                 // Atualiza Logs Recentes
                 // logContainerEl.innerHTML = ''; // Limpa TUDO e recria - pode causar flicker
                 const existingLogTimestamps = new Set(Array.from(logContainerEl.querySelectorAll('.log-entry')).map(el => el.dataset.timestamp));
                 const logsToAdd = [];
                 const receivedLogTimestamps = new Set();

                 if (data.recent_logs && data.recent_logs.length > 0) {
                     noLogsMessageEl.style.display = 'none';
                     const fragment = document.createDocumentFragment();

                     data.recent_logs.forEach(log => {
                         const timestampKey = log.timestamp; // Usar o timestamp ISO como chave única
                         receivedLogTimestamps.add(timestampKey);

                         if (!existingLogTimestamps.has(timestampKey)) {
                             const logDiv = document.createElement('div');
                             logDiv.className = 'log-entry'; // Começa invisível
                             logDiv.dataset.timestamp = timestampKey; // Armazena timestamp para identificar
                             logDiv.dataset.rawlog = log.raw_line || ''; // Armazena raw log para modal

                             const timeStr = new Date(log.timestamp).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                             const level = log.level || 'UNKNOWN';
                             const thread = log.thread || '-';
                             const module = log.module || '-';
                             const message = log.message || '';

                             logDiv.innerHTML = `
                                 <span class="log-time">${timeStr}</span>
                                 <span class="log-level-cont">
                                     <span class="log-level ${getLevelClass(level)}">
                                        <i class="${getLevelIcon(level)} ${getIconClass(level)} mr-1.5" title="${level}"></i>${level}
                                     </span>
                                 </span>
                                 <span class="log-thread" title="${thread}">${thread}</span>
                                 <span class="log-module" title="${module}">${module}</span>
                                 <span class="log-message">${escapeHtml(message)}</span>
                             `;
                             logsToAdd.push(logDiv); // Adiciona ao array para inserir depois
                          }
                     });

                    // Remover logs antigos que não estão mais na lista recebida
                    Array.from(logContainerEl.querySelectorAll('.log-entry')).forEach(el => {
                        if (!receivedLogTimestamps.has(el.dataset.timestamp)) {
                            el.remove();
                        }
                    });

                    // Adicionar novos logs no início (ou fim, dependendo da ordem desejada)
                    logsToAdd.reverse().forEach(logDiv => { // Reverte para adicionar os mais novos no topo
                        fragment.appendChild(logDiv);
                    });
                    logContainerEl.prepend(fragment); // Adiciona os novos no topo

                    // Aciona a transição CSS para os novos logs
                    requestAnimationFrame(() => {
                        logsToAdd.forEach(el => el.classList.add('visible'));
                    });

                 } else {
                     logContainerEl.innerHTML = ''; // Limpa se não há logs
                     noLogsMessageEl.textContent = 'Nenhuma entrada de log encontrada no período.';
                     noLogsMessageEl.style.display = 'block';
                 }

                // Atualiza Chart
                if (logChart && data.chart_data) {
                    logChart.data.labels = data.chart_data.labels;
                    logChart.data.datasets[0].data = data.chart_data.info;
                    logChart.data.datasets[1].data = data.chart_data.warning;
                    logChart.data.datasets[2].data = data.chart_data.error_critical;
                    logChart.update('none'); // Atualização suave sem reanimar tudo
                } else if (logChart) {
                    // Limpa o gráfico se não houver dados
                    logChart.data.labels = [];
                    logChart.data.datasets.forEach(ds => ds.data = []);
                    logChart.update('none');
                }

            } catch (error) {
                console.error("Erro ao buscar ou processar dados:", error);
                 noLogsMessageEl.textContent = 'Erro crítico ao carregar dados do dashboard. Verifique o console.';
                 noLogsMessageEl.style.display = 'block'; noLogsMessageEl.style.color = 'var(--color-red)';
            }
        }

        // --- Lógica do Modal ---
        function openModal(rawLogContent) {
            if (!logModal || !modalLogContent) return;
            modalLogContent.textContent = rawLogContent; // Usa textContent para segurança e preservar formatação <pre>
            logModal.classList.add('active');
        }

        function closeModal() {
            if (!logModal) return;
            logModal.classList.remove('active');
        }

        // Event Listener para abrir o modal (delegação de evento)
        logContainerEl.addEventListener('click', (event) => {
            const logEntry = event.target.closest('.log-entry');
            if (logEntry && logEntry.dataset.rawlog) {
                openModal(logEntry.dataset.rawlog);
            }
        });

        // Event Listeners para fechar o modal
        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', closeModal);
        }
        if (logModal) {
            logModal.addEventListener('click', (event) => {
                // Fecha se clicar no overlay (fora do content)
                if (event.target === logModal) {
                    closeModal();
                }
            });
        }
        // Fecha modal com a tecla Escape
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && logModal && logModal.classList.contains('active')) {
                closeModal();
            }
        });


        // --- Inicialização ---
        document.addEventListener('DOMContentLoaded', () => {
            initializeChart();
            fetchDataAndUpdate(); // Busca inicial
            setInterval(fetchDataAndUpdate, UPDATE_INTERVAL); // Busca periódica
            if (currentYearEl) currentYearEl.textContent = new Date().getFullYear();

            // Ajusta a altura do container de logs dinamicamente (opcional)
            const chartContainer = document.querySelector('.lg\\:col-span-3 .glass-card');
            if (chartContainer && logContainerEl) {
                 const resizeObserver = new ResizeObserver(entries => {
                     for (let entry of entries) {
                         const chartHeight = entry.contentRect.height;
                         // Tenta igualar a altura da área de scroll do log à altura total do card do gráfico
                         // Ajuste o '- 50' conforme necessário para compensar padding/título no card de logs
                         const targetLogHeight = chartHeight - (document.querySelector('.lg\\:col-span-2 .glass-card h2')?.offsetHeight || 30) - 40; // 40px para p-4/p-5
                         logContainerEl.style.setProperty('--log-container-height', `${Math.max(200, targetLogHeight)}px`); // Min height 200px
                     }
                 });
                 resizeObserver.observe(chartContainer);
             }
        });

    </script>

</body>
</html>
"""

# --- Endpoint da API (sem grandes mudanças, apenas garante 'raw_line') ---
@app.route('/api/data')
def get_api_data():
    """Fornece dados de log filtrados, estatísticas e informações do gráfico para o frontend."""
    # dashboard_logger.debug("Requisição API /api/data recebida") # Log muito verboso
    with log_data_lock:
        current_df = log_data_df.copy()
        file_processed = last_file_processed
        update_time = last_update_time
        parse_errors = log_parsing_errors

    now_naive = datetime.datetime.now()
    cutoff_time_naive = now_naive - datetime.timedelta(minutes=DATA_RETENTION_MINUTES)

    filtered_df = pd.DataFrame(columns=current_df.columns) # Default vazio
    if 'timestamp' in current_df.columns and not current_df.empty:
         try:
             # Garante datetime e remove NaT se ainda existir algum
             if not pd.api.types.is_datetime64_any_dtype(current_df['timestamp']):
                 current_df['timestamp'] = pd.to_datetime(current_df['timestamp'], errors='coerce')
                 current_df.dropna(subset=['timestamp'], inplace=True)

             if not current_df.empty: # Verifica se ainda há dados após dropna
                filtered_df = current_df[current_df['timestamp'] >= cutoff_time_naive].copy()
             # dashboard_logger.debug(f"Filtragem completa. {len(filtered_df)} linhas nos últimos {DATA_RETENTION_MINUTES} minutos.")
         except Exception as e:
             dashboard_logger.error(f"Erro durante a filtragem de dados: {e}", exc_info=True)
             filtered_df = pd.DataFrame(columns=current_df.columns)

    # --- Estatísticas (dos dados filtrados) ---
    level_counts_filtered = filtered_df['level'].value_counts() if not filtered_df.empty else pd.Series(dtype=int)
    stats = {
        'total_filtered': len(filtered_df),
        'info_filtered': int(level_counts_filtered.get('INFO', 0)),
        'warning_filtered': int(level_counts_filtered.get('WARNING', 0)),
        'error_filtered': int(level_counts_filtered.get('ERROR', 0)),
        'critical_filtered': int(level_counts_filtered.get('CRITICAL', 0)),
        'debug_filtered': int(level_counts_filtered.get('DEBUG', 0)),
        # Adicionar UNPARSED se for rastreado
        'unparsed_filtered': int(level_counts_filtered.get('UNPARSED', 0)),
    }

    # --- Logs Recentes (dos dados filtrados, incluindo raw_line) ---
    recent_logs_list = []
    if not filtered_df.empty:
        recent_logs_display_df = filtered_df.sort_values(by='timestamp', ascending=False).head(MAX_RECENT_LOGS_DISPLAY)
        # Usa to_isoformat() que é padrão e bem parseado pelo JS Date()
        recent_logs_display_df['timestamp_iso'] = recent_logs_display_df['timestamp'].apply(lambda x: x.isoformat())
        # Garante que todas as colunas necessárias existem antes de selecionar
        cols_to_send = ['timestamp_iso', 'level', 'thread', 'module', 'message', 'raw_line']
        for col in cols_to_send:
             if col not in recent_logs_display_df.columns and col != 'timestamp_iso': # timestamp_iso é criado agora
                  recent_logs_display_df[col] = '' # Adiciona coluna vazia se faltar

        recent_logs_list = recent_logs_display_df[cols_to_send].rename(columns={'timestamp_iso': 'timestamp'}).to_dict(orient='records')
        # dashboard_logger.debug(f"Preparadas {len(recent_logs_list)} entradas de log recentes.")


    # --- Dados do Gráfico (Agrupado por minuto, dos dados filtrados) ---
    chart_data = {'labels': [], 'info': [], 'warning': [], 'error_critical': []}
    if not filtered_df.empty and 'timestamp' in filtered_df.columns:
        try:
            # Reamostragem mais robusta
             chart_df_resample = filtered_df.set_index('timestamp').resample('min')
             # Contar níveis dentro de cada grupo de minuto
             level_counts_per_minute = chart_df_resample['level'].value_counts().unstack(fill_value=0)

             # Garante colunas essenciais
             for level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                 if level not in level_counts_per_minute.columns:
                     level_counts_per_minute[level] = 0

             # Prepara dados para Chart.js
             chart_data['labels'] = level_counts_per_minute.index.strftime('%H:%M').tolist()
             chart_data['info'] = level_counts_per_minute['INFO'].tolist()
             chart_data['warning'] = level_counts_per_minute['WARNING'].tolist()
             chart_data['error_critical'] = (level_counts_per_minute['ERROR'] + level_counts_per_minute['CRITICAL']).tolist()
             # dashboard_logger.debug(f"Preparados dados do gráfico com {len(chart_data['labels'])} pontos.")

        except Exception as e:
             dashboard_logger.error(f"Erro ao preparar dados do gráfico: {e}", exc_info=True)


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
        'recent_logs': recent_logs_list, # Já ordenado do mais recente para o mais antigo
        'chart_data': chart_data
    }
    return jsonify(response_data)


# --- Rota Principal ---
@app.route('/')
def index():
    """Serve a página HTML principal do dashboard."""
    # Injeta variáveis de configuração no template
    return render_template_string(
        html_template,
        # Passa as variáveis necessárias para o template JS/HTML
        FRONTEND_UPDATE_INTERVAL_MS=FRONTEND_UPDATE_INTERVAL_MS,
        MONITORED_LOG_DIR=MONITORED_LOG_DIR,
        DATA_RETENTION_MINUTES=DATA_RETENTION_MINUTES,
        MAX_RECENT_LOGS_DISPLAY=MAX_RECENT_LOGS_DISPLAY
    )

# --- Execução Principal ---
if __name__ == '__main__':
    dashboard_logger.info(f"--- Iniciando Dashboard de Logs na Porta {DASHBOARD_PORT} ---")
    monitored_dir_path = Path(MONITORED_LOG_DIR)
    if not monitored_dir_path.exists():
        dashboard_logger.warning(f"Diretório monitorado '{MONITORED_LOG_DIR}' não existe. Tentando criar...")
        try:
            monitored_dir_path.mkdir(parents=True, exist_ok=True)
            dashboard_logger.info(f"Diretório '{MONITORED_LOG_DIR}' criado.")
        except Exception as e:
             dashboard_logger.error(f"Falha ao criar diretório monitorado '{MONITORED_LOG_DIR}': {e}")
             # Considerar sair se o diretório for essencial e não puder ser criado? Por enquanto continua.

    dashboard_logger.info("Iniciando thread de atualização de log em background...")
    update_thread = threading.Thread(target=update_log_data_periodically, name="LogUpdateThread", daemon=True)
    update_thread.start()

    print("\n" + "="*60)
    print(f"🚀 Dashboard de Monitoramento de Logs Iniciado!")
    print(f"   ➡️ Acesse em: http://127.0.0.1:{DASHBOARD_PORT}")
    print(f"   📂 Monitorando diretório: '{MONITORED_LOG_DIR}'")
    print(f"   📄 Logs do Dashboard: '{DASHBOARD_LOG_FILE}'")
    print(f"   🕒 Retenção de dados: {DATA_RETENTION_MINUTES} minutos")
    print(f"   🔄 Intervalo de atualização: {FRONTEND_UPDATE_INTERVAL_MS / 1000} segundos (frontend), {FILE_SCAN_INTERVAL_SECONDS} segundo(s) (backend)")
    print(f"   Pressione CTRL+C para parar.")
    print("="*60 + "\n")

    dashboard_logger.info(f"Servidor Waitress escutando em http://0.0.0.0:{DASHBOARD_PORT}")
    # Usar Waitress para produção
    serve(app, host='0.0.0.0', port=DASHBOARD_PORT, threads=6) # Aumentado número de threads