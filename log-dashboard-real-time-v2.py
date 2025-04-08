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

import pandas as pd
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS
from waitress import serve # Use waitress for a more robust server

from typing import Optional, List, Dict, Any

# --- Configuration ---
DASHBOARD_PORT: int = 8444 # <--- Porta pode ser alterada
MONITORED_LOG_DIR: str = "valuation_v3_web_log" # <--- Diretório monitorado pode ser alterado
FILENAME_TIMESTAMP_REGEX = re.compile(r'_(\d{8})_(\d{6})\.log$') # Para encontrar o mais recente pelo nome
# <<< ALTERADO: Intervalos menores para atualização "em tempo real" >>>
FILE_SCAN_INTERVAL_SECONDS: int = 1 # Com que frequência verificar novo arquivo / reler (Mais curto)
FRONTEND_UPDATE_INTERVAL_MS: int = 1000 # Com que frequência o JS do frontend busca dados (em milissegundos)
# <<< FIM DA ALTERAÇÃO >>>
DATA_RETENTION_MINUTES: int = 30 # Quanto tempo para trás manter/exibir logs na UI
MAX_RECENT_LOGS_DISPLAY: int = 100 # Limitar linhas exibidas em "Recent Logs"

LOG_LEVEL = logging.INFO # Logging para o próprio dashboard
# Garante que os logs do dashboard vão para um diretório separado
DASHBOARD_LOG_DIR = "dashboard_logs"
os.makedirs(DASHBOARD_LOG_DIR, exist_ok=True)
DASHBOARD_LOG_FILE = os.path.join(DASHBOARD_LOG_DIR, f"dashboard_monitor_{DASHBOARD_PORT}.log")

# --- Dashboard Logging Setup ---
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

# --- Global Data Store ---
log_data_lock = threading.Lock()
# Inicializar com DataFrame vazio correspondendo ao formato do log alvo
log_data_df = pd.DataFrame(columns=[
    'timestamp', 'level', 'thread', 'module', 'message', 'raw_line'
])
log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp']) # Garantir dtype correto

last_file_processed: Optional[str] = None
last_file_mod_time: Optional[float] = None # Armazenar o tempo de modificação do arquivo processado
log_parsing_errors: int = 0
last_update_time: Optional[datetime.datetime] = None

# Regex para analisar o formato de log VALUATION
# Exemplo: 2025-04-02 00:05:12,975 - INFO - [MainThread] - <module> - --- Application Start ---
# Exemplo: 2025-04-02 00:05:56,561 - INFO - [waitress-0] - index - Request received for index route.
LOG_LINE_REGEX = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})" # Timestamp
    r"\s+-\s+"
    r"(?P<level>\w+)" # Log Level
    r"\s+-\s+"
    r"(?:\[(?P<thread>.*?)\])?" # Opcional Thread Name (non-greedy)
    r"\s+-\s+"
    # Opcional Module/Function Name (permite '<', '>', '.', '-')
    r"(?:<?(?P<module>[\w.-]+)>?)?"
    r"\s+-\s+"
    r"(?P<message>.*)$" # O resto é a mensagem
)

# --- Helper Functions ---

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
    latest_effective_ts: Optional[datetime.datetime] = None # Timestamp a ser usado para comparação
    dir_path = Path(directory)

    if not dir_path.is_dir():
        dashboard_logger.warning(f"Diretório de logs não encontrado: {directory}")
        return None

    try:
        log_files = list(dir_path.glob('*.log'))

        if not log_files:
            # Não é um aviso se está apenas começando, pode ser normal
            # dashboard_logger.info(f"Nenhum arquivo .log encontrado em {directory}")
            return None

        for file_path in log_files:
            if not file_path.is_file():
                continue

            current_effective_ts = None
            try:
                # Prioriza timestamp do nome do arquivo
                filename_ts = parse_filename_timestamp(file_path.name)
                if filename_ts:
                    current_effective_ts = filename_ts
                else:
                    # Fallback para tempo de modificação
                     mod_time_ts = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                     current_effective_ts = mod_time_ts

                # Compara com o melhor encontrado até agora
                if latest_effective_ts is None or (current_effective_ts and current_effective_ts > latest_effective_ts):
                    latest_effective_ts = current_effective_ts
                    latest_file_path = file_path

            except OSError as e:
                dashboard_logger.warning(f"Não foi possível obter atributos para {file_path.name}: {e}")
            except Exception as e:
                dashboard_logger.warning(f"Erro ao processar o arquivo {file_path.name}: {e}")


        if latest_file_path:
            dashboard_logger.debug(f"Arquivo de log mais recente identificado: {latest_file_path.name} (Timestamp Efetivo: {latest_effective_ts})")
        else:
             # Só loga se havia arquivos mas nenhum pôde ser processado
             if log_files:
                 dashboard_logger.warning(f"Não foi possível determinar o arquivo de log mais recente em {directory}. Verifique as permissões ou nomes.")

        return latest_file_path

    except Exception as e:
        dashboard_logger.error(f"Erro ao encontrar o arquivo de log mais recente em '{directory}': {e}", exc_info=True)
        return None

def parse_log_file(file_path: Path) -> pd.DataFrame:
    """Analisa o arquivo de log inteiro em um DataFrame Pandas."""
    global log_parsing_errors
    parsed_lines_data = []
    errors_in_parse_session = 0
    lines_read = 0
    start_time = time.monotonic()

    try:
        # Use utf-8, ignore erros para resiliência
        with file_path.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                lines_read += 1
                line = line.strip()
                if not line:
                    continue

                match = LOG_LINE_REGEX.match(line)
                if match:
                    log_entry = match.groupdict()
                    # Limpa campos opcionais que podem ser None do regex
                    log_entry['thread'] = log_entry.get('thread', '') or ''
                    log_entry['module'] = log_entry.get('module', '') or ''
                    log_entry['raw_line'] = line # Armazena linha bruta

                    try:
                        # Usa pandas to_datetime para análise robusta
                        log_entry['timestamp'] = pd.to_datetime(log_entry['timestamp'], format='%Y-%m-%d %H:%M:%S,%f', errors='coerce')
                        if pd.isna(log_entry['timestamp']):
                            # Não incrementa erro aqui, apenas avisa, será tratado depois
                            dashboard_logger.debug(f"File '{file_path.name}', Line ~{i+1}: Timestamp resultou em NaT - Line: '{line[:100]}...'")
                        parsed_lines_data.append(log_entry)
                    except ValueError as ve: # Captura erros específicos de conversão de data/hora se to_datetime falhar de outra forma
                        dashboard_logger.warning(f"File '{file_path.name}', Line ~{i+1}: Erro de análise de timestamp inesperado: {ve} - Line: '{line[:100]}...'")
                        errors_in_parse_session += 1
                else:
                    # Loga linhas que não correspondem se necessário para depuração
                    dashboard_logger.debug(f"Linha {i+1} não correspondeu ao regex em {file_path.name}: {line[:100]}...")
                    errors_in_parse_session += 1

        duration = time.monotonic() - start_time
        dashboard_logger.debug(f"Analisou {len(parsed_lines_data)} linhas de {lines_read} em {file_path.name}. {errors_in_parse_session} problemas de análise. Duração: {duration:.3f}s")

    except FileNotFoundError:
        dashboard_logger.error(f"Arquivo de log não encontrado durante a análise: {file_path}")
        return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line']) # Retorna vazio
    except PermissionError:
         dashboard_logger.error(f"Permissão negada ao ler o arquivo de log: {file_path}")
         return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
    except Exception as e:
        dashboard_logger.error(f"Erro ao ler/analisar o arquivo de log '{file_path}': {e}", exc_info=True)
        return pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line']) # Retorna vazio

    # Atualiza contagem global de erros (thread-safe se necessário, mas apenas atualizado aqui)
    # Nota: Isso é CUMULATIVO entre arquivos e sessões. Resetar se desejado.
    log_parsing_errors += errors_in_parse_session # Adiciona erros desta sessão

    if not parsed_lines_data:
        # Retorna DF vazio, mas com colunas corretas
        df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Garante tipo correto
        return df

    # Cria DataFrame
    df = pd.DataFrame(parsed_lines_data)

    # <<< IMPORTANTE: Tratar timestamps inválidos (NaT) antes de prosseguir >>>
    initial_rows = len(df)
    df.dropna(subset=['timestamp'], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        dashboard_logger.warning(f"Removidas {dropped_rows} linhas com timestamp inválido (NaT) de '{file_path.name}'.")
        log_parsing_errors += dropped_rows # Conta como erros de parse? Sim.

    # Garante dtypes corretos novamente após a criação e limpeza
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['level', 'thread', 'module', 'message', 'raw_line']:
         if col in df.columns:
              # Converte para string, tratando nulos explicitamente se necessário
              df[col] = df[col].astype(str).fillna('')

    return df


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
                    # Obter o tempo de modificação atual do arquivo mais recente
                    current_mod_time = latest_file.stat().st_mtime
                except Exception as e:
                    dashboard_logger.warning(f"Não foi possível obter o tempo de modificação para {latest_file.name}: {e}. Assumindo necessidade de processamento.")
                    needs_processing = True # Processa se não conseguir verificar

                # Determinar se precisamos processar o arquivo
                if last_file_processed is None:
                    needs_processing = True # Primeira execução
                    dashboard_logger.info(f"Primeira execução, processando: {latest_file.name}")
                elif latest_file.name != last_file_processed:
                    needs_processing = True # Arquivo mais recente mudou
                    dashboard_logger.info(f"Novo arquivo de log detectado: {latest_file.name}")
                elif current_mod_time is not None and (last_file_mod_time is None or current_mod_time > last_file_mod_time):
                    needs_processing = True # Mesmo arquivo, mas foi modificado
                    dashboard_logger.debug(f"Arquivo {latest_file.name} modificado desde a última leitura.")
                # else: O mesmo arquivo e não modificado (ou não conseguimos verificar), não faz nada

                if needs_processing:
                    dashboard_logger.info(f"Processando arquivo de log: {latest_file.name}")
                    # Resetar contagem de erros por sessão de arquivo? Mantendo cumulativo por enquanto.
                    # log_parsing_errors = 0 # Descomente se quiser resetar por arquivo

                    new_df = parse_log_file(latest_file)
                    parse_session_errors = log_parsing_errors # Obter contagem de erros atual

                    # --- Atualizar Dados Globais (Bloqueado) ---
                    with log_data_lock:
                        if not new_df.empty:
                            log_data_df = new_df
                            last_file_processed = latest_file.name
                            last_file_mod_time = current_mod_time # Armazena o tempo de modificação que acabamos de ler
                            last_update_time = current_time

                            # Logar resumo após atualização
                            level_counts = log_data_df['level'].value_counts()
                            total_lines = len(log_data_df)
                            errors = level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)
                            warnings = level_counts.get('WARNING', 0)
                            dashboard_logger.info(f"Dados atualizados de '{latest_file.name}'. Linhas totais: {total_lines}, Erros/Críticos: {errors}, Avisos: {warnings}. Erros de Parse Totais: {parse_session_errors}")
                        else:
                            # O parse retornou vazio, mas o arquivo existe.
                            # Poderia ser um arquivo vazio ou falha de parse total.
                            # Mantém os dados antigos ou limpa? Limpar parece mais seguro se o parse falhou completamente.
                            # No entanto, se o arquivo estiver *legitimamente* vazio, queremos mostrar vazio.
                            # Por segurança, se o parse falhou totalmente (new_df vazio), limpamos.
                            # Se o parse foi ok mas o arquivo não tinha linhas válidas, o DF já estará vazio.
                            dashboard_logger.warning(f"Análise de '{latest_file.name}' resultou em DataFrame vazio. Verifique o arquivo ou erros de parse.")
                            # Limpa o DF se o arquivo foi processado mas resultou vazio
                            log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                            log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                            last_file_processed = latest_file.name # Registra que tentamos este arquivo
                            last_file_mod_time = current_mod_time
                            last_update_time = current_time
                    # --- Fim do Lock ---

            else:
                # Nenhum arquivo de log encontrado
                if last_file_processed is not None: # Só loga se tínhamos um arquivo antes
                    dashboard_logger.warning(f"Nenhum arquivo de log encontrado em '{MONITORED_LOG_DIR}'. Limpando dados anteriores.")
                    with log_data_lock:
                        # Limpa o DataFrame
                        log_data_df = pd.DataFrame(columns=['timestamp', 'level', 'thread', 'module', 'message', 'raw_line'])
                        log_data_df['timestamp'] = pd.to_datetime(log_data_df['timestamp'])
                        last_file_processed = None
                        last_file_mod_time = None
                        last_update_time = current_time
                        # Opcionalmente resetar erros de parse?
                        # log_parsing_errors = 0

        except Exception as e:
            dashboard_logger.error(f"Erro crítico no loop de atualização em background: {e}", exc_info=True)
            # Espera um pouco mais em caso de erro crítico para não spammar logs
            time.sleep(FILE_SCAN_INTERVAL_SECONDS * 5)

        # Espera antes da próxima verificação
        time.sleep(FILE_SCAN_INTERVAL_SECONDS)


# --- Flask App ---
app = Flask(__name__)
CORS(app) # Permitir todas as origens por padrão

# --- HTML Template (Adaptado) ---
html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log Monitor Dashboard ({MONITORED_LOG_DIR})</title>
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
            -webkit-backdrop-filter: blur(12px) saturate(150%); /* Safari */
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
             gap: 0.6rem; /* Increased gap slightly */
             align-items: baseline; /* Align items better */
             /* <<< ALTERADO: Use transition para suavidade >>> */
             opacity: 0;
             transition: opacity 0.4s ease-out, background-color 0.2s ease;
        }
         .log-entry.visible {
             opacity: 1;
         }
         .log-entry:hover {
             background-color: rgba(26, 32, 44, 0.7);
         }
         /* <<< FIM DA ALTERAÇÃO >>> */
         .log-time { color: #718096; flex-shrink: 0; width: 70px; } /* gray-600 */
         .log-level { font-weight: bold; width: 80px; flex-shrink: 0; text-align: right;}
         .log-thread { color: #a0aec0; width: 100px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } /* gray-500 */
         .log-module { color: #9f7aea; width: 120px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } /* purple-500 - Renamed from func */
         .log-message { color: #e2e8f0; flex-grow: 1; word-break: break-word; white-space: pre-wrap;} /* Allow message wrap */


        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(74, 85, 104, 0.3); border-radius: 3px;} /* gray-600ish transparent */
        ::-webkit-scrollbar-thumb { background: rgba(160, 174, 192, 0.5); border-radius: 3px; } /* gray-500ish */
        ::-webkit-scrollbar-thumb:hover { background: rgba(160, 174, 192, 0.7); }

        /* Fade-in animation para seções */
         @keyframes fadeInSection { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in-section { animation: fadeInSection 0.5s ease-out forwards; }
    </style>
</head>
<body class="p-4 md:p-8 min-h-screen">

    <!-- Header -->
    <header class="mb-8 flex flex-col sm:flex-row justify-between items-center gap-2">
        <h1 class="text-2xl md:text-3xl font-orbitron font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
           <i class="fas fa-wave-square mr-2"></i>{MONITORED_LOG_DIR} Monitor
        </h1>
        <div class="text-xs text-gray-400 text-center sm:text-right">
             <p>Monitorando: <code id="monitoringFile" class="text-gray-300 bg-gray-700 px-1 rounded">N/A</code></p>
             <p>Última Atualização: <span id="lastUpdateTime" class="text-gray-300">Nunca</span></p>
             <p><i class="fas fa-bug mr-1 text-red-500"></i> Erros de Parse Totais: <span id="parseErrorCount" class="text-red-400 font-semibold">0</span></p>
         </div>
    </header>

    <!-- Stats Cards -->
    <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 mb-8">
        <!-- Total Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.1s;">
            <div class="text-3xl text-blue-400"><i class="fas fa-file-alt fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-sm uppercase tracking-wider">Total Logs ({DATA_RETENTION_MINUTES}min)</div>
                <div id="totalLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
        <!-- Info Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.2s;">
             <div class="text-3xl text-blue-400"><i class="fas fa-info-circle fa-fw"></i></div>
             <div>
                 <div class="text-gray-400 text-sm uppercase tracking-wider">Info</div>
                 <div id="infoLogs" class="text-2xl font-bold text-gray-100">0</div>
             </div>
         </div>
        <!-- Warning Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.3s;">
            <div class="text-3xl text-yellow-400"><i class="fas fa-exclamation-triangle fa-fw"></i></div>
            <div>
                <div class="text-gray-400 text-sm uppercase tracking-wider">Warnings</div>
                <div id="warningLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
        <!-- Error Logs Card -->
        <div class="glass-card p-5 flex items-center space-x-4 fade-in-section" style="animation-delay: 0.4s;">
            <div class="text-3xl text-red-400"><i class="fas fa-times-circle fa-fw"></i></div>
            <div>
                <!-- <<< ALTERADO: Título mais claro >>> -->
                <div class="text-gray-400 text-sm uppercase tracking-wider">Errors & Critical</div>
                <!-- <<< FIM DA ALTERAÇÃO >>> -->
                <div id="errorLogs" class="text-2xl font-bold text-gray-100">0</div>
            </div>
        </div>
    </section>

    <!-- Chart and Recent Logs -->
    <section class="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8">
        <!-- Chart Column -->
        <div class="lg:col-span-2">
             <div class="glass-card p-4 h-full fade-in-section" style="animation-delay: 0.5s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300"><i class="fas fa-chart-bar mr-2 text-cyan-400"></i>Atividade de Log (Últimos {DATA_RETENTION_MINUTES} Minutos)</h2>
                <div class="relative h-64 md:h-80">
                   <canvas id="logChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Recent Logs Column -->
        <div class="lg:col-span-1">
             <div class="glass-card p-4 h-full flex flex-col fade-in-section" style="animation-delay: 0.6s;">
                <h2 class="text-lg font-semibold mb-3 text-gray-300"><i class="fas fa-stream mr-2 text-cyan-400"></i>Entradas Recentes (Max {MAX_RECENT_LOGS_DISPLAY})</h2>
                <div id="logContainer" class="flex-grow overflow-y-auto pr-2 space-y-1" style="max-height: 400px;">
                    <!-- Log entries will be injected here -->
                     <p id="noLogsMessage" class="text-center text-gray-500 italic mt-4">Aguardando dados de log...</p>
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
        const UPDATE_INTERVAL = {{ FRONTEND_UPDATE_INTERVAL_MS }}; // Obtido da config Flask
        let logChart = null; // Instância do Chart

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

        // --- Configuração do Chart ---
        function initializeChart() {
            const ctx = document.getElementById('logChart').getContext('2d');
            if (logChart) { logChart.destroy(); } // Destroi anterior se existir
            logChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: [], // Timestamps (e.g., HH:MM)
                    datasets: [
                        {
                            label: 'Info', data: [],
                            backgroundColor: 'rgba(99, 179, 237, 0.6)', borderColor: 'rgba(99, 179, 237, 1)', borderWidth: 1
                        },
                        {
                            label: 'Warning', data: [],
                             backgroundColor: 'rgba(246, 224, 94, 0.6)', borderColor: 'rgba(246, 224, 94, 1)', borderWidth: 1
                         },
                         {
                             label: 'Error/Critical', data: [], // Combinado Errors e Critical
                             backgroundColor: 'rgba(252, 129, 129, 0.6)', borderColor: 'rgba(252, 129, 129, 1)', borderWidth: 1
                        }
                     ]
                 },
                 options: {
                     responsive: true, maintainAspectRatio: false,
                     scales: {
                         x: { stacked: true, ticks: { color: '#a0aec0' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                         y: { stacked: true, beginAtZero: true, ticks: { color: '#a0aec0', precision: 0 }, grid: { color: 'rgba(255, 255, 255, 0.1)' } } // Use precision 0 for integer ticks
                     },
                     plugins: {
                         legend: { position: 'bottom', labels: { color: '#e2e8f0' } },
                         tooltip: {
                             mode: 'index', intersect: false, backgroundColor: 'rgba(26, 32, 44, 0.9)', titleColor: '#e2e8f0', bodyColor: '#e2e8f0',
                         }
                     },
                     // <<< ALTERADO: Animação suave mas rápida >>>
                     animation: { duration: 300, easing: 'easeOutQuad' } // Animação mais rápida e suave
                     // <<< FIM DA ALTERAÇÃO >>>
                 }
             });
        }

        // --- Formatação Nível Log ---
        function getLevelClass(level) { return `level-${level.toUpperCase()}`; }
        function getIconClass(level) { return `icon-${level.toUpperCase()}`; }
        function getLevelIcon(level) {
             level = level.toUpperCase();
             switch (level) {
                 case 'DEBUG': return 'fas fa-bug';
                 case 'INFO': return 'fas fa-info-circle';
                 case 'WARNING': return 'fas fa-exclamation-triangle';
                 case 'ERROR': return 'fas fa-times-circle';
                 case 'CRITICAL': return 'fas fa-skull-crossbones'; // Icone mais forte para critical
                 default: return 'fas fa-question-circle';
             }
         }

         // --- Atualização de Dados ---
         async function fetchDataAndUpdate() {
             console.debug("Buscando dados...");
             try {
                 const response = await fetch(API_ENDPOINT);
                 if (!response.ok) {
                     console.error("Falha ao buscar dados:", response.status, response.statusText);
                     noLogsMessageEl.textContent = `Erro ${response.status} ao buscar dados.`;
                     noLogsMessageEl.style.display = 'block'; noLogsMessageEl.style.color = '#fc8181';
                    return;
                 }
                 const data = await response.json();
                 // console.debug("Dados recebidos:", data); // Descomentar para depuração

                 // Atualiza Cards de Estatísticas
                 totalLogsEl.textContent = data.stats.total_filtered || 0;
                 infoLogsEl.textContent = data.stats.info_filtered || 0;
                 warningLogsEl.textContent = data.stats.warning_filtered || 0;
                 errorLogsEl.textContent = data.stats.error_filtered + data.stats.critical_filtered || 0; // Combina error/critical

                 // Atualiza Informações do Header
                monitoringFileEl.textContent = data.status.last_file || 'N/A';
                parseErrorCountEl.textContent = data.status.parse_errors || 0;
                lastUpdateTimeEl.textContent = data.status.last_update ? new Date(data.status.last_update).toLocaleString() : 'Nunca';

                 // Atualiza Logs Recentes
                 logContainerEl.innerHTML = ''; // Limpa logs anteriores
                 if (data.recent_logs && data.recent_logs.length > 0) {
                     noLogsMessageEl.style.display = 'none';
                     const fragment = document.createDocumentFragment(); // Usa fragmento para performance
                     data.recent_logs.forEach((log, index) => {
                         const logDiv = document.createElement('div');
                         // <<< ALTERADO: Apenas adiciona a classe base, a visibilidade é via CSS/JS >>>
                         logDiv.className = 'log-entry';
                         // <<< FIM DA ALTERAÇÃO >>>

                         const timeStr = new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                         const level = log.level || 'UNKNOWN';
                         const thread = log.thread || '';
                         const module = log.module || '';
                         const message = log.message || '';

                         logDiv.innerHTML = `
                             <span class="log-time">${timeStr}</span>
                             <span class="log-level ${getLevelClass(level)}">
                                 <i class="${getLevelIcon(level)} ${getIconClass(level)} mr-1" title="${level}"></i>${level}
                             </span>
                             <span class="log-thread" title="${thread}">${thread}</span>
                             <span class="log-module" title="${module}">${module}</span>
                             <span class="log-message" title="${escapeHtml(log.raw_line || '')}">${escapeHtml(message)}</span>
                        `;
                         fragment.appendChild(logDiv);
                     });
                     logContainerEl.appendChild(fragment); // Adiciona fragmento de uma vez

                     // <<< ALTERADO: Aciona a transição CSS adicionando a classe 'visible' com um pequeno atraso >>>
                     // Isso permite que o navegador renderize os elementos antes de iniciar a transição
                     setTimeout(() => {
                         logContainerEl.querySelectorAll('.log-entry').forEach(el => {
                             el.classList.add('visible');
                         });
                     }, 10); // Pequeno atraso

                 } else {
                     noLogsMessageEl.textContent = 'Nenhuma entrada de log recente encontrada na janela de tempo.';
                     noLogsMessageEl.style.display = 'block';
                 }

                // Atualiza Chart
                if (logChart && data.chart_data) {
                    logChart.data.labels = data.chart_data.labels;
                    logChart.data.datasets[0].data = data.chart_data.info;
                    logChart.data.datasets[1].data = data.chart_data.warning;
                    logChart.data.datasets[2].data = data.chart_data.error_critical; // Combinado
                    // <<< ALTERADO: Usar 'none' para atualização mais suave sem resetar animação >>>
                    logChart.update('none');
                    // <<< FIM DA ALTERAÇÃO >>>
                } else {
                     console.warn("Objeto Chart ou chart_data não disponível para atualização.");
                 }

            } catch (error) {
                console.error("Erro ao buscar ou processar dados:", error);
                 noLogsMessageEl.textContent = 'Erro ao carregar dados do dashboard. Verifique o console.';
                 noLogsMessageEl.style.display = 'block'; noLogsMessageEl.style.color = '#fc8181';
            }
        }

        // <<< CORRIGIDO: Função escapeHtml >>>
         function escapeHtml(unsafe) {
             if (typeof unsafe !== 'string') return '';
             return unsafe
                  .replace(/&/g, "&")
                  .replace(/</g, "<")
                  .replace(/>/g, ">")
                  .replace(/'/g, "'");
          }
        // <<< FIM DA CORREÇÃO >>>

        // --- Inicialização ---
        document.addEventListener('DOMContentLoaded', () => {
            initializeChart();
            fetchDataAndUpdate(); // Busca inicial
            setInterval(fetchDataAndUpdate, UPDATE_INTERVAL); // Busca periódica
            if (currentYearEl) currentYearEl.textContent = new Date().getFullYear();
        });

    </script>

</body>
</html>
"""

# --- API Endpoint ---
@app.route('/api/data')
def get_api_data():
    """Fornece dados de log filtrados, estatísticas e informações do gráfico para o frontend."""
    dashboard_logger.debug("Requisição API /api/data recebida")
    with log_data_lock:
        # Faz uma cópia para trabalhar fora do lock
        current_df = log_data_df.copy()
        file_processed = last_file_processed
        update_time = last_update_time
        parse_errors = log_parsing_errors

    # Filtra dados para os últimos N minutos para exibição/gráfico
    # Usar tempo naive para comparação simples, assumindo logs e servidor no mesmo fuso
    # Para robustez em fusos diferentes, converter tudo para UTC seria melhor
    now_naive = datetime.datetime.now()
    cutoff_time_naive = now_naive - datetime.timedelta(minutes=DATA_RETENTION_MINUTES)

    # Verificação defensiva se a coluna 'timestamp' existe e tem dados
    if 'timestamp' not in current_df.columns or current_df.empty:
         filtered_df = pd.DataFrame(columns=current_df.columns) # DF vazio com mesmas colunas
         if 'timestamp' not in current_df.columns and not current_df.empty:
              dashboard_logger.warning("DataFrame de logs não possui a coluna 'timestamp'.")
         # Se estiver vazio, não é necessariamente um aviso, pode ser normal no início.
    else:
         try:
             # Garante que timestamp é datetime e remove NaTs antes de filtrar
             # A conversão e remoção de NaT já deve ter sido feita no parse_log_file
             # Mas uma verificação extra não faz mal.
             if not pd.api.types.is_datetime64_any_dtype(current_df['timestamp']):
                 dashboard_logger.warning("Coluna 'timestamp' não é do tipo datetime. Tentando converter novamente.")
                 current_df['timestamp'] = pd.to_datetime(current_df['timestamp'], errors='coerce')
                 current_df.dropna(subset=['timestamp'], inplace=True)

             # Filtra baseado no tempo naive
             # Certifique-se de que cutoff_time_naive é comparável com current_df['timestamp']
             # Se current_df['timestamp'] for timezone-aware (e.g., UTC), ajuste cutoff_time_naive
             filtered_df = current_df[current_df['timestamp'] >= cutoff_time_naive].copy()
             dashboard_logger.debug(f"Filtragem completa. {len(filtered_df)} linhas nos últimos {DATA_RETENTION_MINUTES} minutos.")
         except Exception as e:
             dashboard_logger.error(f"Erro durante a filtragem de dados: {e}", exc_info=True)
             filtered_df = pd.DataFrame(columns=current_df.columns) # Retorna vazio em erro

    # --- Prepara Estatísticas (dos dados filtrados) ---
    level_counts_filtered = filtered_df['level'].value_counts() if not filtered_df.empty else pd.Series(dtype=int)
    stats = {
        'total_filtered': len(filtered_df),
        'info_filtered': int(level_counts_filtered.get('INFO', 0)),
        'warning_filtered': int(level_counts_filtered.get('WARNING', 0)),
        'error_filtered': int(level_counts_filtered.get('ERROR', 0)),
        'critical_filtered': int(level_counts_filtered.get('CRITICAL', 0)),
        # Adicionar outros níveis se necessário (e.g., DEBUG)
        'debug_filtered': int(level_counts_filtered.get('DEBUG', 0)),
    }

    # --- Prepara Logs Recentes (limita para exibição, dos dados filtrados) ---
    recent_logs_list = []
    if not filtered_df.empty:
        # Ordena por timestamp descendente para obter os mais recentes primeiro
        recent_logs_display_df = filtered_df.sort_values(by='timestamp', ascending=False).head(MAX_RECENT_LOGS_DISPLAY)
        # Converte timestamp de volta para formato ISO string para JSON
        # O JS Date() consegue analisar este formato.
        recent_logs_display_df['timestamp_iso'] = recent_logs_display_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3]
        # Seleciona colunas relevantes e converte para dict
        # Inclui 'raw_line' que é usado no tooltip da mensagem no JS
        recent_logs_list = recent_logs_display_df[['timestamp_iso', 'level', 'thread', 'module', 'message', 'raw_line']].rename(columns={'timestamp_iso': 'timestamp'}).to_dict(orient='records')
        dashboard_logger.debug(f"Preparadas {len(recent_logs_list)} entradas de log recentes para exibição.")


    # --- Prepara Dados do Gráfico (Agrupa por minuto, dos dados filtrados) ---
    chart_data = {'labels': [], 'info': [], 'warning': [], 'error_critical': []}
    if not filtered_df.empty and 'timestamp' in filtered_df.columns:
        try:
            # Define timestamp como índice para reamostragem (faz cópia implícita se necessário)
            chart_df = filtered_df.set_index('timestamp')

            # Reamostra por minuto ('min' ou 'T') e conta níveis. Preenche NaNs com 0.
            # Usar 'min' é mais explícito que 'T'
            level_counts_per_minute = chart_df.groupby(pd.Grouper(freq='min'))['level'].value_counts().unstack(fill_value=0)

            # Garante que todas as colunas de nível chave existam, adiciona se faltar
            for level in ['INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                if level not in level_counts_per_minute.columns:
                    level_counts_per_minute[level] = 0

            # Prepara dados para Chart.js
            chart_data['labels'] = level_counts_per_minute.index.strftime('%H:%M').tolist()
            chart_data['info'] = level_counts_per_minute['INFO'].tolist()
            chart_data['warning'] = level_counts_per_minute['WARNING'].tolist()
            # Combina Error e Critical para o gráfico
            chart_data['error_critical'] = (level_counts_per_minute['ERROR'] + level_counts_per_minute['CRITICAL']).tolist()
            dashboard_logger.debug(f"Preparados dados do gráfico com {len(chart_data['labels'])} pontos de tempo.")

        except Exception as e:
             dashboard_logger.error(f"Erro ao preparar dados do gráfico: {e}", exc_info=True)
             # chart_data permanecerá vazio

    # --- Informações de Status ---
    status = {
         'last_file': file_processed,
         'last_update': update_time.isoformat() if update_time else None,
         'parse_errors': parse_errors, # Erros de parse cumulativos
         'monitoring_dir': MONITORED_LOG_DIR,
         'retention_minutes': DATA_RETENTION_MINUTES
     }

    # --- Combina e Retorna JSON ---
    response_data = {
        'status': status,
        'stats': stats,
        'recent_logs': recent_logs_list,
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
        dashboard_logger.warning(f"Diretório monitorado '{MONITORED_LOG_DIR}' não existe. Criando...")
        try:
            monitored_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             dashboard_logger.error(f"Falha ao criar diretório monitorado '{MONITORED_LOG_DIR}': {e}")
             # Continuar pode ser ok se a aplicação que gera os logs o criar depois.

    # Inicia a thread de background
    dashboard_logger.info("Iniciando thread de atualização de log em background...")
    update_thread = threading.Thread(target=update_log_data_periodically, name="LogUpdateThread", daemon=True)
    update_thread.start()

    dashboard_logger.info(f"Iniciando servidor Waitress em http://0.0.0.0:{DASHBOARD_PORT}")
    print(f"\n🚀 Dashboard de Logs Disponível em: http://127.0.0.1:{DASHBOARD_PORT}")
    print(f"   Monitorando logs no diretório: '{MONITORED_LOG_DIR}'")
    print(f"   Logs do dashboard são escritos em: '{DASHBOARD_LOG_FILE}'")
    print(f"   Pressione CTRL+C para parar.")

    # Usa o servidor Waitress
    # O comentário em português foi removido da chamada serve()
    serve(app, host='0.0.0.0', port=DASHBOARD_PORT, threads=4)