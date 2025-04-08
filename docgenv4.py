# -*- coding: utf-8 -*-
import os
import platform
import sqlite3
import json
import yaml
import time
from datetime import datetime
import google.generativeai as genai
from colorama import Fore, Style, init
import ast
import re # Importa o módulo de expressões regulares
from typing import List, Dict, Any, Optional, Tuple, Union # Para type hints

# Inicializa o Colorama para colorir os logs
init(autoreset=True)

# --- ATENÇÃO: Chave de API ---
# NUNCA coloque chaves de API diretamente no código em produção.
# Use variáveis de ambiente ou um sistema de gerenciamento de segredos.
API_KEY = os.getenv("GEMINI_API_KEY") # Tenta pegar da env
if not API_KEY:
    # Use uma chave placeholder SOMENTE para desenvolvimento local e NUNCA com dados sensíveis.
    # Substitua 'SUA_CHAVE_PLACEHOLDER_AQUI' por uma chave real ou remova esta linha em produção.
    API_KEY = 'AIzaSyC7dAwSyLKaVO2E-PA6UaacLZ4aLGtrXbY' # CHAVE PLACEHOLDER - RISCO!
    print(f"{Fore.RED}{'#'*66}")
    print(f"### {Fore.YELLOW}ALERTA:{Style.RESET_ALL}{Fore.RED} USANDO CHAVE DE API PADRÃO/PLACEHOLDER!                 ###")
    print(f"### Defina a variável de ambiente 'GEMINI_API_KEY' com sua chave ###")
    print(f"### real para segurança e funcionamento adequado em produção.    ###")
    print(f"### {Style.BRIGHT}NÃO USE ESTA CHAVE EM AMBIENTES PÚBLICOS OU COM DADOS REAIS.{Style.NORMAL} ###")
    print(f"{'#'*66}{Style.RESET_ALL}")
    # Considerar adicionar exit(1) aqui se a chave for absolutamente essencial e não houver fallback funcional.
    # Ex: descomente a linha abaixo se o script NÃO DEVE rodar sem uma chave real:
    # exit(1)

try:
    genai.configure(api_key=API_KEY)
    print(f"{Fore.GREEN}🔑 Configuração da API do Gemini bem-sucedida.")
except Exception as e:
    print(f"{Fore.RED}ERRO FATAL: Falha ao configurar a API do Gemini.")
    print(f"{Fore.RED}   Erro: {e}")
    print(f"{Fore.RED}   Verifique se a chave de API ('{API_KEY[:4]}...{API_KEY[-4:] if API_KEY else ''}') é válida e se há conectividade.")
    exit(1) # Para a execução se não puder configurar a API

# Modelo da IA - Verifique a disponibilidade e adequação do modelo
# Modelos comuns: gemini-1.5-flash-latest, gemini-1.5-pro-latest, gemini-1.0-pro
# NOME_MODELO = "gemini-1.5-pro-latest" # Usar o modelo mais capaz se necessário
NOME_MODELO = "gemini-2.0-flash" # Modelo mais rápido e geralmente suficiente
print(f"{Fore.BLUE}ℹ️ Usando modelo de IA: {NOME_MODELO}")

# Arquivos e diretórios a serem ignorados
# Adicione padrões conforme necessário
IGNORE_LIST = [
    os.path.basename(__file__), # Ignora o próprio script dinamicamente
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".md", # Ignora arquivos Markdown existentes
    ".html", # Ignora arquivos HTML existentes (incluindo o gerado)
    ".yaml", # Ignora arquivos YAML de estrutura gerados anteriormente
    "requirements.txt",
    "LICENSE",
    "build", # Diretório comum de build
    "dist",  # Diretório comum de distribuição
    "*.log", # Arquivos de log (padrão de exemplo)
    ".idea", # Configurações do IntelliJ/Android Studio
    ".vscode",# Configurações do VS Code
    "target", # Diretório comum de build Java/Rust
    "docs",   # Diretório comum de documentação
    "tests",  # Diretório comum de testes (pode querer analisar)
    "test",   # Diretório comum de testes (pode querer analisar)
    ".pytest_cache",
    ".mypy_cache",
    "site",   # Diretório comum do MkDocs/Sphinx
]

# --- Funções de Análise de Arquivos ---

def get_file_size(file_path: str) -> int:
    """Retorna o tamanho do arquivo em bytes ou -1 em caso de erro."""
    try:
        return os.path.getsize(file_path)
    except OSError as e:
        # print(f"{Fore.YELLOW}Aviso: Não foi possível obter o tamanho de '{file_path}': {e}{Style.RESET_ALL}")
        return -1

def count_lines(file_path: str) -> int:
    """Conta as linhas de um arquivo, tentando várias codificações.
    Retorna:
        >= 0: número de linhas
        -1: Erro de IO/Permissão
        -2: Erro de decodificação com codificações comuns
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return sum(1 for _ in f)
        except UnicodeDecodeError:
            continue
        except (OSError, IOError) as e:
            # print(f"{Fore.YELLOW}Aviso: Erro ao abrir/ler '{file_path}' para contagem de linhas: {e}{Style.RESET_ALL}")
            return -1
        except Exception as e: # Captura genérica para outros erros inesperados
            # print(f"{Fore.YELLOW}Aviso: Erro inesperado ao contar linhas de '{file_path}': {e}{Style.RESET_ALL}")
            return -1
    # print(f"{Fore.YELLOW}Aviso: Não foi possível decodificar '{file_path}' para contar linhas com {encodings_to_try}.{Style.RESET_ALL}")
    return -2

def read_file_content(file_path: str, max_lines: int = 50) -> Optional[str]:
    """Lê as primeiras `max_lines` de um arquivo, tentando várias codificações."""
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    content = None
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                lines = [next(f) for _ in range(max_lines)]
                content = "".join(lines)
            return content # Retorna assim que conseguir ler
        except UnicodeDecodeError:
            continue
        except StopIteration: # Arquivo tem menos que max_lines
            try:
                 with open(file_path, 'r', encoding=enc) as f:
                     content = f.read()
                 return content # Retorna o conteúdo completo
            except Exception: # Erro na segunda tentativa de leitura completa
                continue
        except (OSError, IOError):
             # Não loga erro aqui, pois é comum tentar e falhar com encodings
             return None # Erro de IO impede leitura
        except Exception:
             # Log genérico pode ser útil aqui se necessário
             return None # Outro erro impede leitura

    # print(f"{Fore.YELLOW}Aviso: Não foi possível ler o conteúdo de '{file_path}' com codificações comuns.{Style.RESET_ALL}")
    return None # Retorna None se todas as tentativas falharem

def get_python_info(file_path: str) -> Dict[str, Any]:
    """Analisa um arquivo Python usando AST para extrair funções, classes e imports."""
    source_code: Optional[str] = None
    read_error: Optional[str] = None
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                source_code = f.read()
            break # Sucesso na leitura
        except UnicodeDecodeError:
            continue
        except (OSError, IOError) as e:
            read_error = f"Erro de IO ao ler o arquivo com {enc}: {e}"
            break # Erro de IO provavelmente não será resolvido com outro encoding
        except Exception as e:
            read_error = f"Erro inesperado ao ler o arquivo com {enc}: {e}"
            # Continuar tentando outros encodings pode ser útil em casos raros

    if source_code is None:
        err_msg = read_error if read_error else f"Não foi possível decodificar o arquivo '{os.path.basename(file_path)}' com as codificações testadas."
        return {"error": err_msg, "analysis_status": "failed_read"}

    try:
        tree = ast.parse(source_code)
        functions: List[Dict[str, Any]] = []
        classes: List[Dict[str, Any]] = []
        imports: List[Dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Usa getattr para end_lineno que pode não estar presente em todas as versões/nós
                functions.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "", # Garante string vazia se não houver docstring
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, 'end_lineno', None)
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, 'end_lineno', None)
                })
            elif isinstance(node, ast.Import):
                imports.extend([
                    {"type": "import", "name": alias.name, "asname": alias.asname}
                    for alias in node.names
                ])
            elif isinstance(node, ast.ImportFrom):
                 module_name = node.module if node.module else '.relative.' # Indica import relativo se module for None
                 imports.append({
                     "type": "from",
                     "module": module_name,
                     "level": node.level, # Nível do import relativo (0 para absoluto)
                     "names": [(alias.name, alias.asname) for alias in node.names]
                 })
        return {"functions": functions, "classes": classes, "imports": imports, "analysis_status": "success"}
    except SyntaxError as e:
        return {"error": f"Erro de sintaxe Python: {e}", "lineno": e.lineno, "col_offset": e.offset, "analysis_status": "failed_syntax"}
    except Exception as e:
        return {"error": f"Erro inesperado ao analisar AST: {e}", "analysis_status": "failed_ast"}

def get_sqlite_info(file_path: str, max_rows: int = 5) -> Dict[str, Any]:
    """Extrai esquema e amostra de dados de um banco de dados SQLite."""
    # Usar mode=ro (read-only) é mais seguro
    db_uri = f'file:{file_path}?mode=ro'
    tables_data: Dict[str, Any] = {}
    try:
        # Timeout pode ser útil para bancos de dados bloqueados
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        cursor = conn.cursor()

        # Listar tabelas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';") # Ignora tabelas do sistema
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            conn.close()
            return {"info": "Banco de dados vazio ou sem tabelas de usuário.", "tables": {}}

        for table_name in tables:
            try:
                # Obter esquema da tabela
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns = [{"name": row[1], "type": row[2], "notnull": bool(row[3]), "default": row[4], "pk": bool(row[5])}
                           for row in cursor.fetchall()]

                # Contar linhas (pode ser lento em tabelas enormes)
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
                    row_count = cursor.fetchone()[0]
                except sqlite3.OperationalError as count_e:
                    # Pode falhar se a tabela for corrupta ou virtual sem implementação
                    print(f"{Fore.YELLOW}Aviso: Não foi possível contar linhas da tabela SQLite '{table_name}': {count_e}{Style.RESET_ALL}")
                    row_count = "Erro ao Contar"

                # Obter amostra de linhas
                sample_rows_safe = []
                if row_count == 0 or row_count == "Erro ao Contar":
                     sample_rows = []
                else:
                    try:
                        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT {max_rows};")
                        sample_rows = cursor.fetchall()
                        # Tratar BLOBs para evitar problemas na serialização (JSON/YAML)
                        sample_rows_safe = [
                            tuple(f"<bytes {len(item)}>" if isinstance(item, bytes) else item for item in row)
                            for row in sample_rows
                        ]
                    except sqlite3.OperationalError as select_e:
                        print(f"{Fore.YELLOW}Aviso: Não foi possível obter amostra da tabela SQLite '{table_name}': {select_e}{Style.RESET_ALL}")
                        sample_rows = ["Erro ao Ler Amostra"]

                tables_data[table_name] = {
                    "columns": columns,
                    "total_rows": row_count,
                    "sample_rows": sample_rows_safe
                }

            except sqlite3.OperationalError as table_e:
                 # Erro específico da tabela (ex: tabela corrupta)
                print(f"{Fore.YELLOW}Aviso: Erro operacional ao processar tabela SQLite '{table_name}': {table_e}{Style.RESET_ALL}")
                tables_data[table_name] = {"error": f"Erro operacional: {table_e}", "columns": [], "total_rows": "Erro", "sample_rows": []}

        conn.close()
        return {"tables": tables_data, "analysis_status": "success"}

    except sqlite3.OperationalError as conn_e:
         # Erro ao conectar (arquivo não é DB, bloqueado, permissão)
         return {"error": f"Erro SQLite ao conectar (arquivo inválido, bloqueado ou sem permissão?): {conn_e}", "analysis_status": "failed_connect"}
    except Exception as e:
        # Outros erros inesperados
        return {"error": f"Erro inesperado ao processar SQLite: {e}", "analysis_status": "failed_other"}

def get_json_info(file_path: str) -> Dict[str, Any]:
    """Obtém informações básicas sobre um arquivo JSON."""
    info: Dict[str, Any] = {}
    file_size = get_file_size(file_path)
    line_count = count_lines(file_path)

    info["tamanho_formatado"] = format_size(file_size)
    info["tamanho_bytes"] = file_size if file_size >= 0 else "Erro"
    info["numero_de_linhas"] = line_count if line_count >= 0 else ("Erro Codificação" if line_count == -2 else "Erro Leitura")

    # Tenta inferir a estrutura raiz (objeto ou lista) lendo uma pequena parte
    structure_type = "desconhecido"
    if file_size > 0 and file_size < 5 * 1024 * 1024: # Tenta analisar apenas arquivos menores para performance
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Lê um pedaço inicial, suficiente para detectar a estrutura
                # Cuidado com JSONs muito grandes onde o início não é representativo
                preview = f.read(2048)
                # Tenta parsear o preview (pode falhar se o preview cortar no meio de uma string ou estrutura)
                try:
                    # Remove vírgulas ou lixo no final que pode impedir o parse do preview
                    preview_trimmed = preview.strip().rstrip(',').rstrip(']').rstrip('}')
                    # Adiciona fechamento se necessário para tentar parsear
                    if preview_trimmed.startswith('[') and not preview_trimmed.endswith(']'):
                        data = json.loads(preview_trimmed + ']')
                    elif preview_trimmed.startswith('{') and not preview_trimmed.endswith('}'):
                        data = json.loads(preview_trimmed + '}')
                    else:
                         data = json.loads(preview_trimmed)

                    if isinstance(data, list):
                        structure_type = "lista"
                    elif isinstance(data, dict):
                        structure_type = "objeto"
                    else:
                        structure_type = "outro (valor primitivo?)"
                except json.JSONDecodeError:
                    # Falha ao parsear o preview, acontece
                    structure_type = "não foi possível determinar pelo preview"
        except Exception:
            # Erro ao ler o arquivo para preview
            structure_type = "erro ao ler preview"

    info["tipo_estrutura_inferida"] = structure_type
    info["analysis_status"] = "success" # Mesmo que o preview falhe, infos básicas foram coletadas
    return info

def get_yaml_info(file_path: str) -> Dict[str, Any]:
    """Obtém informações básicas sobre um arquivo YAML."""
    info: Dict[str, Any] = {}
    file_size = get_file_size(file_path)
    line_count = count_lines(file_path)

    info["tamanho_formatado"] = format_size(file_size)
    info["tamanho_bytes"] = file_size if file_size >= 0 else "Erro"
    info["numero_de_linhas"] = line_count if line_count >= 0 else ("Erro Codificação" if line_count == -2 else "Erro Leitura")

    # Poderia adicionar uma tentativa de parsear o início com PyYAML para inferir estrutura, similar ao JSON
    # Ex:
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         preview_data = yaml.safe_load(f) # Cuidado: lê o arquivo todo!
    #         # Ou ler preview: preview = f.read(2048); preview_data = yaml.safe_load(preview)
    #     if isinstance(preview_data, list): info["tipo_estrutura_inferida"] = "lista"
    #     elif isinstance(preview_data, dict): info["tipo_estrutura_inferida"] = "objeto/mapa"
    # except Exception: info["tipo_estrutura_inferida"] = "erro ao analisar"

    info["analysis_status"] = "success"
    return info

def format_size(size_in_bytes: int) -> str:
    """Formata o tamanho em bytes para KB, MB, GB."""
    if size_in_bytes < 0: return "Erro"
    if size_in_bytes == 0: return "0 Bytes"
    if size_in_bytes < 1024: return f"{size_in_bytes} Bytes"
    if size_in_bytes < 1024**2: return f"{size_in_bytes/1024:.2f} KB"
    if size_in_bytes < 1024**3: return f"{size_in_bytes/(1024**2):.2f} MB"
    return f"{size_in_bytes/(1024**3):.2f} GB"

# Configurações de geração da IA
AI_TEMPERATURE = 0.7 # Um pouco mais determinístico para seguir instruções complexas
AI_TOP_P = 0.9
AI_TOP_K = 50
AI_MAX_TOKENS = 32800 # Ajuste conforme o modelo e a necessidade (modelos 1.5 suportam mais)

def configurar_geracao(
    temperatura: float = AI_TEMPERATURE,
    top_p: float = AI_TOP_P,
    top_k: int = AI_TOP_K,
    max_tokens: int = AI_MAX_TOKENS
) -> genai.types.GenerationConfig:
    """Cria o objeto de configuração de geração para a IA."""
    return genai.types.GenerationConfig(
        temperature=temperatura,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_tokens,
        # response_mime_type="text/plain" # Garante texto puro se necessário
    )

# Função para enviar mensagens para a IA com retentativas
def enviar_mensagem(sessao_chat: genai.ChatSession, mensagem: str) -> Optional[str]:
    """Envia uma mensagem para a sessão de chat da IA com retentativas."""
    MAX_RETRIES = 3 # Aumenta um pouco as retentativas
    RETRY_DELAY_SECONDS = 7 # Aumenta o delay

    for attempt in range(MAX_RETRIES):
        try:
            print(f"{Fore.YELLOW}🧠 Enviando prompt para IA ({NOME_MODELO}) - Tentativa {attempt + 1}/{MAX_RETRIES}...")
            # Medir tempo da chamada da IA
            start_ai_time = time.time()
            resposta = sessao_chat.send_message(mensagem)
            end_ai_time = time.time()
            print(f"{Fore.GREEN}✅ Resposta da IA recebida em {end_ai_time - start_ai_time:.2f}s.")

            # Verificar bloqueios ou falta de conteúdo
            if not resposta.candidates:
                block_reason = "Não especificado"
                safety_ratings = "N/A"
                try:
                    block_reason = resposta.prompt_feedback.block_reason.name
                    safety_ratings = str(resposta.prompt_feedback.safety_ratings)
                except AttributeError: pass # Nem sempre esses atributos existem
                print(f"{Fore.RED}❗ Resposta da IA bloqueada ou sem candidatos.")
                print(f"{Fore.RED}   Razão: {block_reason}")
                print(f"{Fore.RED}   Safety Ratings: {safety_ratings}")
                # Não tentar novamente se for bloqueado por segurança
                if block_reason != "BLOCK_REASON_UNSPECIFIED":
                    return None
                # Se for não especificado, talvez valha a pena tentar novamente? Ou retornar None?
                # Por segurança, retornamos None aqui também.
                return None

            # Extrair texto da primeira candidata (geralmente a única)
            if resposta.candidates[0].content and resposta.candidates[0].content.parts:
                texto_resposta = "".join(part.text for part in resposta.candidates[0].content.parts if hasattr(part, 'text'))
                if texto_resposta.strip():
                    # Verificar safety ratings da resposta também
                    try:
                        response_safety = str(resposta.candidates[0].safety_ratings)
                        if "BLOCK" in response_safety.upper(): # Verifica se algum rating indica bloqueio
                             print(f"{Fore.YELLOW}⚠️ Resposta da IA contém conteúdo potencialmente problemático (Safety: {response_safety}).")
                             # Decidir se retorna ou não. Por segurança, pode ser melhor retornar None ou uma msg de erro.
                             # return f"Erro: Resposta da IA marcada com problemas de segurança: {response_safety}"
                    except AttributeError: pass

                    return texto_resposta
                else:
                    print(f"{Fore.YELLOW}⚠️ Resposta da IA está vazia ou não contém texto.")
                    # Considerar retentativa ou retornar vazio? Retornar vazio é mais seguro.
                    return "" # Retorna vazia, mas não None
            else:
                 print(f"{Fore.YELLOW}⚠️ Resposta da IA não contém partes de conteúdo ou texto.")
                 return "" # Retorna vazia

        except Exception as e:
            print(f"{Fore.RED}❗Erro ao comunicar com a IA (Tentativa {attempt + 1}/{MAX_RETRIES}): {e}")
            # Verifica se é um erro específico que não justifica retentativa (ex: API Key inválida)
            if "API key not valid" in str(e):
                print(f"{Fore.RED}   Erro de chave de API. Interrompendo retentativas.")
                return None
            if attempt < MAX_RETRIES - 1:
                print(f"{Fore.CYAN}    Aguardando {RETRY_DELAY_SECONDS}s para tentar novamente...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"{Fore.RED}❗ Falha ao comunicar com a IA após {MAX_RETRIES} tentativas.")
                return None
    return None # Caso o loop termine sem sucesso

# Função para varrer diretórios
def scan_directory(root_path: str = ".") -> Dict[str, Any]:
    """Varre o diretório raiz, analisa arquivos e retorna um dicionário com a estrutura."""
    report: Dict[str, Any] = {}
    abs_root_path = os.path.abspath(root_path)
    print(f"{Fore.CYAN}🔍 Varrendo diretório: {abs_root_path}")
    print(f"{Fore.CYAN}   Ignorando nomes/extensões: {IGNORE_LIST}")
    items_processed, items_ignored, errors_occurred = 0, 0, 0

    # Cria uma lista de extensões e nomes completos para ignorar
    ignore_extensions = {item for item in IGNORE_LIST if item.startswith('.')}
    ignore_filenames = {item for item in IGNORE_LIST if not item.startswith('*') and not item.startswith('.')}
    ignore_patterns = [item for item in IGNORE_LIST if item.startswith('*')] # Para futuras melhorias com fnmatch

    for current_root, dirs, files in os.walk(root_path, topdown=True):
        # Filtra diretórios ignorados (modifica dirs in-place)
        dirs[:] = [d for d in dirs if d not in ignore_filenames and not d.startswith('.') and d not in IGNORE_LIST] # Adiciona checagem explícita na IGNORE_LIST

        # Calcula o caminho relativo para o relatório
        relative_root = os.path.relpath(current_root, root_path)
        if relative_root == '.':
            relative_root = 'Raiz do Projeto' # Nome mais amigável para a raiz

        root_report: Dict[str, Any] = {} # Relatório para este diretório específico

        for filename in files:
            file_path = os.path.join(current_root, filename)
            _, file_ext = os.path.splitext(filename)
            file_ext_lower = file_ext.lower()
            filename_lower = filename.lower()

            # Verifica se o arquivo deve ser ignorado
            ignore_file = False
            if filename in ignore_filenames or filename_lower in ignore_filenames:
                ignore_file = True
            elif file_ext_lower in ignore_extensions:
                ignore_file = True
            elif filename.startswith('.'): # Ignora arquivos ocultos
                ignore_file = True
            else:
                 # Checa padrões de extensão como '*.log'
                 if any(pat.endswith(file_ext_lower) for pat in ignore_patterns if pat.startswith('*.')):
                      ignore_file = True

            if ignore_file:
                items_ignored += 1
                continue # Pula para o próximo arquivo

            # Processa o arquivo válido
            items_processed += 1
            file_info: Dict[str, Any] = {}
            has_error = False

            try:
                file_size = get_file_size(file_path)
                line_count = count_lines(file_path) # Pode retornar < 0

                file_info = {
                    "caminho_relativo": os.path.relpath(file_path, root_path),
                    "tamanho_bytes": file_size if file_size >= 0 else -1, # Usa -1 para erro consistente
                    "tamanho_formatado": format_size(file_size),
                    "numero_de_linhas": line_count if line_count >= 0 else (-2 if line_count == -2 else -1), # -1 Erro IO, -2 Erro Decode
                    "extensao": file_ext_lower
                }

                # Análise específica por tipo de arquivo
                if file_ext_lower == ".py":
                    py_analysis = get_python_info(file_path)
                    if "error" in py_analysis: has_error = True
                    file_info["python_analysis"] = py_analysis
                elif file_ext_lower in (".db", ".sqlite", ".sqlite3"):
                    sqlite_analysis = get_sqlite_info(file_path)
                    if "error" in sqlite_analysis: has_error = True
                    file_info["sqlite_info"] = sqlite_analysis
                elif file_ext_lower == ".json":
                    json_analysis = get_json_info(file_path)
                    # get_json_info não sinaliza erro explicitamente, mas pode conter msgs
                    file_info["json_info"] = json_analysis
                elif file_ext_lower in (".yaml", ".yml"):
                    yaml_analysis = get_yaml_info(file_path)
                    file_info["yaml_info"] = yaml_analysis
                # Adiciona preview para arquivos de texto razoavelmente pequenos
                elif line_count > 0 and file_size > 0 and file_size < 1 * 1024 * 1024: # Preview até 1MB
                    preview = read_file_content(file_path, max_lines=15) # Aumenta um pouco as linhas do preview
                    if preview is not None:
                        file_info["content_preview"] = preview
                    # else: # Opcional: indicar que a leitura do preview falhou
                    #     file_info["content_preview_error"] = True

                # Se ocorreu erro em alguma análise específica, marca no log
                if has_error:
                    errors_occurred += 1
                    print(f"{Fore.RED}  -> Erro ao analisar: {file_info.get('caminho_relativo')}{Style.RESET_ALL}")
                # else: # Log verboso opcional para sucesso
                #     print(f"{Fore.GREEN}  + Ok: {file_info.get('caminho_relativo')}{Style.RESET_ALL}")


            except Exception as e:
                errors_occurred += 1
                print(f"{Fore.RED}  -> Erro INESPERADO ao processar {filename}: {e}")
                file_info["processing_error"] = str(e) # Adiciona erro geral ao info

            # Adiciona informações do arquivo ao relatório do diretório atual
            root_report[filename] = file_info

        # Adiciona o relatório deste diretório ao relatório principal, se não estiver vazio
        if root_report:
            report[relative_root] = root_report

    print(f"{Fore.GREEN}📊 Varredura concluída.")
    print(f"   Itens analisados: {items_processed}")
    print(f"   Itens ignorados: {items_ignored}")
    if errors_occurred > 0:
        print(f"{Fore.RED}   Erros durante análise: {errors_occurred}")
    if items_processed == 0:
        print(f"{Fore.YELLOW}⚠️ Nenhum arquivo relevante encontrado para análise detalhada.")

    return report

# --- TEMPLATE HTML (Android Arch - Atualizado - Apenas como GUIA DE ESTILO para a IA) ---
# Este template NÃO será mais usado diretamente para gerar o arquivo final.
# Ele será incluído no prompt para que a IA entenda o estilo visual desejado.
HTML_STYLE_GUIDE_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arquitetura Android Detalhada - Estilo Moderno</title>
    <style>
        /* Reset Básico e Estilos Globais (Mantido do seu exemplo) */
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 74%);
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: flex-start; /* Alinha ao topo para diagramas longos */
            min-height: 100vh;
            padding: 60px 20px; /* Mais padding vertical */
            overflow-x: hidden;
        }

        /* Container Principal do Diagrama */
        .diagram-container {
            width: 95%; /* Um pouco mais largo para comportar mais detalhes */
            max-width: 1100px; /* Aumentado para mais conteúdo */
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 35px; /* Mais padding interno */
            display: flex;
            flex-direction: column;
            gap: 20px; /* Espaço ligeiramente maior entre camadas */
            perspective: 1800px; /* Perspectiva 3D um pouco mais acentuada */
        }

        /* Estilo das Camadas */
        .layer {
            padding: 25px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            transform-style: preserve-3d;
        }

        .layer:hover {
           /* transform: translateY(-3px) translateZ(8px); */ /* Efeito hover opcional na camada */
        }

        /* Cores e Gradientes das Camadas (Mantido) */
        .layer-apps { background: linear-gradient(145deg, #00a9cc, #007bff); } /* Azul Ciano -> Azul */
        .layer-framework { background: linear-gradient(145deg, #5cb85c, #4cae4c); } /* Verde Claro -> Verde */
        .layer-native-libs { background: linear-gradient(145deg, #9d6ac9, #8a2be2); } /* Roxo -> Azul Violeta */
        .layer-runtime { background: linear-gradient(145deg, #f0ad4e, #ec971f); } /* Laranja Claro -> Laranja */
        .layer-hal { background: linear-gradient(145deg, #22b8c2, #1a98a1); } /* Azul-Verde -> Teal */
        .layer-kernel { background: linear-gradient(145deg, #d9534f, #c9302c); } /* Vermelho -> Vermelho Escuro */

        .layer-title {
            font-size: 1.5em; /* Título um pouco maior */
            font-weight: 600;
            color: #ffffff;
            text-shadow: 0 2px 5px rgba(0,0,0,0.4);
            margin-bottom: 25px; /* Mais espaço abaixo do título */
            text-align: center;
            padding-bottom: 10px; /* Mais padding abaixo */
            border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        }

         /* Título secundário dentro de uma camada */
        .sub-layer-title {
             font-size: 1.1em;
             font-weight: 500;
             color: rgba(255, 255, 255, 0.9);
             text-align: center;
             margin-top: 15px;
             margin-bottom: 15px;
             padding-bottom: 5px;
             border-bottom: 1px dashed rgba(255, 255, 255, 0.2);
        }

        /* Grid para os Componentes dentro das Camadas */
        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); /* Componentes um pouco maiores */
            gap: 18px; /* Espaço ligeiramente maior */
        }

        /* Layout especial para Framework (Managers vs Resto) */
         .framework-layout {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); /* Colunas adaptáveis */
            gap: 25px;
            align-items: start;
         }
         .framework-managers-grid { /* Grid específico para managers */
             display: grid;
             grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
             gap: 12px;
         }

         /* Layout para Nativo/Runtime (Lado a Lado onde couber) */
         .native-runtime-layout {
             display: grid;
             grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); /* Adapta melhor */
             gap: 25px;
             align-items: start;
             /* Removemos a cor da camada externa aqui, cada sub-camada tem a sua */
         }
         /* Estilos para as sub-camadas dentro do layout Nativo/Runtime */
         .native-runtime-layout > .layer {
              padding: 20px; /* Adiciona padding interno às sub-camadas */
              border-radius: 15px; /* Bordas arredondadas */
              /* Mantém o fundo individual definido por .layer-native-libs ou .layer-runtime */
         }
         .native-runtime-layout .layer-title { /* Título das sub-camadas */
              font-size: 1.3em;
              margin-bottom: 20px;
              border-bottom-color: rgba(255, 255, 255, 0.25);
         }

        /* Estilo dos Componentes Individuais */
        .component {
            background-color: rgba(255, 255, 255, 0.15);
            color: #f0f8ff;
            padding: 18px 15px; /* Mais padding vertical */
            border-radius: 12px;
            font-size: 0.9em;
            text-align: center;
            box-shadow: 0 5px 12px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.2);
            cursor: default;
            transition: transform 0.35s cubic-bezier(0.25, 0.8, 0.25, 1),
                        box-shadow 0.35s cubic-bezier(0.25, 0.8, 0.25, 1),
                        background-color 0.35s ease;
            opacity: 0;
            animation: fadeInScale 0.5s ease-out forwards;
            display: flex;
            flex-direction: column; /* Para permitir descrição opcional abaixo */
            align-items: center;
            justify-content: center;
            min-height: 70px; /* Altura mínima maior */
            transform-style: preserve-3d;
            position: relative; /* Adicionado para tooltip potencial */
        }

         /* Estilo para a descrição opcional dentro do componente (exemplo simples) */
         .component-desc {
             font-size: 0.75em;
             color: rgba(224, 224, 224, 0.7);
             margin-top: 5px;
             font-style: italic;
         }

         /* Efeito Hover com Tooltip (Exemplo mais elaborado pode ser usado pela IA) */
        .component .tooltiptext {
            visibility: hidden;
            width: 160px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 8px;
            position: absolute;
            z-index: 1;
            bottom: 110%; /* Position above the component */
            left: 50%;
            margin-left: -80px; /* Center the tooltip */
            opacity: 0;
            transition: opacity 0.3s, visibility 0.3s;
            font-size: 0.8em;
            pointer-events: none; /* Prevent tooltip from interfering */
        }

        .component:hover {
            transform: scale(1.07) translateZ(18px) rotateY(4deg); /* Efeito 3D sutilmente ajustado */
            background-color: rgba(255, 255, 255, 0.28);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
            z-index: 10;
        }

        /* Mostra tooltip no hover (se a IA implementar com esta classe) */
        .component:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }


        /* Placeholder para reticências */
        .component.placeholder {
             font-size: 1.6em;
             font-weight: bold;
             background-color: rgba(0, 0, 0, 0.1);
             border-style: dashed;
             color: rgba(255, 255, 255, 0.5);
        }

        /* Kernel específico: Power Management e Binder */
        .kernel-driver-special {
            grid-column: 1 / -1; /* Ocupa a largura toda da grid */
            margin-top: 10px;
             background-color: rgba(0, 0, 0, 0.25);
             font-weight: bold;
        }

        /* Animação Fade-in com Escala (Mantido) */
        @keyframes fadeInScale {
            from { opacity: 0; transform: scale(0.95) translateY(10px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        /* Responsividade (Ajustada para mais detalhes) */
        @media (max-width: 992px) { /* Ajuste breakpoint */
             .framework-layout { grid-template-columns: 1fr; } /* Empilha antes */
             .native-runtime-layout { grid-template-columns: 1fr; } /* Empilha antes */
        }
        @media (max-width: 768px) {
            .diagram-container { width: 95%; padding: 25px; }
            .layer { padding: 20px; }
            .layer-title { font-size: 1.3em; }
            .sub-layer-title { font-size: 1.0em; }
            .components-grid { grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 12px; }
            .component { font-size: 0.85em; padding: 15px 10px; min-height: 60px; }
             .component-desc { font-size: 0.7em; }
        }
         @media (max-width: 480px) {
            body { padding: 20px 10px; } /* Reduz padding do body */
            .diagram-container { border-radius: 15px; padding: 15px; }
             .layer { border-radius: 12px; padding: 15px;}
             .layer-title { font-size: 1.15em; }
             .sub-layer-title { font-size: 0.9em; }
             .components-grid { grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 10px; }
             .component { font-size: 0.78em; padding: 12px 8px; border-radius: 8px; min-height: 55px; }
             .component:hover { transform: scale(1.04) translateZ(8px) rotateY(0deg); } /* Reduz hover no mobile */
              .component-desc { display: none; } /* Oculta descrição em telas muito pequenas */
              .component .tooltiptext { display: none; } /* Oculta tooltip em telas muito pequenas */
         }

    </style>
</head>
<body>

    <div class="diagram-container">

        <!-- Camada System Apps -->
        <!-- Aplicações que vêm pré-instaladas ou são essenciais para o sistema. Usam as APIs do Framework. -->
        <div class="layer layer-apps">
            <div class="layer-title">System Apps Layer</div>
            <div class="components-grid">
                <div class="component" style="animation-delay: 0.1s;">
                    Phone (Dialer)
                    <span class="component-desc">Chamadas</span>
                    <span class="tooltiptext">Gerencia chamadas de voz e vídeo.</span>
                </div>
                <div class="component" style="animation-delay: 0.15s;">
                    Contacts
                    <span class="component-desc">Agenda</span>
                    <span class="tooltiptext">Armazena e gerencia contatos.</span>
                </div>
                <div class="component" style="animation-delay: 0.2s;">
                    Browser (Chrome/AOSP)
                    <span class="component-desc">Navegador Web</span>
                    <span class="tooltiptext">Renderiza páginas web.</span>
                </div>
                <div class="component" style="animation-delay: 0.25s;">
                    Camera
                    <span class="component-desc">Captura de Mídia</span>
                    <span class="tooltiptext">Interface para fotos e vídeos.</span>
                </div>
                <div class="component" style="animation-delay: 0.3s;">
                    Settings
                    <span class="component-desc">Configurações</span>
                    <span class="tooltiptext">Ajustes do sistema e apps.</span>
                </div>
                <div class="component" style="animation-delay: 0.35s;">
                    Clock
                    <span class="component-desc">Alarme/Timer</span>
                    <span class="tooltiptext">Relógio, alarme, cronômetro.</span>
                 </div>
                <div class="component" style="animation-delay: 0.4s;">
                    Calendar
                    <span class="component-desc">Agenda Pessoal</span>
                    <span class="tooltiptext">Gerencia eventos e lembretes.</span>
                 </div>
                <div class="component" style="animation-delay: 0.45s;">
                    Email/Gmail
                    <span class="component-desc">Cliente Email</span>
                    <span class="tooltiptext">Envia e recebe emails.</span>
                </div>
                <div class="component" style="animation-delay: 0.5s;">
                    Messages (SMS/MMS)
                    <span class="component-desc">Mensagens</span>
                    <span class="tooltiptext">Envia e recebe SMS/MMS/RCS.</span>
                </div>
                <div class="component placeholder" style="animation-delay: 0.55s;">
                    ...
                    <span class="component-desc">Outros Apps</span>
                    <span class="tooltiptext">Launcher, Play Store, etc.</span>
                </div>
            </div>
        </div>

        <!-- Camada Java API Framework -->
        <!-- Conjunto rico de APIs escritas em Java/Kotlin que os desenvolvedores usam para criar aplicativos. -->
        <div class="layer layer-framework">
            <div class="layer-title">Java API Framework Layer</div>
            <div class="framework-layout">
                <!-- Coluna Esquerda: Sistemas Fundamentais -->
                <div>
                    <h3 class="sub-layer-title">Core Systems & Utilities</h3>
                    <div class="components-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));"> <!-- Grid um pouco diferente aqui -->
                        <div class="component" style="animation-delay: 0.6s;">
                            Activity Manager
                            <span class="component-desc">Ciclo de vida das Activities</span>
                            <span class="tooltiptext">Gerencia o ciclo de vida e a pilha de Activities.</span>
                        </div>
                        <div class="component" style="animation-delay: 0.65s;">
                            Window Manager
                            <span class="component-desc">Gerencia janelas/superfícies</span>
                             <span class="tooltiptext">Controla a visibilidade e ordem das janelas.</span>
                        </div>
                        <div class="component" style="animation-delay: 0.7s;">
                            View System
                            <span class="component-desc">Widgets UI (Button, TextView)</span>
                            <span class="tooltiptext">Blocos de construção para interfaces de usuário.</span>
                        </div>
                        <div class="component" style="animation-delay: 0.75s;">
                            Content Providers
                            <span class="component-desc">Compartilhamento de dados</span>
                             <span class="tooltiptext">Permite que apps compartilhem dados entre si.</span>
                         </div>
                        <div class="component" style="animation-delay: 0.8s;">
                            Resource Manager
                            <span class="component-desc">Acesso a assets (layouts, strings)</span>
                            <span class="tooltiptext">Gerencia recursos não-compilados (imagens, layouts, etc.).</span>
                        </div>
                        <div class="component" style="animation-delay: 0.85s;">
                            Package Manager
                            <span class="component-desc">Info/instalação de apps</span>
                            <span class="tooltiptext">Obtém informações sobre apps instalados.</span>
                        </div>
                    </div>
                </div>
                <!-- Coluna Direita: Managers Específicos -->
                <div>
                     <h3 class="sub-layer-title">Feature-Specific Managers</h3>
                     <div class="framework-managers-grid">
                        <div class="component" style="animation-delay: 0.9s;">
                            Telephony Mgr
                            <span class="component-desc">Info de rede/chamada</span>
                            <span class="tooltiptext">Acessa status do telefone e rede celular.</span>
                        </div>
                        <div class="component" style="animation-delay: 0.95s;">
                            Location Mgr
                            <span class="component-desc">GPS/Rede/Fused</span>
                            <span class="tooltiptext">Obtém a localização do dispositivo.</span>
                        </div>
                        <div class="component" style="animation-delay: 1.0s;">
                            Notification Mgr
                            <span class="component-desc">Alertas na status bar</span>
                            <span class="tooltiptext">Exibe notificações para o usuário.</span>
                         </div>
                        <div class="component" style="animation-delay: 1.05s;">
                            Sensor Manager
                            <span class="component-desc">Acelerômetro, Giro, etc.</span>
                             <span class="tooltiptext">Acessa os sensores de hardware.</span>
                        </div>
                        <div class="component" style="animation-delay: 1.1s;">
                            Connectivity Mgr
                            <span class="component-desc">Estado de rede (WiFi/Dados)</span>
                            <span class="tooltiptext">Verifica e gerencia conexões de rede.</span>
                        </div>
                        <div class="component" style="animation-delay: 1.15s;">
                            Power Manager
                            <span class="component-desc">Wakelocks, Doze</span>
                            <span class="tooltiptext">Gerencia o estado de energia do dispositivo.</span>
                        </div>
                        <div class="component" style="animation-delay: 1.2s;">
                            Storage Manager
                            <span class="component-desc">Info de armazenamento</span>
                            <span class="tooltiptext">Acessa informações sobre o armazenamento.</span>
                        </div>
                         <div class="component" style="animation-delay: 1.25s;">
                             Download Manager
                             <span class="component-desc">Downloads em background</span>
                             <span class="tooltiptext">Gerencia downloads longos.</span>
                         </div>
                         <div class="component" style="animation-delay: 1.3s;">
                             Account Manager
                             <span class="component-desc">Gerencia contas online</span>
                              <span class="tooltiptext">Centraliza o gerenciamento de contas de usuário.</span>
                          </div>
                         <div class="component placeholder" style="animation-delay: 1.35s;">
                             ...
                             <span class="component-desc">Muitos outros</span>
                             <span class="tooltiptext">AudioManager, WifiManager, BluetoothManager, etc.</span>
                         </div>
                     </div>
                </div>
            </div>
        </div>

        <!-- Layout Agrupado: Native Libraries & Android Runtime -->
        <!-- Esta seção contém bibliotecas C/C++ nativas e o ambiente de execução ART. -->
         <div class="native-runtime-layout">
            <!-- Sub-Camada Native C/C++ Libraries -->
            <!-- Bibliotecas escritas em C/C++ que fornecem funcionalidades de baixo nível, acessadas pelo Framework via JNI. -->
            <div class="layer layer-native-libs">
                <div class="layer-title">Native C/C++ Libraries Layer</div>
                <div class="components-grid">
                    <div class="component" style="animation-delay: 1.4s;">
                        libc (Bionic)
                        <span class="component-desc">Standard C Library</span>
                         <span class="tooltiptext">Implementação otimizada da libc para Android.</span>
                     </div>
                    <div class="component" style="animation-delay: 1.45s;">
                        Media Framework
                        <span class="component-desc">Codecs (AAC, H.264)</span>
                        <span class="tooltiptext">Suporte a reprodução e gravação de mídia.</span>
                    </div>
                    <div class="component" style="animation-delay: 1.5s;">
                        Surface Manager
                        <span class="component-desc">Composição de telas</span>
                        <span class="tooltiptext">Compõe diferentes superfícies gráficas na tela.</span>
                    </div>
                    <div class="component" style="animation-delay: 1.55s;">
                        OpenGL ES / Vulkan
                        <span class="component-desc">APIs Gráficas 2D/3D</span>
                        <span class="tooltiptext">APIs padrão para renderização gráfica acelerada.</span>
                    </div>
                    <div class="component" style="animation-delay: 1.6s;">
                        SQLite
                        <span class="component-desc">Banco de dados local</span>
                         <span class="tooltiptext">Engine de banco de dados relacional leve.</span>
                    </div>
                    <div class="component" style="animation-delay: 1.65s;">
                        WebKit / Chromium
                        <span class="component-desc">Motor de renderização Web</span>
                        <span class="tooltiptext">Usado pelo WebView para exibir conteúdo web.</span>
                     </div>
                    <div class="component" style="animation-delay: 1.7s;">
                        Skia
                        <span class="component-desc">Engine Gráfica 2D</span>
                         <span class="tooltiptext">Biblioteca para desenho 2D (usada pela UI).</span>
                    </div>
                    <div class="component" style="animation-delay: 1.75s;">
                        SSL
                        <span class="component-desc">Segurança (HTTPS)</span>
                         <span class="tooltiptext">Implementa TLS/SSL para conexões seguras.</span>
                     </div>
                    <div class="component" style="animation-delay: 1.8s;">
                        FreeType
                        <span class="component-desc">Renderização de Fontes</span>
                        <span class="tooltiptext">Renderiza fontes TrueType e OpenType.</span>
                    </div>
                    <div class="component placeholder" style="animation-delay: 1.85s;">
                        ...
                        <span class="component-desc">Outras libs</span>
                         <span class="tooltiptext">libjpeg, libpng, zlib, etc.</span>
                     </div>
                </div>
            </div>

            <!-- Sub-Camada Android Runtime (ART) -->
            <!-- Ambiente de execução onde os aplicativos Android rodam. Inclui a VM e bibliotecas Java base. -->
            <div class="layer layer-runtime">
                 <div class="layer-title">Android Runtime (ART) Layer</div>
                 <div class="components-grid" style="grid-template-columns: 1fr 1fr; gap: 20px;"> <!-- Layout 2 colunas fixas aqui -->
                    <!-- Coluna ART VM -->
                    <div class="component" style="animation-delay: 1.9s; grid-row: span 2; display: flex; flex-direction: column; justify-content: space-around; align-items: center; padding: 25px;">
                         <span style="font-size: 1.2em; font-weight: bold;">ART Virtual Machine</span>
                         <span class="component-desc" style="font-size: 0.85em; text-align: center;">Executa bytecode DEX. Usa AOT/JIT/GC otimizados.</span>
                          <span class="tooltiptext">Android Runtime: Compilação Ahead-of-Time (AOT) e Just-in-Time (JIT), Garbage Collection (GC).</span>
                     </div>
                     <!-- Coluna Core Libraries -->
                     <div class="component" style="animation-delay: 1.95s;">
                         Core Java/Kotlin Libraries
                         <span class="component-desc">Classes base (java.lang, kotlin.*, Collections, I/O, etc.)</span>
                          <span class="tooltiptext">Fornece a funcionalidade das bibliotecas padrão Java/Kotlin.</span>
                     </div>
                     <div class="component" style="animation-delay: 2.0s;">
                         Dalvik Executable (DEX)
                         <span class="component-desc">Formato de bytecode</span>
                         <span class="tooltiptext">Formato compacto de bytecode otimizado para dispositivos móveis.</span>
                      </div>
                </div>
            </div>
         </div>


        <!-- Camada Hardware Abstraction Layer (HAL) -->
        <!-- Define interfaces padrão que expõem as capacidades do hardware do dispositivo para o Framework API. -->
        <div class="layer layer-hal">
            <div class="layer-title">Hardware Abstraction Layer (HAL)</div>
            <div class="components-grid" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">
                <div class="component" style="animation-delay: 2.05s;">
                    Audio HAL
                    <span class="component-desc">Interface p/ áudio</span>
                    <span class="tooltiptext">Interface padrão para hardware de áudio.</span>
                </div>
                <div class="component" style="animation-delay: 2.1s;">
                    Camera HAL
                    <span class="component-desc">Interface p/ câmera</span>
                     <span class="tooltiptext">Interface padrão para hardware de câmera.</span>
                 </div>
                <div class="component" style="animation-delay: 2.15s;">
                    Sensors HAL
                    <span class="component-desc">Interface p/ sensores</span>
                    <span class="tooltiptext">Interface padrão para sensores (acel., giro., etc.).</span>
                </div>
                <div class="component" style="animation-delay: 2.2s;">
                    Bluetooth HAL
                    <span class="component-desc">Interface p/ BT</span>
                     <span class="tooltiptext">Interface padrão para hardware Bluetooth.</span>
                 </div>
                <div class="component" style="animation-delay: 2.25s;">
                    GPS/Location HAL
                    <span class="component-desc">Interface p/ localização</span>
                    <span class="tooltiptext">Interface padrão para hardware de GPS/GNSS.</span>
                </div>
                <div class="component" style="animation-delay: 2.3s;">
                    WiFi HAL
                    <span class="component-desc">Interface p/ Wi-Fi</span>
                     <span class="tooltiptext">Interface padrão para hardware Wi-Fi.</span>
                 </div>
                <div class="component" style="animation-delay: 2.35s;">
                    Graphics HAL (Gralloc/Composer)
                    <span class="component-desc">Alocação/Composição Gráfica</span>
                    <span class="tooltiptext">Interfaces para alocação de buffers gráficos (Gralloc) e composição de hardware (HW Composer).</span>
                 </div>
                <div class="component" style="animation-delay: 2.4s;">
                    RIL HAL (Telephony)
                    <span class="component-desc">Interface p/ Rádio</span>
                     <span class="tooltiptext">Radio Interface Layer: Comunica com o modem/baseband.</span>
                 </div>
                 <div class="component" style="animation-delay: 2.45s;">
                     NFC HAL
                     <span class="component-desc">Interface p/ NFC</span>
                     <span class="tooltiptext">Interface padrão para hardware NFC.</span>
                 </div>
                 <div class="component" style="animation-delay: 2.5s;">
                     Biometrics HAL
                     <span class="component-desc">Impressão digital, etc.</span>
                     <span class="tooltiptext">Interface padrão para hardware biométrico.</span>
                 </div>
                 <div class="component placeholder" style="animation-delay: 2.55s;">
                     ...
                     <span class="component-desc">Outros HALs</span>
                     <span class="tooltiptext">USB HAL, Power HAL, DRM HAL, etc.</span>
                 </div>
            </div>
        </div>

        <!-- Camada Linux Kernel -->
        <!-- O coração do sistema, baseado no Kernel Linux. Gerencia processos, memória, dispositivos, rede, etc. -->
        <div class="layer layer-kernel">
            <div class="layer-title">Linux Kernel Layer</div>
             <h3 class="sub-layer-title">Hardware Device Drivers</h3>
            <div class="components-grid" style="grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));"> <!-- Grid mais denso -->
                <div class="component" style="animation-delay: 2.6s;">
                    Display Driver
                    <span class="component-desc">GPU/Tela</span>
                     <span class="tooltiptext">Controla o hardware de vídeo.</span>
                </div>
                <div class="component" style="animation-delay: 2.65s;">
                    Camera Driver
                    <span class="component-desc">Sensor Câmera</span>
                     <span class="tooltiptext">Controla o sensor da câmera.</span>
                </div>
                <div class="component" style="animation-delay: 2.7s;">
                    Bluetooth Driver
                    <span class="component-desc">Chip BT</span>
                     <span class="tooltiptext">Controla o chip Bluetooth.</span>
                </div>
                <div class="component" style="animation-delay: 2.75s;">
                    WiFi Driver
                    <span class="component-desc">Chip Wi-Fi</span>
                    <span class="tooltiptext">Controla o chip Wi-Fi.</span>
                </div>
                <div class="component" style="animation-delay: 2.8s;">
                    Audio Driver
                    <span class="component-desc">Codec/DSP Áudio</span>
                     <span class="tooltiptext">Controla o hardware de áudio.</span>
                </div>
                <div class="component" style="animation-delay: 2.85s;">
                    USB Driver
                    <span class="component-desc">Porta USB</span>
                     <span class="tooltiptext">Controla a porta USB.</span>
                 </div>
                <div class="component" style="animation-delay: 2.9s;">
                    Keypad/Touch Driver
                    <span class="component-desc">Entrada Tela</span>
                     <span class="tooltiptext">Controla a tela de toque e botões.</span>
                </div>
                <div class="component" style="animation-delay: 2.95s;">
                    Sensor Drivers
                    <span class="component-desc">Físicos</span>
                    <span class="tooltiptext">Controla os sensores físicos.</span>
                </div>
                <div class="component" style="animation-delay: 3.0s;">
                    Memory Mgmt Driver
                    <span class="component-desc">Flash/RAM</span>
                    <span class="tooltiptext">Controla a memória flash e RAM.</span>
                </div>
                 <div class="component placeholder" style="animation-delay: 3.05s;">
                     ...
                     <span class="component-desc">Outros drivers</span>
                     <span class="tooltiptext">Drivers para GPS, NFC, Bateria, etc.</span>
                 </div>
            </div>
             <h3 class="sub-layer-title">Core Kernel Services</h3>
             <div class="components-grid" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));">
                 <!-- Binder e Power Management como itens especiais -->
                 <div class="component kernel-driver-special" style="animation-delay: 3.1s;">
                     Binder (IPC) Driver
                     <span class="component-desc">Comunicação Inter-Processos fundamental no Android</span>
                     <span class="tooltiptext">Mecanismo principal de IPC (Inter-Process Communication) do Android.</span>
                 </div>
                 <div class="component kernel-driver-special" style="animation-delay: 3.15s;">
                     Power Management (PM)
                     <span class="component-desc">Gerenciamento de energia, CPU frequency scaling, suspend</span>
                      <span class="tooltiptext">Gerencia o consumo de energia (wakelocks, suspend-to-RAM).</span>
                  </div>
                 <div class="component" style="animation-delay: 3.2s;">
                     Process Management
                     <span class="component-desc">Criação/gerência de processos</span>
                     <span class="tooltiptext">Gerencia a execução e o ciclo de vida dos processos.</span>
                 </div>
                 <div class="component" style="animation-delay: 3.25s;">
                     Memory Management
                     <span class="component-desc">Alocação/liberação de memória (OOM Killer)</span>
                     <span class="tooltiptext">Gerencia a memória RAM, incluindo o Low Memory Killer.</span>
                  </div>
                 <div class="component" style="animation-delay: 3.3s;">
                     Security (SELinux)
                     <span class="component-desc">Controle de acesso mandatório</span>
                     <span class="tooltiptext">Impõe políticas de segurança (Mandatory Access Control).</span>
                 </div>
                 <div class="component" style="animation-delay: 3.35s;">
                     Networking Stack
                     <span class="component-desc">TCP/IP, Sockets</span>
                     <span class="tooltiptext">Pilha de protocolos de rede padrão do Linux.</span>
                 </div>
             </div>
        </div>

    </div>

    <!-- Script JS opcional (Mantido do seu exemplo) -->
    <script>
        // Pequeno script para adicionar delays de animação dinamicamente se não definidos inline
        // e potencialmente inicializar tooltips JS mais complexos se a IA os gerar.
        document.addEventListener('DOMContentLoaded', () => {
            const components = document.querySelectorAll('.component');
            components.forEach((comp, index) => {
                // Aplica delay de animação CSS se não estiver definido inline
                if (!comp.style.animationDelay) {
                    comp.style.animationDelay = `${index * 0.05 + 0.1}s`; // Adiciona um pequeno delay base
                }
            });
             // console.log("Diagrama Carregado e Animações Aplicadas.");
        });
    </script>

</body>
</html>
"""

# --- Função Principal de Geração ---

def gerar_relatorio_ia(
    project_data: Dict[str, Any],
    project_name: str
) -> None:
    """
    Gera relatórios: Markdown (descrição do projeto) e HTML (diagrama de arquitetura do projeto).
    O HTML é gerado pela IA baseando-se nos dados do projeto e usando um template como guia de estilo.
    """
    try:
        model = genai.GenerativeModel(
            model_name=NOME_MODELO,
            generation_config=configurar_geracao(),
            # Safety settings podem ser ajustados aqui se necessário
            # Exemplo: Bloquear conteúdo mais sensível
            # safety_settings={
            #     'HATE': 'BLOCK_MEDIUM_AND_ABOVE',
            #     'HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
            #     'SEXUAL' : 'BLOCK_MEDIUM_AND_ABOVE',
            #     'DANGEROUS' : 'BLOCK_MEDIUM_AND_ABOVE'
            # }
        )
        sessao_chat = model.start_chat(history=[])
        print(f"{Fore.CYAN}🤖 Iniciando sessão de chat com {NOME_MODELO}...")
    except Exception as e:
        print(f"{Fore.RED}❗Erro fatal ao iniciar sessão de chat com a IA: {e}")
        print(f"{Fore.RED}   Verifique o nome do modelo ('{NOME_MODELO}') e as configurações.")
        return

    generation_info = {
        "os_system": platform.system(),
        "os_release": platform.release(),
        "hostname": platform.node(),
        "generation_timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "ai_model": NOME_MODELO,
    }

    # Constrói o prompt final com instruções revisadas
    prompt = f"""
    
    sempre escolha o tema de cores, ou clara, ou escura, sempre escolha usa, se inspire no template, mas crie algo 100 vezes melhor
    
    o nivel de detalhe é 9999999 
    
    o nivel de design é 999999999
    
    
    o css js e tudo estilo tailwind mais de 200 componentes tudo ao estado da arte
    
    mantenha consistencia de cores
    
    capture ao fundo toda as camadas o mais detalhe da mais rasa a mais profunda
    
    capture as camadas da arquitetura de software, nada é raso, tudo é profundo e literal, os baloes de infroamcao sao lognos e completos, fala o que é o que faz como faz como foi feito a escolha daquele componente, logica, etc, 
    
    use todos meus promtps cmo base e me surpreenda
    
    voce é o maior arquiteto de software do mundo - entenda a fundo o projeto, entenda o mais detalhado o projeto, os arquivos, o codigo fonte, funcao, o que faz, como faz, modulos, tudo que existe no projeto
    
    crie na sua memoria uma enorome matriz de dados do projeto 6666 x6666 x6666 blocos de dados unicos e chunks do projeto, codigo fonte, estrutura, funcoes, logica, etc, ate mesmo a nivel de evolucao, sempre entenda o todo mas tambem o que a de amis novo cofnorme nome e versoes, 
    
    sempre entende o padrao dos nomes pra entender o que era poc, mvp e atualizacoes e variacoes
    
    nos blocos de detalhes, sempre use muito icones, emojis, melhore muito use muito grid, 3d, blur, fade, box, flexbo,x grid, tailwind, repsonsividade, gradiente, glass, 3d, fade blur
    
    
    crie o html altamente longo, completo, com inumeras sessoes, com blocos, com inumeros elementos por blocos, cubra a fundo o projeto, entenda a fundo o projeto, crie sem footer e header
    
    --
    
    detalhe muito a fundo a arquitetura, crie inspierado, mas adapte tudo que precisar
    
    nunca invente
    nunca crie abreviado
    
    nunca deixe coisas faltando ou como .... etc tudo é declarado e ja esta presente
    
    
    **Sua Identidade:** Você é a Replika AI DocGen, um especialista sênior em arquitetura de software e documentação técnica, criado por Elias Andrade. Sua missão é analisar dados de projetos e gerar documentação clara e visualizações de arquitetura impressionantes.

    **Tarefa Dupla:** Analise os dados estruturados de um projeto (fornecido em YAML abaixo) e gere DOIS artefatos distintos:

    **1. Documentação Técnica em Markdown (`DOCUMENTACAO-PROJETO.md`):**
        *   **Objetivo:** Criar uum html snipept detalhado sobre o projeto '{project_name}', baseado EXCLUSIVAMENTE nos dados YAML fornecidos.
        *   **Conteúdo Essencial:**
            *   **Visão Geral:** Propósito principal do projeto, o problema que resolve.
            *   **Funcionalidades Chave:** Liste as principais capacidades identificadas.
            *   **Tecnologias e Dependências:** Mencione linguagens, bibliotecas, frameworks, bancos de dados identificados (resuma os imports/análises).
            *   **Estrutura do Projeto:** Descreva brevemente a organização dos diretórios e arquivos importantes (use os caminhos relativos).
            *   **Pontos de Atenção:** Se a análise (YAML) indicou erros (syntax errors, DB errors, etc.), mencione-os.
            *   **(Opcional) Como Executar:** Se possível inferir dos arquivos (ex: main.py, package.json), sugira comandos básicos.
            *   **Estado Inferido:** (Ex: Protótipo, Em desenvolvimento, Maduro) com base na complexidade e completude aparente.
        *   **Estilo e Tom:** Técnico, claro, conciso e direto. Use formatação Markdown (cabeçalhos, listas, `inline code`).
        *   **Restrição CRÍTICA:** NÃO inclua blocos de código fonte do projeto no Markdown. Descreva a estrutura e o propósito, não copie o código.
        *   **Autoria:** Inclua uma nota no final: "Documentação gerada por Replika AI DocGen (Elias Andrade) em {generation_info['generation_timestamp']}."
        *   **Emojis:** Use emojis relevantes para ilustrar seções (ex: 🎯 Propósito, 🛠️ Tecnologias, 📁 Estrutura, 🚀 Execução, ⚠️ Atenção).

    **2. Diagrama de Arquitetura em HTML (`doc-web-diagram-data-hashtagh unica .html`):**
        *   **Objetivo:** Gerar um ARQUIVO HTML COMPLETO (`<!DOCTYPE html>...</html>`) que visualize a arquitetura do software do projeto '{project_name}', baseando-se nos dados do YAML (estrutura de diretórios, arquivos Python, DBs, etc.).
        *   **Conteúdo do Diagrama:** O diagrama deve representar as **camadas lógicas ou componentes principais** do *projeto analisado*. Use os dados do YAML para identificar:
            *   Módulos Python principais (baseado em arquivos .py com classes/funções significativas).
            *   Possíveis camadas (ex: UI, Lógica de Negócios, Acesso a Dados - inferir pela estrutura e nomes de arquivos/diretórios).
            *   Componentes de dados (bancos SQLite, arquivos JSON/YAML importantes).
            *   Dependências chave (imports relevantes).
            *   Interconexões (descreva ou mostre visualmente se possível, como um módulo usa outro ou acessa um DB).
        *   **GUIA DE ESTILO VISUAL (CRÍTICO):** Use o template HTML fornecido abaixo **APENAS COMO INSPIRAÇÃO VISUAL**. **NÃO REPLIQUE O CONTEÚDO DO ANDROID.** Sua tarefa é criar um diagrama com visual semelhante ao do exemplo Android:
            *   **Tema Escuro:** Use fundos escuros e texto claro, seguindo a paleta do exemplo.
            *   **Layout:** Use divs para representar camadas (`.layer`) e componentes (`.component`). Organize-os usando CSS Grid ou Flexbox de forma lógica (inspire-se nos layouts `.components-grid`, `.framework-layout`, `.native-runtime-layout` do exemplo para organizar os componentes do projeto real). Use títulos de camada (`.layer-title`) e sub-títulos (`.sub-layer-title`) se apropriado.
            *   **Estilo Moderno:** Use gradientes sutis nas camadas (você pode escolher cores diferentes, mas mantenha o estilo), cantos arredondados, sombras (`box-shadow`), efeito de vidro/blur (`backdrop-filter` no container principal).
            *   **Interatividade (Hover + Tooltip):** Implemente tooltips ou descrições que aparecem ao passar o mouse sobre os componentes (`.component`). Use uma estrutura similar a `<span class="tooltiptext">...</span>` dentro do `.component` (como no exemplo ATUALIZADO) para mostrar detalhes extraídos do YAML (ex: docstrings de funções/classes, colunas de tabelas DB, tamanho/linhas de arquivos, previews de conteúdo). Adapte o CSS `.component .tooltiptext` e `.component:hover .tooltiptext` do exemplo. Inclua também uma breve descrição visível com `<span class="component-desc">...</span>`.
            *   **Responsividade:** O layout deve se adaptar a diferentes tamanhos de tela (use media queries como no exemplo, ajustando breakpoints e estilos conforme necessário para o SEU diagrama).
            *   **Animações Sutis:** Use animações de entrada (`@keyframes fadeInScale`) para tornar a apresentação mais agradável, aplicando delays diferentes para cada componente.
            *   **Profundidade e Detalhe:** Esforce-se para criar um diagrama DETALHADO e PROFUNDO, cobrindo o máximo possível de aspectos relevantes do projeto identificados no YAML. Crie muitos blocos/componentes se a análise permitir. **Não use "..." ou "outros"**; represente explicitamente os elementos importantes identificados nos dados YAML.
        *   **Idioma:** Todo o texto visível no HTML (títulos, nomes de componentes, descrições, tooltips) deve ser em **Português (pt-br)**.
        *   **Formato:** Gere o HTML completo como um bloco de código delimitado estritamente por ```html no início e ``` no final.
        *   **NÃO INCLUA UM RODAPÉ (FOOTER) PADRÃO NO HTML.** O script pós-processará e adicionará um rodapé se necessário.
        *   **Scripts JS:** Mantenha o pequeno script JS no final do HTML (como no exemplo) para aplicar delays de animação dinamicamente, caso não sejam definidos inline.


depois de criar toda a arquitetura de softwae, dai crie de funcao, dai otura de bibliotecas, dai outra e pipeline dai outra de estrutura de dados 
    **Guia de Estilo Visual HTML (Use para INSPIRAÇÃO DE ESTILO E ESTRUTURA, NÃO COPIE o conteúdo Android):**
    ```html
    {HTML_STYLE_GUIDE_TEMPLATE}
    ```

    **Dados da Análise do Projeto '{project_name}' (Base para AMBAS as tarefas):**
    ```yaml
    # Dados da análise do projeto: {project_name}
    # Gerado em: {generation_info['generation_timestamp']}
    # Ambiente: {generation_info['hostname']} ({generation_info['os_system']} {generation_info['os_release']}, Python {generation_info['python_version']})
    ---
    {yaml.dump(project_data, allow_unicode=True, default_flow_style=False, sort_keys=False, width=150, indent=2)}
    ---
    ```

    **Instrução Final:** Gere primeiro o conteúdo Markdown completo para o `DOCUMENTACAO-PROJETO.md`. Após o Markdown, gere o bloco de código HTML completo e válido para o `doc-web-diagram-data-hashtagh unica .html`, seguindo TODAS as instruções acima, especialmente sobre usar os dados do YAML e o guia de estilo visual ATUALIZADO. Delimite o HTML estritamente com ```html ... ```.
    """

    resposta_completa = enviar_mensagem(sessao_chat, prompt)

    if not resposta_completa:
        print(f"{Fore.RED}❗Erro: Nenhuma resposta recebida da IA ou resposta vazia/bloqueada após retentativas.")
        return

    print(f"{Fore.CYAN}📝 Processando a resposta da IA...")

    # Extrair o bloco HTML da resposta
    html_pattern = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    html_match = html_pattern.search(resposta_completa)

    markdown_content = resposta_completa # Assume inicialmente que tudo é Markdown
    html_content: Optional[str] = None
    html_filename = f"doc-web-diagram-{__import__('datetime').datetime.now().strftime('%Y%m%d-%H%M%S')}-{__import__('hashlib').sha256(__import__('datetime').datetime.now().isoformat().encode()).hexdigest()[:8]}.html"

    markdown_filename = "DOCUMENTACAO-PROJETO.md"

    if html_match:
        html_content = html_match.group(1).strip()
        # Remove o bloco HTML e os delimitadores do conteúdo Markdown
        markdown_content = html_pattern.sub("", resposta_completa).strip()
        print(f"{Fore.GREEN}📄 Bloco HTML encontrado e extraído.")

        # Validação básica do HTML extraído
        if not html_content:
            print(f"{Fore.YELLOW}⚠️ O bloco HTML extraído está vazio.")
            html_content = None # Descarta se estiver vazio
        elif not (html_content.lower().startswith("<!doctype") or html_content.lower().startswith("<html")):
             print(f"{Fore.YELLOW}⚠️ O HTML extraído não parece ser um documento completo (sem DOCTYPE/html inicial). Será salvo assim mesmo.")
        elif len(html_content) < 500: # Um diagrama útil provavelmente será maior
             print(f"{Fore.YELLOW}⚠️ O conteúdo HTML parece muito curto ({len(html_content)} caracteres). Pode estar incompleto ou a IA não seguiu as instruções de detalhe.")

    else:
        print(f"{Fore.YELLOW}⚠️ Bloco HTML (```html ... ```) não encontrado na resposta da IA.")
        print(f"{Fore.YELLOW}   Todo o conteúdo recebido será salvo como Markdown em {markdown_filename}.")

    # --- Salvando o arquivo Markdown ---
    try:
        # Adiciona a nota de autoria ao final se não estiver presente
        autoria_ia = f"Documentação gerada por Replika AI DocGen (Elias Andrade) em {generation_info['generation_timestamp']}"
        if autoria_ia not in markdown_content:
            markdown_content += f"\n\n---\n*{autoria_ia}*"

        with open(markdown_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        print(f"{Fore.GREEN}✅ Arquivo Markdown salvo: {markdown_filename}")
    except IOError as e:
        print(f"{Fore.RED}❗Erro de IO ao salvar {markdown_filename}: {e}")
    except Exception as e:
        print(f"{Fore.RED}❗Erro inesperado ao salvar {markdown_filename}: {e}")

    # --- Salvando e Pós-Processando o arquivo HTML ---
    if html_content:
        try:
            final_html = html_content
            now_str = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            hostname_str = generation_info['hostname']
            os_str = generation_info['os_system']

            # 1. Atualizar o Título (sobrescreve o da IA se existir)
            title_tag = f"<title>Diagrama de Arquitetura | {project_name} | Replika AI</title>"
            # Tenta substituir um <title> existente, senão insere no <head>
            if re.search(r"<head.*?>", final_html, re.IGNORECASE | re.DOTALL):
                 if re.search(r"<title>.*?</title>", final_html, re.IGNORECASE | re.DOTALL):
                     final_html = re.sub(r"<title>.*?</title>", title_tag, final_html, count=1, flags=re.IGNORECASE | re.DOTALL)
                 else:
                     # Insere o title dentro do head se não existir
                     final_html = re.sub(r"(<head.*?>)", r"\1\n    " + title_tag, final_html, count=1, flags=re.IGNORECASE | re.DOTALL)
            else:
                 # Adiciona head e title se nem head existir (improvável para HTML completo)
                 final_html = final_html.replace("<html>", f"<html>\n<head>\n    {title_tag}\n</head>", 1)


            # 2. Lógica do Rodapé: Comentar o da IA se existir, senão adicionar o nosso
            footer_added_by_script = False
            # Padrão para detectar um rodapé gerado pela IA ou pelo script (flexível)
            footer_pattern = re.compile(
                r"<footer.*?>(.*?(?:Gerado por|Diagrama|Projeto Associado|Replika AI|{project_name}).*?)</footer>".format(project_name=re.escape(project_name)),
                re.IGNORECASE | re.DOTALL
            )
            # Padrão para encontrar especificamente o rodapé *comentado* pelo script
            commented_footer_pattern = re.compile(r"<!--\s*<footer.*?Gerado por Replika AI DocGen.*?</footer>\s*-->", re.IGNORECASE | re.DOTALL)

            footer_match = footer_pattern.search(final_html)

            # Verifica se NÃO é um rodapé já comentado por uma execução anterior deste script
            if footer_match and not commented_footer_pattern.search(footer_match.group(0)):
                print(f"{Fore.YELLOW}⚠️ Rodapé detectado na resposta da IA. Comentando-o.")
                ai_footer_content = footer_match.group(0)
                # Comenta o bloco do rodapé encontrado
                commented_footer = f"\n<!-- Rodapé original da IA (comentado pelo script):\n{ai_footer_content}\n-->\n"
                final_html = final_html.replace(ai_footer_content, commented_footer, 1)
                # Mesmo comentando o da IA, adicionamos o nosso para garantir consistência
                print(f"{Fore.CYAN}ℹ️ Adicionando rodapé padrão ao HTML.")
                footer_html = f"""
        <!-- Footer Adicionado pelo Script -->
        <footer style="text-align: center; margin-top: 30px; font-size: 0.85em; color: rgba(224, 224, 224, 0.6); padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            Diagrama da Arquitetura | Projeto: {project_name} <br>
            Gerado por Replika AI DocGen em {now_str} | Host: {hostname_str} ({os_str})
        </footer>
"""
                # Tenta inserir o footer antes de </body>, senão antes de </script> final, senão no fim
                if '</body>' in final_html.lower():
                    final_html = re.sub(r"</body>", f"{footer_html}\n</body>", final_html, count=1, flags=re.IGNORECASE)
                    footer_added_by_script = True
                elif '</script>' in final_html.lower():
                     parts = final_html.rsplit('</script>', 1)
                     if len(parts) == 2:
                         final_html = parts[0] + '</script>' + footer_html + parts[1]
                         footer_added_by_script = True
                # Se não encontrar body ou script, tenta anexar antes de </html>
                if not footer_added_by_script and '</html>' in final_html.lower():
                     final_html = re.sub(r"</html>", f"{footer_html}\n</html>", final_html, count=1, flags=re.IGNORECASE)
                     footer_added_by_script = True
                # Último recurso: anexar ao final
                if not footer_added_by_script:
                    final_html += footer_html

            else:
                # Se não encontrou rodapé da IA (ou era um já comentado), adiciona o do script
                print(f"{Fore.CYAN}ℹ️ Adicionando rodapé padrão ao HTML.")
                footer_html = f"""
        <!-- Footer Adicionado pelo Script -->
        <footer style="text-align: center; margin-top: 30px; font-size: 0.85em; color: rgba(224, 224, 224, 0.6); padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            Diagrama da Arquitetura | Projeto: {project_name} <br>
            Gerado por Replika AI DocGen em {now_str} | Host: {hostname_str} ({os_str})
        </footer>
"""
                # Tenta inserir o footer antes de </body>, senão antes de </script> final, senão no fim
                if '</body>' in final_html.lower():
                    final_html = re.sub(r"</body>", f"{footer_html}\n</body>", final_html, count=1, flags=re.IGNORECASE)
                    footer_added_by_script = True
                elif '</script>' in final_html.lower():
                     parts = final_html.rsplit('</script>', 1)
                     if len(parts) == 2:
                         final_html = parts[0] + '</script>' + footer_html + parts[1]
                         footer_added_by_script = True
                # Se não encontrar body ou script, tenta anexar antes de </html>
                if not footer_added_by_script and '</html>' in final_html.lower():
                     final_html = re.sub(r"</html>", f"{footer_html}\n</html>", final_html, count=1, flags=re.IGNORECASE)
                     footer_added_by_script = True
                # Último recurso: anexar ao final
                if not footer_added_by_script:
                    final_html += footer_html

            # 3. Salvar o HTML final
            with open(html_filename, 'w', encoding='utf-8') as html_file:
                html_file.write(final_html)
            print(f"{Fore.GREEN}✅ Arquivo HTML (Diagrama do Projeto) salvo: {html_filename}")

        except IOError as e:
            print(f"{Fore.RED}❗Erro de IO ao pós-processar ou salvar {html_filename}: {e}")
        except Exception as e:
            print(f"{Fore.RED}❗Erro inesperado ao pós-processar ou salvar {html_filename}: {e}")
            import traceback
            traceback.print_exc() # Imprime stack trace para depuração

# --- Função Principal ---
def main():
    """Função principal que orquestra a varredura e geração dos relatórios."""
    start_time = time.time()
    project_path = "." # Diretório atual como padrão
    try:
        # Tenta obter um nome mais significativo para o projeto a partir do caminho absoluto
        abs_path = os.path.abspath(project_path)
        project_name = os.path.basename(abs_path)
        if not project_name or project_name == '.': # Caso rode de C:\ ou /
             project_name = "Projeto Raiz Desconhecido"
    except Exception:
        project_name = "Projeto Desconhecido"

    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"🚀 {Style.BRIGHT}Iniciando Replika AI DocGen - Geração de Documentação e Diagrama{Style.NORMAL}")
    print(f"   Projeto Alvo: {Fore.CYAN}{project_name}{Style.RESET_ALL}")
    print(f"   Diretório:    {Fore.CYAN}{abs_path}{Style.RESET_ALL}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    project_data = scan_directory(project_path)

    # Mesmo que a análise não encontre arquivos, prossegue para a IA (ela pode lidar com dados vazios)
    if not project_data:
        print(f"{Fore.YELLOW}⚠️ Análise do diretório não encontrou arquivos relevantes ou retornou vazia.")
        # Cria uma entrada mínima para enviar à IA, indicando que nada foi encontrado
        project_data = {
            "analise_info": {
                "status": "Diretório vazio ou sem arquivos analisáveis.",
                "timestamp": datetime.now().isoformat(),
                "diretorio_raiz": abs_path,
                "arquivos_processados": 0,
             }
        }

    # Salva o YAML de estrutura (mesmo que vazio/mínimo, pode ser útil para depuração)
    yaml_filename = "estrutura_projeto_analisado.yaml" # Nome fixo para facilitar
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hostname = platform.node().replace(' ', '_').lower().replace('.', '_') # Nome de arquivo mais seguro
        # Limita o tamanho do nome do projeto no arquivo yaml para evitar nomes muito longos
        safe_project_name = "".join(c if c.isalnum() else "_" for c in project_name)[:30]
        # Mantém nome fixo, mas poderia ser dinâmico:
        # yaml_filename = f"estrutura_{safe_project_name}_{hostname}_{timestamp}.yaml"

        with open(yaml_filename, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(project_data, yaml_file, allow_unicode=True, default_flow_style=False, sort_keys=False, width=150, indent=2)
        print(f"{Fore.GREEN}💾 Dados da estrutura do projeto (para análise da IA) salvos em: {yaml_filename}")
    except IOError as e:
         print(f"{Fore.RED}❗Erro de IO ao salvar YAML de estrutura: {e}")
    except Exception as e:
         print(f"{Fore.RED}❗Erro inesperado ao salvar YAML de estrutura: {e}")

    # Envia os dados (ou a mensagem de erro/vazio) para a IA gerar os relatórios
    gerar_relatorio_ia(project_data, project_name)

    end_time = time.time()
    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"⏱️ {Style.BRIGHT}Processo concluído em {end_time - start_time:.2f} segundos.{Style.NORMAL}")
    print(f"   Verifique os arquivos gerados:")
    print(f"   - Documentação: {Fore.CYAN}DOCUMENTACAO-PROJETO.md{Style.RESET_ALL}")
    print(f"   - Diagrama HTML: {Fore.CYAN}doc-web-diagram-data-hashtagh unica .html{Style.RESET_ALL}")
    print(f"   - Dados YAML:    {Fore.CYAN}{yaml_filename}{Style.RESET_ALL}")
    print(f"{'='*70}{Style.RESET_ALL}\n")


# --- Execução ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}❗ Processo interrompido pelo usuário (Ctrl+C).{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}{Style.BRIGHT}💥 Erro crítico inesperado na execução principal: {e}{Style.RESET_ALL}")
        import traceback
        print(f"{Fore.RED}--- Stack Trace ---")
        traceback.print_exc()
        print(f"{Fore.RED}-------------------{Style.RESET_ALL}")