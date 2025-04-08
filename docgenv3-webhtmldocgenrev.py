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
NOME_MODELO = "gemini-2.0-flash"
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
AI_TEMPERATURE = 0.6 # Um pouco mais determinístico para seguir instruções complexas
AI_TOP_P = 0.9
AI_TOP_K = 40
AI_MAX_TOKENS = 8000 # Ajuste conforme o modelo e a necessidade (modelos 1.5 suportam mais)

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

# --- TEMPLATE HTML (Android Arch - Apenas como GUIA DE ESTILO para a IA) ---
# Este template NÃO será mais usado diretamente para gerar o arquivo final.
# Ele será incluído no prompt para que a IA entenda o estilo visual desejado.
HTML_STYLE_GUIDE_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exemplo de Estilo de Diagrama de Arquitetura</title>
    <style>
        /* Reset Básico e Estilos Globais */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 74%); /* Tema Escuro */
            color: #e0e0e0; display: flex; justify-content: center;
            align-items: flex-start; min-height: 100vh; padding: 40px 20px; /* Reduzido padding */
            overflow-x: hidden;
        }
        /* Container Principal */
        .diagram-container {
            width: 95%; max-width: 1200px; /* Aumentado max-width */
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 20px; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 30px; display: flex; flex-direction: column; gap: 25px; /* Aumentado gap */
            perspective: 1500px;
        }
        /* Camadas (Layers) */
        .layer {
            padding: 20px 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3); transition: transform 0.3s ease, box-shadow 0.3s ease;
            transform-style: preserve-3d;
            /* Cores podem ser definidas por camada ou globalmente */
             background: linear-gradient(145deg, rgba(0, 169, 204, 0.3), rgba(0, 123, 255, 0.3)); /* Exemplo de cor base suave */
        }
        /* Títulos */
        .layer-title {
            font-size: 1.4em; font-weight: 600; color: #ffffff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 20px;
            text-align: center; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.25);
        }
        .sub-layer-title {
             font-size: 1.1em; font-weight: 500; color: rgba(255, 255, 255, 0.85);
             text-align: center; margin-top: 15px; margin-bottom: 15px; padding-bottom: 5px;
             border-bottom: 1px dashed rgba(255, 255, 255, 0.15);
        }
        /* Grid de Componentes */
        .components-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); /* Ajustado minmax */
            gap: 15px;
        }
        /* Componente Individual */
        .component {
            background-color: rgba(255, 255, 255, 0.1); color: #f0f8ff; padding: 15px 12px;
            border-radius: 10px; font-size: 0.88em; text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); border: 1px solid rgba(255, 255, 255, 0.15);
            cursor: default; transition: all 0.3s ease;
            opacity: 0; animation: fadeInItem 0.5s ease-out forwards;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            min-height: 65px; transform-style: preserve-3d; position: relative; /* Para tooltip */
        }
        /* Efeito Hover com Tooltip Simples (CSS Puro) */
        .component .component-desc {
            visibility: hidden; /* Escondido por padrão */
            width: 180px; background-color: rgba(0, 0, 0, 0.8); color: #fff;
            text-align: center; border-radius: 6px; padding: 8px;
            position: absolute; z-index: 10; bottom: 115%; /* Posição acima */
            left: 50%; margin-left: -90px; /* Centralizar */
            opacity: 0; transition: opacity 0.3s, visibility 0.3s; font-size: 0.8em;
            pointer-events: none; /* Não interfere com o hover no componente pai */
        }
        .component:hover {
            transform: scale(1.05) translateZ(10px); background-color: rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35); z-index: 5; /* Garante que fique sobre outros */
        }
        .component:hover .component-desc {
            visibility: visible; opacity: 1;
        }
        /* Animação de Entrada */
        @keyframes fadeInItem { from { opacity: 0; transform: translateY(8px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }
        /* Responsividade */
        @media (max-width: 768px) {
            .diagram-container { padding: 20px; } .layer { padding: 15px 20px; }
            .layer-title { font-size: 1.25em; } .sub-layer-title { font-size: 1.0em; }
            .components-grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }
            .component { font-size: 0.82em; padding: 12px 10px; min-height: 60px; }
            .component .component-desc { width: 150px; margin-left: -75px; font-size: 0.75em; }
        }
         @media (max-width: 480px) {
            body { padding: 20px 10px; }
            .diagram-container { padding: 15px; border-radius: 15px;} .layer { padding: 12px 15px; border-radius: 12px;}
            .layer-title { font-size: 1.1em; } .sub-layer-title { font-size: 0.9em; }
            .components-grid { grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; }
            .component { font-size: 0.78em; padding: 10px 8px; border-radius: 8px; min-height: 55px; }
            .component:hover { transform: scale(1.03) translateZ(5px); }
            .component .component-desc { display: none; } /* Esconde tooltip em telas muito pequenas */
         }
    </style>
</head>
<body>
    <div class="diagram-container">
        <!-- Exemplo de Camada -->
        <div class="layer" style="background: linear-gradient(145deg, rgba(92, 184, 92, 0.3), rgba(76, 174, 76, 0.3));">
            <div class="layer-title">Exemplo Camada 'Presentation'</div>
            <div class="components-grid">
                <div class="component" style="animation-delay: 0.1s;">
                    Componente A
                    <span class="component-desc">Descrição detalhada do Componente A. Pode incluir tecnologias, propósito, etc.</span>
                </div>
                <div class="component" style="animation-delay: 0.15s;">
                    Componente B
                    <span class="component-desc">Informações sobre o Componente B.</span>
                </div>
                <!-- Adicionar mais componentes conforme necessário -->
                 <div class="component" style="animation-delay: 0.2s;">Componente C<span class="component-desc">Detalhes C.</span></div>
                 <div class="component" style="animation-delay: 0.25s;">Componente D<span class="component-desc">Detalhes D.</span></div>
            </div>
        </div>
         <!-- Exemplo de outra Camada -->
        <div class="layer" style="background: linear-gradient(145deg, rgba(240, 173, 78, 0.3), rgba(236, 151, 31, 0.3));">
             <div class="layer-title">Exemplo Camada 'Business Logic'</div>
             <div class="components-grid">
                 <div class="component" style="animation-delay: 0.3s;">Serviço X<span class="component-desc">Processa regras de negócio X.</span></div>
                 <div class="component" style="animation-delay: 0.35s;">Serviço Y<span class="component-desc">Coordena tarefas Y.</span></div>
                 <div class="component" style="animation-delay: 0.4s;">Validador Z<span class="component-desc">Valida dados de entrada Z.</span></div>
             </div>
        </div>
        <!-- Footer (Exemplo, o script irá adicionar/gerenciar o real) -->
        <footer style="text-align: center; margin-top: 25px; font-size: 0.8em; color: rgba(224, 224, 224, 0.5); padding-top: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            Diagrama de Exemplo | Projeto Exemplo <br>
            Gerado em DD/MM/AAAA HH:MM:SS
        </footer>
    </div>
    <script>
        // Pequeno script para adicionar delays de animação dinamicamente se necessário
        document.addEventListener('DOMContentLoaded', () => {
            const components = document.querySelectorAll('.component');
            components.forEach((comp, index) => {
                if (!comp.style.animationDelay) { // Aplica delay se não definido inline
                    comp.style.animationDelay = `${index * 0.05}s`;
                }
            });
            // console.log("Diagrama de Exemplo Carregado e Animações Aplicadas.");
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
            # safety_settings=...
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
    **Sua Identidade:** Você é a Replika AI DocGen, um especialista sênior em arquitetura de software e documentação técnica, criado por Elias Andrade. Sua missão é analisar dados de projetos e gerar documentação clara e visualizações de arquitetura impressionantes.

    **Tarefa Dupla:** Analise os dados estruturados de um projeto (fornecido em YAML abaixo) e gere DOIS artefatos distintos:

    **1. Documentação Técnica em Markdown (`DOCUMENTACAO-PROJETO.md`):**
        *   **Objetivo:** Criar um README.md abrangente sobre o projeto '{project_name}', baseado EXCLUSIVAMENTE nos dados YAML fornecidos.
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
        *   **GUIA DE ESTILO VISUAL (CRÍTICO):** Use o template HTML fornecido abaixo **APENAS COMO INSPIRAÇÃO VISUAL**. **NÃO REPLIQUE O CONTEÚDO DO ANDROID.** Sua tarefa é criar um diagrama com visual semelhante:
            *   **Tema Escuro:** Use fundos escuros e texto claro.
            *   **Layout:** Use divs para representar camadas e componentes. Organize-os usando CSS Grid ou Flexbox de forma lógica.
            *   **Estilo Moderno:** Use gradientes sutis, cantos arredondados, sombras (`box-shadow`), efeito de vidro/blur (`backdrop-filter` se apropriado).
            *   **Interatividade (Hover):** Implemente tooltips ou descrições que aparecem ao passar o mouse sobre os componentes (`.component` e `.component-desc` no exemplo), mostrando detalhes extraídos do YAML (ex: docstrings de funções/classes, colunas de tabelas DB, tamanho/linhas de arquivos).
            *   **Responsividade:** O layout deve se adaptar a diferentes tamanhos de tela (use media queries como no exemplo).
            *   **Animações Sutis:** Use animações de entrada (`@keyframes`) para tornar a apresentação mais agradável.
            *   **Profundidade e Detalhe:** Esforce-se para criar um diagrama DETALHADO e PROFUNDO, cobrindo o máximo possível de aspectos relevantes do projeto identificados no YAML. Crie muitos blocos/componentes se a análise permitir. **Não use "..." ou "outros"**; represente explicitamente os elementos importantes.
        *   **Idioma:** Todo o texto visível no HTML (títulos, nomes de componentes, descrições) deve ser em **Português (pt-br)**.
        *   **Formato:** Gere o HTML completo como um bloco de código delimitado estritamente por ```html no início e ``` no final.
        *   **NÃO INCLUA UM RODAPÉ (FOOTER) PADRÃO NO HTML.** O script pós-processará e adicionará um rodapé se necessário.

    **Guia de Estilo Visual HTML (Use para INSPIRAÇÃO, NÃO COPIE o conteúdo):**
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

    **Instrução Final:** Gere primeiro o conteúdo Markdown completo para o `DOCUMENTACAO-PROJETO.md`. Após o Markdown, gere o bloco de código HTML completo e válido para o `doc-web-diagram-data-hashtagh unica .html`, seguindo TODAS as instruções acima, especialmente sobre usar os dados do YAML e o guia de estilo visual. Delimite o HTML estritamente com ```html ... ```.
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
    html_filename = "doc-web-diagram-data-hashtagh unica .html"
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
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hostname = platform.node().replace(' ', '_').lower().replace('.', '_') # Nome de arquivo mais seguro
        # Limita o tamanho do nome do projeto no arquivo yaml para evitar nomes muito longos
        safe_project_name = "".join(c if c.isalnum() else "_" for c in project_name)[:30]
        yaml_filename = f"estrutura_{safe_project_name}_{hostname}_{timestamp}.yaml"

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