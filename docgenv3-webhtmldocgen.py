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
import re
from typing import List, Dict, Any, Optional, Tuple, Union

# Inicializa o Colorama
init(autoreset=True)

# --- Chave de API ---
# (Mesma lógica de aviso e carregamento da versão anterior)
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    API_KEY = 'AIzaSyC7dAwSyLKaVO2E-PA6UaacLZ4aLGtrXbY' # PLACEHOLDER - MUITO CUIDADO!
    print(f"{Fore.RED}{'#'*66}")
    print(f"### {Fore.YELLOW}ALERTA:{Style.RESET_ALL}{Fore.RED} USANDO CHAVE DE API PADRÃO/PLACEHOLDER!                 ###")
    print(f"### Defina a variável de ambiente 'GEMINI_API_KEY' com sua chave ###")
    print(f"### real para segurança e funcionamento adequado em produção.    ###")
    print(f"### {Style.BRIGHT}NÃO USE ESTA CHAVE EM AMBIENTES PÚBLICOS OU COM DADOS REAIS.{Style.NORMAL} ###")
    print(f"{'#'*66}{Style.RESET_ALL}")
    # Descomente para exigir uma chave real:
    # exit(1)

try:
    genai.configure(api_key=API_KEY)
    print(f"{Fore.GREEN}🔑 Configuração da API do Gemini bem-sucedida.")
except Exception as e:
    print(f"{Fore.RED}ERRO FATAL: Falha ao configurar a API do Gemini.")
    print(f"{Fore.RED}   Erro: {e}")
    print(f"{Fore.RED}   Verifique a chave de API ('{API_KEY[:4]}...{API_KEY[-4:] if API_KEY else ''}') e a conectividade.")
    exit(1)

# Modelo da IA
NOME_MODELO = "gemini-2.0-flash"
print(f"{Fore.BLUE}ℹ️ Usando modelo de IA: {NOME_MODELO}")

# Lista de Ignorados
IGNORE_LIST = [
    os.path.basename(__file__), ".git", ".venv", "venv", "__pycache__",
    "node_modules", ".md", ".html", ".yaml", "requirements.txt", "LICENSE",
    "build", "dist", "*.log", ".idea", ".vscode", "target", "docs",
    "tests", "test", ".pytest_cache", ".mypy_cache", "site",
]

# --- Funções de Análise de Arquivos ---
# (get_file_size, count_lines, read_file_content, get_python_info,
#  get_sqlite_info, get_json_info, get_yaml_info, format_size)
# (Mantidas exatamente como na versão anterior - sem necessidade de alteração aqui)
# ... (cole aqui as funções de análise da versão anterior) ...
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
                # Usar buffer maior pode ser um pouco mais rápido para arquivos grandes
                return sum(1 for line in f)
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
            lines = []
            with open(file_path, 'r', encoding=enc) as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
            content = "".join(lines)
            return content # Retorna assim que conseguir ler
        except UnicodeDecodeError:
            continue
        except StopIteration: # Arquivo tem menos que max_lines, já leu tudo
             if content: # Se StopIteration ocorreu, lines já contém o arquivo todo
                 return content
             else: # Caso o arquivo esteja vazio
                 return ""
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
                functions.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
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
                 module_name = node.module if node.module else '.relative.'
                 imports.append({
                     "type": "from",
                     "module": module_name,
                     "level": node.level,
                     "names": [(alias.name, alias.asname) for alias in node.names]
                 })
        return {"functions": functions, "classes": classes, "imports": imports, "analysis_status": "success"}
    except SyntaxError as e:
        return {"error": f"Erro de sintaxe Python: {e}", "lineno": e.lineno, "col_offset": e.offset, "analysis_status": "failed_syntax"}
    except Exception as e:
        return {"error": f"Erro inesperado ao analisar AST: {e}", "analysis_status": "failed_ast"}

def get_sqlite_info(file_path: str, max_rows: int = 5) -> Dict[str, Any]:
    """Extrai esquema e amostra de dados de um banco de dados SQLite."""
    db_uri = f'file:{file_path}?mode=ro'
    tables_data: Dict[str, Any] = {}
    try:
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            conn.close()
            return {"info": "Banco de dados vazio ou sem tabelas de usuário.", "tables": {}, "analysis_status": "success_empty"}

        for table_name in tables:
            try:
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns = [{"name": row[1], "type": row[2], "notnull": bool(row[3]), "default": row[4], "pk": bool(row[5])}
                           for row in cursor.fetchall()]
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
                    row_count = cursor.fetchone()[0]
                except sqlite3.OperationalError as count_e:
                    print(f"{Fore.YELLOW}Aviso: Não foi possível contar linhas da tabela SQLite '{table_name}': {count_e}{Style.RESET_ALL}")
                    row_count = "Erro ao Contar"

                sample_rows_safe = []
                if isinstance(row_count, int) and row_count > 0:
                    try:
                        cursor.execute(f"SELECT * FROM '{table_name}' LIMIT {max_rows};")
                        sample_rows = cursor.fetchall()
                        sample_rows_safe = [
                            tuple(f"<bytes {len(item)}>" if isinstance(item, bytes) else item for item in row)
                            for row in sample_rows
                        ]
                    except sqlite3.OperationalError as select_e:
                        print(f"{Fore.YELLOW}Aviso: Não foi possível obter amostra da tabela SQLite '{table_name}': {select_e}{Style.RESET_ALL}")
                        sample_rows_safe = ["Erro ao Ler Amostra"]
                elif row_count == 0:
                     sample_rows_safe = [] # Lista vazia é apropriado
                # else: # row_count é "Erro ao Contar"
                #     sample_rows_safe já é []

                tables_data[table_name] = {
                    "columns": columns,
                    "total_rows": row_count,
                    "sample_rows": sample_rows_safe
                }
            except sqlite3.OperationalError as table_e:
                print(f"{Fore.YELLOW}Aviso: Erro operacional ao processar tabela SQLite '{table_name}': {table_e}{Style.RESET_ALL}")
                tables_data[table_name] = {"error": f"Erro operacional: {table_e}", "columns": [], "total_rows": "Erro", "sample_rows": []}

        conn.close()
        return {"tables": tables_data, "analysis_status": "success"}
    except sqlite3.OperationalError as conn_e:
         return {"error": f"Erro SQLite ao conectar (arquivo inválido, bloqueado ou sem permissão?): {conn_e}", "analysis_status": "failed_connect"}
    except Exception as e:
        return {"error": f"Erro inesperado ao processar SQLite: {e}", "analysis_status": "failed_other"}

def get_json_info(file_path: str) -> Dict[str, Any]:
    """Obtém informações básicas sobre um arquivo JSON."""
    info: Dict[str, Any] = {}
    file_size = get_file_size(file_path)
    line_count = count_lines(file_path)

    info["tamanho_formatado"] = format_size(file_size)
    info["tamanho_bytes"] = file_size if file_size >= 0 else "Erro"
    info["numero_de_linhas"] = line_count if line_count >= 0 else ("Erro Codificação" if line_count == -2 else "Erro Leitura")

    structure_type = "desconhecido"
    preview_error = None
    if file_size > 0 and file_size < 5 * 1024 * 1024:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                preview = f.read(2048)
                try:
                    # Tenta parsear o preview (pode falhar se cortar no meio)
                    data = json.loads(preview)
                    if isinstance(data, list): structure_type = "lista"
                    elif isinstance(data, dict): structure_type = "objeto"
                    else: structure_type = "outro (valor primitivo?)"
                except json.JSONDecodeError as json_e:
                    # Tenta uma abordagem mais robusta para preview cortado
                    if preview.strip().startswith('['): structure_type = "lista (preview incompleto?)"
                    elif preview.strip().startswith('{'): structure_type = "objeto (preview incompleto?)"
                    else: structure_type = f"não foi possível parsear preview ({json_e})"
        except Exception as read_e:
            structure_type = "erro ao ler preview"
            preview_error = str(read_e)

    info["tipo_estrutura_inferida"] = structure_type
    if preview_error: info["preview_error"] = preview_error
    info["analysis_status"] = "success"
    return info

def get_yaml_info(file_path: str) -> Dict[str, Any]:
    """Obtém informações básicas sobre um arquivo YAML."""
    info: Dict[str, Any] = {}
    file_size = get_file_size(file_path)
    line_count = count_lines(file_path)

    info["tamanho_formatado"] = format_size(file_size)
    info["tamanho_bytes"] = file_size if file_size >= 0 else "Erro"
    info["numero_de_linhas"] = line_count if line_count >= 0 else ("Erro Codificação" if line_count == -2 else "Erro Leitura")
    info["analysis_status"] = "success"
    # Poderia adicionar parse YAML do preview aqui também, com tratamento de erro
    return info

def format_size(size_in_bytes: int) -> str:
    """Formata o tamanho em bytes para KB, MB, GB."""
    if size_in_bytes < 0: return "Erro"
    if size_in_bytes == 0: return "0 Bytes"
    if size_in_bytes < 1024: return f"{size_in_bytes} Bytes"
    if size_in_bytes < 1024**2: return f"{size_in_bytes/1024:.2f} KB"
    if size_in_bytes < 1024**3: return f"{size_in_bytes/(1024**2):.2f} MB"
    return f"{size_in_bytes/(1024**3):.2f} GB"

# --- Configurações e Funções da IA ---
AI_TEMPERATURE = 0.5 # Mais focado ainda para seguir instruções detalhadas
AI_TOP_P = 0.9
AI_TOP_K = 40
AI_MAX_TOKENS = 16000 # Aumentar se o modelo permitir e precisar de mais detalhes

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
    )

# (Função enviar_mensagem mantida da versão anterior)
def enviar_mensagem(sessao_chat: genai.ChatSession, mensagem: str) -> Optional[str]:
    """Envia uma mensagem para a sessão de chat da IA com retentativas."""
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 7

    for attempt in range(MAX_RETRIES):
        try:
            print(f"{Fore.YELLOW}🧠 Enviando prompt para IA ({NOME_MODELO}) - Tentativa {attempt + 1}/{MAX_RETRIES}...")
            start_ai_time = time.time()
            resposta = sessao_chat.send_message(mensagem)
            end_ai_time = time.time()
            print(f"{Fore.GREEN}✅ Resposta da IA recebida em {end_ai_time - start_ai_time:.2f}s.")

            if not resposta.candidates:
                block_reason = "Não especificado"
                safety_ratings = "N/A"
                try:
                    block_reason = resposta.prompt_feedback.block_reason.name
                    safety_ratings = str(resposta.prompt_feedback.safety_ratings)
                except AttributeError: pass
                print(f"{Fore.RED}❗ Resposta da IA bloqueada ou sem candidatos.")
                print(f"{Fore.RED}   Razão: {block_reason}")
                print(f"{Fore.RED}   Safety Ratings: {safety_ratings}")
                if block_reason != "BLOCK_REASON_UNSPECIFIED": return None
                return None

            if resposta.candidates[0].content and resposta.candidates[0].content.parts:
                texto_resposta = "".join(part.text for part in resposta.candidates[0].content.parts if hasattr(part, 'text'))
                if texto_resposta.strip():
                    try:
                        response_safety = str(resposta.candidates[0].safety_ratings)
                        if "BLOCK" in response_safety.upper():
                             print(f"{Fore.YELLOW}⚠️ Resposta da IA contém conteúdo potencialmente problemático (Safety: {response_safety}).")
                             # return f"Erro: Resposta da IA marcada com problemas de segurança: {response_safety}"
                    except AttributeError: pass
                    return texto_resposta
                else:
                    print(f"{Fore.YELLOW}⚠️ Resposta da IA está vazia ou não contém texto.")
                    return ""
            else:
                 print(f"{Fore.YELLOW}⚠️ Resposta da IA não contém partes de conteúdo ou texto.")
                 return ""

        except Exception as e:
            print(f"{Fore.RED}❗Erro ao comunicar com a IA (Tentativa {attempt + 1}/{MAX_RETRIES}): {e}")
            if "API key not valid" in str(e):
                print(f"{Fore.RED}   Erro de chave de API. Interrompendo retentativas.")
                return None
            # Adicionar tratamento para outros erros específicos se necessário (ex: RateLimit, Quota)
            # if "429" in str(e): # Exemplo de Rate Limit
            #     print(f"{Fore.RED}   Rate limit atingido. Aumentando delay...")
            #     RETRY_DELAY_SECONDS *= 1.5 # Aumenta delay exponencialmente (simples)

            if attempt < MAX_RETRIES - 1:
                print(f"{Fore.CYAN}    Aguardando {RETRY_DELAY_SECONDS:.1f}s para tentar novamente...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"{Fore.RED}❗ Falha ao comunicar com a IA após {MAX_RETRIES} tentativas.")
                return None
    return None

# (Função scan_directory mantida da versão anterior)
def scan_directory(root_path: str = ".") -> Dict[str, Any]:
    """Varre o diretório raiz, analisa arquivos e retorna um dicionário com a estrutura."""
    report: Dict[str, Any] = {}
    abs_root_path = os.path.abspath(root_path)
    print(f"{Fore.CYAN}🔍 Varrendo diretório: {abs_root_path}")
    print(f"{Fore.CYAN}   Ignorando nomes/extensões: {IGNORE_LIST}")
    items_processed, items_ignored, errors_occurred = 0, 0, 0

    ignore_extensions = {item for item in IGNORE_LIST if item.startswith('.')}
    ignore_filenames = {item for item in IGNORE_LIST if not item.startswith('*') and not item.startswith('.')}
    ignore_patterns = [item for item in IGNORE_LIST if item.startswith('*')]

    for current_root, dirs, files in os.walk(root_path, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_filenames and not d.startswith('.') and d not in IGNORE_LIST]

        relative_root = os.path.relpath(current_root, root_path)
        if relative_root == '.': relative_root = 'Raiz do Projeto'

        root_report: Dict[str, Any] = {}

        for filename in files:
            file_path = os.path.join(current_root, filename)
            _, file_ext = os.path.splitext(filename)
            file_ext_lower = file_ext.lower()
            filename_lower = filename.lower()

            ignore_file = False
            if filename in ignore_filenames or filename_lower in ignore_filenames: ignore_file = True
            elif file_ext_lower in ignore_extensions: ignore_file = True
            elif filename.startswith('.'): ignore_file = True
            else:
                 if any(pat.endswith(file_ext_lower) for pat in ignore_patterns if pat.startswith('*.')): ignore_file = True

            if ignore_file:
                items_ignored += 1
                continue

            items_processed += 1
            file_info: Dict[str, Any] = {}
            has_error = False

            try:
                file_size = get_file_size(file_path)
                line_count = count_lines(file_path)

                file_info = {
                    "caminho_relativo": os.path.relpath(file_path, root_path),
                    "tamanho_bytes": file_size if file_size >= 0 else -1,
                    "tamanho_formatado": format_size(file_size),
                    "numero_de_linhas": line_count if line_count >= 0 else (-2 if line_count == -2 else -1),
                    "extensao": file_ext_lower
                }

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
                    file_info["json_info"] = json_analysis
                elif file_ext_lower in (".yaml", ".yml"):
                    yaml_analysis = get_yaml_info(file_path)
                    file_info["yaml_info"] = yaml_analysis
                elif line_count > 0 and file_size > 0 and file_size < 1 * 1024 * 1024:
                    preview = read_file_content(file_path, max_lines=15)
                    if preview is not None: file_info["content_preview"] = preview

                if has_error:
                    errors_occurred += 1
                    # print(f"{Fore.RED}  -> Erro ao analisar: {file_info.get('caminho_relativo')}{Style.RESET_ALL}")
                # else:
                #     print(f"{Fore.GREEN}  + Ok: {file_info.get('caminho_relativo')}{Style.RESET_ALL}")

            except Exception as e:
                errors_occurred += 1
                print(f"{Fore.RED}  -> Erro INESPERADO ao processar {filename}: {e}")
                file_info["processing_error"] = str(e)

            root_report[filename] = file_info

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

# (Template HTML Guia de Estilo mantido da versão anterior)
HTML_STYLE_GUIDE_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exemplo de Estilo de Diagrama de Arquitetura</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 74%); color: #e0e0e0; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; padding: 40px 20px; overflow-x: hidden; }
        .diagram-container { width: 95%; max-width: 1200px; background-color: rgba(255, 255, 255, 0.05); border-radius: 20px; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.15); padding: 30px; display: flex; flex-direction: column; gap: 25px; perspective: 1500px; }
        .layer { padding: 20px 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.3); transition: transform 0.3s ease, box-shadow 0.3s ease; transform-style: preserve-3d; background: linear-gradient(145deg, rgba(0, 169, 204, 0.3), rgba(0, 123, 255, 0.3)); }
        .layer-title { font-size: 1.4em; font-weight: 600; color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.3); margin-bottom: 20px; text-align: center; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.25); }
        .sub-layer-title { font-size: 1.1em; font-weight: 500; color: rgba(255, 255, 255, 0.85); text-align: center; margin-top: 15px; margin-bottom: 15px; padding-bottom: 5px; border-bottom: 1px dashed rgba(255, 255, 255, 0.15); }
        .components-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; }
        .component { background-color: rgba(255, 255, 255, 0.1); color: #f0f8ff; padding: 15px 12px; border-radius: 10px; font-size: 0.88em; text-align: center; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); border: 1px solid rgba(255, 255, 255, 0.15); cursor: default; transition: all 0.3s ease; opacity: 0; animation: fadeInItem 0.5s ease-out forwards; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 65px; transform-style: preserve-3d; position: relative; }
        .component .component-desc { visibility: hidden; width: 200px; background-color: rgba(10, 20, 40, 0.9); color: #fff; text-align: left; border-radius: 6px; padding: 10px; position: absolute; z-index: 10; bottom: 115%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s, visibility 0.3s; font-size: 0.8em; pointer-events: none; border: 1px solid rgba(255, 255, 255, 0.2); }
        .component:hover { transform: scale(1.05) translateZ(10px); background-color: rgba(255, 255, 255, 0.2); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35); z-index: 5; }
        .component:hover .component-desc { visibility: visible; opacity: 1; }
        @keyframes fadeInItem { from { opacity: 0; transform: translateY(8px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }
        @media (max-width: 768px) { .diagram-container { padding: 20px; } .layer { padding: 15px 20px; } .layer-title { font-size: 1.25em; } .sub-layer-title { font-size: 1.0em; } .components-grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; } .component { font-size: 0.82em; padding: 12px 10px; min-height: 60px; } .component .component-desc { width: 160px; margin-left: -80px; font-size: 0.78em; } }
        @media (max-width: 480px) { body { padding: 20px 10px; } .diagram-container { padding: 15px; border-radius: 15px;} .layer { padding: 12px 15px; border-radius: 12px;} .layer-title { font-size: 1.1em; } .sub-layer-title { font-size: 0.9em; } .components-grid { grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; } .component { font-size: 0.78em; padding: 10px 8px; border-radius: 8px; min-height: 55px; } .component:hover { transform: scale(1.03) translateZ(5px); } .component .component-desc { display: none; } }
    </style>
</head>
<body><div class="diagram-container">
<div class="layer" style="background: linear-gradient(145deg, rgba(92, 184, 92, 0.3), rgba(76, 174, 76, 0.3));"><div class="layer-title">Exemplo Camada 'UI/Presentation'</div><div class="components-grid">
<div class="component" style="animation-delay: 0.1s;">Componente View X<span class="component-desc">Renderiza a tela X. Usa dados do ViewModel Y. Interage com o usuário via eventos A, B. (Detalhes específicos do projeto aqui)</span></div>
<div class="component" style="animation-delay: 0.15s;">ViewModel Y<span class="component-desc">Prepara e fornece dados para a View X. Observa mudanças no Repositório Z. Expõe estado (LiveData/StateFlow). Lógica de UI simples.</span></div>
<div class="component" style="animation-delay: 0.2s;">Adapter Lista<span class="component-desc">Responsável por exibir itens de uma lista na UI. Usa ViewHolder padrão. Conectado ao ViewModel Y.</span></div>
</div></div>
<div class="layer" style="background: linear-gradient(145deg, rgba(240, 173, 78, 0.3), rgba(236, 151, 31, 0.3));"><div class="layer-title">Exemplo Camada 'Domain/Business Logic'</div><div class="components-grid">
<div class="component" style="animation-delay: 0.3s;">UseCase 'Processar Pedido'<span class="component-desc">Orquestra a lógica de negócio para processar um pedido. Usa Repositório Z e Serviço Externo W. Valida dados.</span></div>
<div class="component" style="animation-delay: 0.35s;">Repositório Z (Interface)<span class="component-desc">Define contrato para acesso aos dados de Pedidos. Implementado na camada de Dados.</span></div>
<div class="component" style="animation-delay: 0.4s;">Entidade 'Pedido'<span class="component-desc">Representa um pedido no domínio. Contém regras de negócio intrínsecas (validações).</span></div>
</div></div>
<div class="layer" style="background: linear-gradient(145deg, rgba(217, 83, 79, 0.3), rgba(201, 48, 44, 0.3));"><div class="layer-title">Exemplo Camada 'Data/Infrastructure'</div><div class="components-grid">
<div class="component" style="animation-delay: 0.45s;">Repositório Z (Impl)<span class="component-desc">Implementação concreta do Repositório Z. Usa DataSource Local (SQLite) e DataSource Remoto (API). Gerencia cache.</span></div>
<div class="component" style="animation-delay: 0.5s;">DataSource Local (SQLite)<span class="component-desc">Acessa a tabela 'pedidos' no banco `main.db`. Usa Room/SQLAlchemy. Contém DAOs/mappers.</span></div>
<div class="component" style="animation-delay: 0.55s;">DataSource Remoto (API)<span class="component-desc">Comunica com a API REST `/orders`. Usa Retrofit/Requests. Trata erros de rede. DTOs definidos.</span></div>
<div class="component" style="animation-delay: 0.6s;">Serviço Externo W (Client)<span class="component-desc">Cliente para o serviço de Pagamentos. SDK fornecido. Trata autenticação.</span></div>
</div></div>
</div></body></html>
"""

# --- Função Principal de Geração (com prompt aprimorado) ---

def gerar_relatorio_ia(
    project_data: Dict[str, Any],
    project_name: str,
    user_base_prompt: Optional[str] = None # Novo parâmetro
) -> None:
    """
    Gera relatórios: Markdown (descrição profunda) e HTML (diagrama detalhado).
    Utiliza um prompt base do usuário (opcional) e instruções aprimoradas para a IA.
    """
    try:
        model = genai.GenerativeModel(
            model_name=NOME_MODELO,
            generation_config=configurar_geracao(),
        )
        sessao_chat = model.start_chat(history=[])
        print(f"{Fore.CYAN}🤖 Iniciando sessão de chat com {NOME_MODELO}...")
    except Exception as e:
        print(f"{Fore.RED}❗Erro fatal ao iniciar sessão de chat com a IA: {e}")
        return

    generation_info = {
        "os_system": platform.system(), "os_release": platform.release(),
        "hostname": platform.node(), "generation_timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(), "ai_model": NOME_MODELO,
    }

    # Prepara a seção do prompt base do usuário
    user_prompt_section = ""
    if user_base_prompt and user_base_prompt.strip():
        print(f"{Fore.MAGENTA}ℹ️ Usando prompt base fornecido pelo usuário.")
        user_prompt_section = f"""
--- INSTRUÇÕES ADICIONAIS DO USUÁRIO ---
{user_base_prompt.strip()}
--- FIM DAS INSTRUÇÕES DO USUÁRIO ---

"""
    else:
        print(f"{Fore.MAGENTA}ℹ️ Nenhum prompt base do usuário fornecido.")

    # Constrói o prompt final com instruções super aprimoradas
    prompt = f"""{user_prompt_section}
**Sua Identidade e Missão:** Você é a Replika AI DocGen, um Arquiteto de Software e Documentador Técnico Sênior meticuloso, criado por Elias Andrade. Sua missão é analisar PROFUNDAMENTE os dados estruturados de um projeto de software (fornecidos em YAML) e gerar documentação e diagramas de arquitetura que sejam EXTREMAMENTE DETALHADOS, CONTEXTUALIZADOS e 100% FIÉIS aos dados fornecidos. **ABANDONE QUALQUER SUPERFICIALIDADE.**

**Princípios Fundamentais (Aplicáveis a AMBAS as Tarefas Abaixo):**

*   **PROFUNDIDADE ABSOLUTA:** Vá muito além do óbvio. Não diga apenas "usa Flask". Explique *COMO* o Flask é usado *NESTE PROJETO ESPECÍFICO*: quais rotas existem (identificadas na análise AST), como os blueprints (se houver) estão organizados, qual o propósito das funções associadas às rotas. Para cada componente significativo (módulo Python, classe, função importante, tabela de banco de dados, arquivo de configuração chave), explique:
    *   **O quê:** O que é este elemento?
    *   **Porquê:** Qual o seu propósito *dentro da arquitetura deste projeto*? Que problema ele resolve aqui?
    *   **Como:** Como ele funciona internamente (com base no código analisado - funções, imports)? Como ele interage com outros componentes *deste projeto*?
*   **FIDELIDADE TOTAL AO PROJETO (100%):** Sua análise e descrição devem se basear *EXCLUSIVAMENTE* nos dados fornecidos no bloco YAML abaixo. **NUNCA INVENTE** funcionalidades, componentes, tecnologias ou relações que não estejam explicitamente presentes ou fortemente implícitas nos dados. **NÃO FAÇA SUPOSIÇÕES.** Se algo não está nos dados, não mencione.
*   **CONTEXTUALIZAÇÃO É TUDO:** Cada descrição deve estar firmemente ancorada no contexto do projeto '{project_name}'. Evite definições genéricas de tecnologias. O foco é *a aplicação da tecnologia neste projeto*.
*   **ZERO SUPERFICIALIDADE:** Rejeite descrições rasas. "Script que faz X" não é suficiente. Detalhe *como* ele faz X, quais dados ele usa/gera, quais outros scripts/módulos ele chama *neste projeto*.
*   **COMPLETUDE E DETALHE:** Cubra *todos* os aspectos relevantes identificados na análise YAML. **NÃO ABREVIE.** **NÃO USE reticências ("...") ou placeholders como "outros".** Se múltiplos arquivos/funções têm propósitos similares, descreva os mais importantes explicitamente e mencione o padrão, se aplicável, mas evite agrupar genericamente sem detalhar exemplos chave. Seja METICULOSO.

**Tarefa Dupla (Com Base nos Princípios Acima):**

    *   **Objetivo:** GerarEXTREMAMENTE DETALHADO e PROFUNDO sobre o projeto '{project_name}', aplicando rigorosamente os princípios fundamentais acima e usando *apenas* os dados do YAML.
    *   **Conteúdo Mandatório (com Profundidade):**
        *   **🎯 Visão Geral e Propósito Contextualizado:** Qual o objetivo central *deste projeto específico*? Que problema real ele visa solucionar, conforme inferido dos dados?
        *   **⚙️ Arquitetura e Componentes Principais:** Descreva a arquitetura inferida (camadas? módulos? serviços?). Para cada componente chave (diretórios importantes, arquivos Python centrais, bancos de dados), explique seu papel e funcionamento *neste projeto* (O quê, Porquê, Como). Use os dados da análise AST (funções, classes, imports) para substanciar a descrição.
        *   **🛠️ Tecnologias em Ação:** Liste as tecnologias (linguagens, libs, frameworks, DBs) identificadas. Para cada uma, explique *como ela é usada especificamente aqui*. Ex: "Usa SQLite através do arquivo `data/main.db` para armazenar X, Y, Z (tabelas identificadas), acessado principalmente pelo módulo `database_manager.py` (funções A, B)".
        *   **📁 Estrutura Detalhada:** Apresente a estrutura de diretórios relevante. Para arquivos/diretórios importantes, explique seu conteúdo e propósito *no contexto do projeto*.
        *   **⚠️ Pontos de Atenção e Erros:** Se o YAML indica erros (syntax, DB, leitura), detalhe-os e explique o possível impacto *neste projeto*.
        *   **🚀 Fluxo de Execução (Inferido):** Se possível inferir dos arquivos (`main.py`, scripts, etc.), descreva o fluxo principal de execução ou como interagir com o projeto, referenciando os arquivos/funções específicas identificadas.
    *   **Estilo:** Técnico, ultra-detalhado, claro. Use Markdown avançado se necessário (tabelas, blocos de código para exemplos curtos de *nomes* de arquivos/funções, NUNCA código fonte real).
    *   **Restrição CRÍTICA:** JAMAIS copie blocos de código fonte. Descreva funcionalidade e estrutura baseada na análise AST e outros metadados do YAML.
    *   **Autoria:** Inclua nota no final: "Documentação detalhada gerada por Replika AI DocGen (Elias Andrade) em {generation_info['generation_timestamp']}."

**2. Diagrama de Arquitetura HTML (`doc-web-diagram-data-hashtagh unica .html`):**
    *   **Objetivo:** Gerar um ARQUIVO HTML COMPLETO (`<!DOCTYPE html>...</html>`) que visualize a arquitetura do projeto '{project_name}' de forma DETALHADA e FIEL, baseando-se *estritamente* nos dados do YAML e nos princípios fundamentais.
    *   **Conteúdo do Diagrama (Mapeado 1:1 com o YAML):** O diagrama DEVE representar visualmente os componentes e relações *reais* identificados no YAML. Cada elemento visual (bloco, camada, componente, linha de conexão implícita) PRECISA corresponder a algo concreto nos dados:
        *   Diretórios/Módulos Chave: Represente visualmente os diretórios ou arquivos .py significativos (com base na análise AST - presença de muitas classes/funções).
        *   Camadas Inferidas (SE EVIDENTE no YAML): Se a estrutura de diretórios/nomes sugerir camadas (ex: `ui/`, `data/`, `domain/`), represente-as. Se não for óbvio, represente a estrutura de módulos.
        *   Componentes de Dados: Inclua blocos para bancos de dados SQLite (mostrando tabelas chave), arquivos JSON/YAML importantes (indicando seu propósito inferido).
        *   Relações e Dependências (Inferidas): Use a proximidade visual, linhas (se possível gerar SVG/Canvas via JS, senão descrição no tooltip) ou descrições nos tooltips para indicar quais módulos usam outros (baseado nos imports do AST), ou quais módulos acessam quais bancos/arquivos de dados.
    *   **GUIA DE ESTILO VISUAL (Usar como INSPIRAÇÃO para o layout e estilo, NÃO para o conteúdo):** Siga o *estilo visual* do template HTML abaixo, mas popule-o com a arquitetura *real* do projeto '{project_name}'.
        *   **Estilo:** Tema escuro, moderno, gradientes, sombras, cantos arredondados, possivelmente `backdrop-filter`.
        *   **Interatividade VITAL:** Cada componente visual DEVE ter um tooltip (`.component-desc`) que aparece no hover, mostrando DETALHES EXTRAÍDOS DO YAML:
            *   Para módulos/arquivos .py: Nomes de classes/funções importantes, resumo do docstring (se houver), imports chave.
            *   Para DBs: Nomes das tabelas, número de linhas (se disponível), nomes das colunas principais.
            *   Para JSON/YAML: Propósito inferido, tamanho/linhas.
            *   Use `<br>` ou listas `<ul>` dentro do tooltip para formatar múltiplos detalhes.
        *   **Layout:** Use CSS Grid/Flexbox para organizar os componentes logicamente.
        *   **Responsividade:** Adapte o layout para diferentes telas.
        *   **Animações Sutis:** Use `@keyframes`.
        *   **DETALHE E COMPLETUDE VISUAL:** Represente o máximo possível de elementos relevantes do YAML. **NÃO USE "..." ou "outros".** Crie tantos blocos quantos forem necessários para cobrir a complexidade revelada pelo YAML.
    *   **Idioma:** Todo texto visível no HTML (títulos, nomes, tooltips) em **Português (pt-br)**.
    *   **Formato:** Bloco de código HTML completo e válido (```html ... ```).
    *   **SEM RODAPÉ:** NÃO adicione um footer ao HTML.

    **Guia de Estilo Visual HTML (INSPIRAÇÃO VISUAL APENAS):**
    ```html
    {HTML_STYLE_GUIDE_TEMPLATE}
    ```

    **Dados da Análise do Projeto '{project_name}' (BASE ESTRITA PARA AMBAS AS TAREFAS):**
    ```yaml
    # Dados da análise detalhada do projeto: {project_name}
    # Gerado em: {generation_info['generation_timestamp']}
    # Ambiente: {generation_info['hostname']} ({generation_info['os_system']} {generation_info['os_release']}, Python {generation_info['python_version']})
    # Modelo IA: {generation_info['ai_model']}
    ---
    {yaml.dump(project_data, allow_unicode=True, default_flow_style=False, sort_keys=False, width=150, indent=2)}
    ---
    ```

    **Instrução Final:** Priorize a PROFUNDIDADE, CONTEXTUALIZAÇÃO e FIDELIDADE ABSOLUTA aos dados YAML. Gere primeiro o conteúdo Markdown completo e ultra-detalhado. Em seguida, gere o bloco de código HTML completo e válido, mapeando visualmente os dados do YAML e seguindo o guia de estilo. Delimite o HTML estritamente com ```html ... ```. **EVITE SUPERFICIALIDADE A TODO CUSTO.**
    """

    resposta_completa = enviar_mensagem(sessao_chat, prompt)

    if not resposta_completa:
        print(f"{Fore.RED}❗Erro: Nenhuma resposta recebida da IA ou resposta vazia/bloqueada.")
        return

    print(f"{Fore.CYAN}📝 Processando a resposta da IA...")

    # (Extração de HTML e Markdown mantida da versão anterior)
    html_pattern = re.compile(r"```html\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    html_match = html_pattern.search(resposta_completa)

    markdown_content = resposta_completa
    html_content: Optional[str] = None
    html_filename = "doc-web-diagram-data-hashtagh unica .html"
    markdown_filename = "DOCUMENTACAO-PROJETO.md"

    if html_match:
        html_content = html_match.group(1).strip()
        markdown_content = html_pattern.sub("", resposta_completa).strip()
        print(f"{Fore.GREEN}📄 Bloco HTML encontrado e extraído.")
        if not html_content:
            print(f"{Fore.YELLOW}⚠️ O bloco HTML extraído está vazio.")
            html_content = None
        elif not (html_content.lower().startswith("<!doctype") or html_content.lower().startswith("<html")):
             print(f"{Fore.YELLOW}⚠️ O HTML extraído não parece ser um documento completo (sem DOCTYPE/html inicial).")
        # Não verificar mais o tamanho mínimo, deixar a IA decidir o quão detalhado ser
    else:
        print(f"{Fore.YELLOW}⚠️ Bloco HTML (```html ... ```) não encontrado na resposta. Todo o conteúdo será salvo como Markdown.")

    # --- Salvando Markdown ---
    try:
        autoria_ia = f"Documentação detalhada gerada por Replika AI DocGen (Elias Andrade) em {generation_info['generation_timestamp']}"
        if autoria_ia not in markdown_content:
            markdown_content += f"\n\n---\n*{autoria_ia}*"
        with open(markdown_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown_content)
        print(f"{Fore.GREEN}✅ Arquivo Markdown salvo: {markdown_filename}")
    except Exception as e:
        print(f"{Fore.RED}❗Erro ao salvar {markdown_filename}: {e}")

    # --- Salvando e Pós-Processando HTML ---
    if html_content:
        try:
            final_html = html_content
            now_str = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            hostname_str = generation_info['hostname']
            os_str = generation_info['os_system']

            # 1. Atualizar Título
            title_tag = f"<title>Diagrama Arquitetura Detalhado | {project_name} | Replika AI</title>"
            if re.search(r"<head.*?>", final_html, re.IGNORECASE | re.DOTALL):
                 if re.search(r"<title>.*?</title>", final_html, re.IGNORECASE | re.DOTALL):
                     final_html = re.sub(r"<title>.*?</title>", title_tag, final_html, count=1, flags=re.IGNORECASE | re.DOTALL)
                 else:
                     final_html = re.sub(r"(<head.*?>)", r"\1\n    " + title_tag, final_html, count=1, flags=re.IGNORECASE | re.DOTALL)
            else:
                 final_html = final_html.replace("<html>", f"<html>\n<head>\n    {title_tag}\n</head>", 1)

            # 2. Lógica do Rodapé (Comentar IA, Adicionar Nosso)
            footer_added_by_script = False
            footer_pattern = re.compile(
                r"<footer.*?>(.*?(?:Gerado por|Diagrama|Projeto|Replika AI|{project_name}).*?)</footer>".format(project_name=re.escape(project_name)),
                re.IGNORECASE | re.DOTALL
            )
            commented_footer_pattern = re.compile(r"<!--\s*<footer.*?Gerado por Replika AI DocGen.*?</footer>\s*-->", re.IGNORECASE | re.DOTALL)
            footer_match = footer_pattern.search(final_html)

            if footer_match and not commented_footer_pattern.search(footer_match.group(0)):
                print(f"{Fore.YELLOW}⚠️ Rodapé detectado na resposta da IA. Comentando-o.")
                ai_footer_content = footer_match.group(0)
                commented_footer = f"\n<!-- Rodapé original da IA (comentado pelo script):\n{ai_footer_content}\n-->\n"
                final_html = final_html.replace(ai_footer_content, commented_footer, 1)

            # Adiciona nosso rodapé SE nenhum foi encontrado ou o da IA foi comentado
            if not footer_match or (footer_match and commented_footer_pattern.search(commented_footer)):
                print(f"{Fore.CYAN}ℹ️ Adicionando rodapé padrão ao HTML.")
                footer_html = f"""
        <!-- Footer Adicionado pelo Script -->
        <footer style="text-align: center; margin-top: 30px; font-size: 0.85em; color: rgba(224, 224, 224, 0.6); padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            Diagrama da Arquitetura | Projeto: {project_name} <br>
            Gerado por Replika AI DocGen em {now_str} | Host: {hostname_str} ({os_str}) | Modelo IA: {NOME_MODELO}
        </footer>"""
                # Tenta inserir o footer de forma inteligente
                inserted = False
                if '</body>' in final_html.lower():
                    final_html = re.sub(r"</body>", f"{footer_html}\n</body>", final_html, count=1, flags=re.IGNORECASE)
                    inserted = True
                if not inserted and '</script>' in final_html.lower():
                     parts = final_html.rsplit('</script>', 1)
                     if len(parts) == 2:
                         final_html = parts[0] + '</script>' + footer_html + parts[1]
                         inserted = True
                if not inserted and '</html>' in final_html.lower():
                     final_html = re.sub(r"</html>", f"{footer_html}\n</html>", final_html, count=1, flags=re.IGNORECASE)
                     inserted = True
                if not inserted:
                    final_html += footer_html

            # 3. Salvar o HTML
            with open(html_filename, 'w', encoding='utf-8') as html_file:
                html_file.write(final_html)
            print(f"{Fore.GREEN}✅ Arquivo HTML (Diagrama Detalhado) salvo: {html_filename}")

        except Exception as e:
            print(f"{Fore.RED}❗Erro ao pós-processar ou salvar {html_filename}: {e}")
            import traceback
            traceback.print_exc()

# --- Função Principal (com input de prompt base) ---
def main():
    """Função principal: solicita prompt base, varre, gera YAML e chama a IA."""
    start_time = time.time()
    project_path = "."
    try:
        abs_path = os.path.abspath(project_path)
        project_name = os.path.basename(abs_path)
        if not project_name or project_name == '.': project_name = "Projeto Raiz Desconhecido"
    except Exception: project_name = "Projeto Desconhecido"

    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"🚀 {Style.BRIGHT}Iniciando Replika AI DocGen - Geração de Documentação e Diagrama{Style.NORMAL}")
    print(f"   Projeto Alvo: {Fore.CYAN}{project_name}{Style.RESET_ALL}")
    print(f"   Diretório:    {Fore.CYAN}{abs_path}{Style.RESET_ALL}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    # --- Solicitar Prompt Base do Usuário ---
    print(f"{Fore.CYAN}❓ Deseja fornecer instruções/contexto adicional para a IA analisar o projeto '{project_name}'?")
    print(f"{Fore.CYAN}   (Ex: 'Focar na API de usuários', 'Descrever a lógica de cálculo X', 'Ignorar testes')")
    print(f"{Fore.CYAN}   Deixe em branco e pressione Enter para pular.")
    try:
        user_base_prompt = input(f"{Fore.GREEN}> {Style.RESET_ALL}")
    except EOFError: # Caso o input seja redirecionado ou interrompido
        user_base_prompt = None
        print("\nEntrada não disponível ou interrompida. Continuando sem prompt base.")

    project_data = scan_directory(project_path)

    if not project_data:
        print(f"{Fore.YELLOW}⚠️ Análise do diretório não encontrou arquivos relevantes.")
        project_data = {
            "analise_info": { "status": "Diretório vazio ou sem arquivos analisáveis.", "timestamp": datetime.now().isoformat() }
        }

    # Salvar YAML
    yaml_filename = "" # Inicializa para garantir que a variável exista
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hostname = platform.node().replace(' ', '_').lower().replace('.', '_')
        safe_project_name = "".join(c if c.isalnum() else "_" for c in project_name)[:30]
        yaml_filename = f"estrutura_{safe_project_name}_{hostname}_{timestamp}.yaml"

        with open(yaml_filename, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(project_data, yaml_file, allow_unicode=True, default_flow_style=False, sort_keys=False, width=150, indent=2)
        print(f"{Fore.GREEN}💾 Dados da estrutura do projeto (para IA) salvos em: {yaml_filename}")
    except Exception as e:
         print(f"{Fore.RED}❗Erro ao salvar YAML de estrutura: {e}")

    # Chamar a IA com os dados e o prompt base opcional
    gerar_relatorio_ia(project_data, project_name, user_base_prompt)

    end_time = time.time()
    print(f"\n{Fore.YELLOW}{'='*70}")
    print(f"⏱️ {Style.BRIGHT}Processo concluído em {end_time - start_time:.2f} segundos.{Style.NORMAL}")
    print(f"   Verifique os arquivos gerados:")
    print(f"   - Documentação: {Fore.CYAN}DOCUMENTACAO-PROJETO.md{Style.RESET_ALL}")
    print(f"   - Diagrama HTML: {Fore.CYAN}doc-web-diagram-data-hashtagh unica .html{Style.RESET_ALL}")
    if yaml_filename:
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