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

# Inicializa o Colorama para colorir os logs
init(autoreset=True)

# Configuração da API de IA
API_KEY = 'AIzaSyC7dAwSyLKaVO2E-PA6UaacLZ4aLGtrXbY'  # Chave da API fornecida
genai.configure(api_key=API_KEY)
NOME_MODELO = "gemini-2.0-flash"  # Modelo atualizado

# Arquivos e diretórios a serem ignorados
IGNORE_LIST = [
    "docgenv1.py",  # Ignora o próprio script
    ".git",  # Ignora diretórios .git
    ".venv",  # Ignora ambientes virtuais
    "venv",  # Ignora ambientes virtuais
    "__pycache__", # Ignora diretórios de cache
    "node_modules",  # Ignora diretórios node_modules
    ".md" # Ignora arquivos Markdown
]

# Função para calcular o tamanho de um arquivo
def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except OSError:
        return -1  # Retorna -1 se o arquivo não existir ou ocorrer um erro

# Função para calcular o número de linhas em um arquivo de texto
def count_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return -1

# Função para ler o conteúdo de um arquivo de texto (com limite)
def read_file_content(file_path, max_lines=500): # Aumentei o limite de linhas
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(max_lines)]  # Lê até max_lines linhas
            return "".join(lines)
    except Exception:
        return None

# Função para analisar o código Python e extrair informações relevantes
def analyze_python_code(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        
        functions = []
        classes = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno if hasattr(node, 'end_lineno') else None
                })
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno if hasattr(node, 'end_lineno') else None
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "name": alias.name,
                        "asname": alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    "module": node.module,
                    "names": [alias.name for alias in node.names]
                })
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "source_code": source_code # Mantém o código fonte completo
        }

    except Exception as e:
        return {"error": str(e)}

# Função para obter informações de um arquivo .db (SQLite)
def get_sqlite_info(file_path, max_rows=5): # Aumentei o número de exemplos
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_data = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}');")
            columns = [(row[1], row[2]) for row in cursor.fetchall()] # (nome, tipo)
            
            try:
                cursor.execute(f"SELECT * FROM '{table}' LIMIT {max_rows};")
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                print(f"{Fore.YELLOW}Aviso: Não foi possível selecionar dados da tabela '{table}': {e}{Style.RESET_ALL}")
                rows = []  # Define rows como uma lista vazia em caso de erro

            table_data[table] = {"columns": columns, "rows": rows}
        
        conn.close()
        return table_data
    except Exception as e:
        return {"error": str(e)}

# Função para obter informações de um arquivo .py
def get_python_info(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        analysis_result = analyze_python_code(file_path)
        # Garante que o código fonte seja armazenado corretamente
        analysis_result['source_code'] = source_code
        return analysis_result
    except Exception as e:
        return {"error": str(e)}

# Função para obter informações de um arquivo .json
def get_json_info(file_path):
     try:
        file_size = get_file_size(file_path)
        line_count = count_lines(file_path)
        return {"tamanho": format_size(file_size), "numero_de_linhas": line_count}
     except Exception as e:
        return {"error": str(e)}

# Função para obter informações de um arquivo .yaml
def get_yaml_info(file_path):
    try:
        file_size = get_file_size(file_path)
        line_count = count_lines(file_path)
        return {"tamanho": format_size(file_size), "numero_de_linhas": line_count}
    except Exception as e:
        return {"error": str(e)}


# Função para formatar o tamanho em MB
def format_size(size_in_bytes):
    size_mb = size_in_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"

# Função para estilizar o cabeçalho de cada unidade
def stylize_header(header):
    return f"📁 --- {header} --- 📁\n"

# Configurações de geração da IA (movido para o escopo global para facilitar a modificação)
AI_TEMPERATURE = 0.9  # Aumentei para respostas mais criativas e variadas
AI_TOP_P = 0.99       # Aumentei para considerar um conjunto maior de palavras possíveis
AI_TOP_K = 80         # Aumentei para uma amostragem mais ampla
AI_MAX_TOKENS = 32768  # Aumentei consideravelmente o limite máximo de tokens

# Função para gerar o payload da IA para relatório Markdown
def configurar_geracao(temperatura=AI_TEMPERATURE, top_p=AI_TOP_P, top_k=AI_TOP_K, max_tokens=AI_MAX_TOKENS):
    return {
        "temperature": temperatura,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

# Função para enviar mensagens para a IA com logging aprimorado
def enviar_mensagem(sessao, mensagem):
    try:
        print(f"{Fore.YELLOW}🧠 Enviando mensagem para a IA...")  # Log colorido
        resposta = sessao.send_message([mensagem])
        print(f"{Fore.GREEN}✅ Resposta recebida!")  # Sucesso
        return resposta.text
    except Exception as e:
        print(f"{Fore.RED}❗Erro ao enviar mensagem para o modelo: {e}")
        return ""

# Função para varrer diretórios e arquivos
def scan_directory(root_path="."):
    report = {}
    for root, dirs, files in os.walk(root_path):
        # Ignora diretórios da lista de ignorados
        dirs[:] = [d for d in dirs if d not in IGNORE_LIST]

        root_report = {}
        for file in files:
            if file in IGNORE_LIST:
                continue
            
            file_path = os.path.join(root, file)
            file_size = get_file_size(file_path)
            line_count = count_lines(file_path)

            file_info = {
                "tamanho": format_size(file_size),
                "numero_de_linhas": line_count,
                "caminho_completo": file_path # Adicionado caminho completo
            }

            if file.endswith(".py"):
                python_info = get_python_info(file_path)
                if python_info:
                    file_info.update(python_info)
            elif file.endswith(".db"):
                sqlite_info = get_sqlite_info(file_path)
                file_info["sqlite_info"] = sqlite_info
            elif file.endswith(".json"):
                json_info = get_json_info(file_path)
                file_info["json_info"] = json_info
            elif file.endswith(".yaml") or file.endswith(".yml"):
                yaml_info = get_yaml_info(file_path)
                file_info["yaml_info"] = yaml_info
            
            root_report[file] = file_info

        report[root] = root_report
    return report

# Função para gerar relatório em YAML e enviar para IA
def gerar_relatorio_ia(data, project_name):
    try:
        sessao_chat = genai.GenerativeModel(
            model_name=NOME_MODELO,
            generation_config=configurar_geracao(),
        ).start_chat(history=[])
    except Exception as e:
        print(f"{Fore.RED}❗Erro ao iniciar sessão de chat: {e}")
        return

    prompt = f"""
    nao repita dados dos bancos, foque mais em descrever o projeto, o que faz, como faz, etc quem faz, etc, o projeto no espectro amplo é o objetivo
    documente o projeto bem comercial e executivo
    nao precisa cobrir cada arquivo cada coisa, documente no amplo, foque no pareto 80 - 20  - foque no 20 relevante dos 80 irrrelevante ignore
    
    naoseja minuciosoo, abranja tudo de forma geral, o projeto como todo, foque no que é relevante
    NUNCA GERE CODIGO NA SUA RESPOSTA, NAO ANALISE O CODIGO A PONTO DE REPETIR, NUNCA CRIE SNIPPETS E CODIGO FONTE NA RESPOSTA
    seu objeitov nao é criar nem ficar citando ocdoigo, mas sim documentar o projeto, documente como produto e projeto
    sempre criado por elias andrade - replika ia solutions- sempre documente o projeto em si, nao fique repetindo codigo
    respondoa mais longo, detlahado, aprofundado, use muitos icones, emojis, documente o projeto, o que faz, como faz, o que faz o que, etc.
    
    Você é um especialista sênior em documentação de projetos de software e agora também um arquiteto de sistemas.
    Sua missão é analisar a estrutura e o conteúdo de um projeto e gerar um README.md completo e detalhado.
    Você deve fornecer uma visão geral abrangente do projeto para facilitar a compreensão e manutenção futura.

    Com base nos dados fornecidos, gere um README.md completo e detalhado para o projeto. Use Markdown para formatar o README.md.

    O README.md deve incluir as seguintes seções (use títulos e subtítulos apropriados):

    1.  **Título do Projeto:** {project_name} (Utilize emojis relacionados ao nome, se aplicável)

    2.  **Descrição Geral:**
        *   Forneça uma descrição concisa e de alto nível do projeto, incluindo seu propósito principal e funcionalidades.
        *   Identifique os principais componentes e tecnologias utilizadas.
        *   Resuma o problema que o projeto resolve ou a necessidade que atende.

    3.  **Estrutura do Projeto (com muitos emojis):**
        *   Apresente uma lista detalhada de todos os diretórios e arquivos.
        *   Para cada diretório, descreva seu papel e responsabilidades dentro do projeto.
        *   Para cada arquivo, inclua:
            *   Nome do arquivo (com emoji indicando o tipo, ex: 🐍 para .py, ⚙️ para config)
            *   Caminho completo (importante para localização)
            *   Tamanho do arquivo
            *   Número de linhas
            *   Descrição da função do arquivo (inferida do nome, comentários no código, etc.)

    4.  **Detalhes Técnicos e Arquiteturais:**
        *   **Código Fonte (Python):**
            *   Apresente o código fonte completo de cada arquivo Python.
            *   Analise o código e destaque os principais componentes:
                *   Funções (nome, docstring, linhas de início e fim)
                *   Classes (nome, docstring, linhas de início e fim)
                *   Imports (módulos importados, aliases)
            *   Explique a lógica por trás dos principais algoritmos e estruturas de dados.
            *   Comente sobre padrões de design utilizados (se houver).
             *  Formate o código de forma legível, removendo quebras de linha desnecessárias e garantindo a codificação UTF-8 correta.
        *   **Bancos de Dados (SQLite):**
            *   Diagrama ER (se possível, descreva a estrutura em texto se não for possível gerar o diagrama).
            *   Lista de tabelas com descrições claras.
            *   Esquema de cada tabela (nome das colunas, tipos de dados, chaves primárias/estrangeiras).
            *   Exemplos de consultas SQL importantes.
            *   Dados de exemplo (até 5 linhas por tabela, com formatação de tabela Markdown).
            *   Observações sobre otimizações de queries (se aplicável).
        *   **Configurações (JSON/YAML):**
            *   Descreva o propósito de cada arquivo de configuração.
            *   Liste as principais chaves de configuração e seus significados.
            *   (Não inclua o conteúdo completo, apenas resumos e exemplos se necessários)
            *   Caminho completo (para facilidade de acesso)
            *    Tamanho do arquivo e número de linhas.

    5.  **Como Executar e Configurar o Projeto:**
        *   Instruções passo a passo para configurar o ambiente (ex: instalar Python, dependências).
        *   Exemplo de como executar o projeto (ex: `python main.py`).
        *   Explique como configurar o projeto (ex: editar arquivos de configuração).
        *   Liste as dependências externas e como instalá-las (use `pip install -r requirements.txt` se aplicável).
        *   Explique como executar os testes (se houver testes automatizados).

    6.  **Considerações Adicionais:**
        *   **Arquitetura do Projeto:** Discuta a arquitetura geral do projeto (ex: MVC, microserviços, etc.).
        *   **Padrões de Codificação:** Mencione se o projeto segue algum padrão de codificação específico (ex: PEP 8).
        *   **Licença:** Informe a licença sob a qual o projeto é distribuído.
        *   **Contribuições:** Explique como outros desenvolvedores podem contribuir para o projeto.
        *   **Próximos Passos:** Liste os próximos passos para o desenvolvimento do projeto.
        *   **Notas:** Inclua quaisquer notas adicionais que sejam relevantes para o projeto.

    7.  **Informações sobre o ambiente que o gerou:**
        *   Sistema Operacional
        *   Data e Hora da geração
        *   Nome do computador

    Formate o README.md de forma clara, concisa e profissional. Use formatação Markdown (títulos, subtítulos, listas, tabelas, blocos de código, links, imagens) para facilitar a leitura e a compreensão. Inclua emojis relevantes para tornar o documento mais visualmente atraente.

    Seja extremamente detalhista e procure inferir o máximo de informações possível sobre o projeto com base nos dados fornecidos.
    O objetivo é criar um README.md que seja útil tanto para desenvolvedores que já conhecem o projeto quanto para aqueles que estão chegando agora.

    Aqui estão os dados do projeto em formato YAML:

    ```yaml
    {yaml.dump(data, allow_unicode=True)}
    ```
    
    nunca crie snippet de codigo, nunca traga na sua resposta pedaços do codigo por que isso é redundante, nao é permitido ter codigo fonte nas respostas
    --
    a documentacao é itnerna rpa mim, pra replika ai, entao nao crie como se fosse um repo publico, é documento interno - use muitos icones e emojis, cubra o projeto e fale altaemnte tecnico
    entenda uqe nivel o projeto est,a parou, etc, o que ja funciona, oq ue nao funciona ainda, o que ja foi criado, entenda nesse aspecto, muito icone e emoji estilizado pro notion
    documente o projeto a propriedade intelectual e que nivel esta o projeto, etc, 
    
    """

    resposta = enviar_mensagem(sessao_chat, prompt)

    if resposta:
        markdown_filename = "DOCUMENTACAO-PROJETO.md"  # Nome fixo do arquivo README
        with open(markdown_filename, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write(resposta)
        print(f"{Fore.GREEN}📄 {markdown_filename} salvo com sucesso!")
    else:
        print(f"{Fore.RED}❗Erro ao gerar relatório com a IA.")

# Função principal para unir tudo
def main():
    project_path = "." # Diretório atual
    project_name = os.path.basename(os.getcwd()) # Nome do diretório atual
    print(f"{Fore.YELLOW}🔄 Iniciando varredura do projeto '{project_name}' em '{project_path}'...")
    project_data = scan_directory(project_path)

    if project_data:
        # Salva o YAML em arquivo para fins de backup
        yaml_filename = f"estrutura_projeto_{project_name}_{platform.node()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(yaml_filename, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(project_data, yaml_file, allow_unicode=True)
        print(f"{Fore.GREEN}📂 Estrutura do projeto salva em: {yaml_filename}")

        # Envia os dados para a IA para gerar relatório final
        gerar_relatorio_ia(project_data, project_name)
    else:
        print(f"{Fore.RED}🚫 Nenhum arquivo ou diretório encontrado.")

# Execução do programa
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"{Fore.RED}❗Processo interrompido pelo usuário.")
    except Exception as e:
        print(f"{Fore.RED}❗Erro inesperado: {e}")