import sqlite3
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import datetime
import hashlib
import os
import random
from colorama import init, Fore, Style
from rich.console import Console

# Inicializar Colorama e Rich
init(autoreset=True)
console = Console()

# Certifique-se de que a pasta 'valuation' existe
if not os.path.exists('valuation'):
    os.makedirs('valuation')

# Carregar o modelo Sentence Transformer
console.print(f"{Fore.YELLOW}Carregando modelo Sentence Transformer...{Style.RESET_ALL}")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Carregar o índice FAISS
console.print(f"{Fore.YELLOW}Carregando índice FAISS...{Style.RESET_ALL}")
indice = faiss.read_index("indice_perfis.faiss")

# Carregar o mapeamento de IDs
console.print(f"{Fore.YELLOW}Carregando mapeamento de IDs...{Style.RESET_ALL}")
try:
    df_mapeamento = pd.read_csv("mapeamento_ids.csv", skiprows=1, header=None, names=['id_perfil', 'indice_faiss'])  # Ignora a primeira linha
except FileNotFoundError:
    console.print(f"{Fore.RED}Arquivo mapeamento_ids.csv não encontrado. Certifique-se de que ele existe e está no diretório correto.{Style.RESET_ALL}")
    exit()

# Converter a coluna 'id_perfil' para inteiro
df_mapeamento['id_perfil'] = df_mapeamento['id_perfil'].astype(int)
df_mapeamento['indice_faiss'] = df_mapeamento['indice_faiss'].astype(int)

# Imprimir informações sobre o DataFrame de mapeamento
print("Tipo de dados da coluna 'id_perfil' no DataFrame:", df_mapeamento['id_perfil'].dtype)
print("Alguns IDs do DataFrame:", df_mapeamento['id_perfil'].head().tolist())

# Conectar ao banco de dados SQLite
console.print(f"{Fore.YELLOW}Conectando ao banco de dados...{Style.RESET_ALL}")
conn = sqlite3.connect('perfis_jogadores.db')
cursor = conn.cursor()

def carregar_perfil_por_id(id_perfil):
    try:
        cursor.execute("SELECT * FROM perfis WHERE id = ?", (id_perfil,))
        perfil = cursor.fetchone()
        if perfil:
            colunas = [column[0] for column in cursor.description]
            return dict(zip(colunas, perfil))
        else:
            print(f"Perfil com ID {id_perfil} não encontrado no banco de dados.")
            return None
    except Exception as e:
        print(f"Erro ao carregar perfil com ID {id_perfil}: {e}")
        return None

# Imprimir informações sobre o banco de dados
cursor.execute("SELECT id FROM perfis LIMIT 5")
ids_banco = [row[0] for row in cursor.fetchall()]
print("Alguns IDs do banco de dados:", ids_banco)

# Imprimir o tipo de dados da coluna 'id' na tabela 'perfis'
cursor.execute("PRAGMA table_info(perfis)")
colunas_info = cursor.fetchall()
for coluna in colunas_info:
    if coluna[1] == 'id':
        print("Tipo de dados da coluna 'id' no banco de dados:", coluna[2])
        break

# Escolher um perfil de origem aleatoriamente
console.print(f"{Fore.YELLOW}Escolhendo um perfil de origem aleatório...{Style.RESET_ALL}")
cursor.execute("SELECT id FROM perfis")
ids_disponiveis = [row[0] for row in cursor.fetchall()]
id_origem = random.choice(ids_disponiveis)

# Carregar os dados completos do perfil de origem
perfil_origem = carregar_perfil_por_id(id_origem)

if perfil_origem:
    # Preparar o texto para gerar o embedding do perfil de origem
    texto_origem = f"{perfil_origem['nome']}, {perfil_origem['cidade']}, {perfil_origem['sexo']}, {perfil_origem['interesses_musicais']}, {perfil_origem['jogos_favoritos']}, {perfil_origem['plataformas_possuidas']}, {perfil_origem['estilos_preferidos']}, {perfil_origem['descricao']}"

    # Gerar o embedding do perfil de origem
    console.print(f"{Fore.YELLOW}Gerando embedding para o perfil de origem...{Style.RESET_ALL}")
    embedding_origem = model.encode(texto_origem, convert_to_tensor=False).reshape(1, -1)

    # Buscar os 10 perfis mais similares
    console.print(f"{Fore.YELLOW}Buscando os 10 perfis mais similares...{Style.RESET_ALL}")
    D, I = indice.search(embedding_origem, 10)  # D são as distâncias, I são os índices

    # Imprimir os índices retornados pelo FAISS para depuração
    print("Índices retornados pelo FAISS:", I)

    # Recuperar os IDs dos perfis similares
    ids_similares = []
    for indice_faiss in I[0]:
        # Converter o índice FAISS para inteiro (redundante, mas seguro)
        indice_faiss = int(indice_faiss)

        # Encontrar o ID do perfil correspondente ao índice FAISS
        try:
            id_similares = df_mapeamento[df_mapeamento['indice_faiss'] == indice_faiss]['id_perfil'].values[0]
            ids_similares.append(id_similares)
        except IndexError:
            print(f"Índice FAISS {indice_faiss} não encontrado no mapeamento.")


    # Imprimir os IDs similares encontrados para depuração
    print("IDs similares encontrados:", ids_similares)

    # Recuperar os dados completos dos perfis similares
    console.print(f"{Fore.YELLOW}Recuperando dados dos perfis similares...{Style.RESET_ALL}")
    perfis_similares = []
    for id_similar in ids_similares:
        perfil_similar = carregar_perfil_por_id(id_similar)
        if perfil_similar:
            perfis_similares.append(perfil_similar)

    # Imprimir os perfis similares encontrados para depuração
    print("Perfis similares encontrados:", perfis_similares)

    # Criar o JSON com os resultados
    console.print(f"{Fore.YELLOW}Criando JSON com os resultados...{Style.RESET_ALL}")
    data_hora = datetime.datetime.now().isoformat()
    dados = {
        'data_hora': data_hora,
        'perfil_origem': perfil_origem,
        'perfis_similares': perfis_similares
    }

    # Calcular o hash dos dados
    dados_str = json.dumps(dados, sort_keys=True, ensure_ascii=False).encode('utf-8')
    hash_dados = hashlib.sha256(dados_str).hexdigest()

    # Adicionar o hash aos dados
    dados['hash'] = hash_dados

    # Salvar o JSON em um arquivo
    nome_arquivo = f"valuation/valuation_{data_hora.replace(':', '-')}.json"
    console.print(f"{Fore.YELLOW}Salvando JSON em {nome_arquivo}...{Style.RESET_ALL}")
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)

    console.print(f"{Fore.GREEN}Valuation concluída e salva em {nome_arquivo} com sucesso!{Style.RESET_ALL}")
else:
    console.print(f"{Fore.RED}Perfil de origem não encontrado!{Style.RESET_ALL}")

# Fechar a conexão com o banco de dados
conn.close()