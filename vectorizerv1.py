import sqlite3
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from colorama import init, Fore, Style
from rich.console import Console
from rich.progress import track

# Inicializar Colorama e Rich
init(autoreset=True)
console = Console()

# Carregar o modelo Sentence Transformer
console.print(f"{Fore.YELLOW}Carregando modelo Sentence Transformer...{Style.RESET_ALL}")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # Um modelo pré-treinado multilingue

# Conectar ao banco de dados SQLite
console.print(f"{Fore.YELLOW}Conectando ao banco de dados...{Style.RESET_ALL}")
conn = sqlite3.connect('perfis_jogadores.db')
cursor = conn.cursor()

# Ler os perfis do banco de dados
console.print(f"{Fore.YELLOW}Lendo perfis do banco de dados...{Style.RESET_ALL}")
cursor.execute("SELECT id, nome, cidade, sexo, interesses_musicais, jogos_favoritos, plataformas_possuidas, estilos_preferidos, descricao FROM perfis")
perfis = cursor.fetchall()

# Fechar a conexão com o banco de dados
conn.close()

# Preparar os dados para o Sentence Transformer
console.print(f"{Fore.YELLOW}Preparando dados para gerar embeddings...{Style.RESET_ALL}")
textos_para_embedding = []
ids_para_embedding = []
for perfil in perfis:
    id_perfil, nome, cidade, sexo, interesses_musicais, jogos_favoritos, plataformas_possuidas, estilos_preferidos, descricao = perfil
    texto = f"{nome}, {cidade}, {sexo}, {interesses_musicais}, {jogos_favoritos}, {plataformas_possuidas}, {estilos_preferidos}, {descricao}"
    textos_para_embedding.append(texto)
    ids_para_embedding.append(id_perfil)

# Gerar os embeddings
console.print(f"{Fore.YELLOW}Gerando embeddings com Sentence Transformer...{Style.RESET_ALL}")
embeddings = model.encode(textos_para_embedding, convert_to_tensor=False)

# Dimensão dos embeddings
dimensao_embedding = embeddings.shape[1]

# Criar o índice FAISS
console.print(f"{Fore.YELLOW}Criando índice FAISS...{Style.RESET_ALL}")
indice = faiss.IndexFlatL2(dimensao_embedding)

# Adicionar os embeddings ao índice FAISS
console.print(f"{Fore.YELLOW}Adicionando embeddings ao índice FAISS...{Style.RESET_ALL}")
indice.add(embeddings)

# Criar um mapeamento entre IDs dos perfis e o índice no FAISS
console.print(f"{Fore.YELLOW}Criando mapeamento entre IDs dos perfis e o índice no FAISS...{Style.RESET_ALL}")
id_para_indice = {id_perfil: i for i, id_perfil in enumerate(ids_para_embedding)}

# Salvar o índice FAISS em disco
console.print(f"{Fore.YELLOW}Salvando índice FAISS em disco...{Style.RESET_ALL}")
faiss.write_index(indice, "indice_perfis.faiss")

# Salvar o mapeamento de IDs em um arquivo (pode ser um arquivo CSV, JSON, etc.)
console.print(f"{Fore.YELLOW}Salvando mapeamento de IDs em disco...{Style.RESET_ALL}")
df_mapeamento = pd.DataFrame(list(id_para_indice.items()), columns=['id_perfil', 'indice_faiss'])
df_mapeamento.to_csv("mapeamento_ids.csv", index=False)

console.print(f"{Fore.GREEN}Índice FAISS e mapeamento de IDs criados e salvos com sucesso!{Style.RESET_ALL}")