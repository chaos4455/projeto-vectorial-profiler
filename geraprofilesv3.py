# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import faiss
from faker import Faker
import random
from colorama import init, Fore, Style
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import os
import platform
import sys
from datetime import datetime
import math
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import time
import argparse
from functools import lru_cache

# --- Dependency Check ---
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = None
    silhouette_score = None
    davies_bouldin_score = None

# --- Python Version Check ---
if sys.version_info < (3, 6):
    print("ERROR: This script requires Python 3.6 or higher.", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
LOG_DIR_BASE = "logs_v6"
DB_DIR_BASE = "databases_v6"
FAISS_DIR_BASE = "faiss_indices_v6"
DB_PERFIS_NAME = 'perfis_jogadores_v6.db'
DB_VETORES_NAME = 'vetores_perfis_v6.db'
DB_EMBEDDINGS_NAME = 'embeddings_perfis_v6.db'
DB_CLUSTERS_NAME = 'clusters_perfis_v6.db'
FAISS_INDEX_PREFIX = 'faiss_kmeans_index_'
CENTROIDS_PREFIX = 'kmeans_centroids_'
KMEANS_MAX_POINTS_DEFAULT = 1000000 # Default, adjust if needed
CHUNK_SIZE_FACTOR = 4 # Factor for multiprocessing chunks

# --- Initialization ---
init(autoreset=True)
console = Console(log_time=True, log_path=False, record=True)
_worker_faker_instances = {}

# --- Base Data ---
# (Keep your large lists populated here as before)
CIDADES_ESTADOS = { "São Paulo": "SP", "Rio de Janeiro": "RJ", "Recife": "PE", "Curitiba": "PR", "Manaus":"AM" } # Example, use your full lists
CIDADES_BRASIL = list(CIDADES_ESTADOS.keys()) * 10
random.shuffle(CIDADES_BRASIL)
ESTADOS_BRASIL = sorted(list(set(CIDADES_ESTADOS.values()))) * 10
random.shuffle(ESTADOS_BRASIL)
JOGOS_MAIS_JOGADOS = ["LoL", "CS2", "Valorant", "BG3", "Helldivers 2"] * 10 # Example, use your full lists
random.shuffle(JOGOS_MAIS_JOGADOS)
PLATAFORMAS = ["PC", "PS5", "Xbox Series", "Switch", "Mobile"] * 10 # Example, use your full lists
random.shuffle(PLATAFORMAS)
ESTILOS_JOGO = ["FPS", "RPG", "MOBA", "Aventura", "Estratégia"] * 10 # Example, use your full lists
random.shuffle(ESTILOS_JOGO)
ESTILOS_MUSICA = ["Rock", "Pop", "Eletrônica", "Hip Hop", "Funk"] * 10 # Example, use your full lists
random.shuffle(ESTILOS_MUSICA)
SEXOS = ["Masculino", "Feminino", "Não Binário", "Prefiro não informar"] * 10 # Example, use your full lists
random.shuffle(SEXOS)
INTERACAO = ["Apenas Online", "Online e Presencial", "Grupo Fixo", "Indiferente"] * 10 # Example, use your full lists
random.shuffle(INTERACAO)
DISPONIBILIDADE_LISTA = ["Manhã", "Tarde", "Noite", "Madrugada", "FDS"] * 10 # Example, use your full lists
random.shuffle(DISPONIBILIDADE_LISTA)
OBJETIVO_PRINCIPAL_LISTA = ["Competitivo", "Casual", "Social", "Exploração"] * 10 # Example, use your full lists
random.shuffle(OBJETIVO_PRINCIPAL_LISTA)
NOMES_MASCULINOS = ["Miguel", "Arthur", "Heitor"] * 10 # Example, use your full lists
NOMES_FEMININOS = ["Alice", "Sophia", "Helena"] * 10 # Example, use your full lists
NOMES_NAO_BINARIOS = ["Alex", "Kim", "Sam"] * 10 # Example, use your full lists
random.shuffle(NOMES_MASCULINOS)
random.shuffle(NOMES_FEMININOS)
random.shuffle(NOMES_NAO_BINARIOS)
IDIOMAS = ["Português", "Inglês", "Espanhol", "Português, Inglês", "Português, Espanhol", "Inglês, Espanhol", "Português, Inglês, Espanhol"] * 10
random.shuffle(IDIOMAS)
NIVEL_COMPETITIVO = ["Iniciante", "Bronze", "Prata", "Ouro", "Platina", "Diamante", "Mestre", "Grão-Mestre", "Desafiante", "Lendário", "Não jogo ranked", "Aprendendo", "Casual Competitivo"] * 10
random.shuffle(NIVEL_COMPETITIVO)
ESTILO_COMUNICACAO = ["Direto(a) e Objetivo(a)", "Calmo(a) e Analítico(a)", "Estratégico(a) (calls)", "Bem-humorado(a) e Descontraído(a)", "Quieto(a)/Observador(a)", "Líder Natural", "Apenas Texto", "Prefiro Ouvir", "Depende do Jogo/Grupo"] * 10
random.shuffle(ESTILO_COMUNICACAO)

# Description Generation Keywords (Example, keep your full lists)
ADJETIVOS_POSITIVOS = ["amigável", "parceiro", "dedicado"]
ADJETIVOS_CASUAIS = ["tranquilo", "de boa", "flexível"]
VERBOS_GOSTAR = ["curto", "gosto de", "prefiro"]
VERBOS_JOGAR = ["jogar", "mandar um", "me dedicar a"]
TERMOS_PLATAFORMA = ["no PC", "no console", "principalmente no PC"]
TERMOS_ESTILO = ["{}", "jogos do tipo {}", "focado em {}"]
TERMOS_MUSICA = ["ouvindo {}", "curto um som {}", "gosto de {}"]
TERMOS_DISPONIBILIDADE = ["geralmente {}", "mais {}", "livre {}"]
OBJETIVOS_JOGO = ["me divertir", "evoluir", "fazer amigos", "competir"]
TERMOS_INTERACAO_ONLINE = ["online", "via Discord", "pela net"]
TERMOS_INTERACAO_OFFLINE = ["encontrar a galera", "eventos presenciais"]
ADJETIVOS_EXPERIENCIA = ["veterano", "experiente", "macaco velho"]
FOCO_JOGO = ["gameplay", "estratégia", "zoeira", "história"]
COMUNICACAO = ["Me comunico bem", "Sou bom em calls", "Falo o necessário"]
HUMOR = ["bem-humorado(a)", "engraçado(a)", "sério(a) quando precisa"]
ABERTURA_CONTATO = ["Passo contato fácil", "Add aí!", "Bora trocar ideia"]
PACIENCIA = ["paciente", "compreensivo(a)", "sem rage"]


# --- Helper Functions ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerador de Perfis de Jogadores Otimizado V6")
    parser.add_argument("-n", "--num-profiles", type=int, default=30000, help="Número total de perfis a gerar.")
    parser.add_argument("-w", "--workers", type=int, default=max(1, cpu_count() - 1), help="Número de workers para multiprocessing.")
    parser.add_argument("--locale", type=str, default="pt_BR", help="Locale para o Faker.")
    parser.add_argument("--seed", type=int, default=None, help="Seed global para random e numpy.")
    parser.add_argument("--dim-vector", type=int, default=20, help="Dimensão do vetor de características numéricas.")
    parser.add_argument("--dim-embedding", type=int, default=192, help="Dimensão do embedding simulado.")
    parser.add_argument("--cluster-method", type=str, default="sqrt", choices=["sqrt", "fixed"], help="Método para determinar o número de clusters.")
    parser.add_argument("--fixed-clusters", type=int, default=150, help="Número fixo de clusters (usado se --cluster-method=fixed).")
    parser.add_argument("--kmeans-niter", type=int, default=25, help="Número de iterações do KMeans.")
    parser.add_argument("--kmeans-nredo", type=int, default=3, help="Número de execuções do KMeans com seeds diferentes.")
    parser.add_argument("--kmeans-seed", type=int, default=42, help="Seed específico para o KMeans.")
    parser.add_argument("--kmeans-gpu", action="store_true", help="Tentar usar GPU para KMeans.")
    parser.add_argument("--db-dir", type=str, default=DB_DIR_BASE, help="Diretório para salvar os bancos de dados.")
    parser.add_argument("--faiss-dir", type=str, default=FAISS_DIR_BASE, help="Diretório para salvar índices e centróides FAISS.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR_BASE, help="Diretório para salvar os arquivos de log.")
    parser.add_argument("--skip-clustering", action="store_true", help="Pular etapas de clustering e salvamento relacionado.")
    parser.add_argument("--skip-vector-scaling", action="store_true", help="Pular etapa de escalonamento MinMax dos vetores.")
    parser.add_argument("--skip-cluster-metrics", action="store_true", help="Pular cálculo de métricas de cluster.")
    parser.add_argument("--no-save-faiss-index", action="store_true", help="Não salvar o índice FAISS treinado.")
    parser.add_argument("--no-save-centroids", action="store_true", help="Não salvar os centróides do KMeans.")
    parser.add_argument("--vacuum-dbs", action="store_true", help="Executar VACUUM nos DBs no final.")
    parser.add_argument("--detailed-log", action="store_true", help="Habilitar logging mais detalhado (DEBUG level).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suprimir a maioria dos outputs do console.")
    args = parser.parse_args()
    args.save_faiss_index = not args.no_save_faiss_index
    args.save_centroids = not args.no_save_centroids
    return args

def setup_logging(log_dir: str, detailed: bool) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"profile_generator_v6_{timestamp}.log")
    log_level = logging.DEBUG if detailed else logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8')]
    )
    logging.info(f"Logging configurado para nível {logging.getLevelName(log_level)} em {log_file}")
    return log_file

def console_print(message: str, args: argparse.Namespace, style: Optional[str] = None):
    if not args.quiet:
        if style:
            console.print(message, style=style)
        else:
            console.print(message)

def get_db_path(args: argparse.Namespace, db_name: str) -> str:
     return os.path.join(args.db_dir, db_name)

def escolher_com_peso(lista: List[Any], pesos: List[float]) -> Any:
    if not lista: return None
    if not pesos or len(lista) != len(pesos): return random.choice(lista)
    soma_pesos = sum(pesos)
    if soma_pesos <= 0: return random.choice(lista)
    pesos_normalizados = [p / soma_pesos for p in pesos]
    return random.choices(lista, weights=pesos_normalizados, k=1)[0]

@lru_cache(maxsize=len(CIDADES_BRASIL) if CIDADES_BRASIL else 128)
def obter_estado_por_cidade(cidade: str) -> str:
    return CIDADES_ESTADOS.get(cidade, "??")

def gerar_horario_disponivel(idade: int) -> str:
    if not DISPONIBILIDADE_LISTA: return "Horário Indefinido"
    if idade < 25: pesos = [0.1, 0.2, 0.3, 0.4, 0.3]
    elif idade > 45: pesos = [0.2, 0.2, 0.3, 0.1, 0.4]
    else: pesos = [0.2, 0.3, 0.3, 0.2, 0.3]
    # Pad weights to match list length if necessary
    pesos.extend([0.1] * (len(DISPONIBILIDADE_LISTA) - len(pesos)))
    pesos = pesos[:len(DISPONIBILIDADE_LISTA)] # Ensure it doesn't exceed length
    return escolher_com_peso(DISPONIBILIDADE_LISTA, pesos)

def gerar_nome(sexo: str, fake_instance: Faker) -> str:
    try:
        nome_base = None
        if sexo == "Masculino" and NOMES_MASCULINOS: nome_base = random.choice(NOMES_MASCULINOS)
        elif sexo == "Feminino" and NOMES_FEMININOS: nome_base = random.choice(NOMES_FEMININOS)
        elif sexo == "Não Binário" and NOMES_NAO_BINARIOS: nome_base = random.choice(NOMES_NAO_BINARIOS)

        if nome_base is None:
            todos_nomes = NOMES_MASCULINOS + NOMES_FEMININOS + NOMES_NAO_BINARIOS
            if todos_nomes: nome_base = random.choice(todos_nomes)

        if nome_base is None: return fake_instance.name()
        return f"{nome_base} {fake_instance.last_name()}"
    except Exception as e:
        logging.error(f"Erro inesperado em gerar_nome: {e}", exc_info=True)
        return fake_instance.name()

def gerar_descricao_consistente(perfil: Dict) -> str:
    try:
        nome_curto = perfil.get('nome', 'Jogador(a)').split(' ')[0]
        idade = perfil.get('idade', 30)
        jogos_fav_lista = [j.strip() for j in perfil.get('jogos_favoritos', '').split(',') if j.strip()]
        jogo_fav1 = jogos_fav_lista[0] if jogos_fav_lista else "jogos variados"
        jogo_fav2 = random.choice(jogos_fav_lista[1:]) if len(jogos_fav_lista) > 1 else None
        estilos_pref_lista = [s.strip() for s in perfil.get('estilos_preferidos', '').split(',') if s.strip()]
        estilo_pref1 = estilos_pref_lista[0] if estilos_pref_lista else "diversos gêneros"
        estilo_pref2 = random.choice(estilos_pref_lista[1:]) if len(estilos_pref_lista) > 1 else None
        plataforma_pref = perfil.get('plataformas_possuidas', '').split(', ')[0] if perfil.get('plataformas_possuidas') else "várias plataformas"
        musica_pref = perfil.get('interesses_musicais', '').split(', ')[0] if perfil.get('interesses_musicais') else "música boa"
        disponibilidade = perfil.get('disponibilidade', "Horários Variados")
        interacao = perfil.get('interacao_desejada', "Indiferente")
        anos_exp = perfil.get('anos_experiencia', random.randint(0, 20))
        objetivo_p = perfil.get('objetivo_principal', "Casual")
        usa_mic = perfil.get('usa_microfone', False)
        compartilha = perfil.get('compartilhar_contato', False)
        idiomas = perfil.get('idiomas', "Português")
        nivel_comp = perfil.get('nivel_competitivo', "Casual")
        estilo_comm = perfil.get('estilo_comunicacao', "Calmo(a)")

        adj_pos = random.choice(ADJETIVOS_POSITIVOS)
        adj_cas = random.choice(ADJETIVOS_CASUAIS)
        verbo_gostar = random.choice(VERBOS_GOSTAR)
        verbo_jogar = random.choice(VERBOS_JOGAR)
        termo_plat = random.choice(TERMOS_PLATAFORMA)
        termo_estilo1 = random.choice(TERMOS_ESTILO).format(estilo_pref1)
        termo_musica = random.choice(TERMOS_MUSICA).format(musica_pref)
        termo_disp = random.choice(TERMOS_DISPONIBILIDADE).format(disponibilidade.split('(')[0].strip().lower())
        objetivo_jogo = random.choice(OBJETIVOS_JOGO)
        termo_interacao = random.choice(TERMOS_INTERACAO_ONLINE if "Online" in interacao else (TERMOS_INTERACAO_OFFLINE + TERMOS_INTERACAO_ONLINE))
        exp_desc = f"com ~{anos_exp} anos de exp" if anos_exp > 1 else ("comecei há pouco" if anos_exp <= 1 else random.choice(ADJETIVOS_EXPERIENCIA))
        foco = random.choice(FOCO_JOGO)
        comm = f"{random.choice(COMUNICACAO)}. Meu estilo é mais {estilo_comm.split('(')[0].lower()}"
        comm += ". Uso mic!" if usa_mic else ". Prefiro chat."
        humor = random.choice(HUMOR)
        contato_desc = random.choice(ABERTURA_CONTATO) if compartilha else "Contato pessoal só com mais calma."
        paciencia_desc = random.choice(PACIENCIA)
        idiomas_desc = f"Falo {idiomas}." if idiomas != "Português" else "Falo Português."
        nivel_desc = f"Nível: {nivel_comp}." if nivel_comp != "Não jogo ranked" else "Não ligo pra rank."

        part1 = [f"Olá, sou {nome_curto} ({idade}a), {exp_desc}.", f"Jogador(a) {adj_cas}, {exp_desc}. Pode me chamar de {nome_curto}.", f"{adj_pos.capitalize()} e {adj_cas}, {idade} anos."]
        part2 = [f"{verbo_gostar.capitalize()} {termo_estilo1}." + (f" Também me aventuro em {estilo_pref2}." if estilo_pref2 else ""), f"Principalmente {verbo_jogar} {jogo_fav1}." + (f" Às vezes {jogo_fav2}." if jogo_fav2 else ""), f"Foco em {jogo_fav1}, mas topo {estilo_pref1} variados."]
        part3 = [f"Jogo {termo_plat}.", f"Plataforma: {plataforma_pref}.", f"Mais ativo(a) {termo_plat}."]
        part4 = [f"Objetivo: {objetivo_p}.", f"Busco {objetivo_jogo}, principalmente de forma {objetivo_p.lower()}.", f"{nivel_desc}"]
        part5 = [f"Comunicação: {comm}", f"Sou {humor} e {paciencia_desc}. {comm}", f"{idiomas_desc} {comm}"]
        part6 = [f"Prefiro interação {termo_interacao}. {contato_desc}", f"Disponibilidade: {termo_disp}.", f"Tô online {termo_disp}. Música: {termo_musica}."]

        desc = f"{random.choice(part1)} {random.choice(part2)} {random.choice(part3)} {random.choice(part4)} {random.choice(part5)} {random.choice(part6)}"
        return desc[:700]
    except Exception as e:
        logging.error(f"Erro ao gerar descrição V6: {e}", exc_info=True)
        return f"Jogador(a) de {perfil.get('cidade', '?')}. Curte jogos. Nível {perfil.get('nivel_competitivo', '?')}. Disponível {perfil.get('disponibilidade', '?')}."

def get_fake_instance(args: argparse.Namespace) -> Faker:
    pid = os.getpid()
    if pid not in _worker_faker_instances:
        if args.detailed_log:
             logging.debug(f"Criando nova instância Faker para PID {pid} com locale {args.locale}")
        _worker_faker_instances[pid] = Faker(args.locale)
        if args.seed is not None:
             faker_seed = pid + args.seed
             random_seed = pid + args.seed + 1
             numpy_seed = (pid + args.seed + 2) % (2**32 - 1)
             Faker.seed(faker_seed)
             random.seed(random_seed)
             np.random.seed(numpy_seed)
             if args.detailed_log:
                  logging.debug(f"Worker PID {pid} seeded: Faker={faker_seed}, Random={random_seed}, Numpy={numpy_seed}")
    return _worker_faker_instances[pid]

def generate_profile_worker(worker_info: Tuple[int, int, argparse.Namespace]) -> Optional[Dict]:
    worker_id, profile_index, args = worker_info
    try:
        if args.seed is None:
            seed = os.getpid() + int(time.time_ns() / 1000) + profile_index * args.workers + worker_id
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            if args.detailed_log:
                 logging.debug(f"Worker {worker_id} (PID {os.getpid()}) gerando perfil {profile_index} com seed {seed}")

        fake = get_fake_instance(args)

        idade = random.randint(16, 75)
        cidade = random.choice(CIDADES_BRASIL) if CIDADES_BRASIL else "N/A"
        estado = obter_estado_por_cidade(cidade)
        sexo = random.choice(SEXOS) if SEXOS else "N/A"
        nome = gerar_nome(sexo, fake)

        num_musica = random.randint(1, min(6, len(ESTILOS_MUSICA)))
        interesses_musicais_list = random.sample(ESTILOS_MUSICA, k=num_musica) if ESTILOS_MUSICA else []
        num_jogos = random.randint(1, min(8, len(JOGOS_MAIS_JOGADOS)))
        jogos_favoritos_list = random.sample(JOGOS_MAIS_JOGADOS, k=num_jogos) if JOGOS_MAIS_JOGADOS else []
        num_plats = random.randint(1, min(4, len(PLATAFORMAS)))
        plataformas_possuidas_list = random.sample(PLATAFORMAS, k=num_plats) if PLATAFORMAS else []
        num_estilos = random.randint(1, min(6, len(ESTILOS_JOGO)))
        estilos_preferidos_list = random.sample(ESTILOS_JOGO, k=num_estilos) if ESTILOS_JOGO else []

        disponibilidade = gerar_horario_disponivel(idade)
        interacao_desejada = random.choice(INTERACAO) if INTERACAO else "N/A"
        compartilhar_contato = random.random() < 0.5

        anos_experiencia = max(0, random.gauss(mu=idade / 4.0, sigma=idade / 5.5))
        anos_experiencia = int(min(max(0, anos_experiencia), max(0, idade - 12)))
        objetivo_principal = random.choice(OBJETIVO_PRINCIPAL_LISTA) if OBJETIVO_PRINCIPAL_LISTA else "N/A"

        nivel_comp = "Não jogo ranked"
        if objetivo_principal == "Competitivo" or (objetivo_principal == "Casual" and random.random() < 0.3):
             if NIVEL_COMPETITIVO:
                 rank_weights = [0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0, 0.01, 0.0]
                 rank_weights.extend([0.01] * (len(NIVEL_COMPETITIVO) - len(rank_weights)))
                 rank_weights = rank_weights[:len(NIVEL_COMPETITIVO)]
                 exp_factor = math.log1p(anos_experiencia) / math.log1p(40)
                 shift = int(exp_factor * 4)
                 shifted_weights = np.roll(rank_weights, shift)
                 nivel_comp = escolher_com_peso(NIVEL_COMPETITIVO, shifted_weights.tolist())

        mic_prob = 0.80 if objetivo_principal == "Competitivo" else (0.65 if objetivo_principal == "Social" else 0.40)
        usa_microfone = random.random() < mic_prob
        idiomas = random.choice(IDIOMAS) if IDIOMAS else "Português"
        estilo_comunicacao = random.choice(ESTILO_COMUNICACAO) if ESTILO_COMUNICACAO else "N/A"

        perfil_data = {
            'nome': nome, 'idade': idade, 'cidade': cidade, 'estado': estado, 'sexo': sexo,
            'interesses_musicais': ', '.join(interesses_musicais_list),
            'jogos_favoritos': ', '.join(jogos_favoritos_list),
            'plataformas_possuidas': ', '.join(plataformas_possuidas_list),
            'estilos_preferidos': ', '.join(estilos_preferidos_list),
            'disponibilidade': disponibilidade,
            'interacao_desejada': interacao_desejada,
            'compartilhar_contato': compartilhar_contato,
            'anos_experiencia': anos_experiencia,
            'objetivo_principal': objetivo_principal,
            'usa_microfone': usa_microfone,
            'idiomas': idiomas,
            'nivel_competitivo': nivel_comp,
            'estilo_comunicacao': estilo_comunicacao,
            'descricao': ""
        }
        perfil_data['descricao'] = gerar_descricao_consistente(perfil_data)
        return perfil_data
    except Exception as e:
        logging.error(f"Erro worker {worker_id} (PID {os.getpid()}) perfil {profile_index}: {e}", exc_info=True)
        return None

# --- Vectorization and Embedding ---
_vector_maps_cache = {}

def _get_vector_maps():
    if not _vector_maps_cache:
        _vector_maps_cache['sexo'] = {s: i for i, s in enumerate(SEXOS)} if SEXOS else {}
        _vector_maps_cache['interacao'] = {it: i for i, it in enumerate(INTERACAO)} if INTERACAO else {}
        _vector_maps_cache['objetivo'] = {o: i for i, o in enumerate(OBJETIVO_PRINCIPAL_LISTA)} if OBJETIVO_PRINCIPAL_LISTA else {}
        _vector_maps_cache['nivel'] = {n: i for i, n in enumerate(NIVEL_COMPETITIVO)} if NIVEL_COMPETITIVO else {}
        _vector_maps_cache['comm'] = {c: i for i, c in enumerate(ESTILO_COMUNICACAO)} if ESTILO_COMUNICACAO else {}
        _vector_maps_cache['lang'] = {lang: i for i, lang in enumerate(IDIOMAS)} if IDIOMAS else {}
    return _vector_maps_cache

def gerar_vetor_perfil(perfil_row: pd.Series, args: argparse.Namespace) -> Optional[np.ndarray]:
    try:
        maps = _get_vector_maps()
        vetor = np.zeros(args.dim_vector, dtype=np.float32)
        idx = 0

        def assign_val(val):
            nonlocal idx # Correct Python 3 syntax
            if idx < args.dim_vector:
                vetor[idx] = val
                idx += 1
            # else: log warning or ignore if dim_vector is fixed

        assign_val(np.clip(perfil_row.get('idade', 30) / 80.0, 0, 1))
        assign_val(maps['sexo'].get(perfil_row.get('sexo', "?"), len(maps['sexo'])) / max(1, len(maps['sexo'])))
        assign_val(len(perfil_row.get('interesses_musicais', '').split(',')) / 10.0 if perfil_row.get('interesses_musicais') else 0.0)
        assign_val(len(perfil_row.get('jogos_favoritos', '').split(',')) / 15.0 if perfil_row.get('jogos_favoritos') else 0.0)
        assign_val(len(perfil_row.get('plataformas_possuidas', '').split(',')) / max(1, len(PLATAFORMAS)) if perfil_row.get('plataformas_possuidas') and PLATAFORMAS else 0.0)
        assign_val(len(perfil_row.get('estilos_preferidos', '').split(',')) / 10.0 if perfil_row.get('estilos_preferidos') else 0.0)
        assign_val(maps['interacao'].get(perfil_row.get('interacao_desejada', "?"), len(maps['interacao'])) / max(1, len(maps['interacao'])))
        assign_val(1.0 if perfil_row.get('compartilhar_contato', False) else 0.0)
        assign_val(np.clip(len(perfil_row.get('descricao', '')) / 700.0, 0, 1))
        assign_val(np.clip(perfil_row.get('anos_experiencia', 0) / 50.0, 0, 1))
        assign_val(maps['objetivo'].get(perfil_row.get('objetivo_principal', "?"), len(maps['objetivo'])) / max(1, len(maps['objetivo'])))
        assign_val(1.0 if perfil_row.get('usa_microfone', False) else 0.0)

        idioma_principal = perfil_row.get('idiomas', 'Português').split(',')[0].strip()
        assign_val(maps['lang'].get(idioma_principal, len(maps['lang'])) / max(1, len(maps['lang'])))
        assign_val(len(perfil_row.get('idiomas', 'Português').split(','))/ max(1, len(IDIOMAS)) if IDIOMAS else 0.0) # Normalize num languages
        assign_val(maps['nivel'].get(perfil_row.get('nivel_competitivo', "?"), len(maps['nivel'])) / max(1, len(maps['nivel'])))
        assign_val(maps['comm'].get(perfil_row.get('estilo_comunicacao', "?"), len(maps['comm'])) / max(1, len(maps['comm'])))

        while idx < args.dim_vector:
             assign_val(random.random() * 0.05)

        return np.nan_to_num(vetor).clip(0, 1)

    except Exception as e:
        profile_id = perfil_row.name if hasattr(perfil_row, 'name') else 'N/A'
        logging.error(f"Erro gerar vetor V6 perfil ID {profile_id}: {e}", exc_info=True)
        return None

def gerar_embedding_perfil(perfil_row: pd.Series, args: argparse.Namespace) -> Optional[np.ndarray]:
    try:
        hash_input = (
            f"{perfil_row.get('nome', '')[:10]}|" # Use first part of name for hash stability
            f"{perfil_row.get('descricao', '')[:60]}|"
            f"{perfil_row.get('jogos_favoritos', '')}|{perfil_row.get('estilos_preferidos', '')}|"
            f"{perfil_row.get('objetivo_principal', '')}|{perfil_row.get('nivel_competitivo', '')}|"
            f"{perfil_row.get('plataformas_possuidas', '')}|{perfil_row.get('estilo_comunicacao', '')}|"
            f"{perfil_row.get('idade', 0)}|{perfil_row.get('anos_experiencia', 0)}|{perfil_row.get('idiomas', '')}"
        )
        # Use a more robust hash function if collisions are a concern, but hash() is fine for simulation
        seed = hash(hash_input)
        rng = np.random.RandomState(seed % (2**32 - 1))
        embedding = rng.randn(args.dim_embedding).astype(np.float32)

        factor = (
            np.log1p(len(perfil_row.get('descricao', ''))) *
            (1 + 0.05 * len(perfil_row.get('jogos_favoritos','').split(','))) *
            (1 + 0.03 * len(perfil_row.get('estilos_preferidos','').split(','))) *
            (1 + 0.01 * perfil_row.get('idade', 30)) *
            (1 + 0.04 * np.log1p(perfil_row.get('anos_experiencia', 0))) *
            (1.05 if perfil_row.get('usa_microfone', False) else 0.95) *
            (1 + 0.02 * len(perfil_row.get('idiomas', 'P').split(',')))
        )

        nivel_idx = -1
        if NIVEL_COMPETITIVO and perfil_row.get('nivel_competitivo') in NIVEL_COMPETITIVO:
            nivel_idx = NIVEL_COMPETITIVO.index(perfil_row.get('nivel_competitivo'))

        if nivel_idx != -1 and NIVEL_COMPETITIVO:
             factor *= (1 + 0.1 * (nivel_idx / len(NIVEL_COMPETITIVO)))

        embedding = embedding * (factor / 5.0) # Normalize factor influence

        norm = np.linalg.norm(embedding)
        if norm > 0: embedding = embedding / norm
        return np.nan_to_num(embedding)

    except Exception as e:
        profile_id = perfil_row.name if hasattr(perfil_row, 'name') else 'N/A'
        logging.error(f"Erro gerar embedding V6 perfil ID {profile_id}: {e}", exc_info=True)
        return None

def process_chunk_vectors_embeddings(chunk_data: Tuple[pd.DataFrame, argparse.Namespace]) -> pd.DataFrame:
    df_chunk, args = chunk_data
    if df_chunk.empty: return df_chunk

    # Ensure columns exist before applying
    if 'vetor' not in df_chunk.columns: df_chunk['vetor'] = pd.Series(index=df_chunk.index, dtype=object)
    if 'embedding' not in df_chunk.columns: df_chunk['embedding'] = pd.Series(index=df_chunk.index, dtype=object)

    try:
        df_chunk['vetor'] = df_chunk.apply(lambda row: gerar_vetor_perfil(row, args), axis=1)
        df_chunk['embedding'] = df_chunk.apply(lambda row: gerar_embedding_perfil(row, args), axis=1)
        if args.detailed_log:
             null_vecs = df_chunk['vetor'].isnull().sum()
             null_embs = df_chunk['embedding'].isnull().sum()
             logging.debug(f"PID {os.getpid()} processou chunk {len(df_chunk)}. Nulos V:{null_vecs} E:{null_embs}")
        return df_chunk
    except Exception as e:
         logging.error(f"Erro fatal worker (PID {os.getpid()}) processando chunk: {e}", exc_info=True)
         # Return chunk with potential NaNs/Nones where errors occurred
         return df_chunk

# --- Database Operations ---
def setup_database_pragmas(conn: sqlite3.Connection):
    try:
        cursor = conn.cursor()
        pragmas = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = -16000;",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA foreign_keys = OFF;"
        ]
        logging.debug(f"Aplicando PRAGMAs...")
        for pragma in pragmas:
            try:
                 cursor.execute(pragma)
                 if "journal_mode" in pragma.lower():
                      res = cursor.fetchone()
                      logging.debug(f"Executado: {pragma} -> Resultado: {res}")
            except sqlite3.Error as e:
                 logging.warning(f"Falha ao executar PRAGMA '{pragma}': {e}")
        logging.info("PRAGMAs SQLite configurados.")
    except sqlite3.Error as e:
        logging.warning(f"Não foi possível configurar PRAGMAs: {e}")

def criar_tabelas_otimizadas(args: argparse.Namespace) -> None:
    databases = {
        get_db_path(args, DB_PERFIS_NAME): f'''
            CREATE TABLE IF NOT EXISTS perfis (
                id INTEGER PRIMARY KEY, nome TEXT NOT NULL, idade INTEGER,
                cidade TEXT, estado TEXT, sexo TEXT,
                interesses_musicais TEXT, jogos_favoritos TEXT, plataformas_possuidas TEXT,
                estilos_preferidos TEXT, disponibilidade TEXT, interacao_desejada TEXT,
                compartilhar_contato INTEGER, anos_experiencia INTEGER, -- BOOLEAN as INTEGER
                objetivo_principal TEXT, usa_microfone INTEGER, -- BOOLEAN as INTEGER
                idiomas TEXT, nivel_competitivo TEXT, estilo_comunicacao TEXT,
                descricao TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_perfis_estado ON perfis (estado);
            CREATE INDEX IF NOT EXISTS idx_perfis_idade ON perfis (idade);
            CREATE INDEX IF NOT EXISTS idx_perfis_objetivo ON perfis (objetivo_principal);
            CREATE INDEX IF NOT EXISTS idx_perfis_nivel_comp ON perfis (nivel_competitivo);
        ''',
        get_db_path(args, DB_VETORES_NAME): f'''
            CREATE TABLE IF NOT EXISTS vetores (id INTEGER PRIMARY KEY, vetor BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_vetores_id ON vetores (id);
        ''',
        get_db_path(args, DB_EMBEDDINGS_NAME): f'''
            CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY, embedding BLOB NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_embeddings_id ON embeddings (id);
        ''',
        get_db_path(args, DB_CLUSTERS_NAME): f'''
            CREATE TABLE IF NOT EXISTS clusters (id INTEGER PRIMARY KEY, cluster_id INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON clusters (cluster_id);
            CREATE INDEX IF NOT EXISTS idx_clusters_id ON clusters (id);
        '''
    }
    console_print(f"⚙️ [bold cyan]V6: Verificando/Criando tabelas em '{args.db_dir}'...[/]", args)
    os.makedirs(args.db_dir, exist_ok=True)
    for db_path, create_sql in databases.items():
        console_print(f"   Configurando {os.path.basename(db_path)}...", args, style="dim")
        try:
            with sqlite3.connect(db_path, timeout=15.0) as conn:
                 setup_database_pragmas(conn)
                 conn.executescript(create_sql)
            logging.info(f"Tabela V6 verificada/criada com sucesso em {db_path}")
        except sqlite3.Error as e:
            console_print(f"❌ [bold red]Erro CRÍTICO ao criar/verificar tabela em {db_path}: {e}[/bold red]", args)
            logging.critical(f"Erro SQLite CRÍTICO ao configurar {db_path}: {e}", exc_info=True)
            raise

def inserir_dataframe_no_db(df: pd.DataFrame, db_path: str, table_name: str, args: argparse.Namespace, include_index: bool = False, chunksize: int = 1000) -> bool:
    if df.empty:
        logging.info(f"DataFrame vazio, nada a inserir em '{db_path}.{table_name}'.")
        return True

    logging.info(f"Iniciando inserção V6 de {len(df)} linhas em '{db_path}.{table_name}'...")
    start_time = time.time()
    try:
        df_copy = df.copy()
        # Explicitly convert boolean columns to integer (0 or 1) for SQLite compatibility
        for col in df_copy.select_dtypes(include=['bool', 'boolean']).columns:
             # Handle potential Pandas nullable boolean type
             df_copy[col] = df_copy[col].fillna(0).astype(int)

        with sqlite3.connect(db_path, timeout=30.0) as conn:
             # setup_database_pragmas(conn) # Already active if WAL mode set previously
             df_copy.to_sql(table_name, conn, if_exists='append', index=include_index, chunksize=chunksize, method='multi')
        end_time = time.time()
        logging.info(f"{len(df)} registros V6 inseridos em '{db_path}.{table_name}' em {end_time - start_time:.2f}s.")
        return True
    except (sqlite3.Error, ValueError, AttributeError, TypeError, pd.errors.DatabaseError) as e:
        logging.error(f"Erro ao inserir DataFrame V6 (to_sql) em {db_path}.{table_name}. Erro: {e}", exc_info=True)
        try:
            logging.error(f"Colunas do DataFrame: {df.columns.tolist()}")
            logging.error(f"Tipos de dados do DataFrame:\n{df.dtypes.to_string()}")
            logging.error(f"Primeiras 5 linhas do DF que falhou:\n{df.head().to_string()}")
        except Exception as inspect_err:
             logging.error(f"Erro adicional ao inspecionar DataFrame: {inspect_err}")
        console_print(f"❌ [bold red]Erro ao inserir DataFrame V6 em '{db_path}.{table_name}': {e}[/bold red]", args)
        return False

def salvar_blobs_lote(dados: List[Tuple[int, Optional[bytes]]], db_path: str, table_name: str, column_name: str, args: argparse.Namespace) -> bool:
    dados_validos = [(id_val, blob) for id_val, blob in dados if blob is not None and isinstance(id_val, (int, np.integer))]
    num_total = len(dados)
    num_validos = len(dados_validos)
    num_nulos_invalidos = num_total - num_validos

    if num_nulos_invalidos > 0:
        logging.warning(f"[{table_name}] {num_nulos_invalidos}/{num_total} blobs ou IDs eram nulos/inválidos e foram pulados.")
    if not dados_validos:
        logging.info(f"Nenhum blob válido para salvar em '{db_path}.{table_name}'.")
        return True

    sql = f"INSERT OR REPLACE INTO {table_name} (id, {column_name}) VALUES (?, ?)"
    logging.info(f"Iniciando salvamento V6 de {num_validos} blobs em '{db_path}.{table_name}'...")
    start_time = time.time()
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            # conn.execute("PRAGMA synchronous = OFF") # Could be faster but riskier
            conn.execute("BEGIN;")
            try:
                cursor = conn.cursor()
                cursor.executemany(sql, dados_validos)
                conn.commit()
                end_time = time.time()
                logging.info(f"{num_validos} blobs V6 salvos em '{db_path}.{table_name}' em {end_time - start_time:.2f}s.")
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logging.error(f"Erro SQLite (executemany) V6 em {db_path}.{table_name}: {e}", exc_info=True)
                console_print(f"❌ [bold red]Erro salvamento em lote V6 em '{db_path}.{table_name}': {e}[/bold red]", args)
                return False
    except sqlite3.Error as e:
        logging.error(f"Erro SQLite (conexão/transação) V6 em {db_path}.{table_name}: {e}", exc_info=True)
        console_print(f"❌ [bold red]Erro conexão/transação V6 ao salvar blobs em '{db_path}.{table_name}': {e}[/bold red]", args)
        return False

# --- Clustering ---
def realizar_clustering(embeddings: np.ndarray, num_clusters: int, args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Optional[faiss.Index], Optional[float], Optional[np.ndarray], Dict[str, Optional[float]]]:
    cluster_assignments, faiss_index, inertia, centroids, metrics = None, None, None, None, {'silhouette': None, 'davies_bouldin': None}
    if embeddings is None or embeddings.shape[0] == 0:
        console_print("⚠️ [bold yellow]Nenhum embedding válido para clustering.[/bold yellow]", args)
        return cluster_assignments, faiss_index, inertia, centroids, metrics
    if num_clusters <= 0:
        logging.warning("Número de clusters <= 0, pulando clustering.")
        return cluster_assignments, faiss_index, inertia, centroids, metrics

    actual_profiles, dimension = embeddings.shape
    if actual_profiles < num_clusters:
        logging.warning(f"N={actual_profiles} < k={num_clusters}. Ajustando k={actual_profiles}.")
        num_clusters = actual_profiles
    if num_clusters <= 1:
         logging.warning(f"Número efetivo de clusters é {num_clusters}. Pulando clustering real e métricas.")
         return None, None, None, None, {}

    console_print(f"📊 [bold blue]Iniciando Clustering KMeans V6 (k={num_clusters}, N={actual_profiles}, Dim={dimension})...[/]", args)
    console_print(f"   [blue]Params: niters={args.kmeans_niter}, nredo={args.kmeans_nredo}, seed={args.kmeans_seed}, gpu={args.kmeans_gpu}[/]", args, style="dim")

    embeddings_faiss = np.ascontiguousarray(embeddings, dtype=np.float32)

    gpu_option = args.kmeans_gpu
    res = None
    if gpu_option:
        try:
            if not faiss.get_num_gpus() > 0:
                raise RuntimeError("Nenhuma GPU detectada pelo FAISS.")
            res = faiss.StandardGpuResources()
            logging.info("Recursos GPU FAISS alocados.")
        except (AttributeError, RuntimeError, Exception) as gpu_err:
            logging.warning(f"Falha ao alocar GPU FAISS ou GPU não disponível: {gpu_err}. Usando CPU.")
            gpu_option = False
            res = None

    max_points = getattr(args, 'kmeans_max_points', KMEANS_MAX_POINTS_DEFAULT)

    kmeans = faiss.Kmeans(d=dimension, k=num_clusters, niter=args.kmeans_niter,
                          nredo=args.kmeans_nredo, verbose=args.detailed_log,
                          gpu=gpu_option, max_points_per_centroid=max_points,
                          seed=args.kmeans_seed)
    try:
        train_start = time.time()
        kmeans.train(embeddings_faiss)
        train_time = time.time() - train_start

        if not hasattr(kmeans, 'centroids') or kmeans.centroids is None or kmeans.centroids.shape[0] != num_clusters:
             console_print(f"❌ [bold red]Erro Treinamento KMeans V6: Centróides inválidos ou não gerados.[/bold red]", args)
             logging.error(f"Falha Treinamento KMeans V6. kmeans.centroids: {getattr(kmeans, 'centroids', 'N/A')}")
             return None, None, None, None, {}

        centroids = kmeans.centroids.copy()

        inertia = None
        if hasattr(kmeans, 'obj') and kmeans.obj is not None:
            obj_list = kmeans.obj if isinstance(kmeans.obj, (list, np.ndarray)) else []
            if obj_list: inertia = obj_list[-1]
            elif isinstance(kmeans.obj, (int, float)): inertia = kmeans.obj # Handle scalar case
            else: logging.warning(f"kmeans.obj tipo inesperado: {type(kmeans.obj)}")

        assign_start = time.time()
        if not hasattr(kmeans, 'index') or kmeans.index is None:
            console_print(f"❌ [bold red]Erro Pós-Treinamento KMeans V6: kmeans.index não encontrado.[/bold red]", args)
            logging.error("Falha Pós-Treinamento KMeans V6: kmeans.index não existe.")
            return None, None, inertia, centroids, {}

        # Index might be on GPU, transfer if needed? Usually search handles this.
        # If using GPU, kmeans.index is often a GpuIndex. Search should work.
        D, I = kmeans.index.search(embeddings_faiss, 1)
        cluster_assignments = I.flatten()
        assign_time = time.time() - assign_start

        # If index was on GPU, we might want to copy it back to CPU for saving
        faiss_index = faiss.index_gpu_to_cpu(kmeans.index) if gpu_option and hasattr(faiss, 'index_gpu_to_cpu') and isinstance(kmeans.index, faiss.GpuIndex) else kmeans.index


        console_print(f"✅ [bold green]Clustering concluído em {train_time:.2f}s (T) + {assign_time:.2f}s (A).[/]", args)
        if inertia is not None:
             console_print(f"   [green]Inércia Final (WCSS): {inertia:,.4f}[/]", args)
             logging.info(f"KMeans V6 OK. k={num_clusters}, inertia={inertia:.4f}")
        else:
             logging.info(f"KMeans V6 OK. k={num_clusters} (Inércia N/A)")

        # Optional Metrics Calculation
        if SKLEARN_AVAILABLE and not args.skip_cluster_metrics and num_clusters > 1 and actual_profiles > num_clusters:
            console_print(f"   [blue]Calculando métricas de cluster (pode levar tempo)...[/]", args, style="dim")
            metrics_start = time.time()
            sample_size = min(5000, actual_profiles // 2) if actual_profiles > 10 else actual_profiles

            try:
                 metrics['silhouette'] = silhouette_score(embeddings_faiss, cluster_assignments, metric='cosine', sample_size=sample_size)
                 console_print(f"   [green]Silhouette Score (sample={sample_size}, cosine): {metrics['silhouette']:.4f}[/]", args)
            except ValueError as sil_err:
                 logging.warning(f"Não foi possível calcular Silhouette Score: {sil_err}")
                 console_print(f"   [yellow]Silhouette Score N/A: {sil_err}[/]", args)
            except Exception as e:
                 logging.error(f"Erro inesperado calculando Silhouette Score: {e}", exc_info=True)

            try:
                 metrics['davies_bouldin'] = davies_bouldin_score(embeddings_faiss, cluster_assignments)
                 console_print(f"   [green]Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}[/]", args)
            except ValueError as db_err:
                 logging.warning(f"Não foi possível calcular Davies-Bouldin Score: {db_err}")
                 console_print(f"   [yellow]Davies-Bouldin Score N/A: {db_err}[/]", args)
            except Exception as e:
                 logging.error(f"Erro inesperado calculando Davies-Bouldin Score: {e}", exc_info=True)
            logging.info(f"Métricas de Cluster: Silhouette={metrics['silhouette']:.4f if metrics['silhouette'] else 'N/A'}, Davies-Bouldin={metrics['davies_bouldin']:.4f if metrics['davies_bouldin'] else 'N/A'} (Tempo: {time.time() - metrics_start:.2f}s)")
        elif not SKLEARN_AVAILABLE:
            logging.info("Métricas de cluster puladas (scikit-learn não disponível).")
        elif args.skip_cluster_metrics:
            logging.info("Métricas de cluster puladas (via flag --skip-cluster-metrics).")
        elif num_clusters <= 1 or actual_profiles <= num_clusters:
             logging.info(f"Métricas de cluster puladas (k={num_clusters}, N={actual_profiles}).")

        return cluster_assignments, faiss_index, inertia, centroids, metrics

    except (RuntimeError, MemoryError, Exception) as faiss_err:  # Removido FaissException
        console_print(f"❌ [bold red]Erro FAISS V6: {faiss_err}[/bold red]", args)
        logging.error(f"Erro FAISS V6: {faiss_err}", exc_info=True)
        return None, None, None, None, {}
    except Exception as e:
        console_print(f"❌ [bold red]Erro Inesperado Clustering V6: {e.__class__.__name__}: {e}[/bold red]", args)
        logging.error(f"Erro inesperado Clustering V6: {e}", exc_info=True)
        return None, None, None, None, {}
    finally:
        # Explicitly free GPU resources if they were allocated
        if res is not None:
            del res
            logging.info("Recursos GPU FAISS liberados.")


# --- Saving Functions ---
def salvar_clusters_lote(cluster_data: List[Tuple[int, int]], db_path: str, args: argparse.Namespace) -> bool:
    if not cluster_data:
        logging.info(f"Nenhum dado de cluster para salvar em '{db_path}'.")
        return True

    # Ensure IDs are standard Python ints for executemany
    valid_cluster_data = [(int(id_val), int(clu_id)) for id_val, clu_id in cluster_data]

    sql = "INSERT OR REPLACE INTO clusters (id, cluster_id) VALUES (?, ?)"
    logging.info(f"Iniciando salvamento V6 de {len(valid_cluster_data)} atribuições de cluster em '{db_path}'...")
    start_time = time.time()
    try:
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            conn.execute("BEGIN;")
            try:
                cursor = conn.cursor()
                cursor.executemany(sql, valid_cluster_data)
                conn.commit()
                end_time = time.time()
                logging.info(f"{len(valid_cluster_data)} atribuições cluster V6 salvas em {end_time - start_time:.2f}s.")
                return True
            except sqlite3.Error as e:
                 conn.rollback()
                 logging.error(f"Erro SQLite (executemany clusters) V6 em {db_path}: {e}", exc_info=True)
                 console_print(f"❌ [bold red]Erro salvamento clusters V6 em '{db_path}': {e}[/bold red]", args)
                 return False
    except sqlite3.Error as e:
        logging.error(f"Erro SQLite (conexão/transação clusters) V6 em {db_path}: {e}", exc_info=True)
        console_print(f"❌ [bold red]Erro conexão/transação V6 clusters em '{db_path}': {e}[/bold red]", args)
        return False

def salvar_indice_faiss(index: faiss.Index, filepath: str, args: argparse.Namespace) -> bool:
    if index is None:
        logging.warning("Tentativa de salvar índice FAISS nulo.")
        return False
    console_print(f"💾 [blue]Salvando índice FAISS treinado em '{filepath}'...[/]", args)
    start_time = time.time()
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(index, filepath)
        end_time = time.time()
        console_print(f"✅ [green]Índice FAISS salvo com sucesso em {end_time - start_time:.2f}s.[/]", args)
        logging.info(f"Índice FAISS V6 ({type(index)}) salvo em '{filepath}' em {end_time - start_time:.2f}s.")
        return True
    except Exception as e:
        console_print(f"❌ [bold red]Erro ao salvar índice FAISS V6 em '{filepath}': {e}[/bold red]", args)
        logging.error(f"Erro ao salvar índice FAISS V6: {e}", exc_info=True)
        return False

def salvar_centroides(centroids: np.ndarray, filepath: str, args: argparse.Namespace) -> bool:
    if centroids is None:
        logging.warning("Tentativa de salvar centróides nulos.")
        return False
    console_print(f"💾 [blue]Salvando centróides ({centroids.shape}) em '{filepath}'...[/]", args)
    start_time = time.time()
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, centroids)
        end_time = time.time()
        console_print(f"✅ [green]Centróides salvos com sucesso em {end_time - start_time:.2f}s.[/]", args)
        logging.info(f"Centróides V6 ({centroids.shape}) salvos em '{filepath}' em {end_time - start_time:.2f}s.")
        return True
    except Exception as e:
        console_print(f"❌ [bold red]Erro ao salvar centróides V6 em '{filepath}': {e}[/bold red]", args)
        logging.error(f"Erro ao salvar centróides V6: {e}", exc_info=True)
        return False

def vacuum_database(db_path: str, args: argparse.Namespace):
    if not os.path.exists(db_path):
        logging.warning(f"Banco de dados '{db_path}' não encontrado para VACUUM.")
        return
    console_print(f"🧹 [blue]Executando VACUUM em '{os.path.basename(db_path)}'... (Pode demorar)[/]", args)
    start_time = time.time()
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
             # conn.execute("PRAGMA busy_timeout = 60000;") # 60 seconds timeout if busy
             conn.execute("VACUUM;")
        end_time = time.time()
        console_print(f"✅ [green]VACUUM concluído em '{os.path.basename(db_path)}' em {end_time - start_time:.2f}s.[/]", args)
        logging.info(f"VACUUM V6 executado em '{db_path}' em {end_time - start_time:.2f}s.")
    except sqlite3.Error as e:
        console_print(f"❌ [bold red]Erro ao executar VACUUM V6 em '{os.path.basename(db_path)}': {e}[/bold red]", args)
        logging.error(f"Erro durante VACUUM V6 em {db_path}: {e}", exc_info=True)

# --- Main Execution Pipeline ---
def main(args: argparse.Namespace):
    overall_start_time = time.time()
    step_timers = {}
    global timestamp # Make timestamp global for use in filenames later
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    console_print(f"[bold blue]🚀 INICIANDO GERADOR DE PERFIS OTIMIZADO V6 🚀[/]", args)
    # Print key args
    console_print(f"   Profiles: {args.num_profiles}, Workers: {args.workers}, Locale: {args.locale}, Seed: {args.seed if args.seed is not None else 'None'}", args, style="dim")
    console_print(f"   Dims: Vector={args.dim_vector}, Embedding={args.dim_embedding}", args, style="dim")
    console_print(f"   DB: '{args.db_dir}', FAISS: '{args.faiss_dir}', Logs: '{args.log_dir}'", args, style="dim")
    cluster_method_str = "Skipped" if args.skip_clustering else f"{args.cluster_method} (k={args.fixed_clusters if args.cluster_method == 'fixed' else 'sqrt'})"
    console_print(f"   Clustering: {cluster_method_str}", args, style="dim")
    console_print(f"   Flags: ScaleVec={'No' if args.skip_vector_scaling else 'Yes'}, Metrics={'No' if args.skip_cluster_metrics else 'Yes'}, SaveIdx={'No' if not args.save_faiss_index else 'Yes'}, SaveCentr={'No' if not args.save_centroids else 'Yes'}, Vacuum={'Yes' if args.vacuum_dbs else 'No'}", args, style="dim")
    if not SKLEARN_AVAILABLE:
        console_print("   [yellow]Aviso:[/yellow] scikit-learn não encontrado. Normalização e métricas de cluster desabilitadas.", args)
    console.rule()

    current_step_start_time = time.time()
    console_print("\n stepfather [bold]1. Preparando Bancos de Dados V6...[/]", args)
    try:
        criar_tabelas_otimizadas(args)
        step_timers['1_Setup_DB'] = time.time() - current_step_start_time
        console_print(f"✅ [green]Bancos de dados prontos em {step_timers['1_Setup_DB']:.2f}s.[/green]", args)
    except Exception as e:
        console_print(f"❌ [bold red]Falha CRÍTICA Etapa 1: {e}[/bold red]", args)
        logging.critical(f"Falha CRÍTICA Etapa 1: {e}", exc_info=True)
        return

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]2. Gerando {args.num_profiles} perfis ({args.workers} workers)...[/]", args)
    perfis_gerados_list: List[Dict] = []
    num_gerados = 0
    if args.num_profiles <= 0:
        console_print("ℹ️ [yellow]Nenhum perfil a gerar (num_profiles <= 0).[/yellow]", args)
        step_timers['2_Generate_Profiles'] = 0.0
    else:
        try:
            tasks_args = [(i % args.workers, i, args) for i in range(args.num_profiles)]
            imap_chunksize = max(1, args.num_profiles // (args.workers * 10)) if args.num_profiles > 0 else 1

            with Pool(processes=args.workers) as pool:
                console_print(f"   Lançando tasks de geração V6 (imap chunksize={imap_chunksize})...", args, style="dim")
                results_iterator = pool.imap_unordered(generate_profile_worker, tasks_args, chunksize=imap_chunksize)

                prog_cols = (SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                             TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                             TimeRemainingColumn(), TimeElapsedColumn())
                with Progress(*prog_cols, console=console, disable=args.quiet, transient=False) as progress:
                    task_gen = progress.add_task("[yellow]Gerando perfis...", total=args.num_profiles)
                    processed_count = 0
                    for perfil in results_iterator:
                        if perfil: perfis_gerados_list.append(perfil)
                        processed_count += 1
                        progress.update(task_gen, completed=processed_count)
                        if processed_count % max(1, args.num_profiles // 20) == 0:
                             logging.debug(f"Geração: {processed_count}/{args.num_profiles}")

            num_gerados = len(perfis_gerados_list)
            step_timers['2_Generate_Profiles'] = time.time() - current_step_start_time
            console_print(f"✅ [green]{num_gerados}/{args.num_profiles} perfis gerados em {step_timers['2_Generate_Profiles']:.2f}s.[/green]", args)
            logging.info(f"Etapa 2: {num_gerados}/{args.num_profiles} perfis gerados em {step_timers['2_Generate_Profiles']:.2f}s.")
            if num_gerados < args.num_profiles:
                falhas = args.num_profiles - num_gerados
                console_print(f"⚠️ [yellow]{falhas} perfis falharam na geração (ver logs).[/yellow]", args)
                logging.warning(f"{falhas} falhas na geração (Etapa 2).")

        except Exception as e:
            step_timers['2_Generate_Profiles'] = time.time() - current_step_start_time
            console_print(f"❌ [bold red]Erro Geração Paralela (Etapa 2): {e}[/bold red]", args)
            logging.critical(f"Erro Pool Geração (Etapa 2): {e}", exc_info=True)
            # Continue only if some profiles were generated
            if not perfis_gerados_list: return

    if not perfis_gerados_list:
        console_print("❌ [bold red]Nenhum perfil gerado. Encerrando.[/bold red]", args)
        return

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]3. DataFrame e Salvamento DB Perfis V6...[/]", args)
    perfis_df = pd.DataFrame()
    inserted_ids = []
    try:
        perfis_df = pd.DataFrame(perfis_gerados_list)
        if perfis_df.empty: raise ValueError("DataFrame V6 vazio pós-geração.")
        console_print(f"   DataFrame criado com {len(perfis_df)} linhas.", args, style="dim")
        logging.info(f"DF inicial V6: {len(perfis_df)} linhas, Colunas: {perfis_df.columns.tolist()}")

        if not args.quiet and len(perfis_df) > 0:
             try:
                  console_print("[cyan]Estatísticas Descritivas (Numéricas):[/]", args)
                  console.print(perfis_df.describe(include=np.number).round(2).to_string(), style="dim")
             except Exception as stats_err:
                  console_print(f"⚠️ [yellow]Não foi possível gerar estatísticas: {stats_err}[/yellow]", args)
                  logging.warning(f"Falha gerar estatísticas DF: {stats_err}")

        colunas_db_v6 = ['nome', 'idade', 'cidade', 'estado', 'sexo',
                         'interesses_musicais', 'jogos_favoritos', 'plataformas_possuidas',
                         'estilos_preferidos', 'disponibilidade', 'interacao_desejada',
                         'compartilhar_contato', 'anos_experiencia', 'objetivo_principal',
                         'usa_microfone', 'idiomas', 'nivel_competitivo', 'estilo_comunicacao', 'descricao']

        for col in colunas_db_v6:
             if col not in perfis_df.columns:
                  logging.warning(f"Coluna DB '{col}' ausente no DF. Adicionando com None.")
                  # Determine appropriate fill value based on expected type
                  if col in ['compartilhar_contato', 'usa_microfone']:
                       perfis_df[col] = False # Default boolean to False
                  elif col in ['idade', 'anos_experiencia']:
                       perfis_df[col] = 0 # Default numeric to 0
                  else:
                       perfis_df[col] = None # Default others to None
        perfis_df_to_db = perfis_df[colunas_db_v6]

        db_path_perfis = get_db_path(args, DB_PERFIS_NAME)
        insert_ok = inserir_dataframe_no_db(perfis_df_to_db, db_path_perfis, 'perfis', args)
        if not insert_ok:
             raise RuntimeError("Falha ao inserir DataFrame de perfis no banco de dados.")

        # Retrieve IDs reliably
        with sqlite3.connect(db_path_perfis, timeout=45.0) as conn:
            logging.info("Buscando IDs dos perfis inseridos...")
            cursor = conn.cursor()
            # Fetch IDs corresponding to the number of rows we *attempted* to insert
            cursor.execute("SELECT id FROM perfis ORDER BY rowid DESC LIMIT ?", (len(perfis_df_to_db),))
            fetched_ids = [item[0] for item in cursor.fetchall()]

            if len(fetched_ids) == len(perfis_df):
                inserted_ids = fetched_ids[::-1] # Reverse to match original order
                logging.info(f"IDs recuperados com sucesso ({len(inserted_ids)}).")
            else:
                # Fallback: try fetching all IDs and taking the last N (less reliable if concurrent writes)
                logging.warning(f"Inconsistência IDs ({len(fetched_ids)} vs {len(perfis_df)}). Tentando alternativa (pode falhar com concorrência).")
                cursor.execute("SELECT id FROM perfis")
                all_ids = [item[0] for item in cursor.fetchall()]
                if len(all_ids) >= len(perfis_df):
                     inserted_ids = all_ids[-len(perfis_df):]
                     logging.info(f"Recuperação IDs alternativa OK ({len(inserted_ids)}).")
                else:
                     # This case indicates a severe problem (fewer IDs in DB than inserted rows)
                     raise sqlite3.IntegrityError(f"Falha grave recuperação IDs ({len(all_ids)} vs {len(perfis_df)}). Dados podem estar inconsistentes.")

        perfis_df['id'] = inserted_ids
        perfis_df.set_index('id', inplace=True)
        logging.info(f"DF atualizado com IDs. Exemplo índices: {perfis_df.index[:3].tolist()}...")

        step_timers['3_DataFrame_SavePerfis'] = time.time() - current_step_start_time
        console_print(f"✅ [green]Perfis salvos no DB ({os.path.basename(db_path_perfis)}) em {step_timers['3_DataFrame_SavePerfis']:.2f}s.[/green]", args)
        logging.info(f"Etapa 3: Perfis salvos e DF atualizado em {step_timers['3_DataFrame_SavePerfis']:.2f}s.")

    except (pd.errors.EmptyDataError, ValueError, sqlite3.Error, RuntimeError, Exception) as e:
        step_timers['3_DataFrame_SavePerfis'] = time.time() - current_step_start_time
        console_print(f"❌ [bold red]Erro CRÍTICO Etapa 3: {e}[/bold red]", args)
        logging.critical(f"Erro CRÍTICO Etapa 3: {e}", exc_info=True)
        return

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]4. Vetorização/Embedding V6 ({args.workers} workers)...[/]", args)
    perfis_df_processed = pd.DataFrame()
    if perfis_df.empty:
        console_print("ℹ️ [yellow]DataFrame vazio. Pulando Etapa 4.[/yellow]", args)
        step_timers['4_Vectorize_Embed'] = 0.0
    else:
        try:
            # More robust chunk splitting
            num_rows = len(perfis_df)
            target_chunks = max(1, args.workers * CHUNK_SIZE_FACTOR * 2)
            num_splits = min(num_rows, target_chunks) if num_rows > 0 else 1

            # Create chunks ensuring the DataFrame index is preserved for reassembly
            chunks_with_args = [(perfis_df.iloc[idx], args) for idx in np.array_split(np.arange(num_rows), num_splits) if len(idx) > 0]

            if not chunks_with_args: raise ValueError("Nenhum chunk válido para processamento V6.")

            processed_chunks = []
            with Pool(processes=args.workers) as pool:
                console_print(f"   Lançando tasks Vet/Emb V6 no Pool ({len(chunks_with_args)} chunks)...", args, style="dim")
                # Use imap_unordered for potentially better load balancing
                results_iterator = pool.imap_unordered(process_chunk_vectors_embeddings, chunks_with_args, chunksize=1)

                prog_cols = (SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                             TextColumn("{task.completed}/{task.total} Chunks"), TimeElapsedColumn())
                with Progress(*prog_cols, console=console, disable=args.quiet, transient=False) as progress:
                    task_vec = progress.add_task("[yellow]Processando chunks...", total=len(chunks_with_args))
                    for chunk_result in results_iterator:
                        if chunk_result is not None and not chunk_result.empty:
                            processed_chunks.append(chunk_result)
                        progress.update(task_vec, advance=1)

            if not processed_chunks: raise ValueError("Nenhum chunk V6 processado com sucesso.")

            # Concatenate preserving the original index order
            perfis_df_processed = pd.concat(processed_chunks).sort_index()
            original_count = len(perfis_df_processed)

            # Check for None vectors/embeddings created by errors in worker functions
            valid_mask = perfis_df_processed['vetor'].notna() & perfis_df_processed['embedding'].notna()
            perfis_df_processed = perfis_df_processed[valid_mask]
            processed_count = len(perfis_df_processed)
            failures = original_count - processed_count

            step_timers['4_Vectorize_Embed'] = time.time() - current_step_start_time
            console_print(f"✅ [green]Vetorização/Embedding V6 concluídos em {step_timers['4_Vectorize_Embed']:.2f}s.[/green]", args)
            logging.info(f"Etapa 4: Vet/Emb V6 concluído em {step_timers['4_Vectorize_Embed']:.2f}s.")
            if failures > 0:
                console_print(f"⚠️ [yellow]{failures} perfis removidos por falha vet/emb (ver logs).[/yellow]", args)
                logging.warning(f"{failures} falhas vet/emb (Etapa 4).")

        except (ValueError, TypeError, MemoryError, Exception) as e:
            step_timers['4_Vectorize_Embed'] = time.time() - current_step_start_time
            console_print(f"❌ [bold red]Erro CRÍTICO Vet/Emb (Etapa 4): {e}[/bold red]", args)
            logging.critical(f"Erro Pool Vet/Emb V6 (Etapa 4): {e}", exc_info=True)
            return

    if perfis_df_processed.empty:
         console_print("❌ [bold red]Nenhum perfil válido após Etapa 4. Encerrando.[/bold red]", args)
         return

    current_step_start_time = time.time()
    vector_column_to_save = 'vetor'
    if SKLEARN_AVAILABLE and not args.skip_vector_scaling:
        console_print(f"\n stepfather [bold]4.5. Escalonando Vetores (MinMaxScaler)...[/]", args)
        try:
             # Ensure vectors are valid before stacking
             valid_vectors = perfis_df_processed['vetor'].dropna()
             if not valid_vectors.empty:
                 vector_matrix = np.stack(valid_vectors.values)
                 if vector_matrix.ndim == 2 and vector_matrix.shape[1] == args.dim_vector: # Basic sanity check
                     scaler = MinMaxScaler()
                     scaled_vectors_matrix = scaler.fit_transform(vector_matrix)
                     # Add scaled vectors as a new column or update existing (careful with index alignment)
                     # Creating a new column is safer if there were NaNs dropped
                     scaled_series = pd.Series(list(scaled_vectors_matrix), index=valid_vectors.index)
                     perfis_df_processed['vetor_scaled'] = scaled_series
                     vector_column_to_save = 'vetor_scaled'
                     step_timers['4.5_Scale_Vectors'] = time.time() - current_step_start_time
                     console_print(f"✅ [green]Vetores escalonados ({vector_column_to_save}) em {step_timers['4.5_Scale_Vectors']:.2f}s.[/green]", args)
                     logging.info(f"Etapa 4.5: Vetores escalonados (MinMaxScaler) em {step_timers['4.5_Scale_Vectors']:.2f}s.")
                 else:
                     logging.warning("Matriz de vetores inválida para escalonamento, pulando.")
                     console_print("⚠️ [yellow]Matriz de vetores inválida, pulando escalonamento.[/yellow]", args)
                     step_timers['4.5_Scale_Vectors'] = time.time() - current_step_start_time # Record time even if skipped due to invalid data
             else:
                console_print("ℹ️ [yellow]Nenhum vetor válido para escalonar.[/yellow]", args)
                step_timers['4.5_Scale_Vectors'] = time.time() - current_step_start_time
        except Exception as scale_err:
             step_timers['4.5_Scale_Vectors'] = time.time() - current_step_start_time
             console_print(f"❌ [bold red]Erro ao escalonar vetores: {scale_err}[/bold red]", args)
             logging.error(f"Erro Etapa 4.5 (Scaling): {scale_err}", exc_info=True)
    else:
        step_timers['4.5_Scale_Vectors'] = 0.0
        if not SKLEARN_AVAILABLE: logging.info("Etapa 4.5 pulada (scikit-learn indisponível).")
        elif args.skip_vector_scaling: logging.info("Etapa 4.5 pulada (--skip-vector-scaling).")

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]5. Salvando Vetores ({vector_column_to_save}) e Embeddings V6...[/]", args)
    v_ok, e_ok = False, False
    try:
        # Use .items() for (index, value) pairs, ensuring index is int
        vetores_para_db = [(int(idx), vec.tobytes()) for idx, vec in perfis_df_processed[vector_column_to_save].dropna().items() if vec is not None]
        embeddings_para_db = [(int(idx), emb.tobytes()) for idx, emb in perfis_df_processed['embedding'].dropna().items() if emb is not None]
        logging.info(f"Preparados {len(vetores_para_db)} vetores ({vector_column_to_save}) e {len(embeddings_para_db)} embeddings para salvar.")

        db_path_vectors = get_db_path(args, DB_VETORES_NAME)
        db_path_embeddings = get_db_path(args, DB_EMBEDDINGS_NAME)

        v_ok = salvar_blobs_lote(vetores_para_db, db_path_vectors, "vetores", "vetor", args)
        e_ok = salvar_blobs_lote(embeddings_para_db, db_path_embeddings, "embeddings", "embedding", args)

        step_timers['5_Save_Blobs'] = time.time() - current_step_start_time
        if v_ok and e_ok:
            console_print(f"✅ [green]Vetores/Embeddings V6 salvos em {step_timers['5_Save_Blobs']:.2f}s.[/green]", args)
            logging.info(f"Etapa 5: Blobs salvos em {step_timers['5_Save_Blobs']:.2f}s.")
        else:
            console_print(f"⚠️ [yellow]Erro ao salvar vetores/embeddings V6 (V:{v_ok}, E:{e_ok}). Ver logs.[/yellow]", args)
            logging.warning(f"Falha salvamento blobs (Etapa 5): V={v_ok}, E={e_ok}")

    except Exception as e:
        step_timers['5_Save_Blobs'] = time.time() - current_step_start_time
        console_print(f"❌ [bold red]Erro CRÍTICO Preparo/Salvar Blobs (Etapa 5): {e}[/bold red]", args)
        logging.critical(f"Erro Etapa 5 (Salvar Blobs): {e}", exc_info=True)
        # Decide whether to continue: If blobs failed, clustering might be impossible/useless.
        # Let's stop here if saving failed, as clustering relies on this data.
        if not v_ok or not e_ok:
            console_print("❌ [bold red]Encerrando devido a falha no salvamento de vetores/embeddings.[/bold red]", args)
            return

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]6. Clustering FAISS V6...[/]", args)
    cluster_assignments = None
    faiss_index = None
    inertia = None
    centroids = None
    cluster_metrics = {}
    profile_ids_for_clustering = []
    n_clusters_target = 0 # Initialize

    if args.skip_clustering:
        console_print("ℹ️ [yellow]SKIP_CLUSTERING=True. Pulando Etapa 6.[/yellow]", args)
        logging.info("Etapa 6 pulada (--skip-clustering).")
        step_timers['6_Clustering'] = 0.0
    elif perfis_df_processed.empty:
        console_print("ℹ️ [yellow]DF processado vazio. Pulando Etapa 6.[/yellow]", args)
        logging.info("Etapa 6 pulada - DF processado vazio.")
        step_timers['6_Clustering'] = 0.0
    else:
        valid_embeddings_series = perfis_df_processed['embedding'].dropna()
        if not valid_embeddings_series.empty:
            try:
                embeddings_matrix = np.stack(valid_embeddings_series.values)
                # Ensure it's a 2D array
                if embeddings_matrix.ndim == 1:
                    # This shouldn't happen if embeddings are generated correctly, but handle defensively
                    logging.warning("Embeddings matrix is 1D, attempting reshape.")
                    try:
                        embeddings_matrix = embeddings_matrix.reshape(-1, args.dim_embedding)
                    except ValueError as reshape_err:
                         raise ValueError(f"Cannot reshape embeddings matrix with shape {embeddings_matrix.shape} to (-1, {args.dim_embedding})") from reshape_err

                if embeddings_matrix.size == 0 or embeddings_matrix.shape[0] == 0 or embeddings_matrix.shape[1] != args.dim_embedding:
                     console_print(f"⚠️ [yellow]Matriz embeddings inválida (shape: {embeddings_matrix.shape}) pós-dropna/reshape. Pulando clustering.[/yellow]", args)
                     logging.warning(f"Clustering pulado - matriz inválida: {embeddings_matrix.shape}")
                else:
                     profile_ids_for_clustering = valid_embeddings_series.index.tolist()
                     num_profiles_to_cluster = embeddings_matrix.shape[0]

                     if args.cluster_method == 'sqrt':
                         n_clusters_target = max(1, int(math.sqrt(num_profiles_to_cluster)))
                     else:
                         n_clusters_target = max(1, args.fixed_clusters)

                     # Ensure k is not greater than N
                     n_clusters_target = min(n_clusters_target, num_profiles_to_cluster)

                     logging.info(f"Iniciando clustering V6 com k={n_clusters_target} (N={num_profiles_to_cluster}, Método: {args.cluster_method})")

                     cluster_assignments, faiss_index, inertia, centroids, cluster_metrics = realizar_clustering(
                         embeddings_matrix, n_clusters_target, args
                     )

                     if cluster_assignments is None:
                         console_print("❌ [bold red]Falha no processo FAISS (Etapa 6).[/bold red]", args)
                         logging.error("Falha Etapa 6 (Clustering retornou None).")
                     elif len(cluster_assignments) != len(profile_ids_for_clustering):
                         # This is a critical error indicating mismatch post-clustering
                         console_print(f"❌ CRÍTICO: Mismatch pós-cluster ({len(cluster_assignments)} vs {len(profile_ids_for_clustering)}).", args)
                         logging.critical(f"Falha CRÍTICA pós-cluster: mismatch {len(cluster_assignments)} vs {len(profile_ids_for_clustering)}.")
                         cluster_assignments = None # Invalidate results

            except (ValueError, MemoryError, Exception) as e:
                 console_print(f"❌ [bold red]Erro CRÍTICO Preparo/Execução Clustering (Etapa 6): {e}[/bold red]", args)
                 logging.critical(f"Erro Etapa 6 (Clustering): {e}", exc_info=True)
                 # Invalidate results on critical error
                 cluster_assignments = None
                 faiss_index = None
                 centroids = None
        else:
            console_print("ℹ️ [yellow]Nenhum embedding válido para clustering (Etapa 6).[/yellow]", args)
            logging.info("Clustering pulado (Etapa 6) - sem embeddings válidos.")
        step_timers['6_Clustering'] = time.time() - current_step_start_time

    current_step_start_time = time.time()
    clusters_saved, index_saved, centroids_saved = False, False, False
    faiss_index_path = ""
    centroids_path = ""

    if args.skip_clustering:
        console_print("\nℹ️ [yellow]Pulando Etapa 7 (Clustering foi pulado).[/yellow]", args)
        step_timers['7_Save_Cluster_Results'] = 0.0
    elif cluster_assignments is not None and profile_ids_for_clustering: # Check assignments are valid
        console_print(f"\n stepfather [bold]7. Salvando Resultados Clustering V6...[/]", args)
        db_path_clusters = get_db_path(args, DB_CLUSTERS_NAME)
        faiss_index_path = os.path.join(args.faiss_dir, f'{FAISS_INDEX_PREFIX}{timestamp}.index')
        centroids_path = os.path.join(args.faiss_dir, f'{CENTROIDS_PREFIX}{timestamp}.npy')

        # Check consistency again before saving
        if len(cluster_assignments) == len(profile_ids_for_clustering):
            cluster_data_to_save = list(zip(profile_ids_for_clustering, cluster_assignments.astype(int).tolist()))
            clusters_saved = salvar_clusters_lote(cluster_data_to_save, db_path_clusters, args)
        else:
            # This case should have been caught earlier, but double-check
            console_print(f"❌ CRÍTICO: Mismatch antes de salvar clusters ({len(cluster_assignments)} vs {len(profile_ids_for_clustering)}). Clusters NÃO salvos.", args)
            logging.critical(f"Falha CRÍTICA salvar clusters: mismatch {len(cluster_assignments)} vs {len(profile_ids_for_clustering)}.")
            clusters_saved = False # Ensure it's marked as failed

        if args.save_faiss_index and faiss_index is not None:
            index_saved = salvar_indice_faiss(faiss_index, faiss_index_path, args)
        elif args.save_faiss_index and faiss_index is None:
             logging.warning("Índice FAISS não pôde ser salvo pois é nulo (provavelmente falha no clustering).")

        if args.save_centroids and centroids is not None:
             centroids_saved = salvar_centroides(centroids, centroids_path, args)
        elif args.save_centroids and centroids is None:
             logging.warning("Centróides não puderam ser salvos pois são nulos (provavelmente falha no clustering).")

        step_timers['7_Save_Cluster_Results'] = time.time() - current_step_start_time
        console_print(f"✅ [green]Resultados clustering salvos em {step_timers['7_Save_Cluster_Results']:.2f}s (DB:{clusters_saved}, Idx:{index_saved}, Centr:{centroids_saved}).[/]", args)
        logging.info(f"Etapa 7: Resultados cluster salvos em {step_timers['7_Save_Cluster_Results']:.2f}s.")

    else:
         console_print("\nℹ️ [yellow]Clustering não executado ou falhou. Pulando Etapa 7.[/yellow]", args)
         logging.info("Etapa 7 pulada - clustering não executado ou falhou.")
         step_timers['7_Save_Cluster_Results'] = 0.0

    current_step_start_time = time.time()
    console_print(f"\n stepfather [bold]8. Exibindo Exemplo V6...[/]", args)
    if not perfis_df_processed.empty:
        try:
            example_id = perfis_df_processed.index[0]
            example_profile_series = perfis_df_processed.loc[example_id]
            table = Table(title=f"Perfil Exemplo V6 ID: {example_id}", show_header=True, header_style="bold blue", show_lines=True, expand=False)
            table.add_column("Atributo", style="cyan", max_width=25, no_wrap=True)
            table.add_column("Valor", style="magenta", overflow="fold")

            for col_name, value in example_profile_series.items():
                display_value = "N/A"
                if pd.notna(value):
                    if isinstance(value, (bool, np.bool_)): 
                        display_value = "Sim" if bool(value) else "Não"
                    elif isinstance(value, np.ndarray):
                        # Show limited info for arrays unless detailed log is on
                        display_value = f"Array (Dim: {len(value)}) [{value.dtype}]"
                        if args.detailed_log and not args.quiet:
                            sample_repr = np.array2string(value[:5], precision=3, separator=', ', suppress_small=True)
                            display_value += f": [{sample_repr}{'...' if len(value) > 5 else ''}]"
                    elif isinstance(value, str) and len(value) > 100 and not args.quiet: 
                        display_value = value[:100] + "..."
                    else: 
                        display_value = str(value)

                title = str(col_name).replace('_', ' ').title()
                table.add_row(title, display_value)

            if not args.skip_clustering and clusters_saved: # Only look up if clustering ran and saved
                cluster_id_exemplo = "N/A"
                try:
                    db_path_clusters = get_db_path(args, DB_CLUSTERS_NAME)
                    with sqlite3.connect(db_path_clusters, timeout=5.0) as conn_clu:
                        cur = conn_clu.cursor()
                        # Ensure example_id is Python int
                        cur.execute("SELECT cluster_id FROM clusters WHERE id = ?", (int(example_id),))
                        res = cur.fetchone()
                        cluster_id_exemplo = str(res[0]) if res else "Não Encontrado"
                except Exception as lookup_err:
                     cluster_id_exemplo = f"Erro ({lookup_err.__class__.__name__})"
                     logging.warning(f"Erro buscar cluster exemplo {example_id}: {lookup_err}")
                table.add_row("Cluster ID (do DB)", cluster_id_exemplo)
            elif not args.skip_clustering and not clusters_saved:
                 table.add_row("Cluster ID (do DB)", "Falha ao salvar")

            if not args.quiet: console.print(table)
            logging.info(f"Exemplo perfil ID {example_id} logado/exibido.")

        except (IndexError, KeyError) as e:
             console_print(f"ℹ️ [yellow]Não foi possível obter perfil de exemplo (DF vazio ou ID inválido?).[/yellow]", args)
             logging.warning(f"Falha ao obter exemplo (Index/Key Error): {e}")
        except Exception as e:
            console_print(f"❌ [bold red]Erro ao exibir exemplo (Etapa 8): {e}[/bold red]", args)
            logging.error(f"Erro exibir exemplo V6: {e}", exc_info=True)
    else:
        console_print("ℹ️ Nenhum perfil válido processado para exemplo.", args)
    step_timers['8_Display_Example'] = time.time() - current_step_start_time

    current_step_start_time = time.time()
    if args.vacuum_dbs:
        console_print(f"\n stepfather [bold]9. Executando VACUUM...[/]", args)
        db_files_to_vacuum = [
             get_db_path(args, DB_PERFIS_NAME),
             get_db_path(args, DB_VETORES_NAME),
             get_db_path(args, DB_EMBEDDINGS_NAME),
             get_db_path(args, DB_CLUSTERS_NAME)
        ]
        for db_file in db_files_to_vacuum:
             vacuum_database(db_file, args)
        step_timers['9_Vacuum'] = time.time() - current_step_start_time
        logging.info(f"Etapa 9: VACUUM executado em {step_timers['9_Vacuum']:.2f}s.")
    else:
        step_timers['9_Vacuum'] = 0.0
        logging.info("Etapa 9 pulada (--vacuum-dbs não usado).")

    overall_end_time = time.time()
    total_time = overall_end_time - overall_start_time
    step_timers['Total'] = total_time

    console.rule(f"[bold green]🎉 Processo V6 Concluído em {total_time:.2f} segundos! 🎉")
    logging.info(f"--- Script V6 Finalizado em {total_time:.2f}s ---")

    summary_table = Table(title="Resumo da Execução V6", show_header=True, header_style="bold magenta")
    summary_table.add_column("Etapa", style="cyan")
    summary_table.add_column("Tempo (s)", style="yellow", justify="right")
    summary_table.add_column("Detalhes", style="dim")

    num_db_files = len([f for f in os.listdir(args.db_dir) if f.endswith('.db')]) if os.path.exists(args.db_dir) else 0
    df_len = len(perfis_df) if 'perfis_df' in locals() else 0
    processed_len = len(perfis_df_processed) if 'perfis_df_processed' in locals() else 0
    
    # Corrigindo a formatação do cluster_status para evitar erro com None
    cluster_status = "Skipped" if args.skip_clustering else (
        f"k={n_clusters_target if n_clusters_target > 0 else 'N/A'}, "
        f"Inertia={inertia:.2f if inertia is not None else 'N/A'}"
    )

    summary_table.add_row("1. Setup DB", f"{step_timers.get('1_Setup_DB', 0):.2f}", f"{num_db_files} arquivos DB")
    summary_table.add_row("2. Gerar Perfis", f"{step_timers.get('2_Generate_Profiles', 0):.2f}", f"{num_gerados}/{args.num_profiles} perfis")
    summary_table.add_row("3. Salvar Perfis DB", f"{step_timers.get('3_DataFrame_SavePerfis', 0):.2f}", f"{df_len} linhas no DF")
    summary_table.add_row("4. Vetorizar/Embeddar", f"{step_timers.get('4_Vectorize_Embed', 0):.2f}", f"{processed_len} processados")
    summary_table.add_row("4.5 Escalar Vetores", f"{step_timers.get('4.5_Scale_Vectors', 0):.2f}", "Skipped" if args.skip_vector_scaling or not SKLEARN_AVAILABLE else f"Coluna: {vector_column_to_save}")
    summary_table.add_row("5. Salvar Blobs", f"{step_timers.get('5_Save_Blobs', 0):.2f}", f"V:{'OK' if v_ok else 'ERR'}, E:{'OK' if e_ok else 'ERR'}")
    summary_table.add_row("6. Clustering", f"{step_timers.get('6_Clustering', 0):.2f}", cluster_status)

    if not args.skip_clustering and cluster_metrics:
         sil_score = cluster_metrics.get('silhouette')
         db_score = cluster_metrics.get('davies_bouldin')
         metrics_str = f"Silh={sil_score:.4f if sil_score is not None else 'N/A'}, DB={db_score:.4f if db_score is not None else 'N/A'}"
         if args.skip_cluster_metrics: metrics_str = "Skipped (flag)"
         elif not SKLEARN_AVAILABLE: metrics_str = "Skipped (no sklearn)"
         elif n_clusters_target <= 1 or processed_len <= n_clusters_target : metrics_str = "N/A (k<=1 or N<=k)"
         summary_table.add_row("   Cluster Metrics", "-", metrics_str)

    summary_table.add_row("7. Salvar Cluster Res.", f"{step_timers.get('7_Save_Cluster_Results', 0):.2f}", f"DB:{clusters_saved}, Idx:{index_saved}, Centr:{centroids_saved}")
    summary_table.add_row("8. Exibir Exemplo", f"{step_timers.get('8_Display_Example', 0):.2f}", "-")
    summary_table.add_row("9. Vacuum DBs", f"{step_timers.get('9_Vacuum', 0):.2f}", "Skipped" if not args.vacuum_dbs else "Executado")
    summary_table.add_row("[bold]Total[/]", f"[bold]{step_timers.get('Total', 0):.2f}[/]", "-")

    console.print(summary_table)
    logging.info(f"Resumo Timers (s): {step_timers}")

    console_print(f"\n📄 [cyan]Logs:[/cyan] {log_file_path}", args)
    console_print(f"🗃️ [cyan]Bancos de Dados:[/cyan] {args.db_dir}", args)
    if args.save_faiss_index and not args.skip_clustering and index_saved and faiss_index_path:
         console_print(f"💾 [cyan]Índice FAISS:[/cyan] {faiss_index_path}", args)
    if args.save_centroids and not args.skip_clustering and centroids_saved and centroids_path:
         console_print(f"💾 [cyan]Centróides:[/cyan] {centroids_path}", args)
    console.rule()

    console_output_file = os.path.join(args.log_dir, f"console_output_{timestamp}.html")
    try:
        console.save_html(console_output_file)
        logging.info(f"Output console Rich salvo em '{console_output_file}'")
    except Exception as save_err:
        logging.error(f"Falha ao salvar output console Rich: {save_err}")

# --- Entry Point ---
if __name__ == "__main__":
    args = parse_arguments()
    log_file_path = setup_logging(args.log_dir, args.detailed_log)

    if args.seed is not None:
        seed_value = args.seed
        random.seed(seed_value)
        np.random.seed(seed_value % (2**32 - 1)) # Numpy seed must be within 32-bit unsigned int range
        Faker.seed(seed_value) # Seed Faker globally if used outside workers initially
        logging.info(f"Global random seed set to: {seed_value}")
        console_print(f"ℹ️ Seed global definido: {seed_value}", args, style="dim")

    import multiprocessing as mp
    start_method = None
    try:
        # Set start method explicitly for cross-platform compatibility/safety
        # 'spawn' is safer, especially on macOS and Windows. 'fork' is faster but can cause issues.
        if platform.system() == "Windows":
            start_method = "spawn"
        elif platform.system() == "Darwin": # macOS
             start_method = "spawn" # 'fork' can be problematic on macOS
        else: # Linux/other Unix
             start_method = "fork" # Usually safe and fast on Linux

        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != start_method:
            mp.set_start_method(start_method, force=True) # Force needed if already set implicitly
            logging.info(f"Método start MP definido para: {mp.get_start_method()}")
            console_print(f"ℹ️ Método start MP definido: {mp.get_start_method()}", args, style="dim")
        else:
             logging.info(f"Método start MP já está como: {current_method}")
             console_print(f"ℹ️ Método start MP já definido: {current_method}", args, style="dim")

    except Exception as e:
         logging.warning(f"Falha definir método start MP para '{start_method}': {e}. Usando padrão: {mp.get_start_method(allow_none=True)}")
         console_print(f"⚠️ Falha definir método start MP. Padrão: {mp.get_start_method(allow_none=True)}", args, style="yellow")


    try:
        main(args)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Execução interrompida pelo usuário.[/]")
        logging.warning("Execução interrompida por KeyboardInterrupt.")
    except Exception as critical_error:
        console.print(f"\n[bold red]ERRO CRÍTICO INESPERADO:[/]")
        # Log the full traceback to the log file
        logging.critical(f"Erro crítico inesperado no pipeline principal: {critical_error}", exc_info=True)
        # Print simplified traceback to console
        console.print_exception(show_locals=False, word_wrap=True)
    finally:
        if console.record:
             try:
                 final_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                 console_output_file_final = os.path.join(args.log_dir, f"console_output_{final_timestamp}_final.html")
                 console.save_html(console_output_file_final)
                 logging.info(f"Output console Rich (final) salvo em '{console_output_file_final}'")
             except Exception as final_save_err:
                 logging.error(f"Falha ao salvar output final do console: {final_save_err}")