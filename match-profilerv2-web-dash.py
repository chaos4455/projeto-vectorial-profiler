import sqlite3
import numpy as np
import faiss
import json
import datetime
import hashlib
import os
import random
import logging
from typing import Tuple, List, Dict, Set, Optional, Any
from collections import Counter
from flask import Flask, render_template_string, redirect, url_for
import threading # Para carregar dados em background inicial
from functools import lru_cache # Cache mais robusto

# --- Configurações ---
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
VALUATION_DIR = "valuation_v3_web_log" # Log dir específico
os.makedirs(VALUATION_DIR, exist_ok=True)

FLASK_PORT: int = 8881
FLASK_DEBUG: bool = False # Mude para True para debug, False para produção

NUM_NEIGHBORS_TARGET: int = 10
INITIAL_SEARCH_FACTOR: int = 20
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.05 # Score mínimo para aparecer

LOG_LEVEL = logging.INFO
EXPECTED_EMBEDDING_DIM: int = 64

# Pesos para o Score (Ajuste conforme necessário)
WEIGHTS = {
    "jogos": 0.40,
    "estilos": 0.30,
    "plataformas": 0.15,
    "disponibilidade": 0.10,
    "interacao": 0.05, # Adicionado peso para Interação Desejada
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"

# --- Configuração de Logging ---
LOG_FILE_VAL = os.path.join(VALUATION_DIR, f"matchmaking_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE_VAL,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    encoding='utf-8'
)
# Adiciona handler para console também
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# --- Inicialização do Flask App ---
app = Flask(__name__)

# --- Globais para Dados e Índice (Carregados uma vez) ---
# Usaremos um dicionário para armazenar o estado carregado
app_data = {
    "embeddings_matrix": None,
    "profile_ids_map": None,
    "embedding_dim": None,
    "faiss_index": None,
    "data_loaded": False,
    "loading_error": None,
    "db_path_profiles": DATABASE_PROFILES # Guardar para acesso fácil
}
data_load_lock = threading.Lock()

# --- Funções Auxiliares (Adaptadas do script anterior) ---

# Cache para carregar perfis - LRU é mais robusto que um dict simples
@lru_cache(maxsize=1024) # Cache até 1024 perfis
def carregar_perfil_por_id_cached(db_path: str, profile_id: int) -> Optional[Dict[str, Any]]:
    """Carrega os dados completos de um perfil pelo seu ID com cache LRU."""
    logging.debug(f"Tentando carregar perfil ID {profile_id} de {db_path}")
    try:
        # Timeout maior para evitar locked errors sob carga
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM perfis WHERE id = ?", (profile_id,))
            perfil_row = cursor.fetchone()
            if perfil_row:
                perfil_dict = dict(perfil_row)
                # Converte booleano
                if 'compartilhar_contato' in perfil_dict:
                     perfil_dict['compartilhar_contato'] = bool(perfil_dict['compartilhar_contato'])
                logging.debug(f"Perfil ID {profile_id} carregado com sucesso.")
                return perfil_dict
            else:
                logging.warning(f"Perfil com ID {profile_id} não encontrado em '{db_path}'.")
                return None
    except sqlite3.Error as e:
        logging.error(f"Erro SQLite ao carregar perfil ID {profile_id} de {db_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao carregar perfil ID {profile_id}: {e}", exc_info=True)
        return None

def safe_split_and_strip(text: Optional[str], delimiter: str = ',') -> Set[str]:
    if not text or not isinstance(text, str): return set()
    return {item.strip() for item in text.split(delimiter) if item.strip()}

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def availability_similarity(avail1: Optional[str], avail2: Optional[str]) -> float:
    if not avail1 or not avail2: return 0.0
    a1 = avail1.split('(')[0].strip().lower()
    a2 = avail2.split('(')[0].strip().lower()
    if a1 == a2: return 1.0
    if "fim de semana" in a1 and "fim de semana" in a2: return 0.6
    if "semana" in a1 and "semana" in a2: return 0.4
    if ("manhã" in a1 or "tarde" in a1) and "flexível" in a2: return 0.3
    if ("manhã" in a2 or "tarde" in a2) and "flexível" in a1: return 0.3
    # Poderia adicionar noite/madrugada etc.
    return 0.0

def interaction_similarity(inter1: Optional[str], inter2: Optional[str]) -> float:
    """Similaridade simples para tipo de interação."""
    if not inter1 or not inter2: return 0.0
    i1 = inter1.lower()
    i2 = inter2.lower()
    if i1 == i2: return 1.0
    # Considera "principalmente online" e "online" como muito similares
    if ("online" in i1 and "online" in i2) and ("presencial" not in i1 and "presencial" not in i2):
        return 0.9
    # Considera "aberto ao presencial" e "presencialmente" como similares
    if ("presencial" in i1 and "presencial" in i2):
        return 0.7
    if "indiferente" in i1 or "indiferente" in i2: return 0.3 # Indiferente tem baixa similaridade com opções específicas
    return 0.0

def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Calcula o score de compatibilidade ponderado entre dois perfis."""
    scores = {}
    # Usa .get com default vazio para segurança
    games1 = safe_split_and_strip(profile1.get('jogos_favoritos', ''))
    games2 = safe_split_and_strip(profile2.get('jogos_favoritos', ''))
    scores['jogos'] = jaccard_similarity(games1, games2)

    styles1 = safe_split_and_strip(profile1.get('estilos_preferidos', ''))
    styles2 = safe_split_and_strip(profile2.get('estilos_preferidos', ''))
    scores['estilos'] = jaccard_similarity(styles1, styles2)

    platforms1 = safe_split_and_strip(profile1.get('plataformas_possuidas', ''))
    platforms2 = safe_split_and_strip(profile2.get('plataformas_possuidas', ''))
    scores['plataformas'] = jaccard_similarity(platforms1, platforms2)

    scores['disponibilidade'] = availability_similarity(profile1.get('disponibilidade'), profile2.get('disponibilidade'))
    scores['interacao'] = interaction_similarity(profile1.get('interacao_desejada'), profile2.get('interacao_desejada'))

    total_score = sum(scores[key] * WEIGHTS[key] for key in WEIGHTS if key in scores)
    logging.debug(f"Score entre {profile1.get('id')}->{profile2.get('id')}: T={total_score:.3f}, D={ {k: round(v, 2) for k,v in scores.items()} }")
    return total_score, scores

def buscar_e_rankear_vizinhos(
    id_origem: int,
    num_neighbors_target: int
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Função principal de matchmaking: busca no FAISS, calcula score, filtra e retorna perfis."""

    if not app_data["data_loaded"]:
        logging.error("Dados não carregados, impossível buscar vizinhos.")
        return None, []
    if app_data["loading_error"]:
        logging.error(f"Erro prévio no carregamento de dados: {app_data['loading_error']}")
        return None, []

    logging.info(f"Iniciando busca para perfil de origem ID: {id_origem}")
    perfil_origem = carregar_perfil_por_id_cached(app_data["db_path_profiles"], id_origem)
    if perfil_origem is None:
        logging.error(f"Não foi possível carregar perfil de origem ID {id_origem}")
        # Tentar escolher outro ID? Ou retornar erro? Por ora, erro.
        return None, []

    origin_name = perfil_origem.get('nome', f"NOME_NAO_ENCONTRADO_{id_origem}")

    # Obter embedding da origem
    try:
        source_index = app_data["profile_ids_map"].index(id_origem)
        embedding_origem = app_data["embeddings_matrix"][source_index].reshape(1, -1)
    except (ValueError, IndexError, TypeError) as e:
        logging.error(f"Erro ao obter embedding para ID {id_origem}: {e}", exc_info=True)
        return perfil_origem, [] # Retorna origem, mas sem similares

    # Buscar candidatos iniciais no FAISS
    k_search = num_neighbors_target * INITIAL_SEARCH_FACTOR + 1
    logging.debug(f"Buscando {k_search} vizinhos iniciais no FAISS...")
    candidates_with_scores: List[Tuple[float, int, Dict[str, float]]] = []
    checked_profile_ids: Set[int] = {id_origem}

    try:
        distances, indices = app_data["faiss_index"].search(embedding_origem, k_search)
        faiss_indices_found = indices[0]
        logging.info(f"Busca FAISS retornou {len(faiss_indices_found)} índices.")

        # Calcular scores e filtrar
        for idx in faiss_indices_found:
            if 0 <= idx < len(app_data["profile_ids_map"]):
                potential_profile_id = app_data["profile_ids_map"][idx]
            else:
                 logging.warning(f"Índice FAISS inválido: {idx}. Pulando.")
                 continue

            if potential_profile_id in checked_profile_ids: continue
            checked_profile_ids.add(potential_profile_id)

            potential_profile = carregar_perfil_por_id_cached(app_data["db_path_profiles"], potential_profile_id)

            if potential_profile and potential_profile.get('nome') != origin_name:
                total_score, score_details = calculate_custom_similarity(perfil_origem, potential_profile)
                if total_score >= MIN_CUSTOM_SCORE_THRESHOLD:
                    candidates_with_scores.append((total_score, potential_profile_id, score_details))

        # Ordenar por score
        candidates_with_scores.sort(key=lambda item: item[0], reverse=True)
        top_neighbors_data = candidates_with_scores[:num_neighbors_target]
        logging.info(f"Encontrados {len(top_neighbors_data)} vizinhos válidos após score e filtragem.")

        # Carregar perfis finais rankeados
        perfis_similares_final = []
        for score, pid, score_details in top_neighbors_data:
            perfil_similar = carregar_perfil_por_id_cached(app_data["db_path_profiles"], pid)
            if perfil_similar:
                perfil_similar['score_compatibilidade'] = round(score, 3)
                perfil_similar['score_details'] = {k: round(v, 2) for k, v in score_details.items()} # Adiciona detalhes do score
                perfis_similares_final.append(perfil_similar)

        return perfil_origem, perfis_similares_final

    except Exception as e:
        logging.error(f"Erro durante a busca/ranking para ID {id_origem}: {e}", exc_info=True)
        return perfil_origem, [] # Retorna origem, mas sem similares

# --- Função para Carregar Dados (Executada uma vez) ---
def load_data_and_build_index():
    """Carrega embeddings, IDs e constrói o índice FAISS."""
    global app_data
    # Prevenir execução múltipla
    with data_load_lock:
        if app_data["data_loaded"] or app_data["loading_error"]:
            return

        logging.info("Iniciando carregamento de dados e construção do índice FAISS...")
        try:
            # 1. Carregar Embeddings e IDs
            db_path_emb = DATABASE_EMBEDDINGS
            embeddings_matrix, ids_map, emb_dim = carregar_embeddings_e_ids_internal(db_path_emb)
            if embeddings_matrix is None or ids_map is None or emb_dim is None:
                raise ValueError("Falha ao carregar embeddings ou IDs.")
            app_data["embeddings_matrix"] = embeddings_matrix
            app_data["profile_ids_map"] = ids_map
            app_data["embedding_dim"] = emb_dim
            logging.info(f"{len(ids_map)} embeddings carregados.")

            # 2. Construir Índice FAISS
            faiss_index = construir_indice_faiss_internal(embeddings_matrix, emb_dim)
            if faiss_index is None:
                raise ValueError("Falha ao construir índice FAISS.")
            app_data["faiss_index"] = faiss_index
            logging.info("Índice FAISS construído com sucesso.")

            app_data["data_loaded"] = True
            app_data["loading_error"] = None
            logging.info("Carregamento de dados e índice concluído.")

        except Exception as e:
            logging.critical(f"Erro CRÍTICO durante o carregamento de dados/índice: {e}", exc_info=True)
            app_data["loading_error"] = str(e)
            app_data["data_loaded"] = False # Marcar como não carregado em caso de erro

# Funções internas para evitar conflito de nome com as auxiliares globais (se houver)
def carregar_embeddings_e_ids_internal(db_path: str) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[int]]:
    embeddings_list = []
    ids_list = []
    dimension = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, embedding FROM embeddings ORDER BY id")
            rows = cursor.fetchall()
            if not rows: return None, None, None
            for pid, blob in rows:
                emb = np.frombuffer(blob, dtype=np.float32)
                if dimension is None: dimension = len(emb)
                if len(emb) == dimension:
                    embeddings_list.append(emb)
                    ids_list.append(pid)
            if not embeddings_list: return None, None, None
            matrix = np.ascontiguousarray(embeddings_list, dtype=np.float32)
            return matrix, ids_list, dimension
    except Exception as e:
        logging.error(f"Erro interno ao carregar embeddings: {e}", exc_info=True)
        return None, None, None

def construir_indice_faiss_internal(embeddings: np.ndarray, dimension: int) -> Optional[faiss.Index]:
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension or embeddings.shape[0] == 0:
        logging.error("Matriz de embedding inválida para FAISS.")
        return None
    try:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logging.error(f"Erro interno ao construir índice FAISS: {e}", exc_info=True)
        return None


# --- HTML Template (com Tailwind e Font Awesome via CDN) ---
# Usando f-string multi-linha para facilitar a leitura
index_html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matchmaking Dashboard - Replika AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Custom Styles & Overrides */
        body {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(135deg, #1a0000 0%, #0f0f0f 50%, #1a1a1a 100%);
            background-attachment: fixed;
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }

        /* Gradient Text */
        .text-gradient-red {
            background: linear-gradient(90deg, #ff4d4d, #ff1a1a);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .text-gradient-score {
            background: linear-gradient(90deg, #4ade80, #16a34a); /* Green gradient */
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        /* Card Styling */
        .profile-card {
            background: rgba(31, 41, 55, 0.6); /* gray-800 with opacity */
            backdrop-filter: blur(10px); /* Glassmorphism blur */
            border: 1px solid rgba(220, 38, 38, 0.4); /* red-600 border with opacity */
            box-shadow: 0 8px 32px 0 rgba(220, 38, 38, 0.15); /* Subtle red shadow */
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .profile-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 12px 40px 0 rgba(220, 38, 38, 0.25);
        }

        /* Button Styling */
        .btn-retest {
            background: linear-gradient(90deg, #ef4444, #b91c1c); /* red-500 to red-700 */
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px 0 rgba(239, 68, 68, 0.3);
        }
        .btn-retest:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 6px 20px 0 rgba(239, 68, 68, 0.4);
        }
        .btn-retest:active {
            transform: translateY(0px) scale(1);
            box-shadow: 0 2px 10px 0 rgba(239, 68, 68, 0.2);
        }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }
        /* Apply fade-in with delay to similar cards */
        {% for i in range(10) %}
        .similar-card-{{ i }} { animation-delay: {{ (i + 1) * 0.1 }}s; opacity: 0; } /* Start hidden */
        {% endfor %}

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; } /* gray-800 */
        ::-webkit-scrollbar-thumb { background: #dc2626; border-radius: 4px;} /* red-600 */
        ::-webkit-scrollbar-thumb:hover { background: #ef4444; } /* red-500 */

        .icon-style { color: #fca5a5; margin-right: 0.5rem; } /* red-300 */
        .badge { background-color: rgba(185, 28, 28, 0.7); border: 1px solid rgba(220, 38, 38, 0.5); } /* red-700 bg, red-600 border */
    </style>
</head>
<body class="text-gray-200 min-h-screen">

    <!-- Header -->
    <header class="bg-gray-900/80 backdrop-blur-md shadow-lg sticky top-0 z-50 py-4 px-6">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl md:text-3xl font-orbitron font-bold text-gradient-red">
                <i class="fas fa-users-rays mr-2"></i>Matchmaking Dashboard
            </h1>
            <button id="retestBtn" class="btn-retest text-white font-bold py-2 px-4 rounded-lg text-sm md:text-base flex items-center">
                <i class="fas fa-random mr-2 animate-spin" style="animation-duration: 3s;"></i> Gerar Novo Match
            </button>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto p-4 md:p-8">

        {% if error_message %}
            <div class="bg-red-800 border border-red-600 text-red-100 px-4 py-3 rounded-lg relative mb-6" role="alert">
                <strong class="font-bold"><i class="fas fa-exclamation-triangle mr-2"></i>Erro!</strong>
                <span class="block sm:inline">{{ error_message }}</span>
            </div>
        {% elif perfil_origem %}
            <!-- Perfil de Origem Section -->
            <section class="mb-10">
                <h2 class="text-2xl font-orbitron font-semibold mb-4 border-b-2 border-red-600/50 pb-2 text-gray-300">
                    <i class="fas fa-crosshairs mr-2 text-red-400"></i>Perfil de Origem
                </h2>
                <div class="profile-card rounded-lg p-6 fade-in">
                    <h3 class="text-xl md:text-2xl font-bold font-orbitron text-gradient-red mb-3">{{ perfil_origem.nome }} (ID: {{ perfil_origem.id }})</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3 text-sm md:text-base">
                        <p><i class="fas fa-birthday-cake icon-style"></i> <strong>Idade:</strong> {{ perfil_origem.idade }}</p>
                        <p><i class="fas fa-map-marker-alt icon-style"></i> <strong>Local:</strong> {{ perfil_origem.cidade }}, {{ perfil_origem.estado }}</p>
                        <p><i class="fas fa-venus-mars icon-style"></i> <strong>Sexo:</strong> {{ perfil_origem.sexo }}</p>
                        <p><i class="fas fa-headset icon-style"></i> <strong>Plataformas:</strong>
                           {% set platforms = perfil_origem.plataformas_possuidas.split(',') %}
                           {% for platform in platforms %}<span class="badge text-xs px-2 py-0.5 rounded-full mr-1 whitespace-nowrap">{{ platform.strip() }}</span>{% endfor %}
                        </p>
                         <p class="md:col-span-2"><i class="fas fa-gamepad icon-style"></i> <strong>Jogos Favoritos:</strong> {{ perfil_origem.jogos_favoritos }}</p>
                         <p class="md:col-span-2"><i class="fas fa-tags icon-style"></i> <strong>Estilos Preferidos:</strong> {{ perfil_origem.estilos_preferidos }}</p>
                         <p><i class="fas fa-music icon-style"></i> <strong>Música:</strong> {{ perfil_origem.interesses_musicais }}</p>
                         <p><i class="fas fa-clock icon-style"></i> <strong>Disponibilidade:</strong> {{ perfil_origem.disponibilidade }}</p>
                         <p><i class="fas fa-comments icon-style"></i> <strong>Interação:</strong> {{ perfil_origem.interacao_desejada }}</p>
                         <p><i class="fas fa-share-alt icon-style"></i> <strong>Compartilha Contato:</strong> {{ 'Sim <i class="fas fa-check-circle text-green-400"></i>' if perfil_origem.compartilhar_contato else 'Não <i class="fas fa-times-circle text-red-400"></i>' | safe }}</p>
                         <p class="md:col-span-2 mt-2"><i class="fas fa-info-circle icon-style"></i> <strong>Descrição:</strong> <span class="text-gray-400 italic">{{ perfil_origem.descricao }}</span></p>
                    </div>
                </div>
            </section>

            <!-- Perfis Similares Section -->
            <section>
                <h2 class="text-2xl font-orbitron font-semibold mb-5 border-b-2 border-red-600/50 pb-2 text-gray-300">
                   <i class="fas fa-meteor mr-2 text-red-400"></i>Top {{ perfis_similares|length }} Perfis Similares (Rankeados por Score)
                </h2>
                {% if perfis_similares %}
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {% for perfil in perfis_similares %}
                        <div class="profile-card rounded-lg p-4 flex flex-col h-full fade-in similar-card-{{ loop.index0 }}">
                            <h4 class="text-lg font-bold font-orbitron text-gradient-red mb-2">{{ perfil.nome }} (ID: {{ perfil.id }})</h4>
                            <div class="flex-grow text-xs md:text-sm space-y-1.5 mb-3">
                                <p class="font-semibold text-base mb-2">
                                    <i class="fas fa-star-half-alt icon-style"></i> Score:
                                    <span class="font-bold text-gradient-score text-lg">{{ "%.3f"|format(perfil.score_compatibilidade) }}</span>
                                </p>
                                <p><i class="fas fa-birthday-cake icon-style"></i> {{ perfil.idade }} anos | <i class="fas fa-map-marker-alt ml-2 icon-style"></i> {{ perfil.cidade }}, {{ perfil.estado }}</p>
                                <p><i class="fas fa-gamepad icon-style"></i> <strong>Jogos:</strong> <span class="text-gray-400">{{ perfil.jogos_favoritos.split(',')|map('trim')|join(', ', attribute='strip()')|truncate(60, True) }}</span></p>
                                <p><i class="fas fa-tags icon-style"></i> <strong>Estilos:</strong> <span class="text-gray-400">{{ perfil.estilos_preferidos.split(',')|map('trim')|join(', ', attribute='strip()')|truncate(60, True) }}</span></p>
                                <p><i class="fas fa-headset icon-style"></i> <strong>Plataformas:</strong>
                                   {% set platforms = perfil.plataformas_possuidas.split(',') %}
                                   {% for platform in platforms %}<span class="badge text-xs px-1.5 py-0.5 rounded-full mr-1 whitespace-nowrap">{{ platform.strip() }}</span>{% endfor %}
                                </p>
                                <p><i class="fas fa-clock icon-style"></i> <strong>Disponível:</strong> {{ perfil.disponibilidade }}</p>
                                <p><i class="fas fa-comments icon-style"></i> <strong>Interação:</strong> {{ perfil.interacao_desejada }}</p>
                                <!-- Opcional: Detalhes do Score
                                <details class="text-xs mt-1 text-gray-400">
                                    <summary class="cursor-pointer hover:text-red-400">Detalhes Score</summary>
                                    <ul class="list-disc list-inside ml-2">
                                        {% for key, value in perfil.score_details.items() %}
                                        <li>{{ key|capitalize }}: {{ "%.2f"|format(value) }}</li>
                                        {% endfor %}
                                    </ul>
                                </details>
                                -->
                            </div>
                             <p class="text-xs text-gray-500 mt-auto pt-2 border-t border-gray-700/50">
                                 <i class="fas fa-share-alt mr-1"></i> Contato: {{ 'Sim' if perfil.compartilhar_contato else 'Não' }}
                             </p>
                        </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <p class="text-center text-gray-400 italic py-6 bg-gray-800/50 rounded-lg">
                        <i class="fas fa-ghost mr-2"></i> Nenhum perfil similar encontrado com os critérios atuais. Tente gerar um novo match!
                    </p>
                {% endif %}
            </section>

        {% else %}
             <p class="text-center text-red-400 italic py-6 bg-red-900/30 rounded-lg">
                <i class="fas fa-sync fa-spin mr-2"></i> Carregando dados ou perfil de origem não encontrado... Tente atualizar.
             </p>
        {% endif %}

    </main>

    <!-- Footer -->
    <footer class="text-center py-6 mt-10 text-xs text-gray-500 border-t border-gray-700/30">
        Criado por Replika AI Solutions - Maringá, Paraná <br>
        <i class="fas fa-copyright mr-1"></i> {{ current_year }} - Matchmaking Dashboard v1.0
    </footer>

    <script>
        // Botão para recarregar a página e gerar novo match
        document.getElementById('retestBtn').addEventListener('click', () => {
            const icon = document.getElementById('retestBtn').querySelector('i');
            icon.classList.add('fa-spin'); // Add spin on click
            window.location.reload();
        });

        // Remove spin animation after load (optional)
         window.addEventListener('load', () => {
             const icon = document.getElementById('retestBtn')?.querySelector('i');
             if (icon) {
                 // Keep spinning a bit for visual feedback then stop
                 // setTimeout(() => { icon.classList.remove('fa-spin'); }, 500);
                 // Or just let it spin
             }
         });
    </script>

</body>
</html>
"""

# --- Rota Flask Principal ---
@app.route('/')
def index():
    """Rota principal que carrega, busca e renderiza os perfis."""
    # Garante que os dados sejam carregados (thread-safe)
    if not app_data["data_loaded"] and not app_data["loading_error"]:
        logging.warning("Dados não carregados na requisição. Tentando carregar agora...")
        # Idealmente, isso seria feito antes da primeira requisição ou em background.
        # Fazer aqui pode atrasar a primeira resposta.
        load_data_and_build_index()

    # Verifica se houve erro no carregamento
    if app_data["loading_error"]:
        return render_template_string(
            index_html,
            error_message=f"Erro crítico ao carregar dados: {app_data['loading_error']}. Verifique os logs.",
            perfil_origem=None,
            perfis_similares=[],
            current_year=datetime.datetime.now().year
        )

    # Verifica se o mapa de IDs está carregado
    if not app_data["profile_ids_map"]:
         return render_template_string(
            index_html,
            error_message="Nenhum perfil encontrado nos dados carregados.",
            perfil_origem=None,
            perfis_similares=[],
            current_year=datetime.datetime.now().year
        )

    # Escolhe um ID aleatório
    try:
        id_origem = random.choice(app_data["profile_ids_map"])
    except IndexError:
         return render_template_string(
            index_html,
            error_message="Erro: Lista de IDs de perfil está vazia.",
            perfil_origem=None,
            perfis_similares=[],
            current_year=datetime.datetime.now().year
        )

    # Busca e rankeia os vizinhos
    perfil_origem, perfis_similares = buscar_e_rankear_vizinhos(id_origem, NUM_NEIGHBORS_TARGET)

    # Renderiza o template com os dados
    return render_template_string(
        index_html,
        perfil_origem=perfil_origem,
        perfis_similares=perfis_similares,
        error_message=None, # Sem erro nesta execução da rota
        current_year=datetime.datetime.now().year
    )

# Rota simples para forçar recarregamento (alternativa ao JS reload)
@app.route('/new_match')
def new_match():
    """Redireciona para a raiz para obter um novo match."""
    return redirect(url_for('index'))


# --- Função para iniciar o carregamento em background ---
def start_background_load():
    """Inicia o carregamento dos dados em uma thread separada."""
    if not app_data["data_loaded"] and not app_data["loading_error"]:
        load_thread = threading.Thread(target=load_data_and_build_index, name="DataLoaderThread", daemon=True)
        load_thread.start()
    else:
        logging.info("Dados já carregados ou houve erro anterior. Skipping background load.")

# --- Ponto de Entrada ---
if __name__ == '__main__':
    logging.info(f"Verificando existência dos bancos de dados...")
    db_profiles_exists = os.path.exists(DATABASE_PROFILES)
    db_embeddings_exists = os.path.exists(DATABASE_EMBEDDINGS)

    if not db_profiles_exists or not db_embeddings_exists:
        logging.critical("Erro: Bancos de dados necessários não encontrados.")
        if not db_profiles_exists: logging.critical(f"   - Ausente: '{DATABASE_PROFILES}'")
        if not db_embeddings_exists: logging.critical(f"   - Ausente: '{DATABASE_EMBEDDINGS}'")
        logging.critical("Execute 'profile_generator_v3.py' primeiro.")
        print("\n❌ Erro: Bancos de dados não encontrados. Veja o log.")
        print(f"   Verifique se '{DATABASE_PROFILES}' e '{DATABASE_EMBEDDINGS}' existem.")
        print("   Execute 'profile_generator_v3.py' primeiro.\n")
    else:
        logging.info("Bancos de dados encontrados.")
        # Inicia o carregamento dos dados em background ANTES de iniciar o servidor Flask
        print("Iniciando carregamento de dados e índice em background...")
        start_background_load()

        print(f"🚀 Iniciando servidor Flask na porta {FLASK_PORT}...")
        print(f"   Acesse: http://127.0.0.1:{FLASK_PORT}")
        print(f"   Logs em: {LOG_FILE_VAL}")
        print("   Pressione CTRL+C para parar o servidor.")
        # app.run(host='0.0.0.0', port=FLASK_PORT, debug=FLASK_DEBUG) # '0.0.0.0' para acesso externo se necessário
        # Usando waitress para um servidor um pouco mais robusto que o dev do Flask
        try:
            from waitress import serve
            serve(app, host='127.0.0.1', port=FLASK_PORT)
        except ImportError:
            print("Waitress não instalado. Usando servidor de desenvolvimento Flask.")
            print("Para um ambiente mais estável, instale waitress: pip install waitress")
            app.run(host='127.0.0.1', port=FLASK_PORT, debug=FLASK_DEBUG)