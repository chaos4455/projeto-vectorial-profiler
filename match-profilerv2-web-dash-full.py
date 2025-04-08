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
import threading
from functools import lru_cache

# --- Configurações ---
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
VALUATION_DIR = "valuation_v3_web_log"
os.makedirs(VALUATION_DIR, exist_ok=True)

FLASK_PORT: int = 8881
FLASK_DEBUG: bool = False

NUM_NEIGHBORS_TARGET: int = 10
INITIAL_SEARCH_FACTOR: int = 25 # Aumentado um pouco para ter mais candidatos iniciais
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.05 # Threshold final após ponderação e filtros

LOG_LEVEL = logging.INFO
EXPECTED_EMBEDDING_DIM: int = 64

# --- NOVOS PESOS E THRESHOLDS OBRIGATÓRIOS ---
# Prioridade MÁXIMA para Plataforma e Disponibilidade
WEIGHTS = {
    "plataformas": 0.45,        # Peso MUITO ALTO
    "disponibilidade": 0.35,    # Peso MUITO ALTO
    "jogos": 0.10,              # Peso reduzido
    "estilos": 0.05,            # Peso reduzido
    "interacao": 0.05,          # Peso baixo
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"

# Thresholds MÍNIMOS OBRIGATÓRIOS para considerar um match
# Ajuste estes valores conforme necessário. Valores mais altos = mais restritivo.
MIN_REQUIRED_PLATFORM_SCORE: float = 0.20 # Exige alguma plataforma em comum
MIN_REQUIRED_AVAILABILITY_SCORE: float = 0.30 # Exige alguma compatibilidade de horário (e.g., flexível com manhã/tarde, semana com semana)
# ----------------------------------------------

# --- Configuração de Logging ---
LOG_FILE_VAL = os.path.join(VALUATION_DIR, f"matchmaking_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE_VAL,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s - %(message)s', # Adicionado funcName
    encoding='utf-8'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


# --- Inicialização do Flask App ---
app = Flask(__name__)

# --- Globais para Dados e Índice ---
app_data = {
    "embeddings_matrix": None, "profile_ids_map": None, "embedding_dim": None,
    "faiss_index": None, "data_loaded": False, "loading_error": None,
    "db_path_profiles": DATABASE_PROFILES
}
data_load_lock = threading.Lock()

# --- Funções Auxiliares ---

@lru_cache(maxsize=1024)
def carregar_perfil_por_id_cached(db_path: str, profile_id: int) -> Optional[Dict[str, Any]]:
    # logging.debug(f"Cache miss/load for profile ID {profile_id}") # Reduzindo verbosidade
    try:
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM perfis WHERE id = ?", (profile_id,))
            perfil_row = cursor.fetchone()
            if perfil_row:
                perfil_dict = dict(perfil_row)
                if 'compartilhar_contato' in perfil_dict:
                     perfil_dict['compartilhar_contato'] = bool(perfil_dict['compartilhar_contato'])
                # logging.debug(f"Profile ID {profile_id} loaded successfully.") # Reduzindo verbosidade
                return perfil_dict
            else:
                logging.warning(f"Profile ID {profile_id} not found in '{db_path}'.")
                return None
    except Exception as e:
        logging.error(f"Error loading profile ID {profile_id} from {db_path}: {e}", exc_info=True)
        return None

def safe_split_and_strip(text: Optional[str], delimiter: str = ',') -> Set[str]:
    if not text or not isinstance(text, str): return set()
    return {item.strip().lower() for item in text.split(delimiter) if item.strip()} # Convert to lower for consistency

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2: return 0.0 # Considera 0 se ambos vazios
    intersection = len(set1.intersection(set2)); union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def availability_similarity(avail1_str: Optional[str], avail2_str: Optional[str]) -> float:
    if not avail1_str or not avail2_str: return 0.0
    # Simplifica a string de disponibilidade para comparação
    def simplify_avail(avail_text: str) -> str:
        text = avail_text.lower()
        if "manhã" in text: return "manha"
        if "tarde" in text: return "tarde"
        if "noite" in text: return "noite"
        if "madrugada" in text: return "madrugada"
        if "fim de semana" in text: return "fds"
        if "dia de semana" in text or "durante a semana" in text: return "semana"
        if "flexível" in text or "qualquer" in text: return "flexivel"
        return "outro" # Categoria genérica se não reconhecer

    a1_simple = simplify_avail(avail1_str)
    a2_simple = simplify_avail(avail2_str)

    if a1_simple == a2_simple: return 1.0 # Match perfeito (ex: noite == noite)

    # Matches parciais com pontuação alta para flexibilidade
    if a1_simple == "flexivel" or a2_simple == "flexivel": return 0.7 # Flexível combina bem com tudo
    if a1_simple == "fds" and a2_simple == "fds": return 0.8 # FDS é específico
    if a1_simple == "semana" and a2_simple == "semana": return 0.6 # Semana é mais geral

    # Combinações razoáveis (manhã/tarde/noite com dia de semana)
    if (a1_simple in ["manha", "tarde", "noite"] and a2_simple == "semana") or \
       (a2_simple in ["manha", "tarde", "noite"] and a1_simple == "semana"):
        return 0.4

    # Combinações menos prováveis (manhã/tarde/noite com FDS)
    if (a1_simple in ["manha", "tarde", "noite"] and a2_simple == "fds") or \
       (a2_simple in ["manha", "tarde", "noite"] and a1_simple == "fds"):
        return 0.2

    # Madrugada é muito específico
    if a1_simple == "madrugada" or a2_simple == "madrugada": return 0.1

    return 0.0 # Nenhum match significativo

def interaction_similarity(inter1: Optional[str], inter2: Optional[str]) -> float:
    if not inter1 or not inter2: return 0.0
    i1 = inter1.lower(); i2 = inter2.lower()
    s1 = set(w.strip() for w in i1.split())
    s2 = set(w.strip() for w in i2.split())

    if i1 == i2: return 1.0
    if "indiferente" in s1 or "indiferente" in s2: return 0.5 # Indiferente tem compatibilidade média
    if "online" in s1 and "online" in s2: return 0.9 # Ambos querem online
    if "presencial" in s1 and "presencial" in s2: return 0.8 # Ambos querem presencial
    # Se um quer online e outro presencial, baixa compatibilidade
    if ("online" in s1 and "presencial" in s2) or ("presencial" in s1 and "online" in s2):
        return 0.1
    return 0.2 # Caso base para outras combinações

def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calcula a similaridade personalizada com FOCO em PLATAFORMA e DISPONIBILIDADE.
    Retorna 0.0 se os thresholds mínimos para esses dois não forem atingidos.
    """
    scores = {}
    p1_id = profile1.get('id', 'P1')
    p2_id = profile2.get('id', 'P2')

    # --- Cálculo dos Scores Individuais ---
    platforms1 = safe_split_and_strip(profile1.get('plataformas_possuidas', ''))
    platforms2 = safe_split_and_strip(profile2.get('plataformas_possuidas', ''))
    scores['plataformas'] = jaccard_similarity(platforms1, platforms2)

    scores['disponibilidade'] = availability_similarity(profile1.get('disponibilidade'), profile2.get('disponibilidade'))

    games1 = safe_split_and_strip(profile1.get('jogos_favoritos', ''))
    games2 = safe_split_and_strip(profile2.get('jogos_favoritos', ''))
    scores['jogos'] = jaccard_similarity(games1, games2)

    styles1 = safe_split_and_strip(profile1.get('estilos_preferidos', ''))
    styles2 = safe_split_and_strip(profile2.get('estilos_preferidos', ''))
    scores['estilos'] = jaccard_similarity(styles1, styles2)

    scores['interacao'] = interaction_similarity(profile1.get('interacao_desejada'), profile2.get('interacao_desejada'))

    # --- VERIFICAÇÃO DOS THRESHOLDS OBRIGATÓRIOS ---
    if scores['plataformas'] < MIN_REQUIRED_PLATFORM_SCORE:
        logging.debug(f"Score {p1_id}->{p2_id}: REJECTED. Platform score {scores['plataformas']:.2f} < {MIN_REQUIRED_PLATFORM_SCORE:.2f}")
        # Retorna 0.0 para indicar falha no critério obrigatório, mas mantém os scores para possível análise
        return 0.0, {k: round(v, 2) for k,v in scores.items()}

    if scores['disponibilidade'] < MIN_REQUIRED_AVAILABILITY_SCORE:
        logging.debug(f"Score {p1_id}->{p2_id}: REJECTED. Availability score {scores['disponibilidade']:.2f} < {MIN_REQUIRED_AVAILABILITY_SCORE:.2f}")
        # Retorna 0.0 para indicar falha no critério obrigatório
        return 0.0, {k: round(v, 2) for k,v in scores.items()}

    # --- Cálculo do Score Ponderado Final (APENAS SE PASSOU NOS THRESHOLDS) ---
    total_score = sum(scores[key] * WEIGHTS[key] for key in WEIGHTS if key in scores)

    logging.debug(f"Score {p1_id}->{p2_id}: PASSED Thresholds. T={total_score:.3f}, Details={ {k: round(v, 2) for k,v in scores.items()} }")
    return total_score, scores

def buscar_e_rankear_vizinhos(id_origem: int, num_neighbors_target: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not app_data["data_loaded"] or app_data["loading_error"]:
        logging.error(f"Data not loaded or error exists. Cannot search. Loaded: {app_data['data_loaded']}, Error: {app_data['loading_error']}")
        return None, []

    logging.info(f"Starting match search for origin ID: {id_origem}")
    perfil_origem = carregar_perfil_por_id_cached(app_data["db_path_profiles"], id_origem)
    if perfil_origem is None:
        logging.error(f"Failed to load origin profile ID {id_origem}")
        return None, []

    origin_name = perfil_origem.get('nome', f"NAME_NOT_FOUND_{id_origem}")

    try:
        source_index_in_map = app_data["profile_ids_map"].index(id_origem)
        embedding_origem = app_data["embeddings_matrix"][source_index_in_map].reshape(1, -1)
    except (ValueError, IndexError, TypeError) as e:
        logging.error(f"Error getting embedding for origin ID {id_origem} (index {source_index_in_map if 'source_index_in_map' in locals() else 'N/A'}): {e}", exc_info=True)
        return perfil_origem, []

    # Aumentar k_search para compensar a filtragem mais rigorosa
    k_search = num_neighbors_target * INITIAL_SEARCH_FACTOR + 1
    # Garante que k_search não seja maior que o número total de perfis no índice
    k_search = min(k_search, app_data["faiss_index"].ntotal)

    candidates_with_scores: List[Tuple[float, int, Dict[str, float]]] = []
    checked_profile_ids: Set[int] = {id_origem} # Evita auto-comparação e duplicatas

    try:
        logging.info(f"Performing FAISS search with k={k_search} for ID {id_origem}")
        distances, faiss_indices = app_data["faiss_index"].search(embedding_origem, k_search)

        if faiss_indices is None or len(faiss_indices[0]) == 0:
             logging.warning(f"FAISS search returned no neighbors for ID {id_origem}.")
             return perfil_origem, []

        faiss_indices_found = faiss_indices[0]
        logging.info(f"FAISS search returned {len(faiss_indices_found)} potential candidate indices.")

        processed_count = 0
        for i, idx_in_matrix in enumerate(faiss_indices_found):
            # O índice retornado pelo FAISS é a posição na matriz de embeddings
            if not (0 <= idx_in_matrix < len(app_data["profile_ids_map"])):
                logging.warning(f"Invalid FAISS index {idx_in_matrix} returned (out of bounds for profile_ids_map size {len(app_data['profile_ids_map'])}). Skipping.")
                continue

            potential_profile_id = app_data["profile_ids_map"][idx_in_matrix]

            # Pula se for o próprio perfil de origem ou já verificado
            if potential_profile_id == id_origem or potential_profile_id in checked_profile_ids:
                # logging.debug(f"Skipping profile ID {potential_profile_id} (self or already checked).")
                continue
            checked_profile_ids.add(potential_profile_id)

            # Carrega o perfil do vizinho potencial
            potential_profile = carregar_perfil_por_id_cached(app_data["db_path_profiles"], potential_profile_id)

            if potential_profile:
                # Calcula a similaridade personalizada (com os novos thresholds e pesos)
                total_score, score_details = calculate_custom_similarity(perfil_origem, potential_profile)
                processed_count += 1

                # Adiciona à lista APENAS se passou nos thresholds OBRIGATÓRIOS (total_score > 0)
                # E TAMBÉM no threshold geral final (MIN_CUSTOM_SCORE_THRESHOLD)
                if total_score > 0 and total_score >= MIN_CUSTOM_SCORE_THRESHOLD:
                    candidates_with_scores.append((total_score, potential_profile_id, score_details))
                # else: logging.debug(f"Candidate {potential_profile_id} discarded. Score: {total_score:.3f} (Thresholds passed: {total_score > 0}, Final Threshold: {MIN_CUSTOM_SCORE_THRESHOLD:.2f})")

            else:
                 logging.warning(f"Failed to load potential neighbor profile ID {potential_profile_id} (mapped from FAISS index {idx_in_matrix}). Skipping.")


        logging.info(f"Processed {processed_count} unique candidates from FAISS results.")
        logging.info(f"Found {len(candidates_with_scores)} candidates passing ALL thresholds before final sorting.")

        # Ordena os candidatos pelo score total (decrescente)
        candidates_with_scores.sort(key=lambda item: item[0], reverse=True)

        # Seleciona os top N vizinhos
        top_neighbors_data = candidates_with_scores[:num_neighbors_target]
        logging.info(f"Selected Top {len(top_neighbors_data)} neighbors after final ranking.")

        # Monta a lista final de perfis com os scores
        perfis_similares_final = []
        for score, pid, score_details in top_neighbors_data:
            perfil_similar = carregar_perfil_por_id_cached(app_data["db_path_profiles"], pid)
            if perfil_similar:
                perfil_similar['score_compatibilidade'] = round(score, 3)
                # Garante que score_details está arredondado
                perfil_similar['score_details'] = {k: round(v, 2) for k, v in score_details.items()}
                perfis_similares_final.append(perfil_similar)
            else:
                 # Isso não deveria acontecer se o perfil foi carregado antes, mas adiciona robustez
                 logging.error(f"CRITICAL: Failed to reload profile ID {pid} for final list, although it was scored and selected.")


        return perfil_origem, perfis_similares_final

    except faiss.FaissException as fe:
        logging.error(f"FAISS specific error during search for ID {id_origem}: {fe}", exc_info=True)
        return perfil_origem, []
    except Exception as e:
        logging.error(f"General error during search/ranking for ID {id_origem}: {e}", exc_info=True)
        return perfil_origem, []


# --- Função para Carregar Dados ---
def load_data_and_build_index():
    global app_data
    with data_load_lock:
        if app_data["data_loaded"] or app_data["loading_error"]:
             logging.info(f"Load attempt skipped. Loaded: {app_data['data_loaded']}, Error: {app_data['loading_error']}")
             return # Evita recarregar ou tentar carregar se já houve erro

        logging.info("Initiating data load and FAISS index build...")
        try:
            # 1. Carregar Embeddings e IDs
            db_path_emb = DATABASE_EMBEDDINGS
            logging.info(f"Loading embeddings from: {db_path_emb}")
            embeddings_matrix, ids_map, emb_dim = carregar_embeddings_e_ids_internal(db_path_emb)

            if embeddings_matrix is None or not ids_map or emb_dim is None:
                raise ValueError(f"Failed to load valid embeddings/IDs from {db_path_emb}. Check DB content and integrity.")
            if emb_dim != EXPECTED_EMBEDDING_DIM:
                 logging.warning(f"Loaded embedding dimension ({emb_dim}) differs from expected ({EXPECTED_EMBEDDING_DIM}).")
                 # Decide if this is critical or just a warning
                 # raise ValueError(f"Loaded embedding dimension ({emb_dim}) differs from expected ({EXPECTED_EMBEDDING_DIM}).")

            app_data["embeddings_matrix"] = embeddings_matrix
            app_data["profile_ids_map"] = ids_map # GUARDA A ORDEM DOS IDS CORRESPONDENTE ÀS LINHAS DA MATRIZ
            app_data["embedding_dim"] = emb_dim
            logging.info(f"Successfully loaded {len(ids_map)} embeddings with dimension {emb_dim}.")
            logging.info(f"Shape of embeddings matrix: {embeddings_matrix.shape}")
            logging.info(f"First 5 profile IDs loaded: {ids_map[:5]}")


            # 2. Verificar consistência com banco de perfis (opcional, mas útil)
            logging.info(f"Checking consistency with profiles DB: {app_data['db_path_profiles']}")
            profile_ids_in_db = set()
            try:
                with sqlite3.connect(app_data['db_path_profiles'], timeout=15.0) as conn_prof:
                    cursor_prof = conn_prof.cursor()
                    cursor_prof.execute("SELECT id FROM perfis")
                    profile_ids_in_db = {row[0] for row in cursor_prof.fetchall()}
                logging.info(f"Found {len(profile_ids_in_db)} profiles in the profiles database.")
                # Verifica se todos os IDs dos embeddings existem no banco de perfis
                missing_profiles = set(ids_map) - profile_ids_in_db
                if missing_profiles:
                    logging.warning(f"{len(missing_profiles)} embedding IDs do not have corresponding profiles in the profile DB. Examples: {list(missing_profiles)[:10]}")
                # Verifica se todos os perfis têm embeddings (menos crítico, mas informativo)
                missing_embeddings = profile_ids_in_db - set(ids_map)
                if missing_embeddings:
                    logging.warning(f"{len(missing_embeddings)} profiles in DB do not have corresponding embeddings. Examples: {list(missing_embeddings)[:10]}")

            except Exception as db_check_e:
                 logging.error(f"Error during consistency check with profiles DB: {db_check_e}", exc_info=True)
                 # Pode decidir continuar ou parar aqui dependendo da gravidade


            # 3. Construir Índice FAISS
            logging.info("Building FAISS index...")
            # Normalizar embeddings para IndexFlatIP (produto interno é cosseno em vetores normalizados)
            faiss.normalize_L2(embeddings_matrix)
            logging.info("Embeddings normalized (L2).")

            faiss_index = construir_indice_faiss_internal(embeddings_matrix, emb_dim)
            if faiss_index is None:
                raise ValueError("Failed to build FAISS index.")
            app_data["faiss_index"] = faiss_index
            logging.info(f"FAISS index built successfully. Index type: IndexFlatIP, Is trained: {faiss_index.is_trained}, Total vectors: {faiss_index.ntotal}")

            # 4. Marcar como carregado
            app_data["data_loaded"] = True
            app_data["loading_error"] = None
            logging.info("Data loading and index build complete.")

        except Exception as e:
            logging.critical(f"CRITICAL error during data load/index build: {e}", exc_info=True)
            app_data["loading_error"] = str(e)
            app_data["data_loaded"] = False
            # Limpar dados parcialmente carregados para evitar estado inconsistente
            app_data["embeddings_matrix"] = None
            app_data["profile_ids_map"] = None
            app_data["embedding_dim"] = None
            app_data["faiss_index"] = None


def carregar_embeddings_e_ids_internal(db_path: str) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[int]]:
    embeddings_list = []
    ids_list = []
    dimension = None
    expected_bytes = -1 # Para verificar consistência do blob

    try:
        logging.info(f"Connecting to embedding DB: {db_path}")
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            cursor = conn.cursor()
            # Garante ordem consistente para mapeamento correto com o índice FAISS
            cursor.execute("SELECT id, embedding FROM embeddings ORDER BY id")
            rows = cursor.fetchall()
            if not rows:
                logging.error("No data found in embeddings table.")
                return None, None, None

            logging.info(f"Processing {len(rows)} rows from embeddings table...")
            for i, (pid, blob) in enumerate(rows):
                try:
                    # Verifica se o blob não é nulo e tem tamanho esperado (se já definido)
                    if blob is None:
                         logging.warning(f"Row {i+1}, ID {pid}: Embedding blob is NULL. Skipping.")
                         continue

                    emb = np.frombuffer(blob, dtype=np.float32)

                    if dimension is None:
                        dimension = len(emb)
                        expected_bytes = dimension * 4 # float32 = 4 bytes
                        if dimension == 0:
                             logging.error("First embedding loaded has dimension 0. Cannot proceed.")
                             return None, None, None
                        logging.info(f"Detected embedding dimension: {dimension}. Expected blob size: {expected_bytes} bytes.")
                    elif len(emb) != dimension:
                        logging.warning(f"Row {i+1}, ID {pid}: Incorrect embedding dimension ({len(emb)}). Expected {dimension}. Skipping.")
                        continue
                    elif len(blob) != expected_bytes:
                         logging.warning(f"Row {i+1}, ID {pid}: Incorrect blob size ({len(blob)} bytes). Expected {expected_bytes}. Skipping (potential data corruption).")
                         continue

                    embeddings_list.append(emb)
                    ids_list.append(pid)

                except Exception as row_e:
                    logging.error(f"Error processing row {i+1}, ID {pid}: {row_e}", exc_info=False) # Avoid excessive logging in loop
                    continue # Skip problematic row

            if not embeddings_list:
                logging.error("No valid embeddings could be loaded after processing all rows.")
                return None, None, None

            # Cria a matriz NumPy de forma contígua para FAISS
            matrix = np.vstack(embeddings_list).astype(np.float32)
            logging.info(f"Successfully created embedding matrix with shape: {matrix.shape}")
            return matrix, ids_list, dimension

    except sqlite3.DatabaseError as db_err:
         logging.error(f"SQLite Error connecting to or reading from {db_path}: {db_err}", exc_info=True)
         return None, None, None
    except Exception as e:
        logging.error(f"Internal: Unexpected error loading embeddings: {e}", exc_info=True)
        return None, None, None


def construir_indice_faiss_internal(embeddings: np.ndarray, dimension: int) -> Optional[faiss.Index]:
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        logging.error(f"Invalid embeddings matrix provided for FAISS. Shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else 'N/A'}, Expected Dim: {dimension}")
        return None
    if embeddings.shape[0] == 0:
        logging.error("Cannot build FAISS index with an empty embeddings matrix.")
        return None
    if not embeddings.flags['C_CONTIGUOUS']:
         logging.warning("Embeddings matrix is not C-contiguous. Making a copy.")
         embeddings = np.ascontiguousarray(embeddings)

    try:
        logging.info(f"Initializing FAISS IndexFlatIP with dimension {dimension}")
        # Usamos IndexFlatIP porque normalizamos os vetores L2 antes.
        # O produto interno de vetores L2-normalizados é equivalente à similaridade de cosseno.
        index = faiss.IndexFlatIP(dimension)
        logging.info(f"Adding {embeddings.shape[0]} vectors to the FAISS index...")
        index.add(embeddings)
        logging.info("Vectors added successfully.")
        return index
    except Exception as e:
        logging.error(f"Internal: Error building FAISS index: {e}", exc_info=True)
        return None

# --- HTML Template (Sem alterações significativas, apenas para garantir que está completo) ---
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
        body {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(135deg, #1a0000 0%, #0f0f0f 50%, #1a1a1a 100%);
            background-attachment: fixed; color: #d1d5db; /* gray-300 */
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }
        .text-gradient-red { background: linear-gradient(90deg, #ff4d4d, #ff1a1a); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .text-gradient-score { background: linear-gradient(90deg, #4ade80, #16a34a); -webkit-background-clip: text; background-clip: text; color: transparent; }

        .profile-card {
            background: rgba(31, 41, 55, 0.75); /* gray-800 opacity increased */
            backdrop-filter: blur(10px) saturate(120%);
            border: 1px solid rgba(220, 38, 38, 0.4); /* red-600 border */
            box-shadow: 0 6px 25px 0 rgba(220, 38, 38, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            overflow: hidden; /* Prevent content breaking border radius */
        }
        .profile-card:hover { transform: translateY(-5px); box-shadow: 0 10px 35px 0 rgba(220, 38, 38, 0.25); }

        .btn-retest { background: linear-gradient(90deg, #ef4444, #b91c1c); transition: all 0.3s ease; box-shadow: 0 4px 15px 0 rgba(239, 68, 68, 0.3); }
        .btn-retest:hover { transform: translateY(-2px) scale(1.05); box-shadow: 0 6px 20px 0 rgba(239, 68, 68, 0.4); }
        .btn-retest:active { transform: translateY(0px) scale(1); box-shadow: 0 2px 10px 0 rgba(239, 68, 68, 0.2); }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.6s ease-out forwards; }
        {% for i in range(10) %}
        .similar-card-{{ i }} { animation-delay: {{ (i + 1) * 0.10 }}s; opacity: 0; } /* Slightly faster animation delay */
        {% endfor %}

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(31, 41, 55, 0.3); border-radius: 4px;} /* gray-800 transparent */
        ::-webkit-scrollbar-thumb { background: rgba(220, 38, 38, 0.6); border-radius: 4px;} /* red-600 transparent */
        ::-webkit-scrollbar-thumb:hover { background: rgba(239, 68, 68, 0.8); } /* red-500 transparent */

        .icon-style { color: #fca5a5; /* red-300 */ width: 1.1em; text-align: center; margin-right: 0.5rem; flex-shrink: 0; }
        .data-item { display: flex; align-items: flex-start; margin-bottom: 0.5rem; } /* Align icon with first line of text */
        .data-item i { margin-top: 0.15rem; } /* Adjust icon alignment slightly */
        .data-label { font-weight: 600; color: #9ca3af; /* gray-400 */ margin-right: 0.5rem; }
        .data-value { color: #d1d5db; /* gray-300 */ }

        /* Badge Styling for lists */
        .badge-list { display: flex; flex-wrap: wrap; gap: 0.3rem; padding-top: 0.1rem;}
        .badge-item {
            background-color: rgba(185, 28, 28, 0.7); /* red-700 bg */
            border: 1px solid rgba(220, 38, 38, 0.5); /* red-600 border */
            color: #fecaca; /* red-200 text */
            font-size: 0.7rem; /* Smaller font for badges */
            padding: 0.15rem 0.5rem;
            border-radius: 0.75rem; /* Pill shape */
            white-space: nowrap;
            line-height: 1.2;
            transition: background-color 0.2s ease;
        }
        .badge-item:hover { background-color: rgba(153, 27, 27, 0.8); } /* red-800 on hover */

        .card-section { border-top: 1px solid rgba(220, 38, 38, 0.2); padding-top: 0.75rem; margin-top: 0.75rem; }
        .card-section-title { font-size: 0.8rem; font-weight: 700; color: #fca5a5; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;}
    </style>
</head>
<body class="text-gray-300 min-h-screen">

    <!-- Header -->
    <header class="bg-gray-900/80 backdrop-blur-md shadow-lg sticky top-0 z-50 py-4 px-6">
        <div class="container mx-auto flex justify-between items-center">
            <h1 class="text-2xl md:text-3xl font-orbitron font-bold text-gradient-red">
                <i class="fas fa-user-astronaut mr-2"></i>Matchmaking Dashboard
            </h1>
            <button id="retestBtn" class="btn-retest text-white font-bold py-2 px-4 rounded-lg text-sm md:text-base flex items-center">
                <i class="fas fa-sync-alt mr-2"></i> Gerar Novo Match
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
        {% elif not data_loaded_completely %}
             <div class="bg-yellow-800 border border-yellow-600 text-yellow-100 px-4 py-3 rounded-lg relative mb-6 text-center" role="status">
                 <i class="fas fa-spinner fa-spin mr-2"></i> Carregando dados iniciais e construindo índice... O dashboard será exibido em breve. Por favor, aguarde.
             </div>
        {% elif perfil_origem %}
            <!-- Perfil de Origem Section -->
            <section class="mb-10">
                <h2 class="text-2xl font-orbitron font-semibold mb-4 border-b-2 border-red-600/50 pb-2 text-gray-300">
                    <i class="fas fa-crosshairs mr-2 text-red-400"></i>Perfil de Origem
                </h2>
                <div class="profile-card rounded-lg p-6 fade-in">
                    <h3 class="text-xl md:text-2xl font-bold font-orbitron text-gradient-red mb-4">{{ perfil_origem.nome }} <span class="text-sm font-light text-gray-500">(ID: {{ perfil_origem.id }})</span></h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 text-sm md:text-base">
                        <div class="data-item lg:col-span-1"><i class="fas fa-birthday-cake icon-style"></i><div><span class="data-label">Idade:</span><span class="data-value">{{ perfil_origem.idade }}</span></div></div>
                        <div class="data-item lg:col-span-1"><i class="fas fa-map-marker-alt icon-style"></i><div><span class="data-label">Local:</span><span class="data-value">{{ perfil_origem.cidade }}, {{ perfil_origem.estado }}</span></div></div>
                        <div class="data-item lg:col-span-1"><i class="fas fa-venus-mars icon-style"></i><div><span class="data-label">Sexo:</span><span class="data-value">{{ perfil_origem.sexo }}</span></div></div>
                        <div class="data-item lg:col-span-1"><i class="fas fa-clock icon-style"></i><div><span class="data-label">Disponível:</span><span class="data-value font-semibold text-amber-300">{{ perfil_origem.disponibilidade }}</span></div></div>
                        <div class="data-item lg:col-span-1"><i class="fas fa-comments icon-style"></i><div><span class="data-label">Interação:</span><span class="data-value">{{ perfil_origem.interacao_desejada }}</span></div></div>
                        <div class="data-item lg:col-span-1"><i class="fas fa-share-alt icon-style"></i><div><span class="data-label">Contato:</span><span class="data-value"> {% if perfil_origem.compartilhar_contato %}<span class="text-green-400">Sim <i class="fas fa-check-circle"></i></span>{% else %}<span class="text-red-400">Não <i class="fas fa-times-circle"></i></span>{% endif %}</span></div></div>
                    </div>

                    <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-headset mr-1"></i>Plataformas Possuídas</h4>
                        <div class="badge-list">
                            {% set platforms = safe_split_and_strip(perfil_origem.get('plataformas_possuidas')) %}
                            {% for item in platforms %}<span class="badge-item !bg-blue-800/70 !border-blue-600/50 !text-blue-200">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhuma</span>{% endfor %}
                        </div>
                    </div>
                    <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-gamepad mr-1"></i>Jogos Favoritos</h4>
                        <div class="badge-list">
                            {% set games = safe_split_and_strip(perfil_origem.get('jogos_favoritos')) %}
                            {% for item in games %}<span class="badge-item">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhum</span>{% endfor %}
                        </div>
                    </div>
                    <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-tags mr-1"></i>Estilos Preferidos</h4>
                         <div class="badge-list">
                            {% set styles = safe_split_and_strip(perfil_origem.get('estilos_preferidos')) %}
                            {% for item in styles %}<span class="badge-item">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhum</span>{% endfor %}
                        </div>
                    </div>
                     <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-music mr-1"></i>Interesses Musicais</h4>
                         <div class="badge-list">
                            {% set music = safe_split_and_strip(perfil_origem.get('interesses_musicais')) %}
                            {% for item in music %}<span class="badge-item">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhum</span>{% endfor %}
                        </div>
                    </div>
                    <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-info-circle mr-1"></i>Descrição</h4>
                        <p class="text-gray-400 text-sm italic leading-relaxed">{{ perfil_origem.descricao }}</p>
                    </div>
                </div>
            </section>

            <!-- Perfis Similares Section -->
            <section>
                <h2 class="text-2xl font-orbitron font-semibold mb-5 border-b-2 border-red-600/50 pb-2 text-gray-300">
                   <i class="fas fa-users-viewfinder mr-2 text-red-400"></i>Top {{ perfis_similares|length }} Perfis Similares (Prioridade: Plataforma/Horário)
                </h2>
                {% if perfis_similares %}
                    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                        {% for perfil in perfis_similares %}
                        <div class="profile-card rounded-lg p-4 flex flex-col h-full fade-in similar-card-{{ loop.index0 }}">
                             <div class="flex justify-between items-start mb-2">
                                <h4 class="text-lg font-bold font-orbitron text-gradient-red flex-grow pr-2">{{ perfil.nome }} <span class="text-xs font-light text-gray-500">(ID: {{ perfil.id }})</span></h4>
                                <div class="text-right flex-shrink-0">
                                    <span class="block font-semibold text-sm text-gray-400">Score Final</span>
                                    <span class="font-bold text-gradient-score text-xl">{{ "%.3f"|format(perfil.score_compatibilidade) }}</span>
                                </div>
                            </div>
                            <div class="text-xs md:text-sm space-y-1 mb-3 text-gray-400">
                                <p><i class="fas fa-birthday-cake icon-style"></i> {{ perfil.idade }} anos</p>
                                <p><i class="fas fa-map-marker-alt icon-style"></i> {{ perfil.cidade }}, {{ perfil.estado }}</p>
                                <p><i class="fas fa-clock icon-style"></i> <span class="font-semibold text-amber-300">{{ perfil.disponibilidade }}</span> (Score: {{ "%.2f"|format(perfil.score_details.get('disponibilidade', 0.0)) }})</p>
                                <p><i class="fas fa-comments icon-style"></i> {{ perfil.interacao_desejada }} (Score: {{ "%.2f"|format(perfil.score_details.get('interacao', 0.0)) }})</p>
                                <p><i class="fas fa-share-alt icon-style"></i> Contato: {% if perfil.compartilhar_contato %}<span class="text-green-400">Sim</span>{% else %}<span class="text-red-400">Não</span>{% endif %}</p>
                            </div>

                            <div class="card-section text-xs">
                                <h5 class="card-section-title"><i class="fas fa-headset mr-1"></i>Plataformas (Score: {{ "%.2f"|format(perfil.score_details.get('plataformas', 0.0)) }})</h5>
                                <div class="badge-list">
                                    {% set platforms = safe_split_and_strip(perfil.get('plataformas_possuidas')) %}
                                    {% for item in platforms %}<span class="badge-item !bg-blue-800/70 !border-blue-600/50 !text-blue-200">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhuma</span>{% endfor %}
                                </div>
                            </div>
                             <div class="card-section text-xs">
                                <h5 class="card-section-title"><i class="fas fa-gamepad mr-1"></i>Jogos (Score: {{ "%.2f"|format(perfil.score_details.get('jogos', 0.0)) }})</h5>
                                <div class="badge-list">
                                    {% set games = safe_split_and_strip(perfil.get('jogos_favoritos')) %}
                                    {% for item in games %}<span class="badge-item">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhum</span>{% endfor %}
                                </div>
                            </div>
                             <div class="card-section text-xs">
                                <h5 class="card-section-title"><i class="fas fa-tags mr-1"></i>Estilos (Score: {{ "%.2f"|format(perfil.score_details.get('estilos', 0.0)) }})</h5>
                                 <div class="badge-list">
                                    {% set styles = safe_split_and_strip(perfil.get('estilos_preferidos')) %}
                                    {% for item in styles %}<span class="badge-item">{{ item }}</span>{% else %}<span class="text-gray-500 text-xs italic">Nenhum</span>{% endfor %}
                                </div>
                            </div>

                             <!-- Detalhes do Score com pesos visíveis -->
                            <details class="text-xs mt-3 text-gray-500 opacity-75 hover:opacity-100 transition-opacity cursor-pointer">
                                <summary class="font-semibold outline-none">Detalhes do Score Ponderado</summary>
                                <ul class="list-none ml-2 mt-1 space-y-0.5 pt-1">
                                    {% for key, value in perfil.score_details.items() %}
                                    <li>
                                        <span class="inline-block w-24">{{ key|capitalize }}:</span>
                                        <span class="font-semibold">{{ "%.2f"|format(value) }}</span>
                                        <span class="text-gray-600 text-[0.65rem]">(Peso: {{ "%.2f"|format(WEIGHTS.get(key, 0.0)) }})</span>
                                        {% if key == 'plataformas' and value < MIN_REQUIRED_PLATFORM_SCORE %} <i class="fas fa-exclamation-triangle text-yellow-500" title="Abaixo do threshold mínimo"></i> {% endif %}
                                        {% if key == 'disponibilidade' and value < MIN_REQUIRED_AVAILABILITY_SCORE %} <i class="fas fa-exclamation-triangle text-yellow-500" title="Abaixo do threshold mínimo"></i> {% endif %}
                                    </li>
                                    {% endfor %}
                                     <li class="border-t border-gray-700/50 mt-1 pt-1">
                                        <span class="inline-block w-24 font-bold">Total Final:</span>
                                        <span class="font-bold text-green-400">{{ "%.3f"|format(perfil.score_compatibilidade) }}</span>
                                     </li>
                                </ul>
                            </details>

                        </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <p class="text-center text-gray-400 italic py-6 bg-gray-800/50 rounded-lg">
                        <i class="fas fa-ghost mr-2"></i> Nenhum perfil similar encontrado que satisfaça os critérios prioritários (plataforma/horário) e o score mínimo. Tente gerar um novo match ou ajuste os thresholds/pesos!
                    </p>
                {% endif %}
            </section>

        {% else %}
             <p class="text-center text-red-400 italic py-6 bg-red-900/30 rounded-lg">
                <i class="fas fa-sync fa-spin mr-2"></i> Carregando dados ou perfil de origem (ID: {{ id_origem | default('N/A') }}) não encontrado... Tente atualizar ou aguarde. Verifique os logs se o problema persistir.
             </p>
        {% endif %}
    </main>

    <!-- Footer -->
    <footer class="text-center py-6 mt-10 text-xs text-gray-500 border-t border-gray-700/30">
        Criado por <a href="#" class="text-red-400 hover:text-red-300">Replika AI Solutions</a> - Maringá, Paraná <br>
        <i class="fas fa-copyright mr-1"></i> {{ current_year }} - Matchmaking Dashboard v1.2 (Prioridade Alta: Plataforma/Horário)
    </footer>

    <script>
        document.getElementById('retestBtn').addEventListener('click', () => {
            const btn = document.getElementById('retestBtn');
            const icon = btn.querySelector('i');
            icon.classList.add('fa-spin');
            // Change button text while loading
            btn.disabled = true;
            // Get current button text content excluding the icon
            const currentText = btn.textContent.replace(icon.textContent, '').trim();
            btn.innerHTML = `<i class="fas fa-spinner fa-spin mr-2"></i> Gerando...`;
            window.location.href = '/new_match'; // Use dedicated route
        });

         // Adiciona acesso global às constantes para o template, se necessário
        const WEIGHTS = {{ WEIGHTS | tojson }};
        const MIN_REQUIRED_PLATFORM_SCORE = {{ MIN_REQUIRED_PLATFORM_SCORE }};
        const MIN_REQUIRED_AVAILABILITY_SCORE = {{ MIN_REQUIRED_AVAILABILITY_SCORE }};
    </script>

</body>
</html>
"""


# --- Rota Flask Principal ---
@app.route('/')
def index():
    """Rota principal que carrega, busca e renderiza os perfis."""
    start_time = datetime.datetime.now()
    logging.info("Request received for index route.")

    # Verifica se dados estão carregados ANTES de tentar usá-los
    if not app_data["data_loaded"]:
        # Se ainda estiver carregando (sem erro), mostra mensagem de loading
        if not app_data["loading_error"]:
            logging.info("Data not loaded yet, displaying loading message.")
            # Registra a função utilitária e constantes no ambiente Jinja
            app.jinja_env.globals.update(
                safe_split_and_strip=safe_split_and_strip,
                WEIGHTS=WEIGHTS,
                MIN_REQUIRED_PLATFORM_SCORE=MIN_REQUIRED_PLATFORM_SCORE,
                MIN_REQUIRED_AVAILABILITY_SCORE=MIN_REQUIRED_AVAILABILITY_SCORE
            )
            return render_template_string(
                index_html,
                data_loaded_completely=False, # Flag para mostrar loading
                error_message=None,
                perfil_origem=None,
                perfis_similares=[],
                current_year=datetime.datetime.now().year
            )
        # Se houve erro no carregamento
        else:
            logging.error(f"Displaying data loading error: {app_data['loading_error']}")
             # Registra a função utilitária e constantes no ambiente Jinja
            app.jinja_env.globals.update(
                safe_split_and_strip=safe_split_and_strip,
                WEIGHTS=WEIGHTS,
                MIN_REQUIRED_PLATFORM_SCORE=MIN_REQUIRED_PLATFORM_SCORE,
                MIN_REQUIRED_AVAILABILITY_SCORE=MIN_REQUIRED_AVAILABILITY_SCORE
            )
            return render_template_string(
                index_html,
                data_loaded_completely=False,
                error_message=f"Erro crítico ao carregar dados: {app_data['loading_error']}. Verifique os logs.",
                perfil_origem=None, perfis_similares=[],
                current_year=datetime.datetime.now().year
            )

    # Se chegou aqui, data_loaded é True
    if not app_data["profile_ids_map"]:
        logging.error("Data loaded, but profile_ids_map is empty.")
        app.jinja_env.globals.update(
            safe_split_and_strip=safe_split_and_strip,
            WEIGHTS=WEIGHTS,
            MIN_REQUIRED_PLATFORM_SCORE=MIN_REQUIRED_PLATFORM_SCORE,
            MIN_REQUIRED_AVAILABILITY_SCORE=MIN_REQUIRED_AVAILABILITY_SCORE
        )
        return render_template_string(
            index_html, data_loaded_completely=True,
            error_message="Nenhum perfil encontrado nos dados carregados (lista de IDs vazia).",
            perfil_origem=None, perfis_similares=[],
            current_year=datetime.datetime.now().year
        )

    # Seleciona um ID de origem aleatório
    id_origem = None # Inicializa
    try:
        id_origem = random.choice(app_data["profile_ids_map"])
        logging.info(f"Selected random origin profile ID: {id_origem}")
    except IndexError:
         logging.error("Failed to select random ID - profile_ids_map might be empty despite check.")
         app.jinja_env.globals.update(
            safe_split_and_strip=safe_split_and_strip,
            WEIGHTS=WEIGHTS,
            MIN_REQUIRED_PLATFORM_SCORE=MIN_REQUIRED_PLATFORM_SCORE,
            MIN_REQUIRED_AVAILABILITY_SCORE=MIN_REQUIRED_AVAILABILITY_SCORE
         )
         return render_template_string(
            index_html, data_loaded_completely=True,
            error_message="Erro interno: Não foi possível selecionar um perfil de origem.",
            perfil_origem=None, perfis_similares=[],
            current_year=datetime.datetime.now().year
         )

    # Busca e rankeia os vizinhos
    perfil_origem, perfis_similares = buscar_e_rankear_vizinhos(id_origem, NUM_NEIGHBORS_TARGET)

    # Registra a função utilitária e constantes no ambiente Jinja ANTES de renderizar
    # É importante fazer isso a cada request se o ambiente Jinja não for persistente
    app.jinja_env.globals.update(
        safe_split_and_strip=safe_split_and_strip,
        WEIGHTS=WEIGHTS,
        MIN_REQUIRED_PLATFORM_SCORE=MIN_REQUIRED_PLATFORM_SCORE,
        MIN_REQUIRED_AVAILABILITY_SCORE=MIN_REQUIRED_AVAILABILITY_SCORE
    )

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Finished processing index request for ID {id_origem}. Duration: {duration.total_seconds():.3f}s. Found {len(perfis_similares)} matches.")

    return render_template_string(
        index_html,
        data_loaded_completely=True, # Dados carregados
        perfil_origem=perfil_origem,
        perfis_similares=perfis_similares,
        error_message= None if perfil_origem else f"Perfil de origem (ID: {id_origem}) não encontrado no banco de dados.",
        current_year=datetime.datetime.now().year,
        # Passa o ID de origem para o template caso o perfil não seja encontrado
        id_origem=id_origem
    )

# Rota dedicada para o botão, para evitar F5 re-submetendo a mesma lógica
@app.route('/new_match')
def new_match():
    """Redireciona para a raiz para obter um novo match."""
    logging.info("Redirecting to generate new match.")
    return redirect(url_for('index'))


# --- Função para iniciar o carregamento em background ---
def start_background_load():
    # Verifica se já está carregado OU se já ocorreu um erro para não tentar de novo
    if not app_data["data_loaded"] and not app_data["loading_error"]:
        logging.info("Starting background data loading thread...")
        load_thread = threading.Thread(target=load_data_and_build_index, name="DataLoaderThread", daemon=True)
        load_thread.start()
    elif app_data["data_loaded"]:
        logging.info("Data already loaded. Skipping background load initiation.")
    else: # loading_error is set
         logging.warning("Previous loading error detected. Skipping background load initiation.")

# --- Ponto de Entrada ---
if __name__ == '__main__':
    logging.info(f"--- Application Start ---")
    logging.info(f"Checking database existence...")
    db_profiles_exists = os.path.exists(DATABASE_PROFILES)
    db_embeddings_exists = os.path.exists(DATABASE_EMBEDDINGS)

    if not db_profiles_exists or not db_embeddings_exists:
        msg = "❌ CRITICAL ERROR: Required database files not found."
        logging.critical(msg)
        if not db_profiles_exists: logging.critical(f"   - Missing Profile DB: '{DATABASE_PROFILES}'")
        if not db_embeddings_exists: logging.critical(f"   - Missing Embedding DB: '{DATABASE_EMBEDDINGS}'")
        logging.critical("Please ensure 'profile_generator_v3.py' has been run successfully and the databases are in the correct location ('databases_v3' directory).")
        # Exibe mensagem no console também
        print(f"\n{msg}")
        if not db_profiles_exists: print(f"   - Missing Profile DB: '{DATABASE_PROFILES}'")
        if not db_embeddings_exists: print(f"   - Missing Embedding DB: '{DATABASE_EMBEDDINGS}'")
        print("   Please ensure 'profile_generator_v3.py' has been run successfully")
        print(f"   and the databases are in the expected directory: '{DB_DIR}'.\n")
        # Encerra a aplicação se os bancos não existem
        exit(1)
    else:
        logging.info("Database files found.")
        print("⏳ Initializing: Starting background data load and index build...")
        start_background_load() # Dispara o carregamento em background

        print(f"\n🚀 Starting Flask server on port {FLASK_PORT}...")
        print(f"   Access dashboard at: http://127.0.0.1:{FLASK_PORT}")
        print(f"   (Note: Dashboard may show 'Loading...' initially until background task completes)")
        print(f"   View detailed logs at: {LOG_FILE_VAL}")
        print("   Press CTRL+C to stop the server.")
        try:
            from waitress import serve
            print("   Using Waitress server for production.")
            # Aumentar threads pode ajudar com múltiplas requisições simultâneas
            # Ajuste 'threads' conforme a capacidade do seu sistema
            serve(app, host='127.0.0.1', port=FLASK_PORT, threads=8)
        except ImportError:
            print("   Waitress not found, using Flask development server (WARNING: Not recommended for production).")
            print("   Install Waitress for better performance: pip install waitress")
            # debug=True recarrega em mudanças de código, mas pode ser mais lento e consumir mais memória
            app.run(host='127.0.0.1', port=FLASK_PORT, debug=FLASK_DEBUG)