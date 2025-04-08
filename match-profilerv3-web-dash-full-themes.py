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
from flask import Flask, render_template_string, redirect, url_for, jsonify # Adicionado jsonify
import threading
from functools import lru_cache

# --- Configurações ---
# (Mantidas da versão anterior)
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
VALUATION_DIR = "valuation_v3_web_log"
os.makedirs(VALUATION_DIR, exist_ok=True)

FLASK_PORT: int = 8881
FLASK_DEBUG: bool = False # Mantenha False para produção com Waitress

NUM_NEIGHBORS_TARGET: int = 10
INITIAL_SEARCH_FACTOR: int = 25
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.05

LOG_LEVEL = logging.INFO
EXPECTED_EMBEDDING_DIM: int = 64

# Pesos e Thresholds (Mantidos da versão anterior com prioridade)
WEIGHTS = {
    "plataformas": 0.45, "disponibilidade": 0.35, "jogos": 0.10,
    "estilos": 0.05, "interacao": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"
MIN_REQUIRED_PLATFORM_SCORE: float = 0.20
MIN_REQUIRED_AVAILABILITY_SCORE: float = 0.30
# ----------------------------------------------

# --- Nomes dos Temas ---
AVAILABLE_THEMES = ["red", "purple", "blue", "yellow", "white", "orange"]
DEFAULT_THEME = "red"
# -----------------------

# --- Configuração de Logging ---
# (Mantida da versão anterior)
LOG_FILE_VAL = os.path.join(VALUATION_DIR, f"matchmaking_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE_VAL,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(funcName)s - %(message)s',
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
# (Mantidas da versão anterior)
app_data = {
    "embeddings_matrix": None, "profile_ids_map": None, "embedding_dim": None,
    "faiss_index": None, "data_loaded": False, "loading_error": None,
    "db_path_profiles": DATABASE_PROFILES
}
data_load_lock = threading.Lock()

# --- Funções Auxiliares ---
# carregar_perfil_por_id_cached, safe_split_and_strip, jaccard_similarity,
# availability_similarity, interaction_similarity, calculate_custom_similarity,
# buscar_e_rankear_vizinhos, load_data_and_build_index,
# carregar_embeddings_e_ids_internal, construir_indice_faiss_internal
# (Mantidas EXATAMENTE como na versão anterior com prioridade alta)
# --- (Omitido por brevidade - Cole o código dessas funções aqui da versão anterior) ---
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
        # A busca pelo índice no mapa de IDs deve ser feita aqui
        source_index_in_map = -1
        try:
            source_index_in_map = app_data["profile_ids_map"].index(id_origem)
            embedding_origem = app_data["embeddings_matrix"][source_index_in_map].reshape(1, -1)
        except ValueError:
            logging.error(f"Origin ID {id_origem} not found in profile_ids_map. Cannot perform FAISS search.")
            return perfil_origem, []
        except (IndexError, TypeError) as e:
             logging.error(f"Error getting embedding for origin ID {id_origem} (index {source_index_in_map}): {e}", exc_info=True)
             return perfil_origem, []

    except Exception as e: # Captura outras exceções inesperadas antes da busca FAISS
        logging.error(f"Unexpected error preparing for FAISS search for origin ID {id_origem}: {e}", exc_info=True)
        return perfil_origem, []


    # Aumentar k_search para compensar a filtragem mais rigorosa
    k_search = num_neighbors_target * INITIAL_SEARCH_FACTOR + 1
    # Garante que k_search não seja maior que o número total de perfis no índice
    if app_data["faiss_index"] is None:
         logging.error("FAISS index is not loaded. Cannot search.")
         return perfil_origem, []
    k_search = min(k_search, app_data["faiss_index"].ntotal)

    if k_search <= 1 and app_data["faiss_index"].ntotal > 1:
         logging.warning(f"Adjusted k_search ({k_search}) is too low. Might not find enough neighbors. Index size: {app_data['faiss_index'].ntotal}")
         # Pode aumentar k_search se for muito baixo, ex: k_search = max(k_search, 10)

    candidates_with_scores: List[Tuple[float, int, Dict[str, float]]] = []
    checked_profile_ids: Set[int] = {id_origem} # Evita auto-comparação e duplicatas

    try:
        logging.info(f"Performing FAISS search with k={k_search} for ID {id_origem}")
        # Certifique-se que embedding_origem está normalizado se o índice for IndexFlatIP
        # A normalização já deve ter sido feita no load_data_and_build_index
        distances, faiss_indices = app_data["faiss_index"].search(embedding_origem, k_search)

        if faiss_indices is None or len(faiss_indices) == 0 or len(faiss_indices[0]) == 0:
             logging.warning(f"FAISS search returned no neighbors for ID {id_origem} (k={k_search}).")
             return perfil_origem, []

        faiss_indices_found = faiss_indices[0]
        # O primeiro índice pode ser ele mesmo, especialmente com IndexFlatIP e vetor normalizado
        logging.info(f"FAISS search returned {len(faiss_indices_found)} potential candidate indices. Distances[0]: {distances[0][:5]}") # Log das primeiras distâncias

        processed_count = 0
        for i, idx_in_matrix in enumerate(faiss_indices_found):
            # O índice retornado pelo FAISS é a posição na matriz de embeddings
            if not (0 <= idx_in_matrix < len(app_data["profile_ids_map"])):
                logging.warning(f"Invalid FAISS index {idx_in_matrix} returned (out of bounds for profile_ids_map size {len(app_data['profile_ids_map'])}). Skipping.")
                continue

            potential_profile_id = app_data["profile_ids_map"][idx_in_matrix]

            # Pula se for o próprio perfil de origem ou já verificado
            # Com IndexFlatIP e vetores normalizados, a maior distância (produto interno) deve ser 1.0 para o próprio vetor.
            # Podemos pular o primeiro resultado se a distância for muito próxima de 1.0
            if potential_profile_id == id_origem:
                # logging.debug(f"Skipping index {i} (ID {potential_profile_id}) because it's the origin ID.")
                continue
            if potential_profile_id in checked_profile_ids:
                # logging.debug(f"Skipping profile ID {potential_profile_id} (already checked).")
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
            # logging.info(f"First 5 profile IDs loaded: {ids_map[:5]}") # Maybe too verbose

            # 2. Verificar consistência com banco de perfis (opcional, mas útil)
            logging.info(f"Checking consistency with profiles DB: {app_data['db_path_profiles']}")
            profile_ids_in_db = set()
            try:
                # Adiciona um timeout maior para esta conexão também
                with sqlite3.connect(app_data['db_path_profiles'], timeout=30.0) as conn_prof:
                    cursor_prof = conn_prof.cursor()
                    cursor_prof.execute("SELECT id FROM perfis")
                    profile_ids_in_db = {row[0] for row in cursor_prof.fetchall()}
                logging.info(f"Found {len(profile_ids_in_db)} profiles in the profiles database.")

                # Garante que todos os IDs nos embeddings existem como perfis
                # Se um embedding não tem perfil, ele não pode ser usado corretamente
                missing_profiles_for_embeddings = set(ids_map) - profile_ids_in_db
                if missing_profiles_for_embeddings:
                    logging.error(f"{len(missing_profiles_for_embeddings)} embedding IDs do NOT have corresponding profiles in the profile DB. This will cause errors later. Examples: {list(missing_profiles_for_embeddings)[:10]}")
                    # Decide se quer parar a aplicação aqui
                    raise ValueError("Inconsistency found: Embeddings exist for profiles not present in the profile database.")

                # Verifica se todos os perfis têm embeddings (menos crítico, mas informativo)
                missing_embeddings_for_profiles = profile_ids_in_db - set(ids_map)
                if missing_embeddings_for_profiles:
                    logging.warning(f"{len(missing_embeddings_for_profiles)} profiles in DB do not have corresponding embeddings. They cannot be used as origin or found in matches. Examples: {list(missing_embeddings_for_profiles)[:10]}")

            except sqlite3.Error as db_check_e:
                 logging.error(f"SQLite Error during consistency check with profiles DB: {db_check_e}", exc_info=True)
                 raise # Re-raise para parar o carregamento
            except Exception as consistency_e:
                 logging.error(f"General error during consistency check: {consistency_e}", exc_info=True)
                 raise # Re-raise


            # 3. Construir Índice FAISS
            logging.info("Building FAISS index...")
            # Normalizar embeddings para IndexFlatIP (produto interno é cosseno em vetores normalizados)
            # Fazendo uma cópia para não modificar o original se for necessário em outro lugar
            embeddings_normalized = embeddings_matrix.copy()
            faiss.normalize_L2(embeddings_normalized)
            logging.info("Embeddings normalized (L2) for FAISS index.")

            faiss_index = construir_indice_faiss_internal(embeddings_normalized, emb_dim)
            if faiss_index is None:
                raise ValueError("Failed to build FAISS index.")
            app_data["faiss_index"] = faiss_index
            logging.info(f"FAISS index built successfully. Index type: IndexFlatIP, Is trained: {faiss_index.is_trained}, Total vectors: {faiss_index.ntotal}")

            # 4. Marcar como carregado
            app_data["data_loaded"] = True
            app_data["loading_error"] = None
            logging.info("--- Data loading and index build complete ---")

        except Exception as e:
            logging.critical(f"--- CRITICAL error during data load/index build: {e} ---", exc_info=True)
            app_data["loading_error"] = f"Failed during initialization: {e}" # Mensagem mais amigável
            app_data["data_loaded"] = False
            # Limpar dados parcialmente carregados para evitar estado inconsistente
            app_data["embeddings_matrix"] = None
            app_data["profile_ids_map"] = None
            app_data["embedding_dim"] = None
            app_data["faiss_index"] = None
            # Limpar cache da função de perfil pode ser útil
            carregar_perfil_por_id_cached.cache_clear()

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
            ids_seen = set() # Para detectar IDs duplicados no BD de embeddings
            for i, (pid, blob) in enumerate(rows):
                try:
                    if pid in ids_seen:
                         logging.warning(f"Row {i+1}: Duplicate profile ID {pid} found in embeddings DB. Skipping this instance.")
                         continue
                    ids_seen.add(pid)

                    # Verifica se o blob não é nulo e tem tamanho esperado (se já definido)
                    if blob is None:
                         logging.warning(f"Row {i+1}, ID {pid}: Embedding blob is NULL. Skipping.")
                         continue

                    emb = np.frombuffer(blob, dtype=np.float32)

                    if dimension is None:
                        dimension = len(emb)
                        if dimension == 0:
                             logging.error("First embedding loaded has dimension 0. Cannot proceed.")
                             return None, None, None
                        if dimension != EXPECTED_EMBEDDING_DIM:
                             logging.warning(f"First embedding has dimension {dimension}, but EXPECTED_EMBEDDING_DIM is {EXPECTED_EMBEDDING_DIM}.")
                             # Permitir continuar, mas logar a discrepância
                        expected_bytes = dimension * 4 # float32 = 4 bytes
                        logging.info(f"Detected embedding dimension: {dimension}. Expected blob size: {expected_bytes} bytes.")

                    elif len(emb) != dimension:
                        logging.warning(f"Row {i+1}, ID {pid}: Incorrect embedding dimension ({len(emb)}). Expected {dimension}. Skipping this row.")
                        continue
                    # Verificação de tamanho do blob é redundante se frombuffer funcionou e len(emb) está ok
                    # elif len(blob) != expected_bytes:
                    #      logging.warning(f"Row {i+1}, ID {pid}: Incorrect blob size ({len(blob)} bytes). Expected {expected_bytes}. Skipping (potential data corruption).")
                    #      continue

                    embeddings_list.append(emb)
                    ids_list.append(pid)

                except ValueError as ve: # Erro comum com frombuffer se o tipo/tamanho estiver errado
                     logging.error(f"Error processing row {i+1}, ID {pid} (ValueError, likely buffer issue): {ve}", exc_info=False)
                     continue
                except Exception as row_e:
                    logging.error(f"Error processing row {i+1}, ID {pid}: {row_e}", exc_info=False) # Avoid excessive logging in loop
                    continue # Skip problematic row

            if not embeddings_list:
                logging.error("No valid embeddings could be loaded after processing all rows.")
                return None, None, None

            # Cria a matriz NumPy de forma contígua para FAISS
            # np.vstack é mais robusto que np.array(embeddings_list) se as listas internas não forem perfeitamente uniformes antes da verificação
            matrix = np.vstack(embeddings_list).astype(np.float32)
            if not matrix.flags['C_CONTIGUOUS']:
                 logging.warning("Created embedding matrix is not C-contiguous. Forcing contiguity.")
                 matrix = np.ascontiguousarray(matrix, dtype=np.float32)

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
         logging.warning("Embeddings matrix provided to build_index is not C-contiguous. This might impact performance or compatibility. Trying to proceed...")
         # FAISS pode lidar com isso internamente, mas é bom avisar.

    try:
        logging.info(f"Initializing FAISS IndexFlatIP with dimension {dimension}")
        # Usamos IndexFlatIP porque normalizamos os vetores L2 antes.
        # O produto interno de vetores L2-normalizados é equivalente à similaridade de cosseno.
        index = faiss.IndexFlatIP(dimension)
        logging.info(f"Adding {embeddings.shape[0]} normalized vectors to the FAISS index...")
        index.add(embeddings) # Adiciona os vetores já normalizados
        logging.info("Vectors added successfully.")
        if not index.is_trained:
             logging.warning("FAISS IndexFlatIP should be trained immediately after creation, but reports as not trained. This is unusual.")
        return index
    except faiss.FaissException as fe:
         logging.error(f"FAISS Exception during index building or adding vectors: {fe}", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"Internal: Error building FAISS index: {e}", exc_info=True)
        return None


# --- HTML Template (Expandido com JS para Temas e DOM) ---
index_html = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matchmaking Dashboard Avançado - Replika AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- FontAwesome com fallback -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="stylesheet" href="https://use.fontawesome.com/releases/v6.5.1/css/all.css" media="print" onload="this.media='all'">
    <noscript>
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v6.5.1/css/all.css">
    </noscript>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Base styles - Applied regardless of theme */
        body {
            font-family: 'Rajdhani', sans-serif;
            transition: background-color 0.5s ease, color 0.3s ease;
            /* Default background/text set by theme */
        }
        .font-orbitron { font-family: 'Orbitron', sans-serif; }

        /* Transitions for theme changes */
        .profile-card, .btn-retest, header, .badge-item, .card-section, h1, h2, h3, h4, h5, footer, details summary, details ul, .alert-box {
            transition: background-color 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease, opacity 0.3s ease;
        }

        /* Gradient Text Base - Color applied by theme JS */
        .text-gradient-dynamic {
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
            transition: background 0.3s ease;
        }
        /* Score Gradient - Always Greenish */
        .text-gradient-score { background: linear-gradient(90deg, #4ade80, #16a34a); -webkit-background-clip: text; background-clip: text; color: transparent; }

        /* Profile Card Base */
        .profile-card {
            backdrop-filter: blur(10px) saturate(120%);
            box-shadow: 0 6px 25px 0 rgba(0, 0, 0, 0.1); /* Default shadow, theme might override alpha */
            overflow: hidden;
            border: 1px solid; /* Color set by theme */
        }
        .profile-card:hover {
            transform: translateY(-5px);
            /* Shadow enhancement on hover depends on theme */
        }

        /* Button Base */
        .btn-retest {
            transition: all 0.3s ease;
             /* BG, Shadow, Text color set by theme */
        }
        .btn-retest:hover { transform: translateY(-2px) scale(1.03); }
        .btn-retest:active { transform: translateY(0px) scale(1); }

        /* Animations */
        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.6s ease-out forwards; }
        {% for i in range(10) %}
        .similar-card-{{ i }} { animation-delay: {{ (i + 1) * 0.10 }}s; opacity: 0; }
        {% endfor %}

        /* Scrollbar styling - Adapts slightly based on theme */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(55, 65, 81, 0.3); border-radius: 4px;} /* gray-700ish transparent */
        ::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.5); border-radius: 4px; transition: background-color 0.3s ease;} /* Default thumb, theme overrides */
        ::-webkit-scrollbar-thumb:hover { background: rgba(160, 160, 160, 0.7); } /* Default hover, theme overrides */

        /* Base Icon Style - Color from theme */
        .icon-style { width: 1.1em; text-align: center; margin-right: 0.5rem; flex-shrink: 0; }
        .data-item { display: flex; align-items: flex-start; margin-bottom: 0.5rem; }
        .data-item i { margin-top: 0.15rem; }
        .data-label { font-weight: 600; margin-right: 0.5rem; /* Color from theme */ }
        .data-value { /* Color from theme */ }

        /* Badge Base - BG, Border, Text color from theme */
        .badge-list { display: flex; flex-wrap: wrap; gap: 0.3rem; padding-top: 0.1rem;}
        .badge-item {
            font-size: 0.7rem; padding: 0.15rem 0.5rem; border-radius: 0.75rem;
            white-space: nowrap; line-height: 1.2; border: 1px solid;
        }
        /* Specific badge types might get different base colors */
        .badge-platform { /* Platform-specific base style if needed */ }
        .badge-game { /* Game-specific base style if needed */ }
        .badge-style { /* Style-specific base style if needed */ }
        .badge-music { /* Music-specific base style if needed */ }

        /* Card Section Base */
        .card-section { border-top: 1px solid; padding-top: 0.75rem; margin-top: 0.75rem; /* Border color from theme */ }
        .card-section-title { font-size: 0.8rem; font-weight: 700; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; /* Color from theme */ }

        /* Alert/Message Box Base */
        .alert-box { border: 1px solid; padding: 0.75rem 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; }

        /* Theme Selector Style */
        #themeSelector {
             background-color: rgba(31, 41, 55, 0.8); /* gray-800 base */
             border: 1px solid rgba(107, 114, 128, 0.5); /* gray-500 base */
             color: #d1d5db; /* gray-300 base */
             padding: 0.3rem 0.6rem;
             border-radius: 0.375rem;
             font-size: 0.8rem;
             cursor: pointer;
             appearance: none; /* Remove default system appearance */
             background-image: url('data:image/svg+xml;charset=US-ASCII,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="%239ca3af"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>'); /* Custom dropdown arrow */
             background-repeat: no-repeat;
             background-position: right 0.5rem center;
             background-size: 1.2em;
             padding-right: 2rem; /* Space for the arrow */
             transition: border-color 0.3s ease, background-color 0.3s ease;
        }
         #themeSelector:hover {
             border-color: rgba(156, 163, 175, 0.7); /* gray-400 base hover */
         }
         #themeSelector option {
             background-color: #1f2937; /* gray-800 */
             color: #d1d5db; /* gray-300 */
         }

    </style>
</head>
<body class="min-h-screen"> <!-- Theme class added by JS -->

    <!-- Header -->
    <header id="mainHeader" class="backdrop-blur-md shadow-lg sticky top-0 z-50 py-4 px-6">
        <div class="container mx-auto flex justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-orbitron font-bold text-gradient-dynamic"> <!-- Dynamic Gradient -->
                <i class="fas fa-user-astronaut mr-2 header-icon"></i>Matchmaking Dashboard
            </h1>
            <div class="flex items-center gap-3 md:gap-4">
                 <!-- Theme Selector -->
                <div class="relative">
                     <select id="themeSelector" aria-label="Select Theme">
                         {% for theme_name in available_themes %}
                         <option value="{{ theme_name }}">{{ theme_name|capitalize }}</option>
                         {% endfor %}
                     </select>
                </div>
                <button id="retestBtn" class="btn-retest text-white font-bold py-2 px-4 rounded-lg text-sm md:text-base flex items-center">
                    <i class="fas fa-sync-alt mr-2"></i>
                    <span class="btn-retest-text">Gerar Novo Match</span>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container mx-auto p-4 md:p-8">
        <!-- Loading Message -->
        <div id="loadingMessage" class="alert-box bg-yellow-800/80 border-yellow-600 text-yellow-100 text-center hidden" role="status">
            <i class="fas fa-spinner fa-spin mr-2"></i> Carregando dados iniciais e construindo índice... O dashboard será exibido em breve. Por favor, aguarde.
        </div>
        <!-- Error Message -->
        <div id="errorMessage" class="alert-box bg-red-800/80 border-red-600 text-red-100 hidden" role="alert">
            <strong class="font-bold"><i class="fas fa-exclamation-triangle mr-2"></i>Erro!</strong>
            <span id="errorMessageText"></span>
        </div>

        <!-- Content Area (Populated when data loads) -->
        <div id="mainContentArea" class="hidden">
            <!-- Perfil de Origem Section -->
            <section id="originProfileSection" class="mb-10 hidden">
                <h2 class="section-title text-2xl font-orbitron font-semibold mb-4 border-b-2 pb-2"> <!-- Dynamic Border/Text -->
                   <i class="fas fa-crosshairs mr-2 section-icon"></i>Perfil de Origem <!-- Dynamic Icon Color -->
                </h2>
                <div id="originProfileCard" class="profile-card rounded-lg p-6 fade-in">
                    <!-- Origin profile details will be injected here by JS or rendered by Jinja if doing full page reload -->
                     <h3 id="originName" class="text-xl md:text-2xl font-bold font-orbitron text-gradient-dynamic mb-4"></h3>
                     <div id="originDetailsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-2 text-sm md:text-base">
                         <!-- Data items like idade, local, etc. -->
                         <div class="data-item lg:col-span-1"><i class="fas fa-birthday-cake icon-style"></i><div><span class="data-label">Idade:</span><span class="data-value" id="originIdade"></span></div></div>
                         <div class="data-item lg:col-span-1"><i class="fas fa-map-marker-alt icon-style"></i><div><span class="data-label">Local:</span><span class="data-value" id="originLocal"></span></div></div>
                         <div class="data-item lg:col-span-1"><i class="fas fa-venus-mars icon-style"></i><div><span class="data-label">Sexo:</span><span class="data-value" id="originSexo"></span></div></div>
                         <div class="data-item lg:col-span-1"><i class="fas fa-clock icon-style"></i><div><span class="data-label">Disponível:</span><span class="data-value font-semibold" id="originDisp"></span></div></div>
                         <div class="data-item lg:col-span-1"><i class="fas fa-comments icon-style"></i><div><span class="data-label">Interação:</span><span class="data-value" id="originInteracao"></span></div></div>
                         <div class="data-item lg:col-span-1"><i class="fas fa-share-alt icon-style"></i><div><span class="data-label">Contato:</span><span class="data-value" id="originContato"></span></div></div>
                     </div>
                     <!-- Badge Sections -->
                     <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-headset mr-1"></i>Plataformas Possuídas</h4>
                        <div class="badge-list" id="originPlatforms"></div>
                     </div>
                     <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-gamepad mr-1"></i>Jogos Favoritos</h4>
                        <div class="badge-list" id="originGames"></div>
                     </div>
                     <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-tags mr-1"></i>Estilos Preferidos</h4>
                        <div class="badge-list" id="originStyles"></div>
                     </div>
                      <div class="card-section">
                        <h4 class="card-section-title"><i class="fas fa-music mr-1"></i>Interesses Musicais</h4>
                        <div class="badge-list" id="originMusic"></div>
                     </div>
                     <div class="card-section">
                         <h4 class="card-section-title"><i class="fas fa-info-circle mr-1"></i>Descrição</h4>
                         <p class="text-sm italic leading-relaxed" id="originDesc"></p> <!-- Dynamic Text Color -->
                     </div>
                </div>
            </section>

            <!-- Perfis Similares Section -->
            <section id="similarProfilesSection" class="hidden">
                <h2 id="similarProfilesTitle" class="section-title text-2xl font-orbitron font-semibold mb-5 border-b-2 pb-2"> <!-- Dynamic Border/Text -->
                   <i class="fas fa-users-viewfinder mr-2 section-icon"></i>Top Perfis Similares <!-- Dynamic Icon Color -->
                </h2>
                <div id="similarProfilesGrid" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    <!-- Similar profile cards will be injected here by JS -->
                </div>
                <p id="noSimilarProfilesMessage" class="text-center italic py-6 bg-gray-800/50 rounded-lg hidden"> <!-- Dynamic Text/BG -->
                    <i class="fas fa-ghost mr-2"></i> Nenhum perfil similar encontrado que satisfaça os critérios prioritários (plataforma/horário) e o score mínimo. Tente gerar um novo match ou ajuste os thresholds/pesos!
                </p>
            </section>

             <!-- Fallback message if origin profile itself wasn't found -->
             <p id="originNotFoundMessage" class="alert-box bg-red-900/30 text-red-400 italic text-center hidden">
                <i class="fas fa-exclamation-circle mr-2"></i> Perfil de origem não encontrado no banco de dados. Tente gerar um novo match.
             </p>
        </div>

    </main>

    <!-- Footer -->
    <footer id="pageFooter" class="text-center py-6 mt-10 text-xs border-t"> <!-- Dynamic Border/Text Color -->
        Criado por <a href="#" id="footerLink" class="hover:underline">Replika AI Solutions</a> - Maringá, Paraná <br> <!-- Dynamic Link Color -->
        <i class="fas fa-copyright mr-1"></i> <span id="currentYear"></span> - Matchmaking Dashboard v1.3 (Themes & State)
    </footer>

    <!-- JavaScript -->
    <script>
        // --- Theme Definitions ---
        // Define Tailwind classes for different elements per theme
        const themes = {
            red: { // Default
                name: 'Red',
                bodyBg: 'bg-gradient-to-br from-red-900/10 via-black to-gray-900/50',
                textColor: 'text-gray-300',
                primaryTextColor: 'text-red-400',
                secondaryTextColor: 'text-red-300',
                dataLabelColor: 'text-gray-400',
                dataValueColor: 'text-gray-300',
                descriptionColor: 'text-gray-400',
                iconColor: 'text-red-300', // For most icons
                sectionIconColor: 'text-red-400',
                headerIconColor: 'text-red-400',
                textGradient: 'bg-gradient-to-r from-red-500 to-red-700',
                buttonBg: 'bg-gradient-to-r from-red-600 to-red-800 hover:from-red-700 hover:to-red-900',
                buttonText: 'text-white',
                buttonShadow: 'shadow-md hover:shadow-lg shadow-red-500/30 hover:shadow-red-500/40',
                cardBg: 'bg-gray-800/70',
                cardBorder: 'border-red-600/40',
                cardHoverShadow: 'shadow-lg shadow-red-500/10',
                sectionTitleColor: 'text-red-400',
                sectionBorder: 'border-red-600/50',
                badgeBg: 'bg-red-700/70',
                badgeBorder: 'border-red-600/50',
                badgeText: 'text-red-200',
                badgePlatformBg: 'bg-blue-800/70', // Keep platforms distinct or theme them? Theming them:
                // badgePlatformBg: 'bg-red-800/60',
                // badgePlatformBorder: 'border-red-700/50',
                // badgePlatformText: 'text-red-100',
                alertErrorBg: 'bg-red-800/80',
                alertErrorBorder: 'border-red-600',
                alertErrorText: 'text-red-100',
                alertWarningBg: 'bg-yellow-800/80',
                alertWarningBorder: 'border-yellow-600',
                alertWarningText: 'text-yellow-100',
                footerText: 'text-gray-500',
                footerBorder: 'border-gray-700/30',
                footerLink: 'text-red-400 hover:text-red-300',
                scrollbarThumb: 'bg-red-600/60 hover:bg-red-500/80',
                detailsSummary: 'text-gray-400 hover:text-gray-200',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-400',
                scoreDetailValue: 'text-gray-200',
            },
            purple: {
                name: 'Purple',
                bodyBg: 'bg-gradient-to-br from-purple-900/10 via-black to-gray-900/50',
                textColor: 'text-gray-300',
                primaryTextColor: 'text-purple-400',
                secondaryTextColor: 'text-purple-300',
                dataLabelColor: 'text-gray-400',
                dataValueColor: 'text-gray-300',
                descriptionColor: 'text-gray-400',
                iconColor: 'text-purple-300',
                sectionIconColor: 'text-purple-400',
                headerIconColor: 'text-purple-400',
                textGradient: 'bg-gradient-to-r from-purple-500 to-purple-700',
                buttonBg: 'bg-gradient-to-r from-purple-600 to-purple-800 hover:from-purple-700 hover:to-purple-900',
                buttonText: 'text-white',
                buttonShadow: 'shadow-md hover:shadow-lg shadow-purple-500/30 hover:shadow-purple-500/40',
                cardBg: 'bg-gray-800/70',
                cardBorder: 'border-purple-600/40',
                cardHoverShadow: 'shadow-lg shadow-purple-500/10',
                sectionTitleColor: 'text-purple-400',
                sectionBorder: 'border-purple-600/50',
                badgeBg: 'bg-purple-700/70',
                badgeBorder: 'border-purple-600/50',
                badgeText: 'text-purple-200',
                badgePlatformBg: 'bg-indigo-800/70', // Example: Use Indigo for platforms in Purple theme
                // badgePlatformBg: 'bg-purple-800/60',
                // badgePlatformBorder: 'border-purple-700/50',
                // badgePlatformText: 'text-purple-100',
                alertErrorBg: 'bg-red-800/80', // Keep errors red? Or theme them?
                alertErrorBorder: 'border-red-600',
                alertErrorText: 'text-red-100',
                alertWarningBg: 'bg-yellow-800/80',
                alertWarningBorder: 'border-yellow-600',
                alertWarningText: 'text-yellow-100',
                footerText: 'text-gray-500',
                footerBorder: 'border-gray-700/30',
                footerLink: 'text-purple-400 hover:text-purple-300',
                scrollbarThumb: 'bg-purple-600/60 hover:bg-purple-500/80',
                detailsSummary: 'text-gray-400 hover:text-gray-200',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-400',
                scoreDetailValue: 'text-gray-200',
            },
            blue: {
                name: 'Blue',
                bodyBg: 'bg-gradient-to-br from-blue-900/10 via-black to-gray-900/50',
                textColor: 'text-gray-300',
                primaryTextColor: 'text-blue-400',
                secondaryTextColor: 'text-blue-300',
                dataLabelColor: 'text-gray-400',
                dataValueColor: 'text-gray-300',
                descriptionColor: 'text-gray-400',
                iconColor: 'text-blue-300',
                sectionIconColor: 'text-blue-400',
                headerIconColor: 'text-blue-400',
                textGradient: 'bg-gradient-to-r from-blue-500 to-blue-700',
                buttonBg: 'bg-gradient-to-r from-blue-600 to-blue-800 hover:from-blue-700 hover:to-blue-900',
                buttonText: 'text-white',
                buttonShadow: 'shadow-md hover:shadow-lg shadow-blue-500/30 hover:shadow-blue-500/40',
                cardBg: 'bg-gray-800/70',
                cardBorder: 'border-blue-600/40',
                cardHoverShadow: 'shadow-lg shadow-blue-500/10',
                sectionTitleColor: 'text-blue-400',
                sectionBorder: 'border-blue-600/50',
                badgeBg: 'bg-blue-700/70',
                badgeBorder: 'border-blue-600/50',
                badgeText: 'text-blue-200',
                badgePlatformBg: 'bg-cyan-800/70', // Example: Cyan platforms for Blue theme
                // badgePlatformBg: 'bg-blue-800/60',
                // badgePlatformBorder: 'border-blue-700/50',
                // badgePlatformText: 'text-blue-100',
                alertErrorBg: 'bg-red-800/80',
                alertErrorBorder: 'border-red-600',
                alertErrorText: 'text-red-100',
                alertWarningBg: 'bg-yellow-800/80',
                alertWarningBorder: 'border-yellow-600',
                alertWarningText: 'text-yellow-100',
                footerText: 'text-gray-500',
                footerBorder: 'border-gray-700/30',
                footerLink: 'text-blue-400 hover:text-blue-300',
                scrollbarThumb: 'bg-blue-600/60 hover:bg-blue-500/80',
                detailsSummary: 'text-gray-400 hover:text-gray-200',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-400',
                scoreDetailValue: 'text-gray-200',
            },
             yellow: {
                name: 'Yellow',
                bodyBg: 'bg-gradient-to-br from-yellow-900/10 via-black to-gray-900/50',
                textColor: 'text-gray-200', // Slightly lighter text for contrast
                primaryTextColor: 'text-yellow-400',
                secondaryTextColor: 'text-yellow-300',
                dataLabelColor: 'text-gray-400',
                dataValueColor: 'text-gray-200',
                descriptionColor: 'text-gray-300',
                iconColor: 'text-yellow-400', // Make icons stand out
                sectionIconColor: 'text-yellow-400',
                headerIconColor: 'text-yellow-400',
                textGradient: 'bg-gradient-to-r from-yellow-400 to-yellow-600',
                buttonBg: 'bg-gradient-to-r from-yellow-500 to-yellow-700 hover:from-yellow-600 hover:to-yellow-800',
                buttonText: 'text-gray-900', // Darker text on yellow button
                buttonShadow: 'shadow-md hover:shadow-lg shadow-yellow-500/30 hover:shadow-yellow-500/40',
                cardBg: 'bg-gray-800/75',
                cardBorder: 'border-yellow-600/40',
                cardHoverShadow: 'shadow-lg shadow-yellow-500/15',
                sectionTitleColor: 'text-yellow-400',
                sectionBorder: 'border-yellow-600/50',
                badgeBg: 'bg-yellow-700/70',
                badgeBorder: 'border-yellow-600/50',
                badgeText: 'text-yellow-100',
                badgePlatformBg: 'bg-amber-800/70',
                alertErrorBg: 'bg-red-800/80',
                alertErrorBorder: 'border-red-600',
                alertErrorText: 'text-red-100',
                alertWarningBg: 'bg-orange-800/80', // Use orange for warning in yellow theme
                alertWarningBorder: 'border-orange-600',
                alertWarningText: 'text-orange-100',
                footerText: 'text-gray-500',
                footerBorder: 'border-gray-700/30',
                footerLink: 'text-yellow-400 hover:text-yellow-300',
                scrollbarThumb: 'bg-yellow-600/60 hover:bg-yellow-500/80',
                detailsSummary: 'text-gray-400 hover:text-gray-200',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-400',
                scoreDetailValue: 'text-gray-200',
            },
             white: { // Light theme - requires more overrides
                name: 'White',
                bodyBg: 'bg-gradient-to-br from-gray-100 via-gray-50 to-white',
                textColor: 'text-gray-800', // Dark text
                primaryTextColor: 'text-gray-700', // Less saturated primary
                secondaryTextColor: 'text-gray-600',
                dataLabelColor: 'text-gray-500',
                dataValueColor: 'text-gray-800',
                descriptionColor: 'text-gray-600',
                iconColor: 'text-gray-600',
                sectionIconColor: 'text-gray-700',
                headerIconColor: 'text-gray-700',
                textGradient: 'bg-gradient-to-r from-gray-600 to-gray-800', // Dark gradient
                buttonBg: 'bg-gradient-to-r from-gray-700 to-gray-900 hover:from-gray-800 hover:to-black',
                buttonText: 'text-white', // White text on dark button
                buttonShadow: 'shadow-md hover:shadow-lg shadow-gray-500/30 hover:shadow-gray-500/40',
                cardBg: 'bg-white/80', // White card, slightly transparent
                cardBorder: 'border-gray-300/60',
                cardHoverShadow: 'shadow-lg shadow-gray-400/20',
                sectionTitleColor: 'text-gray-700',
                sectionBorder: 'border-gray-400/50',
                badgeBg: 'bg-gray-600/70',
                badgeBorder: 'border-gray-500/50',
                badgeText: 'text-gray-100',
                badgePlatformBg: 'bg-slate-700/70', // Slate for platforms
                alertErrorBg: 'bg-red-100/80', // Light red bg
                alertErrorBorder: 'border-red-400',
                alertErrorText: 'text-red-800', // Dark red text
                alertWarningBg: 'bg-yellow-100/80',
                alertWarningBorder: 'border-yellow-400',
                alertWarningText: 'text-yellow-800',
                footerText: 'text-gray-600',
                footerBorder: 'border-gray-300/50',
                footerLink: 'text-gray-700 hover:text-black',
                scrollbarThumb: 'bg-gray-500/60 hover:bg-gray-600/80',
                detailsSummary: 'text-gray-600 hover:text-gray-900',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-600',
                scoreDetailValue: 'text-gray-800',
            },
            orange: {
                name: 'Orange',
                bodyBg: 'bg-gradient-to-br from-orange-900/10 via-black to-gray-900/50',
                textColor: 'text-gray-300',
                primaryTextColor: 'text-orange-400',
                secondaryTextColor: 'text-orange-300',
                dataLabelColor: 'text-gray-400',
                dataValueColor: 'text-gray-300',
                descriptionColor: 'text-gray-400',
                iconColor: 'text-orange-300',
                sectionIconColor: 'text-orange-400',
                headerIconColor: 'text-orange-400',
                textGradient: 'bg-gradient-to-r from-orange-500 to-orange-700',
                buttonBg: 'bg-gradient-to-r from-orange-600 to-orange-800 hover:from-orange-700 hover:to-orange-900',
                buttonText: 'text-white',
                buttonShadow: 'shadow-md hover:shadow-lg shadow-orange-500/30 hover:shadow-orange-500/40',
                cardBg: 'bg-gray-800/70',
                cardBorder: 'border-orange-600/40',
                cardHoverShadow: 'shadow-lg shadow-orange-500/10',
                sectionTitleColor: 'text-orange-400',
                sectionBorder: 'border-orange-600/50',
                badgeBg: 'bg-orange-700/70',
                badgeBorder: 'border-orange-600/50',
                badgeText: 'text-orange-200',
                badgePlatformBg: 'bg-amber-800/70', // Amber for platforms
                alertErrorBg: 'bg-red-800/80',
                alertErrorBorder: 'border-red-600',
                alertErrorText: 'text-red-100',
                alertWarningBg: 'bg-yellow-800/80',
                alertWarningBorder: 'border-yellow-600',
                alertWarningText: 'text-yellow-100',
                footerText: 'text-gray-500',
                footerBorder: 'border-gray-700/30',
                footerLink: 'text-orange-400 hover:text-orange-300',
                scrollbarThumb: 'bg-orange-600/60 hover:bg-orange-500/80',
                detailsSummary: 'text-gray-400 hover:text-gray-200',
                detailsContent: 'text-gray-500',
                scoreDetailKey: 'text-gray-400',
                scoreDetailValue: 'text-gray-200',
            },
            // Add more themes (e.g., green, cyan) following the pattern
        };

        // --- Helper Functions ---
        function applyClasses(elements, classesToRemove, classesToAdd) {
            const removeList = classesToRemove.split(' ').filter(c => c);
            const addList = classesToAdd.split(' ').filter(c => c);
            elements.forEach(el => {
                if (removeList.length > 0) el.classList.remove(...removeList);
                if (addList.length > 0) el.classList.add(...addList);
            });
        }

        function applyTheme(themeName) {
            const theme = themes[themeName] || themes.red; // Fallback to red
            console.log(`Applying theme: ${theme.name}`);

            // --- Get All Theme Class Definitions for Removal ---
            let allClassesToRemove = new Set();
            Object.values(themes).forEach(t => {
                Object.values(t).forEach(classString => {
                     if (typeof classString === 'string') {
                         classString.split(' ').filter(c => c).forEach(cls => allClassesToRemove.add(cls));
                     }
                });
            });
            const classesToRemoveStr = Array.from(allClassesToRemove).join(' ');
            // console.debug("Classes to potentially remove:", classesToRemoveStr); // Very verbose

            // --- Apply Theme Classes ---
            // Body background and base text color
            applyClasses([document.body], classesToRemoveStr, `${theme.bodyBg} ${theme.textColor}`);

            // Header & Footer
            applyClasses(document.querySelectorAll('#mainHeader'), classesToRemoveStr, theme.cardBg); // Use cardBg for header consistency
            applyClasses(document.querySelectorAll('.header-icon'), classesToRemoveStr, theme.headerIconColor);
            applyClasses(document.querySelectorAll('h1.text-gradient-dynamic'), classesToRemoveStr, theme.textGradient);
            applyClasses([document.getElementById('pageFooter')], classesToRemoveStr, `${theme.footerText} ${theme.footerBorder}`);
            applyClasses([document.getElementById('footerLink')], classesToRemoveStr, `${theme.footerLink}`);

            // Buttons
            const retestBtn = document.getElementById('retestBtn');
            if (retestBtn) {
                 applyClasses([retestBtn], classesToRemoveStr, `${theme.buttonBg} ${theme.buttonText} ${theme.buttonShadow}`);
            }

            // Section Titles and Borders
            applyClasses(document.querySelectorAll('.section-title'), classesToRemoveStr, `${theme.sectionTitleColor} ${theme.sectionBorder}`);
             applyClasses(document.querySelectorAll('.section-icon'), classesToRemoveStr, theme.sectionIconColor);

            // Profile Cards
            applyClasses(document.querySelectorAll('.profile-card'), classesToRemoveStr, `${theme.cardBg} ${theme.cardBorder}`);
            // Note: Hover shadow might need specific CSS rules per theme if complex

            // Data Items (Labels, Values, Icons)
            applyClasses(document.querySelectorAll('.data-label'), classesToRemoveStr, theme.dataLabelColor);
            applyClasses(document.querySelectorAll('.data-value'), classesToRemoveStr, theme.dataValueColor);
            applyClasses(document.querySelectorAll('.icon-style'), classesToRemoveStr, theme.iconColor);
            applyClasses(document.querySelectorAll('#originDesc'), classesToRemoveStr, theme.descriptionColor); // Origin Description specifically

            // Card Sections
            applyClasses(document.querySelectorAll('.card-section'), classesToRemoveStr, theme.sectionBorder); // Just border color
            applyClasses(document.querySelectorAll('.card-section-title'), classesToRemoveStr, theme.sectionTitleColor);

            // Badges (General)
            applyClasses(document.querySelectorAll('.badge-item:not(.badge-platform)'), classesToRemoveStr, `${theme.badgeBg} ${theme.badgeBorder} ${theme.badgeText}`);
            // Badges (Platform Specific - Optional)
             applyClasses(document.querySelectorAll('.badge-item.badge-platform'), classesToRemoveStr, `${theme.badgePlatformBg || theme.badgeBg} ${theme.badgePlatformBorder || theme.badgeBorder} ${theme.badgePlatformText || theme.badgeText}`); // Fallback to general if platform specific not defined


            // Alert Boxes
            applyClasses(document.querySelectorAll('#errorMessage.alert-box'), classesToRemoveStr, `${theme.alertErrorBg} ${theme.alertErrorBorder} ${theme.alertErrorText}`);
            applyClasses(document.querySelectorAll('#loadingMessage.alert-box'), classesToRemoveStr, `${theme.alertWarningBg} ${theme.alertWarningBorder} ${theme.alertWarningText}`);
            applyClasses(document.querySelectorAll('#noSimilarProfilesMessage'), classesToRemoveStr, theme.dataLabelColor); // Use a neutral color
            applyClasses(document.querySelectorAll('#originNotFoundMessage.alert-box'), classesToRemoveStr, `${theme.alertErrorBg} ${theme.alertErrorBorder} ${theme.alertErrorText}`);


            // Details (Score Breakdown)
             applyClasses(document.querySelectorAll('details summary'), classesToRemoveStr, theme.detailsSummary);
             applyClasses(document.querySelectorAll('details ul'), classesToRemoveStr, theme.detailsContent);
             applyClasses(document.querySelectorAll('.score-detail-key'), classesToRemoveStr, theme.scoreDetailKey);
             applyClasses(document.querySelectorAll('.score-detail-value'), classesToRemoveStr, theme.scoreDetailValue);


            // Scrollbar (Requires dynamic style injection or more complex CSS)
            const styleSheet = document.styleSheets[0]; // Assuming the first stylesheet has the scrollbar rules
            try {
                // Find and modify scrollbar thumb rule (this is fragile)
                let ruleIndex = -1;
                for(let i = 0; i < styleSheet.cssRules.length; i++) {
                    if(styleSheet.cssRules[i].selectorText === '::-webkit-scrollbar-thumb') {
                         ruleIndex = i;
                         break;
                    }
                }
                if (ruleIndex !== -1) {
                    // Tailwind doesn't directly translate theme.scrollbarThumb easily here.
                    // We need to parse the color value. This is simplified.
                    // Example: theme.scrollbarThumb = 'bg-red-600/60 hover:bg-red-500/80'
                    // We need the base color part 'bg-red-600/60' -> parse to rgba or hex
                    // This part is complex to do reliably without a Tailwind color mapping.
                    // For now, let's skip dynamic scrollbar thumb color via JS.
                    // console.warn("Dynamic scrollbar styling via JS is limited.");
                     // Alternative: Add a class to body and use CSS variables for scrollbar
                }
            } catch (e) {
                 console.error("Could not access or modify CSS rules for scrollbar:", e);
            }


            // Update theme selector dropdown to reflect the current theme
            const selector = document.getElementById('themeSelector');
            if (selector) selector.value = themeName;
        }

        function saveTheme(themeName) {
            localStorage.setItem('selectedTheme', themeName);
            console.log(`Theme saved: ${themeName}`);
        }

        function loadTheme() {
            const savedTheme = localStorage.getItem('selectedTheme');
            const themeToApply = savedTheme && themes[savedTheme] ? savedTheme : '{{ DEFAULT_THEME }}'; // Use Flask default
            applyTheme(themeToApply);
            return themeToApply;
        }

        // --- Data Rendering Functions ---
        function clearChildren(element) {
             while (element.firstChild) {
                element.removeChild(element.firstChild);
            }
        }

        function createBadge(text, type = 'default') {
            const span = document.createElement('span');
            span.className = `badge-item ${type === 'platform' ? 'badge-platform' : ''}`; // Base classes, theme adds color
            span.textContent = text;
            return span;
        }

        function renderBadges(containerId, itemsString, type = 'default') {
            const container = document.getElementById(containerId);
            if (!container) return;
            clearChildren(container);
            const items = (itemsString || '').split(',').map(s => s.trim()).filter(s => s);
            if (items.length === 0) {
                container.innerHTML = '<span class="text-xs italic">Nenhum</span>'; // Style this span with theme.dataLabelColor later
            } else {
                items.forEach(item => container.appendChild(createBadge(item, type)));
            }
        }

        function renderOriginProfile(profile) {
            const section = document.getElementById('originProfileSection');
            const card = document.getElementById('originProfileCard');
            const notFoundMsg = document.getElementById('originNotFoundMessage');

            if (!profile || !profile.id) {
                section.classList.add('hidden');
                notFoundMsg.classList.remove('hidden');
                return;
            }

            notFoundMsg.classList.add('hidden');
            section.classList.remove('hidden');

            document.getElementById('originName').innerHTML = `${profile.nome || 'N/A'} <span class="text-sm font-light">(ID: ${profile.id})</span>`; // Color from theme needed for span
            document.getElementById('originIdade').textContent = profile.idade || 'N/A';
            document.getElementById('originLocal').textContent = `${profile.cidade || 'N/A'}, ${profile.estado || 'N/A'}`;
            document.getElementById('originSexo').textContent = profile.sexo || 'N/A';
            document.getElementById('originDisp').textContent = profile.disponibilidade || 'N/A';
            document.getElementById('originInteracao').textContent = profile.interacao_desejada || 'N/A';
            const contatoEl = document.getElementById('originContato');
            if (profile.compartilhar_contato) {
                 contatoEl.innerHTML = `<span class="text-green-400">Sim <i class="fas fa-check-circle"></i></span>`; // Keep green/red for status?
            } else {
                 contatoEl.innerHTML = `<span class="text-red-400">Não <i class="fas fa-times-circle"></i></span>`;
            }

            renderBadges('originPlatforms', profile.plataformas_possuidas, 'platform');
            renderBadges('originGames', profile.jogos_favoritos, 'game');
            renderBadges('originStyles', profile.estilos_preferidos, 'style');
            renderBadges('originMusic', profile.interesses_musicais, 'music');
            document.getElementById('originDesc').textContent = profile.descricao || 'Sem descrição.';

             // Re-apply theme in case elements were just created/modified
             const currentTheme = localStorage.getItem('selectedTheme') || '{{ DEFAULT_THEME }}';
             applyTheme(currentTheme); // Ensures colors are correct after injection
        }

        function renderSimilarProfiles(profiles) {
            const section = document.getElementById('similarProfilesSection');
            const grid = document.getElementById('similarProfilesGrid');
            const noResultsMsg = document.getElementById('noSimilarProfilesMessage');
            const title = document.getElementById('similarProfilesTitle');

            clearChildren(grid); // Clear previous results

            if (!profiles || profiles.length === 0) {
                title.classList.add('hidden'); // Hide title if no results
                grid.classList.add('hidden');
                noResultsMsg.classList.remove('hidden');
                section.classList.remove('hidden'); // Show the section to display the message
                return;
            }

            title.classList.remove('hidden');
            grid.classList.remove('hidden');
            noResultsMsg.classList.add('hidden');
            section.classList.remove('hidden'); // Show the section

            // Update title count
            title.innerHTML = `<i class="fas fa-users-viewfinder mr-2 section-icon"></i>Top ${profiles.length} Perfis Similares (Prioridade: Plataforma/Horário)`;


            profiles.forEach((perfil, index) => {
                const cardDiv = document.createElement('div');
                cardDiv.className = `profile-card rounded-lg p-4 flex flex-col h-full fade-in similar-card-${index}`; // Base classes

                const scoreDetailsHtml = Object.entries(perfil.score_details || {})
                    .map(([key, value]) => `
                        <li>
                            <span class="inline-block w-24 score-detail-key">${key.charAt(0).toUpperCase() + key.slice(1)}:</span>
                            <span class="font-semibold score-detail-value">${parseFloat(value).toFixed(2)}</span>
                             <span class="text-[0.65rem] score-detail-key">(Peso: {{ "%.2f"|format(WEIGHTS.get(key, 0.0)) }})</span>
                             ${ (key === 'plataformas' && value < MIN_REQUIRED_PLATFORM_SCORE) ? '<i class="fas fa-exclamation-triangle text-yellow-500 ml-1" title="Abaixo do threshold mínimo"></i>' : ''}
                             ${ (key === 'disponibilidade' && value < MIN_REQUIRED_AVAILABILITY_SCORE) ? '<i class="fas fa-exclamation-triangle text-yellow-500 ml-1" title="Abaixo do threshold mínimo"></i>' : ''}
                         </li>`)
                    .join('');

                 const contatoHtml = perfil.compartilhar_contato
                    ? `<span class="text-green-400">Sim</span>`
                    : `<span class="text-red-400">Não</span>`;

                cardDiv.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <h4 class="text-lg font-bold font-orbitron text-gradient-dynamic flex-grow pr-2">${perfil.nome} <span class="text-xs font-light">(ID: ${perfil.id})</span></h4>
                        <div class="text-right flex-shrink-0">
                            <span class="block font-semibold text-sm data-label">Score Final</span>
                            <span class="font-bold text-gradient-score text-xl">${parseFloat(perfil.score_compatibilidade).toFixed(3)}</span>
                        </div>
                    </div>
                    <div class="text-xs md:text-sm space-y-1 mb-3 data-value">
                        <p><i class="fas fa-birthday-cake icon-style"></i> ${perfil.idade || 'N/A'} anos</p>
                        <p><i class="fas fa-map-marker-alt icon-style"></i> ${perfil.cidade || 'N/A'}, ${perfil.estado || 'N/A'}</p>
                        <p><i class="fas fa-clock icon-style"></i> <span class="font-semibold text-amber-300">${perfil.disponibilidade || 'N/A'}</span> (Score: ${parseFloat(perfil.score_details?.disponibilidade || 0).toFixed(2)})</p>
                        <p><i class="fas fa-comments icon-style"></i> ${perfil.interacao_desejada || 'N/A'} (Score: ${parseFloat(perfil.score_details?.interacao || 0).toFixed(2)})</p>
                        <p><i class="fas fa-share-alt icon-style"></i> Contato: ${contatoHtml}</p>
                    </div>

                    <div class="card-section text-xs">
                        <h5 class="card-section-title"><i class="fas fa-headset mr-1"></i>Plataformas (Score: ${parseFloat(perfil.score_details?.plataformas || 0).toFixed(2)})</h5>
                        <div class="badge-list" id="sim-platforms-${perfil.id}"></div>
                    </div>
                     <div class="card-section text-xs">
                        <h5 class="card-section-title"><i class="fas fa-gamepad mr-1"></i>Jogos (Score: ${parseFloat(perfil.score_details?.jogos || 0).toFixed(2)})</h5>
                        <div class="badge-list" id="sim-games-${perfil.id}"></div>
                    </div>
                     <div class="card-section text-xs">
                        <h5 class="card-section-title"><i class="fas fa-tags mr-1"></i>Estilos (Score: ${parseFloat(perfil.score_details?.estilos || 0).toFixed(2)})</h5>
                         <div class="badge-list" id="sim-styles-${perfil.id}"></div>
                    </div>

                    <details class="text-xs mt-3 opacity-75 hover:opacity-100 transition-opacity cursor-pointer">
                        <summary class="font-semibold outline-none">Detalhes do Score Ponderado</summary>
                        <ul class="list-none ml-2 mt-1 space-y-0.5 pt-1">
                            ${scoreDetailsHtml}
                             <li class="border-t mt-1 pt-1"> <!-- Dynamic border color needed -->
                                <span class="inline-block w-24 font-bold score-detail-key">Total Final:</span>
                                <span class="font-bold text-green-400">${parseFloat(perfil.score_compatibilidade).toFixed(3)}</span>
                             </li>
                        </ul>
                    </details>
                `;
                grid.appendChild(cardDiv);

                // Render badges for this similar card
                renderBadges(`sim-platforms-${perfil.id}`, perfil.plataformas_possuidas, 'platform');
                renderBadges(`sim-games-${perfil.id}`, perfil.jogos_favoritos, 'game');
                renderBadges(`sim-styles-${perfil.id}`, perfil.estilos_preferidos, 'style');
            });

             // Re-apply theme to ensure all dynamically created elements get styles
             const currentTheme = localStorage.getItem('selectedTheme') || '{{ DEFAULT_THEME }}';
             applyTheme(currentTheme);
        }

        // --- Initialization and Event Listeners ---
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM Loaded. Initializing Dashboard.');

            // Elements
            const themeSelector = document.getElementById('themeSelector');
            const retestBtn = document.getElementById('retestBtn');
            const loadingMessage = document.getElementById('loadingMessage');
            const errorMessage = document.getElementById('errorMessage');
            const errorMessageText = document.getElementById('errorMessageText');
            const mainContentArea = document.getElementById('mainContentArea');
            const currentYearEl = document.getElementById('currentYear');

            // Set current year in footer
            if (currentYearEl) currentYearEl.textContent = new Date().getFullYear();

            // Load initial theme
            const initialTheme = loadTheme();

            // --- Handle Initial Data State (Passed from Flask) ---
            const initialError = {{ error_message | tojson }};
            const initialDataLoaded = {{ data_loaded_completely | tojson }};
            const initialOriginProfile = {{ perfil_origem | tojson }};
            const initialSimilarProfiles = {{ perfis_similares | tojson }};

            if (initialError) {
                errorMessageText.textContent = initialError;
                errorMessage.classList.remove('hidden');
                loadingMessage.classList.add('hidden');
                mainContentArea.classList.add('hidden');
            } else if (!initialDataLoaded) {
                loadingMessage.classList.remove('hidden');
                errorMessage.classList.add('hidden');
                mainContentArea.classList.add('hidden');
                // Maybe add a timeout or check later? For now, just show loading.
            } else {
                // Data is loaded, render content
                loadingMessage.classList.add('hidden');
                errorMessage.classList.add('hidden');
                mainContentArea.classList.remove('hidden');

                // Render initial profiles passed from server
                renderOriginProfile(initialOriginProfile);
                renderSimilarProfiles(initialSimilarProfiles);
            }

            // Theme Selector Logic
            if (themeSelector) {
                themeSelector.value = initialTheme; // Set dropdown to loaded theme
                themeSelector.addEventListener('change', (event) => {
                    const selectedTheme = event.target.value;
                    applyTheme(selectedTheme);
                    saveTheme(selectedTheme);
                });
            }

            // Retest Button Logic
            if (retestBtn) {
                retestBtn.addEventListener('click', () => {
                    const icon = retestBtn.querySelector('i');
                    const btnText = retestBtn.querySelector('.btn-retest-text');
                    icon.classList.add('fa-spin');
                    retestBtn.disabled = true;
                    if(btnText) btnText.textContent = 'Gerando...';

                    // Redirect to the new match route
                    window.location.href = '/new_match';
                    // No need to handle response here, the page will reload
                });
            }
        });

        // --- Global Constants (from Flask/Jinja) ---
        // These are needed for the details rendering logic if done client-side
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

    # Define dados a serem passados para o template
    template_data = {
        "data_loaded_completely": app_data["data_loaded"],
        "error_message": app_data["loading_error"],
        "perfil_origem": None,
        "perfis_similares": [],
        "available_themes": AVAILABLE_THEMES,
        "DEFAULT_THEME": DEFAULT_THEME,
        "current_year": datetime.datetime.now().year,
        "id_origem": None, # Será definido se buscarmos um perfil
        "WEIGHTS": WEIGHTS, # Passa pesos para JS
        "MIN_REQUIRED_PLATFORM_SCORE": MIN_REQUIRED_PLATFORM_SCORE, # Passa threshold para JS
        "MIN_REQUIRED_AVAILABILITY_SCORE": MIN_REQUIRED_AVAILABILITY_SCORE, # Passa threshold para JS
    }

    # Se dados ainda não carregaram ou houve erro, renderiza imediatamente
    if not app_data["data_loaded"] or app_data["loading_error"]:
        log_msg = "Data not loaded yet" if not app_data["loading_error"] else f"Data loading error: {app_data['loading_error']}"
        logging.info(f"{log_msg}, rendering template with status.")
        # Registra a função utilitária no ambiente Jinja (necessário para detalhes do score no HTML inicial)
        app.jinja_env.globals.update(safe_split_and_strip=safe_split_and_strip)
        return render_template_string(index_html, **template_data)

    # Dados carregados, prosseguir com busca
    if not app_data["profile_ids_map"]:
        logging.error("Data loaded, but profile_ids_map is empty.")
        template_data["error_message"] = "Erro: Nenhum perfil encontrado nos dados carregados."
        app.jinja_env.globals.update(safe_split_and_strip=safe_split_and_strip)
        return render_template_string(index_html, **template_data)

    # Seleciona um ID de origem aleatório
    id_origem = None
    try:
        valid_ids = app_data.get("profile_ids_map", [])
        if not valid_ids:
             raise IndexError("Profile ID map is empty.")
        id_origem = random.choice(valid_ids)
        template_data["id_origem"] = id_origem # Guarda o ID escolhido
        logging.info(f"Selected random origin profile ID: {id_origem}")
    except IndexError:
         logging.error("Failed to select random ID - profile_ids_map might be empty.")
         template_data["error_message"] = "Erro interno: Não foi possível selecionar um perfil de origem aleatório."
         app.jinja_env.globals.update(safe_split_and_strip=safe_split_and_strip)
         return render_template_string(index_html, **template_data)
    except Exception as e:
         logging.error(f"Unexpected error selecting random ID: {e}", exc_info=True)
         template_data["error_message"] = f"Erro inesperado ao selecionar perfil: {e}"
         app.jinja_env.globals.update(safe_split_and_strip=safe_split_and_strip)
         return render_template_string(index_html, **template_data)


    # Busca e rankeia os vizinhos
    perfil_origem, perfis_similares = buscar_e_rankear_vizinhos(id_origem, NUM_NEIGHBORS_TARGET)

    # Atualiza dados para o template
    template_data["perfil_origem"] = perfil_origem
    template_data["perfis_similares"] = perfis_similares
    if not perfil_origem:
         # Define a mensagem de erro específica se o perfil de origem não foi carregado
         # mas a busca foi tentada (o id_origem foi selecionado)
         template_data["error_message"] = template_data.get("error_message") or f"Perfil de origem (ID: {id_origem}) não encontrado no banco de dados após seleção."
         logging.warning(f"Origin profile ID {id_origem} selected but not found/loaded by buscar_e_rankear_vizinhos.")


    # Registra a função utilitária no ambiente Jinja ANTES de renderizar
    app.jinja_env.globals.update(safe_split_and_strip=safe_split_and_strip)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Finished processing index request for ID {id_origem}. Duration: {duration.total_seconds():.3f}s. Found {len(perfis_similares)} matches.")

    return render_template_string(index_html, **template_data)


# Rota dedicada para o botão, para evitar F5 re-submetendo a mesma lógica
@app.route('/new_match')
def new_match():
    """Redireciona para a raiz para obter um novo match."""
    logging.info("Redirecting to generate new match via /new_match route.")
    # Adiciona um parâmetro anti-cache para garantir que o navegador realmente recarregue
    # Usa um timestamp ou um número aleatório
    cache_buster = f"?t={datetime.datetime.now().timestamp()}"
    return redirect(url_for('index') + cache_buster)


# --- Função para iniciar o carregamento em background ---
# (Mantida da versão anterior)
def start_background_load():
    # Verifica se já está carregado OU se já ocorreu um erro para não tentar de novo
    if not app_data["data_loaded"] and not app_data["loading_error"]:
        logging.info("Starting background data loading thread...")
        load_thread = threading.Thread(target=load_data_and_build_index, name="DataLoaderThread", daemon=True)
        load_thread.start()
    elif app_data["data_loaded"]:
        logging.info("Data already loaded. Skipping background load initiation.")
    else: # loading_error is set
         logging.warning(f"Previous loading error detected: '{app_data['loading_error']}'. Skipping background load initiation.")


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
        print(f"\n{msg}")
        if not db_profiles_exists: print(f"   - Missing Profile DB: '{DATABASE_PROFILES}'")
        if not db_embeddings_exists: print(f"   - Missing Embedding DB: '{DATABASE_EMBEDDINGS}'")
        print("   Please ensure 'profile_generator_v3.py' has been run successfully")
        print(f"   and the databases are in the expected directory: '{DB_DIR}'.\n")
        exit(1) # Encerra
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
            serve(app, host='127.0.0.1', port=FLASK_PORT, threads=8)
        except ImportError:
            print("   Waitress not found, using Flask development server (WARNING: Not recommended for production).")
            print("   Install Waitress for better performance: pip install waitress")
            app.run(host='127.0.0.1', port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False) # use_reloader=False é importante com threads background