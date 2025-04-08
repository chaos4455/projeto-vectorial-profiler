# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import faiss
import plotly.graph_objects as go
import os
import datetime
import random
import logging
from typing import Tuple, List, Dict, Set, Optional, Any
from functools import lru_cache
import hashlib
import argparse
import textwrap

# --- Dependency Check ---
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    logging.warning("Scikit-learn not found. PCA functionality will be unavailable.")

# --- Configuration ---
# Adjust these paths and parameters according to your setup (e.g., v3 vs v6 data)
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
OUTPUT_DIR = "data-dash-viewer"  # Directory to save the HTML visualizations
LOG_DIR = "logs_v3"              # Directory to save logs for this script
EXPECTED_EMBEDDING_DIM = 64      # Expected dimension of embeddings (e.g., 64 for v3, 192 for v6)

# --- Plotting & Search Parameters ---
NUM_NEIGHBORS_TARGET: int = 10     # How many neighbors to *display* in the plot
# Fetch N * Factor candidates from FAISS initially. Increase if custom scoring filters too many.
FAISS_SEARCH_K_FACTOR: int = 30

# --- Custom Similarity Score Logic (Mirroring apredanesse.py) ---
# Weights for combining different similarity aspects
WEIGHTS = {
    "plataformas": 0.35,      # 35% - critério mais importante
    "disponibilidade": 0.35,  # 35% - segundo mais importante
    "jogos": 0.25,           # 25% - terceiro mais importante
    "estilos": 0.05,         # 5% - menos importante
    "interacao": 0.00,       # 0% - será considerado apenas como informativo
}
# Ensure weights sum to 1 (approximately)
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Similarity weights must sum to 1.0"

# Mandatory minimum scores for certain aspects to even consider a match
# Thresholds mais flexíveis com fallback
MIN_REQUIRED_PLATFORM_SCORE: float = 0.70    # Reduzido para 70% de match em plataformas
MIN_REQUIRED_AVAILABILITY_SCORE: float = 0.70 # Reduzido para 70% de match em disponibilidade
# Minimum final weighted score required to be included in the results
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.50     # Reduzido para 50% de score geral mínimo

# Fallback thresholds (usados se não encontrar vizinhos suficientes)
FALLBACK_PLATFORM_SCORE: float = 0.50    # Fallback para 50% de match em plataformas
FALLBACK_AVAILABILITY_SCORE: float = 0.50 # Fallback para 50% de match em disponibilidade
FALLBACK_CUSTOM_SCORE: float = 0.30      # Fallback para 30% de score geral mínimo

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO
# --------------------------------------------------------------------

# --- Initialization ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Logging Setup ---
timestamp_log = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f"3d_visualization_generator_{timestamp_log}.log")

# Configure logging: File handler + Console handler
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler() # Also print logs to the console
    ]
)
logging.info("Logging initialized.")

# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    # Declare global variables at the start of the function
    global DATABASE_PROFILES, DATABASE_EMBEDDINGS, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(
        description="Generate a 3D Plotly HTML visualization for a profile and its "
                    "neighbors based on a custom similarity score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "-id", "--profile-id", type=int, default=None,
        help="Specify the ID of the origin profile. If None, a random profile ID is chosen."
    )
    parser.add_argument(
        "-n", "--num-neighbors", type=int, default=NUM_NEIGHBORS_TARGET,
        help="Target number of neighbors to find and visualize after custom scoring."
    )
    parser.add_argument(
        "--dim", type=int, default=EXPECTED_EMBEDDING_DIM,
        help="Expected dimension of the profile embeddings."
    )
    parser.add_argument(
        "--search-factor", type=int, default=FAISS_SEARCH_K_FACTOR,
        help="Multiplier for FAISS search (k = num_neighbors * search_factor + 1) "
             "to retrieve initial candidates before custom scoring."
    )
    parser.add_argument(
        '--db-dir', type=str, default=DB_DIR,
        help="Directory containing the database files."
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help="Directory where the output HTML file will be saved."
    )

    args = parser.parse_args()

    # --- Update Global Config from Args ---
    # Allow overriding DB/Output dirs via command line
    # Use the directory from args, but keep the original filename part
    DATABASE_PROFILES = os.path.join(args.db_dir, os.path.basename(DATABASE_PROFILES))
    DATABASE_EMBEDDINGS = os.path.join(args.db_dir, os.path.basename(DATABASE_EMBEDDINGS))
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output dir exists if overridden

    logging.info("Effective Configuration:")
    logging.info(f"  Profile DB: {DATABASE_PROFILES}")
    logging.info(f"  Embeddings DB: {DATABASE_EMBEDDINGS}")
    logging.info(f"  Output Dir: {OUTPUT_DIR}")
    logging.info(f"  Target Neighbors: {args.num_neighbors}")
    logging.info(f"  Embedding Dim: {args.dim}")
    logging.info(f"  FAISS Search Factor: {args.search_factor}")
    logging.info(f"  Origin Profile ID: {'Random' if args.profile_id is None else args.profile_id}")

    return args

# --- Helper Functions (Data Loading & Similarity Calculation) ---

@lru_cache(maxsize=1024) # Cache recently loaded profiles
def carregar_perfil_por_id_cached(db_path: str, profile_id: int) -> Optional[Dict[str, Any]]:
    """
    Loads profile data for a given ID from the profiles database.
    Uses caching to avoid redundant database queries.
    Converts boolean-like integers (0/1) to actual booleans.
    """
    # logging.debug(f"Attempting to load profile ID {profile_id} from cache or DB '{db_path}'")
    try:
        # Connect in read-only mode for safety and potential concurrency benefits
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15.0) as conn:
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM perfis WHERE id = ?", (profile_id,))
            perfil_row = cursor.fetchone()

            if perfil_row:
                perfil_dict = dict(perfil_row)
                # Convert specific integer columns to boolean for clarity
                for key in ['compartilhar_contato', 'usa_microfone']:
                    if key in perfil_dict and isinstance(perfil_dict[key], int):
                        perfil_dict[key] = bool(perfil_dict[key])
                # logging.debug(f"Successfully loaded profile ID {profile_id}")
                return perfil_dict
            else:
                logging.warning(f"Profile ID {profile_id} not found in database '{db_path}'.")
                return None
    except sqlite3.OperationalError as e:
         # More specific error logging for common issues
         if "unable to open database file" in str(e):
              logging.error(f"Database file not found or inaccessible: {db_path}. Error: {e}")
         elif "database is locked" in str(e):
              logging.error(f"Database file is locked: {db_path}. Close other connections. Error: {e}")
         else:
              logging.error(f"SQLite OperationalError loading profile ID {profile_id} from {db_path}: {e}", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"Unexpected error loading profile ID {profile_id} from {db_path}: {e}", exc_info=True)
        return None

def safe_split_and_strip(text: Optional[str], delimiter: str = ',', lower: bool = False) -> List[str]:
    """Safely splits a string by a delimiter, strips whitespace, and optionally lowercases."""
    if not text or not isinstance(text, str):
        return []
    items = [item.strip() for item in text.split(delimiter) if item.strip()]
    if lower:
        items = [item.lower() for item in items]
    return items

def safe_split_and_strip_set(text: Optional[str], delimiter: str = ',') -> Set[str]:
    """Uses safe_split_and_strip to return a set of lowercased items."""
    return set(safe_split_and_strip(text, delimiter, lower=True))

def safe_split_and_strip_list(text: Optional[str], delimiter: str = ',') -> List[str]:
    """Uses safe_split_and_strip to return a sorted list of items (case preserved)."""
    return sorted(safe_split_and_strip(text, delimiter, lower=False))

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculates the Jaccard similarity between two sets."""
    if not isinstance(set1, set) or not isinstance(set2, set):
        logging.warning(f"Jaccard input not sets: type(set1)={type(set1)}, type(set2)={type(set2)}")
        return 0.0
    if not set1 and not set2: # Both empty
        return 1.0 # Or 0.0 depending on definition, let's say 1.0 if goal is matching empties
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union if union > 0 else 0.0

def availability_similarity(avail1_str: Optional[str], avail2_str: Optional[str]) -> float:
    """Calculates similarity based on simplified availability strings."""
    if not avail1_str or not avail2_str: return 0.0

    # Inner function to map variations to canonical terms
    def simplify_avail(avail_text: str) -> str:
        text = avail_text.lower().strip()
        if not text: return "desconhecido"
        if "manhã" in text: return "manha"
        if "tarde" in text: return "tarde"
        if "noite" in text: return "noite"
        if "madrugada" in text: return "madrugada"
        # Order matters: check fds before specific days if both mentioned
        if "fim de semana" in text or "fds" in text: return "fds"
        if "dia de semana" in text or "durante a semana" in text: return "semana"
        if "flexível" in text or "qualquer" in text or "variado" in text: return "flexivel"
        return "outro" # Default if no common pattern matches

    a1_simple = simplify_avail(avail1_str)
    a2_simple = simplify_avail(avail2_str)

    # Define similarity scores based on canonical terms
    if a1_simple == a2_simple: return 1.0
    if a1_simple == "desconhecido" or a2_simple == "desconhecido": return 0.0 # Cannot compare
    if a1_simple == "flexivel" or a2_simple == "flexivel": return 0.7 # High overlap potential
    # Removed redundant checks like fds/fds (covered by ==)
    # Specific period vs broader period
    if (a1_simple in ["manha", "tarde", "noite"] and a2_simple == "semana") or \
       (a2_simple in ["manha", "tarde", "noite"] and a1_simple == "semana"): return 0.4
    if (a1_simple in ["manha", "tarde", "noite", "semana"] and a2_simple == "fds") or \
       (a2_simple in ["manha", "tarde", "noite", "semana"] and a1_simple == "fds"): return 0.2
    # Less common overlaps
    if a1_simple == "madrugada" or a2_simple == "madrugada":
        if a1_simple == "noite" or a2_simple == "noite": return 0.3 # Adjacent-ish
        return 0.1 # Low overlap with most other times
    # Fallback for dissimilar specific times (e.g., manha vs noite)
    return 0.0

def interaction_similarity(inter1: Optional[str], inter2: Optional[str]) -> float:
    """Calculates similarity based on desired interaction type."""
    if not inter1 or not inter2: return 0.0
    i1 = inter1.lower().strip()
    i2 = inter2.lower().strip()
    s1 = set(w.strip() for w in i1.split()) # Use set for keyword checking
    s2 = set(w.strip() for w in i2.split())

    if i1 == i2: return 1.0
    # Handle "indiferente" - high compatibility with anything
    if "indiferente" in s1 or "indiferente" in s2: return 0.6 # Increased score
    # Specific matches
    if "online" in s1 and "online" in s2: return 0.9
    if "presencial" in s1 and "presencial" in s2: return 0.8
    # Mismatches
    if ("online" in s1 and "presencial" in s2) or ("presencial" in s1 and "online" in s2): return 0.1
    # Default low score for other unspecified cases
    return 0.2

def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any], use_fallback: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the weighted custom similarity score between two profiles.
    Applies mandatory thresholds early to optimize. Returns total score and breakdown.
    Returns (0.0, calculated_scores) if any threshold is not met.
    """
    if not profile1 or not profile2:
        logging.warning("Attempted similarity calculation with missing profile(s).")
        return 0.0, {}

    scores = {} # Store individual component scores

    # Seleciona thresholds baseado no modo
    platform_threshold = FALLBACK_PLATFORM_SCORE if use_fallback else MIN_REQUIRED_PLATFORM_SCORE
    availability_threshold = FALLBACK_AVAILABILITY_SCORE if use_fallback else MIN_REQUIRED_AVAILABILITY_SCORE
    custom_threshold = FALLBACK_CUSTOM_SCORE if use_fallback else MIN_CUSTOM_SCORE_THRESHOLD

    # 1. Platforms (Mandatory Threshold)
    platforms1 = safe_split_and_strip_set(profile1.get('plataformas_possuidas'))
    platforms2 = safe_split_and_strip_set(profile2.get('plataformas_possuidas'))
    scores['plataformas'] = jaccard_similarity(platforms1, platforms2)
    if scores['plataformas'] < platform_threshold:
        return 0.0, {k: round(v, 3) for k,v in scores.items()}

    # 2. Availability (Mandatory Threshold)
    scores['disponibilidade'] = availability_similarity(
        profile1.get('disponibilidade'), profile2.get('disponibilidade')
    )
    if scores['disponibilidade'] < availability_threshold:
        return 0.0, {k: round(v, 3) for k,v in scores.items()}

    # --- Thresholds passed, calculate remaining scores ---

    # 3. Favorite Games
    games1 = safe_split_and_strip_set(profile1.get('jogos_favoritos'))
    games2 = safe_split_and_strip_set(profile2.get('jogos_favoritos'))
    scores['jogos'] = jaccard_similarity(games1, games2)

    # 4. Preferred Styles
    styles1 = safe_split_and_strip_set(profile1.get('estilos_preferidos'))
    styles2 = safe_split_and_strip_set(profile2.get('estilos_preferidos'))
    scores['estilos'] = jaccard_similarity(styles1, styles2)

    # 5. Desired Interaction
    scores['interacao'] = interaction_similarity(
        profile1.get('interacao_desejada'), profile2.get('interacao_desejada')
    )

    # --- Calculate Final Weighted Score ---
    total_score = sum(scores.get(key, 0.0) * WEIGHTS[key] for key in WEIGHTS)

    # 6. Final Score Threshold Check
    if total_score < custom_threshold:
        return 0.0, {k: round(v, 3) for k,v in scores.items()}

    return round(total_score, 4), {k: round(v, 3) for k,v in scores.items()}

# --------------------------------------------------------------------

def load_embeddings_and_map(db_path: str, expected_dim: int) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[faiss.Index]]:
    """
    Loads embeddings from the specified SQLite database.
    Creates a mapping from matrix index to profile ID.
    Builds and returns a FAISS IndexFlatL2 for efficient searching.
    """
    embeddings_list = []
    ids_list = []      # Stores profile IDs corresponding to rows in embeddings_list
    faiss_index = None
    detected_dimension = None

    logging.info(f"Connecting to embedding database: {db_path}")
    try:
        # Use read-only mode
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20.0) as conn:
            cursor = conn.cursor()
            # Ensure consistent ordering for index mapping
            cursor.execute("SELECT id, embedding FROM embeddings ORDER BY id")
            rows = cursor.fetchall()

            if not rows:
                logging.error(f"No data found in the 'embeddings' table in {db_path}.")
                return None, None, None

            logging.info(f"Processing {len(rows)} rows from embeddings table...")
            invalid_rows = 0
            dimension_mismatches = 0
            for i, (profile_id, embedding_blob) in enumerate(rows):
                if embedding_blob is None:
                    # logging.warning(f"Row {i+1}: Null embedding found for profile ID {profile_id}. Skipping.")
                    invalid_rows += 1
                    continue
                try:
                    # Assuming embeddings are stored as float32 blobs
                    emb = np.frombuffer(embedding_blob, dtype=np.float32)

                    # Check and set dimension based on the first valid embedding
                    if detected_dimension is None:
                        if len(emb) == 0:
                            raise ValueError("Embedding dimension is zero.")
                        detected_dimension = len(emb)
                        logging.info(f"Detected embedding dimension: {detected_dimension}")
                        if detected_dimension != expected_dim:
                            logging.warning(f"Detected dimension ({detected_dimension}) differs "
                                            f"from expected ({expected_dim}). Using detected dimension.")

                    # Check dimension of current embedding
                    if len(emb) != detected_dimension:
                        # logging.warning(f"Row {i+1}: Embedding for ID {profile_id} has incorrect "
                        #                 f"dimension ({len(emb)} vs {detected_dimension}). Skipping.")
                        dimension_mismatches += 1
                        continue

                    # Add valid embedding and its ID
                    embeddings_list.append(emb)
                    ids_list.append(profile_id)

                except ValueError as ve:
                    logging.warning(f"Row {i+1}: Error processing embedding for ID {profile_id}: {ve}. Skipping.")
                    invalid_rows += 1
                except Exception as row_e:
                    logging.warning(f"Row {i+1}: Unexpected error processing embedding for ID {profile_id}: {row_e}. Skipping.", exc_info=False)
                    invalid_rows += 1

            if invalid_rows > 0:
                 logging.warning(f"Skipped {invalid_rows} rows due to null or processing errors.")
            if dimension_mismatches > 0:
                 logging.warning(f"Skipped {dimension_mismatches} rows due to dimension mismatch.")

        # --- Post-processing and FAISS Indexing ---
        if not embeddings_list or not ids_list:
            logging.error("No valid embeddings could be loaded after processing all rows.")
            return None, None, None

        # Convert list of embeddings to a NumPy matrix
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

        # FAISS requires C-contiguous arrays
        if not embeddings_matrix.flags['C_CONTIGUOUS']:
            logging.debug("Embeddings matrix is not C-contiguous. Converting...")
            embeddings_matrix = np.ascontiguousarray(embeddings_matrix)

        logging.info(f"Successfully loaded {embeddings_matrix.shape[0]} valid embeddings.")
        logging.info(f"Embedding matrix created with shape: {embeddings_matrix.shape}")

        # Build FAISS index (using L2 distance)
        # Use the actual detected dimension
        actual_dimension_for_faiss = embeddings_matrix.shape[1]
        logging.info(f"Building FAISS IndexFlatL2 with dimension {actual_dimension_for_faiss}...")
        index = faiss.IndexFlatL2(actual_dimension_for_faiss)
        index.add(embeddings_matrix) # Add the matrix to the index
        logging.info(f"FAISS index built. Is trained: {index.is_trained}, Total vectors: {index.ntotal}")
        faiss_index = index

        return embeddings_matrix, ids_list, faiss_index

    except sqlite3.Error as db_err:
        logging.error(f"SQLite Error accessing {db_path}: {db_err}", exc_info=True)
        return None, None, None
    except faiss.FaissException as fe:
        logging.error(f"FAISS error during index building: {fe}", exc_info=True)
        return None, None, None
    except Exception as e:
        logging.error(f"Unexpected error loading embeddings or building index: {e}", exc_info=True)
        return None, None, None

# --- Visualization Functions ---

def reduce_dimensionality(embeddings_matrix: np.ndarray, n_components: int = 3) -> Optional[np.ndarray]:
    """
    Reduces the dimensionality of the embeddings matrix using PCA.
    Requires scikit-learn to be installed.
    """
    if not SKLEARN_AVAILABLE:
        logging.error("Cannot perform PCA: Scikit-learn library is not installed.")
        print("ERROR: Scikit-learn is required for PCA visualization. Install with: pip install scikit-learn")
        return None

    if embeddings_matrix is None or not isinstance(embeddings_matrix, np.ndarray) or embeddings_matrix.ndim != 2:
        logging.error(f"Invalid input for PCA: Expected 2D NumPy array, got {type(embeddings_matrix)}.")
        return None

    # Ensure we have enough samples and features for the requested components
    if embeddings_matrix.shape[0] < n_components:
        logging.error(f"Cannot reduce to {n_components} dimensions: "
                      f"Only {embeddings_matrix.shape[0]} data points (samples) available.")
        return None
    if embeddings_matrix.shape[1] < n_components:
        logging.warning(f"Input dimension ({embeddings_matrix.shape[1]}) is less than "
                        f"target components ({n_components}). PCA will output {embeddings_matrix.shape[1]} dimensions.")
        n_components = embeddings_matrix.shape[1] # Adjust target components to the maximum possible

    try:
        logging.info(f"Performing PCA to reduce dimensionality to {n_components}...")
        pca = PCA(n_components=n_components)
        reduced_coords = pca.fit_transform(embeddings_matrix)
        explained_variance = pca.explained_variance_ratio_.sum()
        logging.info(f"PCA completed. Shape of reduced coordinates: {reduced_coords.shape}. "
                     f"Explained variance ratio by {n_components} components: {explained_variance:.3f}")
        return reduced_coords
    except Exception as e:
        logging.error(f"Error during PCA execution: {e}", exc_info=True)
        return None

def format_hover_text(
    profile: Dict[str, Any],
    custom_score: Optional[float] = None,
    score_details: Optional[Dict[str, float]] = None,
    l2_distance: Optional[float] = None
) -> str:
    """
    Creates formatted HTML text for Plotly hover tooltips.
    Includes profile details and optionally similarity scores.
    """
    if not profile: return "Perfil não disponível"

    # Use helper to get display-friendly lists (sorted, original case)
    platforms = safe_split_and_strip_list(profile.get('plataformas_possuidas'))
    jogos = safe_split_and_strip_list(profile.get('jogos_favoritos'))
    estilos = safe_split_and_strip_list(profile.get('estilos_preferidos'))

    # --- Build Hover Text Parts ---
    profile_id = profile.get('id', 'N/A')
    profile_name = profile.get('nome', f'ID {profile_id}') # Use ID if name is missing
    parts = [f"<b>{profile_name} (ID: {profile_id})</b><br>"]

    # Add scores if available (typically for neighbors)
    if custom_score is not None:
        parts.append(f"<b>Score Custom: {custom_score:.3f}</b><br>")
    if score_details:
        # Show key component scores that lead to filtering/ranking
        plat_score = score_details.get('plataformas', -1.0)
        disp_score = score_details.get('disponibilidade', -1.0)
        game_score = score_details.get('jogos', -1.0)
        # Only show if calculated (i.e., >= 0)
        details_parts = []
        if plat_score >= 0: details_parts.append(f"Plat: {plat_score:.2f}")
        if disp_score >= 0: details_parts.append(f"Disp: {disp_score:.2f}")
        if game_score >= 0: details_parts.append(f"Jogos: {game_score:.2f}")
        if details_parts:
             parts.append(f"<i>Scores: {', '.join(details_parts)}</i><br>")

    if l2_distance is not None:
        parts.append(f"Dist. Embedding (L2): {l2_distance:.3f}<br>")

    # Add other profile details
    parts.append(f"Idade: {profile.get('idade', 'N/A')}, {profile.get('cidade', 'N/A')}<br>")
    parts.append(f"Disponível: {profile.get('disponibilidade', 'N/A')}<br>")
    parts.append(f"Objetivo: {textwrap.shorten(profile.get('objetivo_principal', 'N/A'), width=40, placeholder='...')}<br>")
    parts.append(f"Nível Comp.: {profile.get('nivel_competitivo', 'N/A')}<br>")
    if platforms: parts.append(f"Plataformas: {', '.join(platforms)}<br>")
    if jogos: parts.append(f"Jogos Fav.: {', '.join(jogos[:4])}{'...' if len(jogos) > 4 else ''}<br>") # Show a few more games
    if estilos: parts.append(f"Estilos Pref.: {', '.join(estilos[:3])}{'...' if len(estilos) > 3 else ''}<br>")

    # Add description, shortened
    descricao = profile.get('descricao', '')
    if descricao:
         parts.append(f"Descrição: {textwrap.shorten(descricao, width=80, placeholder='...')}")

    # Join all parts into a single HTML string
    return "".join(parts).strip()

def format_legend_text(profile: Dict[str, Any], score: Optional[float] = None) -> str:
    """Formata o texto da legenda com ícones e emojis."""
    parts = []
    
    # Nome e ID
    nome = profile.get('nome', 'N/A')
    id_perfil = profile.get('id', 'N/A')
    parts.append(f"👤 {nome} (ID: {id_perfil})")
    
    # Score (se disponível)
    if score is not None:
        score_icon = "⭐" if score >= 0.7 else "⚡" if score >= 0.5 else "✨"
        parts.append(f"{score_icon} Score: {score:.2f}")
    
    # Idade e Cidade
    idade = profile.get('idade', 'N/A')
    cidade = profile.get('cidade', 'N/A')
    parts.append(f"📅 Idade: {idade} | 🌎 {cidade}")
    
    # Disponibilidade
    disponibilidade = profile.get('disponibilidade', 'N/A')
    parts.append(f"⏰ Disponível: {disponibilidade}")
    
    # Plataformas
    plataformas = safe_split_and_strip_list(profile.get('plataformas_possuidas', ''))
    if plataformas:
        parts.append(f"🎮 Plataformas: {', '.join(plataformas)}")
    
    # Jogos Favoritos
    jogos = safe_split_and_strip_list(profile.get('jogos_favoritos', ''))
    if jogos:
        parts.append(f"🏆 Jogos: {', '.join(jogos[:3])}{'...' if len(jogos) > 3 else ''}")
    
    # Estilos
    estilos = safe_split_and_strip_list(profile.get('estilos_preferidos', ''))
    if estilos:
        parts.append(f"🎯 Estilos: {', '.join(estilos[:3])}{'...' if len(estilos) > 3 else ''}")
    
    # Nível Competitivo
    nivel = profile.get('nivel_competitivo', 'N/A')
    parts.append(f"🏅 Nível: {nivel}")
    
    return "<br>".join(parts)

def create_3d_plot(
    origin_coords_3d: np.ndarray,
    neighbor_coords_3d: np.ndarray,
    origin_profile: Dict[str, Any],
    neighbor_profiles: List[Dict[str, Any]], 
    neighbor_l2_distances: List[float]      
    ) -> go.Figure:
    """
    Creates the 3D Plotly figure using PCA-reduced coordinates.
    Colors neighbors by custom score and sets up informative hover labels.
    """
    # Definição das cores do tema escuro
    DARK_THEME = {
        'bg_color': '#111111',        # Fundo principal quase preto
        'plot_bg': '#1a1a1a',         # Fundo do plot um pouco mais claro
        'grid_color': '#333333',      # Cor da grade mais suave
        'text_color': '#ffffff',      # Texto em branco
        'title_color': '#ffffff',     # Títulos em branco
        'axis_color': '#cccccc',      # Eixos em cinza claro
        'marker_line': '#000000',     # Borda dos marcadores em preto
        'colorscale': 'Viridis'       # Escala de cores que funciona bem no tema escuro
    }

    fig = go.Figure()
    num_neighbors = len(neighbor_profiles)

    # --- Prepare Detailed Legend Text ---
    origin_legend = format_legend_text(origin_profile)
    neighbors_legend = [
        format_legend_text(
            p,
            p.get('score_compatibilidade')
        )
        for p in neighbor_profiles
    ]

    # --- Add Annotation for Legend Box ---
    fig.add_annotation(
        x=0,
        y=1,
        xref="paper",
        yref="paper",
        text="<br>".join([
            "<b>🎯 PERFIL ORIGEM:</b>",
            origin_legend,
            "<br><b>🤝 MATCHES ENCONTRADOS:</b>",
            *neighbors_legend
        ]),
        align="left",
        showarrow=False,
        bgcolor=DARK_THEME['bg_color'],
        bordercolor=DARK_THEME['grid_color'],
        borderwidth=2,
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color=DARK_THEME['text_color']
        ),
        xanchor="left",
        yanchor="top",
        width=400,
        opacity=0.9
    )

    # --- Prepare Data for Plotting ---
    origin_hover = format_hover_text(origin_profile)
    neighbors_hover = [
        format_hover_text(
            p,
            p.get('score_compatibilidade'),
            p.get('score_details'),
            l2_dist
        )
        for p, l2_dist in zip(neighbor_profiles, neighbor_l2_distances)
    ]

    neighbor_names_with_score = []
    for p in neighbor_profiles:
        profile_id = p.get('id', 'N/A')
        name_part = p.get('nome', f"ID {profile_id}")
        score_part = p.get('score_compatibilidade', 0.0)
        label = f"{name_part[:12]} ({score_part:.2f})"
        neighbor_names_with_score.append(label)

    neighbor_custom_scores = [p.get('score_compatibilidade', 0.0) for p in neighbor_profiles]

    if num_neighbors > 0 and neighbor_coords_3d.ndim == 2 and neighbor_coords_3d.shape[0] == num_neighbors:
        fig.add_trace(go.Scatter3d(
            x=neighbor_coords_3d[:, 0],
            y=neighbor_coords_3d[:, 1],
            z=neighbor_coords_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=neighbor_custom_scores,
                colorscale=DARK_THEME['colorscale'],
                cmin=MIN_CUSTOM_SCORE_THRESHOLD,
                cmax=max(max(neighbor_custom_scores), MIN_CUSTOM_SCORE_THRESHOLD + 0.1) if neighbor_custom_scores else 1.0,
                opacity=0.85,
                line=dict(color=DARK_THEME['marker_line'], width=1),
                colorbar=dict(
                    title=dict(
                        text='Score Custom',
                        font=dict(color=DARK_THEME['text_color'], size=12)
                    ),
                    thickness=15,
                    len=0.7,
                    tickformat=".2f",
                    tickfont=dict(color=DARK_THEME['text_color']),
                    bgcolor=DARK_THEME['bg_color'],
                    bordercolor=DARK_THEME['grid_color']
                )
            ),
            text=neighbor_names_with_score,
            textposition='top center',
            textfont=dict(color=DARK_THEME['text_color'], size=10),
            showlegend=False,
            name='Vizinhos'
        ))

    origin_coords_3d = np.array(origin_coords_3d).flatten()
    if len(origin_coords_3d) >= 3:
        fig.add_trace(go.Scatter3d(
            x=[origin_coords_3d[0]],
            y=[origin_coords_3d[1]],
            z=[origin_coords_3d[2]],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='diamond',
                opacity=1.0,
                line=dict(color=DARK_THEME['marker_line'], width=1)
            ),
            showlegend=False,
            name='Origem'
        ))

    # --- Layout Configuration ---
    fig.update_layout(
        title=dict(
            text='Visualização 3D de Perfis Similares',
            x=0.5,
            font=dict(color=DARK_THEME['title_color'], size=16)
        ),
        paper_bgcolor=DARK_THEME['bg_color'],
        plot_bgcolor=DARK_THEME['plot_bg'],
        margin=dict(l=10, r=10, b=10, t=50),
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title='Componente PCA 1',
                titlefont=dict(color=DARK_THEME['axis_color']),
                tickfont=dict(color=DARK_THEME['axis_color']),
                gridcolor=DARK_THEME['grid_color'],
                backgroundcolor=DARK_THEME['plot_bg'],
                showbackground=True
            ),
            yaxis=dict(
                title='Componente PCA 2',
                titlefont=dict(color=DARK_THEME['axis_color']),
                tickfont=dict(color=DARK_THEME['axis_color']),
                gridcolor=DARK_THEME['grid_color'],
                backgroundcolor=DARK_THEME['plot_bg'],
                showbackground=True
            ),
            zaxis=dict(
                title='Componente PCA 3',
                titlefont=dict(color=DARK_THEME['axis_color']),
                tickfont=dict(color=DARK_THEME['axis_color']),
                gridcolor=DARK_THEME['grid_color'],
                backgroundcolor=DARK_THEME['plot_bg'],
                showbackground=True
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.0)
            ),
            bgcolor=DARK_THEME['plot_bg']
        )
    )

    return fig

def generate_html_file(fig: go.Figure, output_path: str):
    """Saves the Plotly figure to an interactive HTML file."""
    try:
        # include_plotlyjs='cdn' uses the online Plotly library (smaller file size)
        # full_html=True creates a self-contained HTML file
        fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
        logging.info(f"Successfully generated HTML visualization: {output_path}")
    except Exception as e:
        logging.error(f"Error writing HTML file to {output_path}: {e}", exc_info=True)
        print(f"ERROR: Could not save HTML file to {output_path}. Check permissions and path.")

# --- Main Execution Logic ---
def main():
    """Main function to orchestrate the visualization generation process."""
    args = parse_arguments()

    # --- Dependency Check ---
    if not SKLEARN_AVAILABLE:
        # Message already printed, exit cleanly if PCA is essential.
        logging.critical("Exiting due to missing critical dependency: scikit-learn (needed for PCA).")
        # If PCA was optional, you might remove the return here.
        return

    # --- 1. Load Embeddings and Build FAISS Index ---
    logging.info("Loading embeddings and building FAISS index...")
    embeddings_matrix, profile_ids_map, faiss_index = load_embeddings_and_map(
        DATABASE_EMBEDDINGS, args.dim
    )

    # Check if loading was successful
    if embeddings_matrix is None or profile_ids_map is None or faiss_index is None:
        logging.critical("Failed to load embeddings or build FAISS index. Aborting.")
        print("ERROR: Failed to load necessary data or build index. Check logs for details.")
        return

    num_total_profiles = len(profile_ids_map)
    logging.info(f"Data loaded: {num_total_profiles} profiles with embeddings.")
    if num_total_profiles == 0:
         logging.critical("No profiles with embeddings found. Cannot proceed.")
         print("ERROR: No profiles with embeddings loaded. Check embedding database.")
         return

    # --- 2. Select Origin Profile ID ---
    origin_id = args.profile_id
    available_ids = set(profile_ids_map) # Use set for faster lookup

    if origin_id is None:
        # Select a random ID from the list of IDs that have embeddings
        if not profile_ids_map: # Should be caught by num_total_profiles check, but extra safety
             logging.critical("Cannot select random ID, profile_ids_map is empty.")
             print("ERROR: Internal error - no profile IDs available.")
             return
        origin_id = random.choice(profile_ids_map)
        logging.info(f"No profile ID provided. Randomly selected origin ID: {origin_id}")
    elif origin_id not in available_ids:
        logging.error(f"Provided origin ID {origin_id} not found among profiles with embeddings ({num_total_profiles} available). Aborting.")
        print(f"ERROR: Profile ID {origin_id} does not have a corresponding embedding or does not exist.")
        if profile_ids_map:
             print(f"       Available IDs with embeddings range roughly from {min(profile_ids_map)} to {max(profile_ids_map)}.")
        return
    else:
        logging.info(f"Using specified origin ID: {origin_id}")

    # --- 3. Get Origin Profile Data and Embedding Vector ---
    origin_profile = carregar_perfil_por_id_cached(DATABASE_PROFILES, origin_id)
    if not origin_profile:
        # This case might occur if the profile exists in embeddings DB but not profiles DB
        logging.error(f"Failed to load profile data for origin ID {origin_id} from '{DATABASE_PROFILES}'. Aborting.")
        print(f"ERROR: Could not load profile details for ID {origin_id}. Check profile database.")
        return

    try:
        # Find the index (row number in the matrix) corresponding to the origin ID
        origin_matrix_index = profile_ids_map.index(origin_id)
        # Extract the embedding vector for the origin profile
        origin_embedding = embeddings_matrix[origin_matrix_index].reshape(1, -1) # Keep as 2D for FAISS search
        logging.info(f"Found embedding for origin ID {origin_id} at matrix index {origin_matrix_index}.")
    except (ValueError, IndexError) as e:
        # This should theoretically not happen if ID check passed, but good to have safety net
        logging.error(f"Internal inconsistency: Could not find origin ID {origin_id} in profile_ids_map "
                      f"or embeddings_matrix after initial checks. Error: {e}. Aborting.", exc_info=True)
        print(f"ERROR: Internal error locating embedding vector for ID {origin_id}.")
        return

    # --- 4. Find Initial Candidate Neighbors using FAISS (Embedding Similarity) ---
    # Calculate how many neighbors to fetch from FAISS (k)
    k_search = args.num_neighbors * args.search_factor + 1 # +1 to include self, will be filtered later
    k_search = min(k_search, faiss_index.ntotal) # Cannot search for more neighbors than exist in the index
    if k_search <= 0:
        logging.warning(f"Calculated k_search is {k_search}. Setting to 1 (search for self only).")
        k_search = 1
    elif k_search == 1 and faiss_index.ntotal > 1:
        logging.warning(f"k_search is 1, but more profiles exist. Increasing k_search to 2 to find at least one neighbor.")
        k_search = 2 # Try to get at least one other profile
    k_search = min(k_search, faiss_index.ntotal) # Ensure k_search doesn't exceed total again

    logging.info(f"Searching for {k_search} nearest neighbors (L2 distance) for ID {origin_id} using FAISS...")
    faiss_candidates_indices = []
    faiss_candidates_distances = []
    if k_search > 0: # No point searching if k=0
        try:
            # Perform the search: find k_search nearest neighbors to the origin_embedding
            distances_sq, initial_indices = faiss_index.search(origin_embedding, k_search)

            # Process FAISS results
            if initial_indices is not None and len(initial_indices) > 0 and len(initial_indices[0]) > 0:
                # FAISS returns indices relative to the matrix used to build the index
                faiss_candidates_indices_raw = initial_indices[0]
                # FAISS returns squared L2 distances, convert to actual L2 distance
                faiss_candidates_distances_sq_raw = distances_sq[0]

                # Filter out invalid indices (-1 can be returned if k > ntotal)
                valid_mask = (faiss_candidates_indices_raw != -1)
                faiss_candidates_indices = faiss_candidates_indices_raw[valid_mask]
                faiss_candidates_distances_sq = faiss_candidates_distances_sq_raw[valid_mask]

                # Calculate actual L2 distances (avoid sqrt of zero or negative if something weird happened)
                faiss_candidates_distances = np.sqrt(np.maximum(0, faiss_candidates_distances_sq))

                logging.info(f"FAISS search returned {len(faiss_candidates_indices)} raw candidates (including self if found).")
            else:
                logging.warning(f"FAISS search returned no results for ID {origin_id}.")
                print("Warning: FAISS search returned no initial neighbors based on embeddings.")
                # Keep lists empty

        except faiss.FaissException as fe:
            logging.error(f"FAISS error during search operation: {fe}", exc_info=True)
            print("ERROR: FAISS search failed. Check logs.")
            return # Cannot proceed without candidates
        except Exception as e:
            logging.error(f"Unexpected error during FAISS search: {e}", exc_info=True)
            print("ERROR: An unexpected error occurred during neighbor search.")
            return
    else:
        logging.warning("k_search is 0, skipping FAISS search.")


    # --- 5. Score and Filter Candidates using Custom Similarity ---
    logging.info("Calculating custom similarity scores for FAISS candidates and filtering...")
    scored_neighbors = [] # List to store dicts of valid neighbors with scores
    checked_ids = {origin_id} # Keep track of processed IDs to avoid duplicates

    # Primeira tentativa com thresholds normais
    for i in range(len(faiss_candidates_indices)):
        matrix_idx = faiss_candidates_indices[i]
        if matrix_idx < 0 or matrix_idx >= len(profile_ids_map):
            continue

        candidate_id = profile_ids_map[matrix_idx]
        if candidate_id in checked_ids:
            continue

        checked_ids.add(candidate_id)
        candidate_profile = carregar_perfil_por_id_cached(DATABASE_PROFILES, candidate_id)
        if not candidate_profile:
            continue

        total_score, score_details = calculate_custom_similarity(origin_profile, candidate_profile, use_fallback=False)
        if total_score >= MIN_CUSTOM_SCORE_THRESHOLD and total_score > 0:
            scored_neighbors.append({
                "profile_id": candidate_id,
                "score": total_score,
                "score_details": score_details,
                "profile_data": candidate_profile,
                "l2_distance": faiss_candidates_distances[i],
                "matrix_index": matrix_idx
            })

    # Se não encontrou vizinhos suficientes, tenta com thresholds mais baixos
    if len(scored_neighbors) < args.num_neighbors:
        logging.info(f"Encontrados apenas {len(scored_neighbors)} vizinhos com thresholds normais. Tentando com thresholds mais baixos...")
        # Limpa checked_ids para permitir reavaliar candidatos
        checked_ids = {origin_id}
        
        for i in range(len(faiss_candidates_indices)):
            matrix_idx = faiss_candidates_indices[i]
            if matrix_idx < 0 or matrix_idx >= len(profile_ids_map):
                continue

            candidate_id = profile_ids_map[matrix_idx]
            if candidate_id in checked_ids:
                continue

            checked_ids.add(candidate_id)
            candidate_profile = carregar_perfil_por_id_cached(DATABASE_PROFILES, candidate_id)
            if not candidate_profile:
                continue

            total_score, score_details = calculate_custom_similarity(origin_profile, candidate_profile, use_fallback=True)
            if total_score >= FALLBACK_CUSTOM_SCORE and total_score > 0:
                scored_neighbors.append({
                    "profile_id": candidate_id,
                    "score": total_score,
                    "score_details": score_details,
                    "profile_data": candidate_profile,
                    "l2_distance": faiss_candidates_distances[i],
                    "matrix_index": matrix_idx
                })

    logging.info(f"Found {len(scored_neighbors)} candidates after all scoring attempts.")

    # --- 6. Sort by Custom Score and Select Top N Neighbors ---
    # Sort candidates in descending order based on the custom score
    scored_neighbors.sort(key=lambda x: x["score"], reverse=True)

    # Select the top N neighbors based on the target number
    top_neighbors_data = scored_neighbors[:args.num_neighbors]

    if not top_neighbors_data:
        logging.info(f"No neighbors passed the custom scoring criteria for ID {origin_id}. Plot will only show origin.") # Info level is fine
        print("\nInfo: No neighbors found matching the specified scoring criteria after filtering.")
        # Prepare empty lists for plotting function if no neighbors are found
        neighbor_profiles_for_plot = []
        neighbor_embeddings_list = []
        final_neighbor_l2_distances = []
    else:
        logging.info(f"Selected top {len(top_neighbors_data)} neighbors based on custom score.")
        neighbor_profiles_for_plot = []
        neighbor_embeddings_list = []
        final_neighbor_l2_distances = []
        # Extract necessary data for plotting
        for data in top_neighbors_data:
            # Add scores directly to the profile dictionary for easy access in plotting functions
            data["profile_data"]["score_compatibilidade"] = data["score"]
            data["profile_data"]["score_details"] = data["score_details"]

            neighbor_profiles_for_plot.append(data["profile_data"])
            # Retrieve embedding using the stored matrix index
            if 0 <= data["matrix_index"] < embeddings_matrix.shape[0]:
                 neighbor_embeddings_list.append(embeddings_matrix[data["matrix_index"]])
            else:
                 logging.error(f"Invalid matrix index {data['matrix_index']} for neighbor ID {data['profile_id']}. Cannot retrieve embedding.")
                 # Handle this error - e.g., skip this neighbor or use a placeholder?
                 # For now, let's skip adding the embedding, which might cause PCA issues later if counts mismatch.
                 # A better approach might be to filter out such neighbors earlier or handle PCA more robustly.
                 continue # Skip adding L2 distance as well if embedding is missing

            final_neighbor_l2_distances.append(data["l2_distance"]) # Keep L2 distance

        # Verify counts after potentially skipping neighbors with invalid indices
        if len(neighbor_profiles_for_plot) != len(neighbor_embeddings_list):
             logging.error("Mismatch between number of profiles and embeddings for neighbors after processing. Aborting plot generation.")
             print("ERROR: Internal data inconsistency preparing neighbors for plotting.")
             return


    # --- 7. Prepare Embeddings for Dimensionality Reduction (PCA) ---
    # Combine origin embedding and *valid* neighbor embeddings into a single matrix
    all_embeddings_list = [origin_embedding.flatten()] + neighbor_embeddings_list
    try:
        # Ensure all elements are numpy arrays before converting the list
        all_embeddings_list_np = [np.array(e, dtype=np.float32) for e in all_embeddings_list if isinstance(e, np.ndarray)]
        if len(all_embeddings_list_np) != len(all_embeddings_list):
             logging.warning("Some neighbor embeddings were invalid and excluded.")
             # Adjust neighbor lists for plotting if needed? Depends on where the error happened.
             # Let's assume the check in step 6 handled this.

        if not all_embeddings_list_np:
             logging.error("No valid embeddings remain after filtering. Cannot proceed with PCA.")
             print("ERROR: No data points available for visualization after filtering.")
             return

        combined_embeddings = np.array(all_embeddings_list_np, dtype=np.float32)
        # Check for potential shape issues (e.g., empty arrays)
        if combined_embeddings.size == 0 or combined_embeddings.ndim != 2:
             raise ValueError(f"Combined embeddings resulted in an invalid shape: {combined_embeddings.shape}")

        logging.info(f"Combined embeddings matrix shape for PCA: {combined_embeddings.shape}")
    except ValueError as e:
         logging.error(f"Error creating combined embeddings matrix: {e}", exc_info=True)
         print("ERROR: Failed to prepare data for dimensionality reduction.")
         return


    # --- 8. Reduce Dimensionality using PCA ---
    coords_3d = None
    origin_coords_3d = np.array([]) # Initialize as empty array
    neighbor_coords_3d = np.empty((0, 3)) # Initialize as empty 2D array

    num_points_for_pca = combined_embeddings.shape[0]

    if num_points_for_pca < 1:
         logging.error("No embeddings available (not even origin). Cannot perform PCA or plot.")
         print("ERROR: No data points available for visualization.")
         return # Cannot proceed
    elif num_points_for_pca == 1:
         logging.warning("Only the origin profile is available. PCA cannot be performed. "
                         "Will attempt to plot origin using first 3 embedding dimensions (or padding).")
         origin_flat = combined_embeddings[0]
         pad_width = max(0, 3 - len(origin_flat))
         origin_coords_3d = np.pad(origin_flat[:3], (0, pad_width))[:3] # Take first 3, pad if needed, ensure size 3
         # neighbor_coords_3d remains empty
    else:
        # We have at least 2 points (origin + >=1 neighbor)
        target_pca_dims = 3
        if num_points_for_pca < target_pca_dims:
            # Not enough points for 3D PCA, try reducing dimensions based on available points
            target_dims_possible = num_points_for_pca -1 # Max components is n_samples - 1
            if target_dims_possible <=0 :
                logging.error(f"Cannot perform PCA with {num_points_for_pca} points.")
                print("Error: Not enough unique data points for PCA.")
                return

            logging.warning(f"Only {num_points_for_pca} points (origin + neighbors). "
                            f"Attempting PCA to {target_dims_possible}D and padding Z coordinate(s).")
            coords_reduced = reduce_dimensionality(combined_embeddings, n_components=target_dims_possible)

        else:
            # Sufficient points for 3D PCA
            logging.info(f"Performing PCA for {num_points_for_pca} points to {target_pca_dims}D visualization...")
            coords_reduced = reduce_dimensionality(combined_embeddings, n_components=target_pca_dims)

        # Process PCA results (whether reduced or full 3D)
        if coords_reduced is not None:
            # Pad with zeros if PCA resulted in fewer than 3 dimensions
            pad_width = target_pca_dims - coords_reduced.shape[1]
            if pad_width < 0: # Should not happen if n_components was handled correctly
                 logging.error(f"PCA returned more dimensions ({coords_reduced.shape[1]}) than requested ({target_pca_dims})")
                 return
            elif pad_width > 0:
                 logging.info(f"Padding PCA results with {pad_width} zero column(s).")
                 coords_3d = np.pad(coords_reduced, ((0, 0), (0, pad_width)), 'constant')
            else:
                 coords_3d = coords_reduced # Already 3D

            # Separate origin and neighbor coordinates from the (potentially padded) result
            origin_coords_3d = coords_3d[0]
            neighbor_coords_3d = coords_3d[1:]

            # Ensure neighbor coords match the number of neighbor profiles plotted
            if neighbor_coords_3d.shape[0] != len(neighbor_profiles_for_plot):
                 logging.warning(f"Mismatch after PCA: {neighbor_coords_3d.shape[0]} coordinate sets vs "
                                 f"{len(neighbor_profiles_for_plot)} neighbor profiles. Plotting may be incorrect.")
                 # Attempt to truncate coords if too many? Or maybe error was earlier.
                 neighbor_coords_3d = neighbor_coords_3d[:len(neighbor_profiles_for_plot), :]

        else:
             logging.error("Dimensionality reduction failed. Cannot generate plot.")
             print("ERROR: Dimensionality reduction (PCA) failed.")
             return


    # --- 9. Create Plotly Figure ---
    logging.info("Creating Plotly 3D figure...")
    try:
        fig = create_3d_plot(
            origin_coords_3d,
            neighbor_coords_3d,
            origin_profile,
            neighbor_profiles_for_plot, # These contain the added scores
            final_neighbor_l2_distances # Pass L2 distances for context in hover text
        )
    except Exception as plot_err:
         logging.error(f"Error during Plotly figure creation: {plot_err}", exc_info=True)
         print("ERROR: Failed to create the Plotly visualization.")
         return

    # --- 10. Generate Output HTML File ---
    # Create a unique filename including origin ID, timestamp, and maybe a hash
    timestamp_file = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create a hash based on the IDs involved for some uniqueness if run multiple times quickly
    neighbor_ids_final = [p['id'] for p in neighbor_profiles_for_plot]
    id_string = str(origin_id) + "".join(map(str, sorted(neighbor_ids_final)))
    data_hash = hashlib.sha1(id_string.encode()).hexdigest()[:8] # Short hash

    output_filename = f"profile_{origin_id}_neighbors_score_{timestamp_file}_{data_hash}.html"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    logging.info(f"Attempting to generate HTML file: {output_path}")
    generate_html_file(fig, output_path)

    # --- Final Summary ---
    print(f"\n✅ Script finished.")
    print(f"   Origin Profile: {origin_profile.get('nome', 'N/A')} (ID: {origin_id})")
    print(f"   Neighbors Found (passing score): {len(neighbor_profiles_for_plot)}")
    if neighbor_profiles_for_plot:
        top_scores = [f"{p['score_compatibilidade']:.3f}" for p in neighbor_profiles_for_plot[:3]]
        print(f"   Top Scores: {', '.join(top_scores)}{'...' if len(neighbor_profiles_for_plot) > 3 else ''}")
    print(f"   Output HTML: {output_path}")
    print(f"   Log File: {LOG_FILE}")


if __name__ == "__main__":
    # Check for dependency *before* running main logic if it's absolutely critical
    if not SKLEARN_AVAILABLE:
        print("-" * 60)
        print("ERROR: Critical dependency 'scikit-learn' is not installed.")
        print("       PCA dimensionality reduction is required for this script.")
        print("       Please install it using: pip install scikit-learn")
        print("-" * 60)
        # Exit if PCA is essential
        exit(1)
    main()