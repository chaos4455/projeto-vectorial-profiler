import sqlite3
import numpy as np
import faiss
import json
import datetime
import hashlib
import os
import random
from colorama import init, Fore, Style
from rich.console import Console
from rich.progress import track
import logging
from typing import Tuple, List, Dict, Set, Optional, Any
from collections import Counter # Para contar itens em comum

# --- Configurações ---
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
VALUATION_DIR = "valuation_v3"
os.makedirs(VALUATION_DIR, exist_ok=True)

NUM_NEIGHBORS_TARGET: int = 10 # Quantos vizinhos finais queremos
INITIAL_SEARCH_FACTOR: int = 2000 # Buscar MUITOS mais (20x) para ter chance de achar bons candidatos
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.1 # Opcional: Exigir um score mínimo para ser incluído

LOG_LEVEL = logging.INFO
EXPECTED_EMBEDDING_DIM: int = 64

# --- Pesos para o Score de Compatibilidade Personalizado ---
# Ajuste estes pesos para dar mais ou menos importância a cada fator
WEIGHTS = {
    "jogos": 0.40,
    "estilos": 0.30,
    "plataformas": 0.20,
    "disponibilidade": 0.10,
    # Adicionar outros se necessário (e ajustar pesos para somar 1)
    # "musica": 0.0,
    # "idade_prox": 0.0,
}
# Garante que os pesos somam aproximadamente 1
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"

# --- Configuração de Logging ---
LOG_FILE_VAL = os.path.join(VALUATION_DIR, f"valuation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=LOG_FILE_VAL,
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Inicializar Colorama e Rich
init(autoreset=True)
console = Console()
logging.info(f"--- Script de Valuation V3 (com Score Personalizado) Iniciado ---")
logging.info(f"Pesos de Score: {WEIGHTS}")

# --- Funções Auxiliares ---

# (carregar_embeddings_e_ids, construir_indice_faiss, carregar_perfil_por_id - permanecem iguais da versão anterior)
def carregar_embeddings_e_ids(db_path: str) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[int]]:
    """Carrega todos os embeddings e seus IDs correspondentes do banco de dados."""
    console.print(f"💾 [cyan]Carregando embeddings de '{db_path}'...[/cyan]")
    embeddings_list = []
    ids_list = []
    dimension = None
    detected_dim = None

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            if total_embeddings == 0:
                 console.print(f"⚠️ [bold yellow]Nenhum embedding encontrado em '{db_path}'.[/bold yellow]")
                 logging.warning(f"Nenhum embedding encontrado em {db_path}")
                 return None, None, None

            cursor.execute("SELECT id, embedding FROM embeddings ORDER BY id")
            rows = cursor.fetchall()

            for profile_id, embedding_blob in track(rows, description="Processando embeddings...", total=total_embeddings):
                try:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    if detected_dim is None:
                        detected_dim = len(embedding)
                        dimension = detected_dim
                        if detected_dim != EXPECTED_EMBEDDING_DIM:
                             console.print(f"⚠️ [bold yellow]Dimensão detectada ({detected_dim}) diferente da esperada ({EXPECTED_EMBEDDING_DIM}). Usando a detectada.[/bold yellow]")
                             logging.warning(f"Dimensão do embedding detectada: {detected_dim} (esperada: {EXPECTED_EMBEDDING_DIM})")
                        else:
                             logging.info(f"Dimensão do embedding confirmada: {dimension}")

                    if len(embedding) == dimension:
                        embeddings_list.append(embedding)
                        ids_list.append(profile_id)
                    else:
                        logging.warning(f"Dimensão incorreta para ID {profile_id}. Esperado: {dimension}, Obtido: {len(embedding)}")

                except Exception as e:
                    console.print(f"⚠️ [bold red]Erro ao processar embedding para ID {profile_id}: {e}[/bold red]")
                    logging.error(f"Erro processando embedding ID {profile_id}: {e}", exc_info=True)
                    continue

        if not embeddings_list:
            console.print(f"❌ [bold red]Nenhum embedding válido carregado de '{db_path}'.[/bold red]")
            logging.error(f"Falha ao carregar embeddings válidos de {db_path}")
            return None, None, None

        embeddings_matrix = np.ascontiguousarray(embeddings_list, dtype=np.float32)
        console.print(f"✅ [green]{len(ids_list)} embeddings carregados com sucesso. Dimensão: {dimension}[/green]")
        logging.info(f"{len(ids_list)} embeddings carregados. Dimensão: {dimension}")
        return embeddings_matrix, ids_list, dimension

    except sqlite3.Error as e:
        console.print(f"❌ [bold red]Erro de banco de dados ao carregar embeddings: {e}[/bold red]")
        logging.critical(f"Erro SQLite ao carregar embeddings de {db_path}: {e}", exc_info=True)
        return None, None, None
    except Exception as e:
        console.print(f"❌ [bold red]Erro inesperado ao carregar embeddings: {e}[/bold red]")
        logging.critical(f"Erro inesperado ao carregar embeddings: {e}", exc_info=True)
        return None, None, None

def construir_indice_faiss(embeddings: np.ndarray, dimension: int) -> Optional[faiss.Index]:
    """Constrói um índice FAISS em memória a partir da matriz de embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        console.print(f"❌ [bold red]Matriz de embeddings tem formato inesperado: {embeddings.shape}. Esperado: (n_vectors, {dimension})[/bold red]")
        logging.error(f"Formato inválido da matriz de embeddings: {embeddings.shape}")
        return None
    if embeddings.shape[0] == 0:
        console.print("❌ [bold red]Matriz de embeddings está vazia. Não é possível construir o índice.[/bold red]")
        logging.error("Tentativa de construir índice FAISS com matriz vazia.")
        return None

    console.print(f"⚙️ [bold blue]Construindo índice FAISS (IndexFlatIP) em memória para {embeddings.shape[0]} vetores...[/bold blue]")
    try:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        console.print(f"✅ [green]Índice FAISS construído com sucesso. Total de vetores indexados: {index.ntotal}[/green]")
        logging.info(f"Índice FAISS (IndexFlatIP) construído. Vetores: {index.ntotal}")
        return index
    except Exception as e:
        console.print(f"❌ [bold red]Erro ao construir índice FAISS: {e}[/bold red]")
        logging.error(f"Erro construindo índice FAISS: {e}", exc_info=True)
        return None

def carregar_perfil_por_id(db_path: str, profile_id: int) -> Optional[Dict[str, Any]]:
    """Carrega os dados completos de um perfil pelo seu ID."""
    # Cache simples para evitar recarregar o mesmo perfil várias vezes na mesma execução
    # ATENÇÃO: Este cache não é persistente entre execuções do script.
    if not hasattr(carregar_perfil_por_id, "cache"):
        carregar_perfil_por_id.cache = {} # type: ignore

    if profile_id in carregar_perfil_por_id.cache: # type: ignore
        return carregar_perfil_por_id.cache[profile_id] # type: ignore

    try:
        with sqlite3.connect(db_path, timeout=10) as conn: # Timeout maior pode ajudar em concorrência
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM perfis WHERE id = ?", (profile_id,))
            perfil_row = cursor.fetchone()
            if perfil_row:
                perfil_dict = dict(perfil_row)
                # Converte booleano aqui mesmo
                if 'compartilhar_contato' in perfil_dict:
                     perfil_dict['compartilhar_contato'] = bool(perfil_dict['compartilhar_contato'])
                carregar_perfil_por_id.cache[profile_id] = perfil_dict # type: ignore
                return perfil_dict
            else:
                logging.debug(f"Perfil com ID {profile_id} não encontrado em '{db_path}'.")
                carregar_perfil_por_id.cache[profile_id] = None # Cache a falha também # type: ignore
                return None
    except sqlite3.Error as e:
        console.print(f"⚠️ [bold red]DB Error loading profile ID {profile_id}: {e}[/bold red]")
        logging.error(f"SQLite Error loading profile ID {profile_id} from {db_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        console.print(f"⚠️ [bold red]Unexpected Error loading profile ID {profile_id}: {e}[/bold red]")
        logging.error(f"Unexpected Error loading profile ID {profile_id}: {e}", exc_info=True)
        return None

def safe_split_and_strip(text: Optional[str], delimiter: str = ',') -> Set[str]:
    """Divide uma string, remove espaços e retorna um set de itens não vazios."""
    if not text or not isinstance(text, str):
        return set()
    return {item.strip() for item in text.split(delimiter) if item.strip()}

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calcula a similaridade Jaccard entre dois sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def availability_similarity(avail1: Optional[str], avail2: Optional[str]) -> float:
    """Calcula uma similaridade simples para disponibilidade."""
    if not avail1 or not avail2:
        return 0.0
    avail1_norm = avail1.split('(')[0].strip().lower()
    avail2_norm = avail2.split('(')[0].strip().lower()

    if avail1_norm == avail2_norm:
        return 1.0
    # Verificações parciais (ex: Fim de Semana)
    elif "fim de semana" in avail1_norm and "fim de semana" in avail2_norm:
        return 0.6 # Match parcial de FDS
    elif "semana" in avail1_norm and "semana" in avail2_norm:
        return 0.4 # Match parcial durante a semana
    # Poderia adicionar mais regras (Manhã/Tarde vs Flexível etc.)
    else:
        return 0.0

def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Calcula o score de compatibilidade ponderado entre dois perfis."""
    scores = {}

    # 1. Jogos Favoritos
    games1 = safe_split_and_strip(profile1.get('jogos_favoritos'))
    games2 = safe_split_and_strip(profile2.get('jogos_favoritos'))
    scores['jogos'] = jaccard_similarity(games1, games2)

    # 2. Estilos Preferidos
    styles1 = safe_split_and_strip(profile1.get('estilos_preferidos'))
    styles2 = safe_split_and_strip(profile2.get('estilos_preferidos'))
    scores['estilos'] = jaccard_similarity(styles1, styles2)

    # 3. Plataformas Possuídas
    platforms1 = safe_split_and_strip(profile1.get('plataformas_possuidas'))
    platforms2 = safe_split_and_strip(profile2.get('plataformas_possuidas'))
    scores['plataformas'] = jaccard_similarity(platforms1, platforms2)

    # 4. Disponibilidade
    scores['disponibilidade'] = availability_similarity(profile1.get('disponibilidade'), profile2.get('disponibilidade'))

    # Calcular score final ponderado
    total_score = sum(scores[key] * WEIGHTS[key] for key in WEIGHTS if key in scores)

    logging.debug(f"Score entre {profile1.get('id')} e {profile2.get('id')}: Total={total_score:.4f}, Detalhes={scores}")
    return total_score, scores # Retorna score total e detalhes

def buscar_e_filtrar_vizinhos_com_score(
    faiss_index: faiss.Index,
    embedding_origem: np.ndarray,
    id_origem: int,
    perfil_origem: Dict[str, Any], # Passa o perfil origem inteiro
    profile_ids_map: List[int],
    num_neighbors_target: int,
    initial_search_factor: int,
    profiles_db_path: str,
    min_score_threshold: float = 0.0 # Score mínimo para considerar
) -> List[Tuple[float, int]]:
    """Busca vizinhos, calcula score customizado, filtra e retorna os top N (score, id)."""

    k_search = num_neighbors_target * initial_search_factor + 1
    origin_name = perfil_origem.get('nome', f"NOME_NAO_ENCONTRADO_{id_origem}")
    console.print(f"🔍 [yellow]Buscando até {k_search-1} vizinhos potenciais (FAISS) para ID {id_origem}...[/yellow]")
    candidates_with_scores: List[Tuple[float, int, Dict[str, float]]] = [] # (score_total, id, score_detalhes)
    checked_profile_ids: Set[int] = {id_origem}

    try:
        distances, indices = faiss_index.search(embedding_origem, k_search)
        faiss_indices_found = indices[0]
        logging.info(f"Busca FAISS inicial retornou {len(faiss_indices_found)} índices.")

        console.print(f"🧐 [cyan]Calculando scores de compatibilidade e filtrando para {num_neighbors_target} perfis...[/cyan]")

        # Usa track para mostrar progresso do cálculo de scores
        for idx in track(faiss_indices_found, description="Calculando scores...", total=len(faiss_indices_found)):
            if 0 <= idx < len(profile_ids_map):
                potential_profile_id = profile_ids_map[idx]
            else:
                 logging.warning(f"Índice FAISS inválido: {idx}. Pulando.")
                 continue

            if potential_profile_id in checked_profile_ids:
                continue
            checked_profile_ids.add(potential_profile_id)

            # Carrega o perfil candidato completo para calcular score
            potential_profile = carregar_perfil_por_id(profiles_db_path, potential_profile_id)

            if potential_profile:
                # Filtro primário: Nome diferente
                if potential_profile.get('nome') != origin_name:
                    # Calcula o score de compatibilidade
                    total_score, score_details = calculate_custom_similarity(perfil_origem, potential_profile)

                    # Adiciona à lista de candidatos se score for suficiente
                    if total_score >= min_score_threshold:
                        candidates_with_scores.append((total_score, potential_profile_id, score_details))
                        logging.debug(f"Candidato ID {potential_profile_id} (Nome: {potential_profile.get('nome')}) com score {total_score:.4f} adicionado.")
                    else:
                        logging.debug(f"Candidato ID {potential_profile_id} descartado por score baixo ({total_score:.4f} < {min_score_threshold}).")

                else:
                    logging.debug(f"Candidato ID {potential_profile_id} descartado por nome igual.")
            else:
                logging.warning(f"Não foi possível carregar perfil potencial ID {potential_profile_id} para cálculo de score.")

        # Ordena os candidatos pelo score total (maior primeiro)
        candidates_with_scores.sort(key=lambda item: item[0], reverse=True)

        # Seleciona os top N vizinhos
        top_neighbors = candidates_with_scores[:num_neighbors_target]

        if len(top_neighbors) < num_neighbors_target:
             console.print(f"⚠️ [yellow]Aviso: Encontrados apenas {len(top_neighbors)} perfis com score >= {min_score_threshold} (meta era {num_neighbors_target}).[/yellow]")
             logging.warning(f"Encontrados apenas {len(top_neighbors)}/{num_neighbors_target} perfis válidos após filtragem e score.")
        else:
             console.print(f"✅ [green]Ranking por score concluído. {len(top_neighbors)} perfis selecionados.[/green]")

        # Retorna lista de (score, id) para os top N
        return [(score, pid) for score, pid, _ in top_neighbors]

    except Exception as e:
        console.print(f"❌ [bold red]Erro durante busca/ranking FAISS/Score: {e}[/bold red]")
        logging.error(f"Erro na busca/ranking FAISS/Score: {e}", exc_info=True)
        return []

def default_serializer(obj):
    """Serializador JSON padrão para tipos NumPy e outros não nativos."""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (datetime.date, datetime.datetime)): return obj.isoformat()
    if isinstance(obj, bytes): return obj.decode('utf-8', errors='replace')
    if isinstance(obj, (Set, set)): return sorted(list(obj)) # Converte sets para listas ordenadas
    try:
        return json.JSONEncoder.default(self, obj) # Tenta o padrão
    except TypeError:
        logging.warning(f"Tipo {type(obj)} não serializável encontrado, usando string.")
        return str(obj)


# --- Script Principal ---
def main():
    console.print("\n" + "="*50)
    console.print(f"🚀 [bold green]INICIANDO VALUATION V3 (COM SCORE PERSONALIZADO)[/bold green] 🚀")
    console.print(f"💾 [cyan]Banco de Perfis:[/cyan] {DATABASE_PROFILES}")
    console.print(f"💾 [cyan]Banco de Embeddings:[/cyan] {DATABASE_EMBEDDINGS}")
    console.print(f"🎯 [cyan]Meta de Vizinhos:[/cyan] {NUM_NEIGHBORS_TARGET} (filtrados e rankeados por score)")
    console.print(f"📈 [cyan]Score Mínimo:[/cyan] {MIN_CUSTOM_SCORE_THRESHOLD}")
    console.print(f"⚖️ [cyan]Pesos Score:[/cyan] {WEIGHTS}")
    console.print(f"📝 [cyan]Log:[/cyan] {LOG_FILE_VAL}")
    console.print(f"📂 [cyan]Saída JSON:[/cyan] {VALUATION_DIR}/")
    console.print("="*50 + "\n")

    # --- Passo 1: Carregar Embeddings e IDs ---
    embeddings_matrix, profile_ids_map, embedding_dim = carregar_embeddings_e_ids(DATABASE_EMBEDDINGS)
    if embeddings_matrix is None or profile_ids_map is None or embedding_dim is None:
        console.print("❌ [bold red]Falha ao carregar embeddings. Encerrando.[/bold red]")
        return

    # --- Passo 2: Construir Índice FAISS ---
    faiss_index = construir_indice_faiss(embeddings_matrix, embedding_dim)
    if faiss_index is None:
        console.print("❌ [bold red]Falha ao construir índice FAISS. Encerrando.[/bold red]")
        return

    # --- Passo 3: Escolher Perfil de Origem ---
    if not profile_ids_map:
         console.print("❌ [bold red]Lista de IDs de perfis vazia. Encerrando.[/bold red]")
         return
    id_origem = random.choice(profile_ids_map)
    console.print(f"👤 [magenta]Perfil de origem escolhido: ID {id_origem}[/magenta]")
    logging.info(f"Perfil de origem selecionado: ID {id_origem}")

    # --- Passo 4: Carregar Dados Completos do Perfil de Origem ---
    perfil_origem = carregar_perfil_por_id(DATABASE_PROFILES, id_origem)
    if perfil_origem is None:
        console.print(f"❌ [bold red]Falha ao carregar perfil de origem (ID: {id_origem}). Encerrando.[/bold red]")
        return

    # --- Passo 5: Obter Embedding do Perfil de Origem ---
    try:
        source_index = profile_ids_map.index(id_origem)
        embedding_origem = embeddings_matrix[source_index].reshape(1, -1)
        logging.info(f"Embedding do perfil de origem (ID {id_origem}) obtido.")
    except (ValueError, IndexError) as e:
        console.print(f"❌ [bold red]Erro ao localizar embedding para ID {id_origem}: {e}. Encerrando.[/bold red]")
        return

    # --- Passo 6: Buscar, Calcular Score e Filtrar Vizinhos ---
    top_neighbors_scored = buscar_e_filtrar_vizinhos_com_score(
        faiss_index=faiss_index,
        embedding_origem=embedding_origem,
        id_origem=id_origem,
        perfil_origem=perfil_origem,
        profile_ids_map=profile_ids_map,
        num_neighbors_target=NUM_NEIGHBORS_TARGET,
        initial_search_factor=INITIAL_SEARCH_FACTOR,
        profiles_db_path=DATABASE_PROFILES,
        min_score_threshold=MIN_CUSTOM_SCORE_THRESHOLD
    )

    # Extrai apenas os IDs na ordem do score
    final_similar_profile_ids = [pid for score, pid in top_neighbors_scored]
    # Mapeia ID para score para adicionar ao JSON depois
    profile_scores = {pid: score for score, pid in top_neighbors_scored}


    # --- Passo 7: Recuperar Dados Completos dos Perfis Similares FINAIS ---
    console.print(f"📄 [cyan]Recuperando dados dos {len(final_similar_profile_ids)} perfis similares rankeados...[/cyan]")
    perfis_similares_final = []
    if final_similar_profile_ids:
        # Reutiliza o cache em carregar_perfil_por_id se possível
        for id_similar in track(final_similar_profile_ids, description="Carregando perfis finais..."):
            perfil_similar = carregar_perfil_por_id(DATABASE_PROFILES, id_similar)
            if perfil_similar:
                # Adiciona o score de compatibilidade ao dicionário do perfil similar
                perfil_similar['score_compatibilidade'] = round(profile_scores.get(id_similar, -1.0), 4) # Adiciona score
                perfis_similares_final.append(perfil_similar)
            else:
                logging.error(f"Falha ao re-carregar dados do perfil similar final ID {id_similar}.")
    else:
        console.print("ℹ️ [yellow]Nenhum perfil similar encontrado após filtragem e ranking por score.[/yellow]")
        logging.info("Nenhum perfil similar encontrado após ranking.")

    # --- Passo 8: Criar e Salvar o JSON de Resultados ---
    console.print(f"💾 [yellow]Criando e salvando JSON com os resultados...[/yellow]")
    data_hora = datetime.datetime.now().isoformat()

    dados = {
        'data_hora': data_hora,
        'perfil_origem': perfil_origem,
        'perfis_similares': perfis_similares_final, # Lista ordenada por score
        'parametros_busca': {
             'num_neighbors_target': NUM_NEIGHBORS_TARGET,
             'num_neighbors_found': len(perfis_similares_final),
             'initial_search_k': NUM_NEIGHBORS_TARGET * INITIAL_SEARCH_FACTOR + 1,
             'min_custom_score': MIN_CUSTOM_SCORE_THRESHOLD,
             'score_weights': WEIGHTS,
             'embedding_source': 'pre_calculated_v3',
             'faiss_index_type': 'IndexFlatIP',
             'filter_logic': 'exclude_self_id, exclude_same_name, rank_by_custom_score, unique_ids'
        }
    }

    # Calcular o hash dos dados
    hash_integridade = None
    try:
        # Usar o serializador customizado
        dados_str = json.dumps(dados, sort_keys=True, ensure_ascii=False, default=default_serializer).encode('utf-8')
        hash_integridade = hashlib.sha256(dados_str).hexdigest()
        dados['hash_integridade'] = hash_integridade
        logging.info(f"Hash de integridade calculado: {hash_integridade}")
    except Exception as e:
         console.print(f"⚠️ [yellow]Erro ao calcular/serializar hash: {e}.[/yellow]")
         logging.warning(f"Erro no cálculo/serialização do hash: {e}", exc_info=True)

    # Salvar o JSON
    safe_data_hora = data_hora.replace(':', '-').replace('.', '_')
    nome_arquivo = os.path.join(VALUATION_DIR, f"valuation_{safe_data_hora}_origem_{id_origem}_scored.json")
    try:
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            # Usar o serializador customizado também no dump final
            json.dump(dados, f, ensure_ascii=False, indent=4, default=default_serializer)
        console.print(f"✅ [bold green]Valuation concluída e salva em '{nome_arquivo}' com sucesso!{Style.RESET_ALL}")
        logging.info(f"Resultados da valuation (com score) salvos em {nome_arquivo}")
    except Exception as e:
        console.print(f"❌ [bold red]Erro Crítico ao salvar o arquivo JSON '{nome_arquivo}': {e}[/bold red]")
        logging.critical(f"Erro Crítico ao salvar JSON em {nome_arquivo}: {e}", exc_info=True)

    console.print("\n" + "="*50)
    console.print(f"🎉 [bold green]Processo de Valuation V3 (com Score) Concluído![/bold green] 🎉")
    console.print("="*50 + "\n")
    logging.info("--- Script de Valuation V3 (com Score) Finalizado ---")


if __name__ == "__main__":
    # Verificação inicial dos bancos de dados
    db_profiles_exists = os.path.exists(DATABASE_PROFILES)
    db_embeddings_exists = os.path.exists(DATABASE_EMBEDDINGS)
    if not db_profiles_exists or not db_embeddings_exists:
        console.print("❌ [bold red]Erro: Bancos de dados necessários não encontrados.[/bold red]")
        if not db_profiles_exists: console.print(f"   - Ausente: '{DATABASE_PROFILES}'")
        if not db_embeddings_exists: console.print(f"   - Ausente: '{DATABASE_EMBEDDINGS}'")
        console.print(f"➡️ [cyan]Execute 'profile_generator_v3.py' primeiro.[/cyan]")
    else:
        # Limpa o cache de perfis ao iniciar (opcional, mas garante dados frescos)
        if hasattr(carregar_perfil_por_id, "cache"):
            del carregar_perfil_por_id.cache # type: ignore
        main()