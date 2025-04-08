# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import json
import datetime
import hashlib
import os
import random
import logging
from typing import Tuple, List, Dict, Set, Optional, Any
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import colorsys # Para interpolação de cores

# --- Configurações ---
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
OUTPUT_DIR = "img-data-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constantes de Similaridade (Idênticas)
NUM_NEIGHBORS_TARGET: int = 10
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.05
WEIGHTS = {
    "plataformas": 0.45, "disponibilidade": 0.35, "jogos": 0.10,
    "estilos": 0.05, "interacao": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"
MIN_REQUIRED_PLATFORM_SCORE: float = 0.20
MIN_REQUIRED_AVAILABILITY_SCORE: float = 0.30

# Configurações da Imagem (Ajustadas para layout horizontal agrupado)
IMG_WIDTH = 3000
MARGIN_TOP = 150  # Aumentado para dar mais espaço ao título
MARGIN_BOTTOM = 120  # Aumentado para mais espaço no final
MARGIN_LEFT = 80  # Aumentado para mais espaço lateral
MARGIN_RIGHT = 80  # Aumentado para mais espaço lateral
PROFILE_SECTION_HEIGHT = 300  # Aumentado para acomodar melhor os elementos
PROFILE_SPACING = 60  # Aumentado para mais separação entre perfis
HEADER_HEIGHT = 120  # Aumentado para mais espaço no cabeçalho
TITLE_FONT_SIZE = 52
PROFILE_NAME_FONT_SIZE = 38
DETAIL_FONT_SIZE = 26
SCORE_LABEL_FONT_SIZE = 22  # Aumentado para melhor legibilidade
SCORE_VALUE_FONT_SIZE = 24  # Aumentado para melhor legibilidade

# Configuração dos Retângulos Horizontais
NUM_RECTS_PER_SCORE = 10
RECT_WIDTH = 28   # Aumentado para retângulos mais largos
RECT_HEIGHT = 18  # Aumentado para retângulos mais altos
RECT_PADDING = 4    # Aumentado para mais espaço entre retângulos

# Largura total de uma visualização de score (10 retângulos + 9 espaçamentos)
SCORE_VIZ_WIDTH = (NUM_RECTS_PER_SCORE * RECT_WIDTH) + ((NUM_RECTS_PER_SCORE - 1) * RECT_PADDING)
SCORE_VIZ_HEIGHT = RECT_HEIGHT

# Espaçamento entre as visualizações de score agrupadas
HORIZONTAL_GROUP_SPACING = 80  # Aumentado para mais espaço horizontal
VERTICAL_GROUP_SPACING = 70    # Aumentado para mais espaço vertical

LABEL_OFFSET_Y = 25  # Aumentado para mais espaço acima dos retângulos
VALUE_OFFSET_Y = 20  # Aumentado para mais espaço abaixo dos retângulos

# Cores (Idênticas)
COLOR_BACKGROUND = (25, 25, 30)
COLOR_TEXT_HEADER = (230, 230, 230)
COLOR_TEXT_PROFILE_NAME = (210, 210, 220)
COLOR_TEXT_DETAIL = (180, 180, 190)
COLOR_TEXT_SCORE_LABEL = (160, 160, 170)
COLOR_TEXT_SCORE_VALUE = (200, 200, 210)
COLOR_SEPARATOR = (60, 60, 70)
COLOR_RECT_DARK = (50, 50, 60) # Cor para retângulos "apagados"

# Fontes (Idênticas)
try:
    FONT_TITLE = ImageFont.truetype("arialbd.ttf", TITLE_FONT_SIZE)
    FONT_PROFILE_NAME = ImageFont.truetype("arialbd.ttf", PROFILE_NAME_FONT_SIZE)
    FONT_DETAIL = ImageFont.truetype("arial.ttf", DETAIL_FONT_SIZE)
    FONT_SCORE_LABEL = ImageFont.truetype("arial.ttf", SCORE_LABEL_FONT_SIZE)
    FONT_SCORE_VALUE = ImageFont.truetype("arialbd.ttf", SCORE_VALUE_FONT_SIZE)
except IOError:
    print("Fontes Arial (bold/regular) não encontradas. Usando fontes padrão.")
    FONT_TITLE = ImageFont.load_default()
    FONT_PROFILE_NAME = ImageFont.load_default()
    FONT_DETAIL = ImageFont.load_default()
    FONT_SCORE_LABEL = ImageFont.load_default()
    FONT_SCORE_VALUE = ImageFont.load_default()


# --- Funções de Similaridade (Idênticas) ---
def safe_split_and_strip(text: Optional[str], delimiter: str = ',') -> Set[str]:
    if not text or not isinstance(text, str): return set()
    return {item.strip().lower() for item in text.split(delimiter) if item.strip()}
def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2: return 0.0
    intersection = len(set1.intersection(set2)); union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0
def availability_similarity(avail1_str: Optional[str], avail2_str: Optional[str]) -> float:
    if not avail1_str or not avail2_str: return 0.0
    def simplify_avail(avail_text: str) -> str:
        text = avail_text.lower(); # ... (código completo omitido por brevidade) ...
        if "manhã" in text: return "manha"
        if "tarde" in text: return "tarde"
        if "noite" in text: return "noite"
        if "madrugada" in text: return "madrugada"
        if "fim de semana" in text: return "fds"
        if "dia de semana" in text or "durante a semana" in text: return "semana"
        if "flexível" in text or "qualquer" in text: return "flexivel"
        return "outro"
    a1_simple = simplify_avail(avail1_str); a2_simple = simplify_avail(avail2_str)
    if a1_simple == a2_simple: return 1.0
    if a1_simple == "flexivel" or a2_simple == "flexivel": return 0.7
    if a1_simple == "fds" and a2_simple == "fds": return 0.8
    if a1_simple == "semana" and a2_simple == "semana": return 0.6
    if (a1_simple in ["manha", "tarde", "noite"] and a2_simple == "semana") or \
       (a2_simple in ["manha", "tarde", "noite"] and a1_simple == "semana"): return 0.4
    if (a1_simple in ["manha", "tarde", "noite"] and a2_simple == "fds") or \
       (a2_simple in ["manha", "tarde", "noite"] and a1_simple == "fds"): return 0.2
    if a1_simple == "madrugada" or a2_simple == "madrugada": return 0.1
    return 0.0
def interaction_similarity(inter1: Optional[str], inter2: Optional[str]) -> float:
    if not inter1 or not inter2: return 0.0
    i1 = inter1.lower(); i2 = inter2.lower()
    s1 = set(w.strip() for w in i1.split()); s2 = set(w.strip() for w in i2.split())
    if i1 == i2: return 1.0
    if "indiferente" in s1 or "indiferente" in s2: return 0.5
    if "online" in s1 and "online" in s2: return 0.9
    if "presencial" in s1 and "presencial" in s2: return 0.8
    if ("online" in s1 and "presencial" in s2) or ("presencial" in s1 and "online" in s2): return 0.1
    return 0.2
def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    scores = {}
    p1_id = profile1.get('id', 'P1'); p2_id = profile2.get('id', 'P2')
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
    if scores['plataformas'] < MIN_REQUIRED_PLATFORM_SCORE: return 0.0, {k: round(v, 2) for k,v in scores.items()}
    if scores['disponibilidade'] < MIN_REQUIRED_AVAILABILITY_SCORE: return 0.0, {k: round(v, 2) for k,v in scores.items()}
    total_score = sum(scores[key] * WEIGHTS[key] for key in WEIGHTS if key in scores)
    if total_score < MIN_CUSTOM_SCORE_THRESHOLD: return 0.0, {k: round(v, 2) for k,v in scores.items()}
    return total_score, scores

# --- Funções de Banco de Dados (Idênticas) ---
def load_all_profiles_for_similarity(db_path: str) -> Dict[int, Dict[str, Any]]:
    profiles = {}
    required_fields = ['id', 'plataformas_possuidas', 'disponibilidade',
                       'jogos_favoritos', 'estilos_preferidos', 'interacao_desejada']
    display_fields = ['nome', 'idade', 'cidade', 'estado', 'sexo']
    all_fields = list(set(required_fields + display_fields)); fields_str = ", ".join(all_fields)
    try:
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            conn.row_factory = sqlite3.Row; cursor = conn.cursor()
            cursor.execute(f"SELECT {fields_str} FROM perfis")
            rows = cursor.fetchall()
            for row in rows: profiles[dict(row)['id']] = dict(row)
        print(f"Carregados {len(profiles)} perfis do banco de dados '{db_path}'.")
        return profiles
    except Exception as e: print(f"Erro ao carregar perfis: {e}"); return {}

# --- Função Principal de Busca (Idêntica) ---
def find_top_matches(origin_id: int, all_profiles: Dict[int, Dict[str, Any]], num_matches: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if origin_id not in all_profiles: print(f"Erro: ID origem {origin_id} não encontrado."); return None, []
    origin_profile = all_profiles[origin_id]; matches = []
    print(f"Calculando similaridade para origem: {origin_profile.get('nome', 'N/A')} (ID: {origin_id})...")
    count = 0; total_profiles = len(all_profiles) - 1
    for target_id, target_profile in all_profiles.items():
        if target_id == origin_id: continue
        count += 1; # if count % 500 == 0: print(f"  Processando {count}/{total_profiles}...")
        total_score, score_details = calculate_custom_similarity(origin_profile, target_profile)
        if total_score > 0:
            match_data = target_profile.copy()
            match_data['score_compatibilidade'] = total_score
            match_data['score_details'] = {k: round(v, 2) for k, v in score_details.items()}
            matches.append(match_data)
    matches.sort(key=lambda x: x['score_compatibilidade'], reverse=True)
    print(f"Cálculo concluído. Encontrados {len(matches)} matches válidos.")
    top_matches = matches[:num_matches]
    return origin_profile, top_matches

# --- Funções de Desenho (PIL) - Adaptadas para Retângulos Horizontais Agrupados ---

def get_color_for_score(score: float, min_hue=0.66, max_hue=0.0) -> Tuple[int, int, int]:
    """ Mapeia score (0.0 a 1.0) para cor (Azul -> Vermelho). """
    score = max(0.0, min(1.0, score))
    hue = min_hue + (max_hue - min_hue) * score
    saturation = 0.7 + 0.3 * score
    lightness = 0.5 + 0.1 * score
    rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)
    rgb_int = tuple(int(c * 255) for c in rgb_float)
    return rgb_int

def draw_text(draw: ImageDraw.Draw, text: str, position: Tuple[int, int], font: ImageFont.FreeTypeFont, color: Tuple[int, int, int], anchor="lt"):
    """Desenha texto com âncora."""
    try:
        draw.text(position, text, fill=color, font=font, anchor=anchor)
    except Exception as e:
        print(f"Erro ao desenhar texto '{text}': {e}")
        try: draw.text(position, text, fill=color, anchor=anchor)
        except Exception as e2: print(f"  Erro fallback fonte: {e2}")

def draw_score_horizontal_rects(draw: ImageDraw.Draw, position: Tuple[int, int], score: float, label: str):
    """Desenha uma linha horizontal de 10 retângulos representando o score."""
    x_start, y_start = position # Top-left da área desta visualização

    num_lit_rects = round(score * NUM_RECTS_PER_SCORE)
    num_lit_rects = max(0, min(NUM_RECTS_PER_SCORE, num_lit_rects)) # Clamp 0-10

    lit_color = get_color_for_score(score)

    # Desenha o rótulo ACIMA da linha de retângulos
    label_pos = (x_start + SCORE_VIZ_WIDTH / 2, y_start - LABEL_OFFSET_Y)
    draw_text(draw, label.capitalize(), label_pos, FONT_SCORE_LABEL, COLOR_TEXT_SCORE_LABEL, anchor="mb") # middle-bottom

    # Desenha os retângulos lado a lado
    current_x = x_start
    for i in range(NUM_RECTS_PER_SCORE):
        rect_color = lit_color if i < num_lit_rects else COLOR_RECT_DARK
        top_left = (current_x, y_start)
        bottom_right = (current_x + RECT_WIDTH, y_start + RECT_HEIGHT)
        draw.rectangle([top_left, bottom_right], fill=rect_color, outline=COLOR_SEPARATOR, width=1)
        current_x += RECT_WIDTH + RECT_PADDING

    # Desenha o valor numérico ABAIXO da linha de retângulos
    score_text = f"{score:.2f}"
    value_pos = (x_start + SCORE_VIZ_WIDTH / 2, y_start + SCORE_VIZ_HEIGHT + VALUE_OFFSET_Y)
    draw_text(draw, score_text, value_pos, FONT_SCORE_VALUE, COLOR_TEXT_SCORE_VALUE, anchor="mt") # middle-top

    # Retorna a altura total ocupada por esta visualização (incluindo labels)
    # Útil para calcular o espaçamento vertical na função chamadora
    total_height = LABEL_OFFSET_Y + SCORE_VIZ_HEIGHT + VALUE_OFFSET_Y + FONT_SCORE_VALUE.getbbox("0.00")[3] # Adiciona altura da fonte do valor
    return total_height


def draw_profile_section_horizontal(draw: ImageDraw.Draw, profile_data: Dict[str, Any], position: Tuple[int, int], is_origin: bool = False):
    """Desenha a seção completa para um perfil usando visualizações horizontais agrupadas."""
    x_section, y_section = position
    section_width = IMG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    text_x_start = x_section + 30
    score_viz_x_start = x_section + section_width * 0.35
    score_viz_y_start_row1 = y_section + HEADER_HEIGHT * 0.7

    # Área para o score máximo (à direita)
    max_score_x = x_section + section_width * 0.85  # Posição X do score máximo
    max_score_y = y_section + HEADER_HEIGHT * 0.7   # Alinhado com a primeira linha de scores

    # --- Detalhes do Perfil (Lado Esquerdo) ---
    profile_name = profile_data.get('nome', 'N/A')
    profile_id = profile_data.get('id', 'N/A')
    name_text = f"{profile_name} (ID: {profile_id})"
    draw_text(draw, name_text, (text_x_start, y_section + 40), FONT_PROFILE_NAME, COLOR_TEXT_PROFILE_NAME)

    details_y = y_section + 100
    details_line_height = 45
    detail_text_1 = f"Idade: {profile_data.get('idade', 'N/A')}, Sexo: {profile_data.get('sexo', 'N/A')}"
    detail_text_2 = f"Local: {profile_data.get('cidade', 'N/A')}, {profile_data.get('estado', 'N/A')}"
    detail_text_3 = f"Disponível: {profile_data.get('disponibilidade', 'N/A')}"
    draw_text(draw, detail_text_1, (text_x_start, details_y), FONT_DETAIL, COLOR_TEXT_DETAIL)
    draw_text(draw, detail_text_2, (text_x_start, details_y + details_line_height), FONT_DETAIL, COLOR_TEXT_DETAIL)
    draw_text(draw, detail_text_3, (text_x_start, details_y + 2 * details_line_height), FONT_DETAIL, COLOR_TEXT_DETAIL)

    # --- Visualização de Scores (Lado Direito, Agrupado) ---
    if is_origin:
        score_details = {
            'plataformas': 1.0 if profile_data.get('plataformas_possuidas') else 0.0,
            'disponibilidade': 1.0 if profile_data.get('disponibilidade') else 0.0,
            'jogos': 1.0 if profile_data.get('jogos_favoritos') else 0.0,
            'interacao': 1.0 if profile_data.get('interacao_desejada') else 0.0,
        }
        final_score_text = "Perfil de Origem (Referência)"
        final_score = 1.0
        final_score_color = COLOR_TEXT_PROFILE_NAME
    else:
        final_score = profile_data.get('score_compatibilidade', 0.0)
        final_score_text = f"Score Final: {final_score:.3f}"
        score_details = profile_data.get('score_details', {})
        final_score_color = get_color_for_score(final_score)

    # Ordem dos scores para desenhar (5 elementos + score final)
    score_order = ["plataformas", "disponibilidade", "jogos", "interacao"]

    current_x = score_viz_x_start
    current_y = score_viz_y_start_row1
    max_height_row1 = 0
    max_height_row2 = 0

    # Desenha os primeiros 3 scores na primeira linha
    for i in range(3):
        if i < len(score_order):
            score_key = score_order[i]
            score_value = score_details.get(score_key, 0.0)
            pos_x = score_viz_x_start + (i * (SCORE_VIZ_WIDTH + HORIZONTAL_GROUP_SPACING))
            pos_y = current_y
            height = draw_score_horizontal_rects(draw, (pos_x, pos_y), score_value, score_key)
            max_height_row1 = max(max_height_row1, height)

    # Calcula Y da segunda linha
    current_y = score_viz_y_start_row1 + max_height_row1 + VERTICAL_GROUP_SPACING

    # Desenha o score restante e o score final na segunda linha
    for i in range(2):
        pos_x = score_viz_x_start + (i * (SCORE_VIZ_WIDTH + HORIZONTAL_GROUP_SPACING))
        pos_y = current_y
        
        if i == 0 and 3 < len(score_order):
            score_key = score_order[3]
            score_value = score_details.get(score_key, 0.0)
            height = draw_score_horizontal_rects(draw, (pos_x, pos_y), score_value, score_key)
        elif i == 1:
            height = draw_score_horizontal_rects(draw, (pos_x, pos_y), final_score, "Score Final")
        
        max_height_row2 = max(max_height_row2, height)

    # Desenha o score máximo na área vazia à direita
    if not is_origin:
        # Fonte maior para o score máximo
        max_score_font = ImageFont.truetype("arialbd.ttf", 72)  # Fonte maior para destaque
        max_score_color = get_color_for_score(final_score)
        
        # Desenha o valor do score com fonte grande
        max_score_text = f"{final_score:.3f}"
        draw_text(draw, max_score_text, (max_score_x, max_score_y), max_score_font, max_score_color, anchor="mm")
        
        # Adiciona o rótulo "Score Máximo" acima
        max_score_label = "Score Máximo"
        draw_text(draw, max_score_label, (max_score_x, max_score_y - 50), FONT_SCORE_LABEL, COLOR_TEXT_SCORE_LABEL, anchor="mm")


def generate_similarity_horizontal_viz_image(origin_profile: Dict[str, Any], top_matches: List[Dict[str, Any]], filename: str):
    """Gera a imagem PNG final com visualizações horizontais agrupadas."""

    num_profiles_total = 1 + len(top_matches)
    # Altura estimada: margens + header + N * (altura_secao + espacamento)
    total_height_needed = MARGIN_TOP + HEADER_HEIGHT + (num_profiles_total * PROFILE_SECTION_HEIGHT) + ((num_profiles_total) * PROFILE_SPACING) + MARGIN_BOTTOM

    try:
        image = Image.new('RGB', (IMG_WIDTH, total_height_needed), COLOR_BACKGROUND)
        draw = ImageDraw.Draw(image)

        # Título
        title_text = f"Visualização de Similaridade - Origem ID: {origin_profile.get('id', 'N/A')}"
        title_pos = (IMG_WIDTH / 2, MARGIN_TOP / 2 + 10)
        draw_text(draw, title_text, title_pos, FONT_TITLE, COLOR_TEXT_HEADER, anchor="mm")

        # Desenha seção da Origem
        current_y = MARGIN_TOP + HEADER_HEIGHT
        draw_profile_section_horizontal(draw, origin_profile, (MARGIN_LEFT, current_y), is_origin=True)
        current_y += PROFILE_SECTION_HEIGHT + PROFILE_SPACING

        # Linha separadora abaixo da origem
        draw.line([(MARGIN_LEFT, current_y - PROFILE_SPACING / 2), (IMG_WIDTH - MARGIN_RIGHT, current_y - PROFILE_SPACING / 2)], fill=COLOR_SEPARATOR, width=3)

        # Desenha seções dos Matches
        for i, match_profile in enumerate(top_matches):
            print(f"  Desenhando match {i+1}/{len(top_matches)}: ID {match_profile['id']} (Score: {match_profile['score_compatibilidade']:.3f})")
            draw_profile_section_horizontal(draw, match_profile, (MARGIN_LEFT, current_y), is_origin=False)
            current_y += PROFILE_SECTION_HEIGHT + PROFILE_SPACING

            # Linha separadora entre matches
            if i < len(top_matches) - 1:
                 draw.line([(MARGIN_LEFT + 40, current_y - PROFILE_SPACING / 2), (IMG_WIDTH - MARGIN_RIGHT - 40, current_y - PROFILE_SPACING / 2)], fill=COLOR_SEPARATOR, width=1)

        # Salva a imagem
        image.save(filename)
        print(f"\nImagem de similaridade (horizontal agrupada) salva como: {filename}")

    except Exception as e:
        print(f"Erro ao gerar ou salvar a imagem: {e}")
        import traceback
        traceback.print_exc()


# --- Execução Principal (Idêntica) ---
if __name__ == "__main__":
    print("--- Iniciando Análise e Geração de Imagem (Horizontal Agrupada) ---")

    # 1. Verificar DB
    if not os.path.exists(DATABASE_PROFILES):
        print(f"❌ Erro Crítico: Banco de dados '{DATABASE_PROFILES}' não encontrado."); exit(1)

    # 2. Carregar Perfis
    print("Carregando perfis...")
    all_profiles_data = load_all_profiles_for_similarity(DATABASE_PROFILES)
    if not all_profiles_data: print("❌ Erro: Nenhum perfil carregado."); exit(1)

    # 3. Escolher Origem
    profile_ids = list(all_profiles_data.keys())
    if not profile_ids: print("❌ Erro: Lista de IDs vazia."); exit(1)
    random_origin_id = random.choice(profile_ids)
    print(f"ID de origem selecionado: {random_origin_id}")

    # 4. Encontrar Matches
    origin_profile_details, top_matches_details = find_top_matches(random_origin_id, all_profiles_data, NUM_NEIGHBORS_TARGET)
    if origin_profile_details is None: print(f"❌ Erro: Origem ID {random_origin_id} não carregada."); exit(1)
    if not top_matches_details: print(f"⚠️ Aviso: Nenhum match encontrado para ID {random_origin_id}. Imagem não gerada."); exit(0)

    print(f"\n--- Top {len(top_matches_details)} Matches para ID {random_origin_id} ({origin_profile_details.get('nome', 'N/A')}) ---")
    for i, match in enumerate(top_matches_details): print(f"  {i+1}. ID: {match['id']} ({match.get('nome', 'N/A')}) - Score: {match['score_compatibilidade']:.4f}")

    # 5. Nome do Arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_str_for_hash = f"{random_origin_id}-{timestamp}"
    file_hash = hashlib.sha1(input_str_for_hash.encode()).hexdigest()[:8]
    # Adiciona "_horizontal" ao nome
    output_filename = os.path.join(OUTPUT_DIR, f"similarity_viz_horizontal_origin_{random_origin_id}_{timestamp}_{file_hash}.png")

    # 6. Gerar Imagem
    print("\nGerando imagem de visualização (horizontal agrupada)...")
    generate_similarity_horizontal_viz_image(origin_profile_details, top_matches_details, output_filename)

    print("\n--- Processo Concluído ---")