# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
# import faiss # Não necessário para este script, pois calcularemos similaridade customizada
import json
import datetime
import hashlib
import os
import random
import logging
from typing import Tuple, List, Dict, Set, Optional, Any
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import colorsys # Para interpolação de cores mais suave

# --- Configurações ---
DB_DIR = "databases_v3"
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
# DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db') # Não usado diretamente aqui
OUTPUT_DIR = "img-data-outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constantes de Similaridade (IMPORTANTE: Devem ser iguais às do Flask App)
NUM_NEIGHBORS_TARGET: int = 10
MIN_CUSTOM_SCORE_THRESHOLD: float = 0.05 # Score final mínimo para ser considerado um match
WEIGHTS = {
    "plataformas": 0.45, "disponibilidade": 0.35, "jogos": 0.10,
    "estilos": 0.05, "interacao": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Pesos devem somar 1"
MIN_REQUIRED_PLATFORM_SCORE: float = 0.20
MIN_REQUIRED_AVAILABILITY_SCORE: float = 0.30

# Configurações da Imagem
IMG_WIDTH = 3000  # Largura alvo (tipo 4K)
MARGIN_TOP = 100
MARGIN_BOTTOM = 100
MARGIN_LEFT = 50
MARGIN_RIGHT = 50
PROFILE_SECTION_HEIGHT = 200 # Altura base para cada perfil (origem + 10 matches)
PROFILE_SPACING = 30 # Espaçamento vertical entre seções de perfil
HEADER_HEIGHT = 80
TITLE_FONT_SIZE = 48
PROFILE_NAME_FONT_SIZE = 36
DETAIL_FONT_SIZE = 24
SCORE_LABEL_FONT_SIZE = 18
SCORE_VALUE_FONT_SIZE = 28
BAR_HEIGHT = 40
BAR_WIDTH = 300 # Largura da barra de score
BAR_SPACING = 20 # Espaçamento horizontal entre barras de score
BAR_LABEL_OFFSET_Y = 5 # Offset para o texto acima da barra
BAR_VALUE_OFFSET_Y = 10 # Offset para o valor dentro/abaixo da barra

# Cores
COLOR_BACKGROUND = (25, 25, 30) # Fundo escuro
COLOR_TEXT_HEADER = (230, 230, 230)
COLOR_TEXT_PROFILE_NAME = (210, 210, 220)
COLOR_TEXT_DETAIL = (180, 180, 190)
COLOR_TEXT_SCORE_LABEL = (150, 150, 160)
COLOR_TEXT_SCORE_VALUE = (255, 255, 255)
COLOR_SEPARATOR = (60, 60, 70)

# Fontes (Tente usar uma fonte comum ou ajuste o caminho)
try:
    # Tente usar fontes específicas se disponíveis
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


# --- Funções de Similaridade (Copiadas/Adaptadas do Flask App) ---

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
        text = avail_text.lower()
        if "manhã" in text: return "manha"
        if "tarde" in text: return "tarde"
        if "noite" in text: return "noite"
        if "madrugada" in text: return "madrugada"
        if "fim de semana" in text: return "fds"
        if "dia de semana" in text or "durante a semana" in text: return "semana"
        if "flexível" in text or "qualquer" in text: return "flexivel"
        return "outro"
    a1_simple = simplify_avail(avail1_str)
    a2_simple = simplify_avail(avail2_str)
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
    s1 = set(w.strip() for w in i1.split())
    s2 = set(w.strip() for w in i2.split())
    if i1 == i2: return 1.0
    if "indiferente" in s1 or "indiferente" in s2: return 0.5
    if "online" in s1 and "online" in s2: return 0.9
    if "presencial" in s1 and "presencial" in s2: return 0.8
    if ("online" in s1 and "presencial" in s2) or ("presencial" in s1 and "online" in s2): return 0.1
    return 0.2

def calculate_custom_similarity(profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Calcula similaridade customizada. Retorna 0.0 se thresholds não forem atingidos."""
    scores = {}
    p1_id = profile1.get('id', 'P1')
    p2_id = profile2.get('id', 'P2')

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

    # VERIFICAÇÃO DOS THRESHOLDS OBRIGATÓRIOS
    if scores['plataformas'] < MIN_REQUIRED_PLATFORM_SCORE:
        return 0.0, {k: round(v, 2) for k,v in scores.items()}

    if scores['disponibilidade'] < MIN_REQUIRED_AVAILABILITY_SCORE:
        return 0.0, {k: round(v, 2) for k,v in scores.items()}

    # Cálculo do Score Ponderado Final
    total_score = sum(scores[key] * WEIGHTS[key] for key in WEIGHTS if key in scores)

    # Verifica threshold mínimo GERAL
    if total_score < MIN_CUSTOM_SCORE_THRESHOLD:
        return 0.0, {k: round(v, 2) for k,v in scores.items()} # Retorna 0 se abaixo do mínimo geral

    return total_score, scores

# --- Funções de Banco de Dados ---

def load_all_profiles_for_similarity(db_path: str) -> Dict[int, Dict[str, Any]]:
    """Carrega todos os perfis com os campos necessários para similaridade."""
    profiles = {}
    required_fields = ['id', 'plataformas_possuidas', 'disponibilidade',
                       'jogos_favoritos', 'estilos_preferidos', 'interacao_desejada']
    # Incluir outros campos que queremos exibir na imagem
    display_fields = ['nome', 'idade', 'cidade', 'estado', 'sexo']
    all_fields = list(set(required_fields + display_fields))
    fields_str = ", ".join(all_fields)

    try:
        with sqlite3.connect(db_path, timeout=20.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"SELECT {fields_str} FROM perfis")
            rows = cursor.fetchall()
            for row in rows:
                profile_dict = dict(row)
                profiles[profile_dict['id']] = profile_dict
        print(f"Carregados {len(profiles)} perfis do banco de dados '{db_path}'.")
        return profiles
    except sqlite3.Error as e:
        print(f"Erro ao carregar perfis do banco de dados {db_path}: {e}")
        return {}
    except Exception as e:
        print(f"Erro inesperado ao carregar perfis: {e}")
        return {}

# --- Função Principal de Busca ---

def find_top_matches(origin_id: int, all_profiles: Dict[int, Dict[str, Any]], num_matches: int) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Encontra os N perfis mais similares ao ID de origem."""
    if origin_id not in all_profiles:
        print(f"Erro: ID de origem {origin_id} não encontrado nos perfis carregados.")
        return None, []

    origin_profile = all_profiles[origin_id]
    matches = []

    print(f"Calculando similaridade para o perfil de origem: {origin_profile.get('nome', 'N/A')} (ID: {origin_id})...")
    count = 0
    total_profiles = len(all_profiles) - 1
    for target_id, target_profile in all_profiles.items():
        if target_id == origin_id:
            continue

        count += 1
        if count % 500 == 0:
             print(f"  Processando {count}/{total_profiles}...")

        total_score, score_details = calculate_custom_similarity(origin_profile, target_profile)

        if total_score > 0: # Já filtra pelos thresholds dentro da função calculate_custom_similarity
            match_data = target_profile.copy() # Copia para não modificar o dict original
            match_data['score_compatibilidade'] = total_score
            match_data['score_details'] = {k: round(v, 2) for k, v in score_details.items()}
            matches.append(match_data)

    # Ordena os matches pelo score final (decrescente)
    matches.sort(key=lambda x: x['score_compatibilidade'], reverse=True)

    print(f"Cálculo de similaridade concluído. Encontrados {len(matches)} matches válidos.")

    top_matches = matches[:num_matches]

    return origin_profile, top_matches

# --- Funções de Desenho (PIL) ---

def get_color_for_score(score: float, min_hue=0.66, max_hue=0.0) -> Tuple[int, int, int]:
    """ Mapeia um score (0.0 a 1.0) para uma cor (Azul -> Verde -> Amarelo -> Vermelho). """
    if not 0.0 <= score <= 1.0:
        score = max(0.0, min(1.0, score)) # Garante que está no range

    # Interpola a matiz (Hue) de Azul (0.66) para Vermelho (0.0)
    # Passando por Ciano, Verde, Amarelo
    hue = min_hue + (max_hue - min_hue) * score
    # Ajusta Saturation e Lightness para ficarem vibrantes
    # Saturação aumenta um pouco com o score, Lightness é média/alta
    saturation = 0.7 + 0.3 * score
    lightness = 0.5 + 0.1 * score # Evita branco puro ou preto puro

    # Converte HSL para RGB
    rgb_float = colorsys.hls_to_rgb(hue, lightness, saturation)
    # Converte de float (0-1) para int (0-255)
    rgb_int = tuple(int(c * 255) for c in rgb_float)
    return rgb_int

def draw_text(draw: ImageDraw.Draw, text: str, position: Tuple[int, int], font: ImageFont.FreeTypeFont, color: Tuple[int, int, int], anchor="lt"):
    """Desenha texto com a âncora especificada (default: left-top)."""
    try:
        draw.text(position, text, fill=color, font=font, anchor=anchor)
    except Exception as e:
        print(f"Erro ao desenhar texto '{text}': {e}")
        # Tenta desenhar com fonte padrão como fallback
        try:
            draw.text(position, text, fill=color, anchor=anchor)
        except Exception as e2:
             print(f"  Erro ao desenhar com fonte padrão: {e2}")


def draw_score_bar(draw: ImageDraw.Draw, position: Tuple[int, int], score: float, label: str):
    """Desenha uma barra colorida representando o score."""
    x, y = position
    bar_color = get_color_for_score(score)
    text_color = COLOR_TEXT_SCORE_LABEL
    value_color = COLOR_TEXT_SCORE_VALUE if score > 0.3 else COLOR_TEXT_DETAIL # Cor do valor mais clara

    # Desenha a barra
    draw.rectangle(
        [x, y, x + BAR_WIDTH, y + BAR_HEIGHT],
        fill=bar_color,
        outline=COLOR_SEPARATOR, # Adiciona um contorno sutil
        width=1
    )

    # Desenha o rótulo acima da barra
    label_pos = (x + BAR_WIDTH / 2, y - BAR_LABEL_OFFSET_Y)
    draw_text(draw, label.capitalize(), label_pos, FONT_SCORE_LABEL, text_color, anchor="mb") # middle-bottom anchor

    # Desenha o valor do score dentro/abaixo da barra
    score_text = f"{score:.2f}"
    value_pos = (x + BAR_WIDTH / 2, y + BAR_HEIGHT / 2)
    draw_text(draw, score_text, value_pos, FONT_SCORE_VALUE, value_color, anchor="mm") # middle-middle anchor


def draw_profile_section(draw: ImageDraw.Draw, profile_data: Dict[str, Any], position: Tuple[int, int], is_origin: bool = False):
    """Desenha a seção completa para um perfil (origem ou match)."""
    x, y = position
    section_width = IMG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    text_x_start = x + 20
    bar_x_start = x + section_width * 0.4 # Início das barras de score (ajuste conforme necessário)
    bar_y_start = y + 70 # Posição Y inicial das barras

    # Nome e ID
    profile_name = profile_data.get('nome', 'N/A')
    profile_id = profile_data.get('id', 'N/A')
    name_text = f"{profile_name} (ID: {profile_id})"
    draw_text(draw, name_text, (text_x_start, y + 15), FONT_PROFILE_NAME, COLOR_TEXT_PROFILE_NAME)

    # Outros Detalhes (Idade, Local, etc.) - Coluna Esquerda
    details_y = y + 60
    details_line_height = 35
    detail_text_1 = f"Idade: {profile_data.get('idade', 'N/A')}, Sexo: {profile_data.get('sexo', 'N/A')}"
    detail_text_2 = f"Local: {profile_data.get('cidade', 'N/A')}, {profile_data.get('estado', 'N/A')}"
    detail_text_3 = f"Disponível: {profile_data.get('disponibilidade', 'N/A')}"
    detail_text_4 = f"Interação: {profile_data.get('interacao_desejada', 'N/A')}"

    draw_text(draw, detail_text_1, (text_x_start, details_y), FONT_DETAIL, COLOR_TEXT_DETAIL)
    draw_text(draw, detail_text_2, (text_x_start, details_y + details_line_height), FONT_DETAIL, COLOR_TEXT_DETAIL)
    draw_text(draw, detail_text_3, (text_x_start, details_y + 2 * details_line_height), FONT_DETAIL, COLOR_TEXT_DETAIL)
    draw_text(draw, detail_text_4, (text_x_start, details_y + 3 * details_line_height), FONT_DETAIL, COLOR_TEXT_DETAIL)

    # Barras de Score (Coluna Direita)
    score_details = profile_data.get('score_details', {})
    if is_origin:
        # Para a origem, calculamos scores contra si mesmo para ter uma linha base (maioria será 1.0)
        self_score, self_score_details = calculate_custom_similarity(profile_data, profile_data)
        score_details = {k: round(v, 2) for k, v in self_score_details.items()}
        final_score_text = "Perfil de Origem"
    else:
        final_score = profile_data.get('score_compatibilidade', 0.0)
        final_score_text = f"Score Final: {final_score:.3f}"

    # Desenha o texto do score final (ou "Perfil de Origem")
    score_text_pos = (bar_x_start, y + 25) # Posição do score final
    draw_text(draw, final_score_text, score_text_pos, FONT_PROFILE_NAME, COLOR_TEXT_SCORE_VALUE if not is_origin else COLOR_TEXT_PROFILE_NAME)


    # Desenha as barras para cada componente de score
    current_bar_x = bar_x_start
    score_order = ["plataformas", "disponibilidade", "jogos", "estilos", "interacao"] # Ordem fixa para comparação
    for score_key in score_order:
        if score_key in score_details:
            score_value = score_details[score_key]
            draw_score_bar(draw, (current_bar_x, bar_y_start), score_value, score_key)
            current_bar_x += BAR_WIDTH + BAR_SPACING


def generate_similarity_image(origin_profile: Dict[str, Any], top_matches: List[Dict[str, Any]], filename: str):
    """Gera a imagem PNG final."""

    num_profiles_total = 1 + len(top_matches)
    total_height_needed = MARGIN_TOP + HEADER_HEIGHT + (num_profiles_total * PROFILE_SECTION_HEIGHT) + ((num_profiles_total - 1) * PROFILE_SPACING) + MARGIN_BOTTOM

    try:
        image = Image.new('RGB', (IMG_WIDTH, total_height_needed), COLOR_BACKGROUND)
        draw = ImageDraw.Draw(image)

        # Título
        title_text = f"Análise de Similaridade - Origem ID: {origin_profile.get('id', 'N/A')}"
        title_pos = (IMG_WIDTH / 2, MARGIN_TOP / 2)
        draw_text(draw, title_text, title_pos, FONT_TITLE, COLOR_TEXT_HEADER, anchor="mm")

        # Desenha seção da Origem
        current_y = MARGIN_TOP + HEADER_HEIGHT
        draw_profile_section(draw, origin_profile, (MARGIN_LEFT, current_y), is_origin=True)
        current_y += PROFILE_SECTION_HEIGHT

        # Linha separadora
        draw.line([(MARGIN_LEFT, current_y + PROFILE_SPACING / 2), (IMG_WIDTH - MARGIN_RIGHT, current_y + PROFILE_SPACING / 2)], fill=COLOR_SEPARATOR, width=2)
        current_y += PROFILE_SPACING


        # Desenha seções dos Matches
        for i, match_profile in enumerate(top_matches):
            print(f"  Desenhando match {i+1}/{len(top_matches)}: ID {match_profile['id']} (Score: {match_profile['score_compatibilidade']:.3f})")
            draw_profile_section(draw, match_profile, (MARGIN_LEFT, current_y), is_origin=False)
            current_y += PROFILE_SECTION_HEIGHT + PROFILE_SPACING

            # Linha separadora (opcional entre matches)
            if i < len(top_matches) - 1:
                 draw.line([(MARGIN_LEFT + 20, current_y - PROFILE_SPACING / 2), (IMG_WIDTH - MARGIN_RIGHT - 20, current_y - PROFILE_SPACING/2)], fill=COLOR_SEPARATOR, width=1)


        # Salva a imagem
        image.save(filename)
        print(f"\nImagem de similaridade salva como: {filename}")

    except Exception as e:
        print(f"Erro ao gerar ou salvar a imagem: {e}")
        import traceback
        traceback.print_exc()


# --- Execução Principal ---
if __name__ == "__main__":
    print("--- Iniciando Análise de Similaridade e Geração de Imagem ---")

    # 1. Verificar se o banco de dados existe
    if not os.path.exists(DATABASE_PROFILES):
        print(f"❌ Erro Crítico: Banco de dados de perfis não encontrado em '{DATABASE_PROFILES}'.")
        print("   Certifique-se que 'profile_generator_v3.py' foi executado e o DB está no lugar correto.")
        exit(1)

    # 2. Carregar todos os perfis
    print("Carregando perfis do banco de dados...")
    all_profiles_data = load_all_profiles_for_similarity(DATABASE_PROFILES)

    if not all_profiles_data:
        print("❌ Erro: Nenhum perfil foi carregado do banco de dados. Abortando.")
        exit(1)

    # 3. Escolher ID de origem aleatório
    profile_ids = list(all_profiles_data.keys())
    if not profile_ids:
        print("❌ Erro: Lista de IDs de perfis está vazia. Abortando.")
        exit(1)

    random_origin_id = random.choice(profile_ids)
    print(f"ID de origem selecionado aleatoriamente: {random_origin_id}")

    # 4. Encontrar os top N matches
    origin_profile_details, top_matches_details = find_top_matches(random_origin_id, all_profiles_data, NUM_NEIGHBORS_TARGET)

    if origin_profile_details is None:
        print(f"❌ Erro: Não foi possível carregar detalhes do perfil de origem ID {random_origin_id}. Abortando.")
        exit(1)

    if not top_matches_details:
        print(f"⚠️ Aviso: Nenhum perfil similar encontrado para o ID {random_origin_id} que atenda aos critérios.")
        # Poderíamos gerar uma imagem apenas com a origem, ou simplesmente sair.
        # Vamos sair por enquanto para evitar imagem vazia.
        print("   Nenhuma imagem será gerada.")
        exit(0)

    print(f"\n--- Top {len(top_matches_details)} Matches Encontrados para ID {random_origin_id} ({origin_profile_details.get('nome', 'N/A')}) ---")
    for i, match in enumerate(top_matches_details):
        print(f"  {i+1}. ID: {match['id']} ({match.get('nome', 'N/A')}) - Score: {match['score_compatibilidade']:.4f}")
        # print(f"     Detalhes: {match['score_details']}") # Opcional: descomentar para ver detalhes no console

    # 5. Gerar o nome do arquivo
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Criar um hash simples baseado no ID de origem e timestamp para alguma unicidade
    input_str_for_hash = f"{random_origin_id}-{timestamp}"
    file_hash = hashlib.sha1(input_str_for_hash.encode()).hexdigest()[:8]
    output_filename = os.path.join(OUTPUT_DIR, f"similarity_map_origin_{random_origin_id}_{timestamp}_{file_hash}.png")

    # 6. Gerar a imagem
    print("\nGerando imagem de visualização...")
    generate_similarity_image(origin_profile_details, top_matches_details, output_filename)

    print("\n--- Processo Concluído ---")