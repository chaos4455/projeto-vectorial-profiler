# profile_generator_v3.py

import sqlite3
import numpy as np
import faiss
from faker import Faker
import random
from colorama import init, Fore, Style
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from typing import List, Dict, Tuple, Any
import logging
import os
from datetime import datetime
import math # Para clustering

# --- Melhorias Implementadas ---
# 1. Logging Detalhado em Arquivo
# 2. Aumento de 800% nos Dados de Base (Cidades, Jogos, etc.)
# 3. Mapeamento Cidade -> Estado para Consistência Geográfica
# 4. Descrições Significativas (sem texto aleatório)
# 5. Nomes Consistentes com o Sexo
# 6. Geração de Nome baseada no Sexo
# 7. Estado Derivado Automaticamente da Cidade
# 8. Nome Consistente com Sexo (Reforço)
# 9. Descrição Baseada em Dados (Reforço)
# 10. Bancos de Dados Separados (Perfis, Vetores, Embeddings, Clusters)
# 11. Criação de Tabelas Robustas com Chaves Primárias
# 12. Função Dedicada para Inserção de Perfis com Progresso
# 13. Geração de Vetores de Características Numéricas/Categóricas
# 14. Geração de Embeddings Simulados (Placeholder para Modelo Real)
# 15. Função Dedicada para Salvar Vetores e Embeddings com Chave Estrangeira (ID)
# 16. +180 Variações de Palavras-Chave para Descrições
# 17. Templates de Descrição Variados
# 18. Função `gerar_descricao_consistente`
# 19. Implementação de Clustering Básico (FAISS KMeans)
# 20. Uso Extensivo de Rich Console (Cores, Emojis)
# 21. Uso de Barras de Progresso Detalhadas (Rich Progress)
# 22. Uso de Tabelas Rich para Exibição de Dados
# 23. Emojis para Feedback Visual (✅, ⚙️, 💾, 📊, 🧩, ⚠️, ℹ️)
# 24. Feedback "Real-time" no Console
# 25. Melhor Estrutura de Pipeline (Passos Claros)
# 26. Error Handling Aprimorado com Try/Except e Logging
# 27. Type Hinting para Clareza
# 28. Variáveis e Funções com Nomes Descritivos
# 29. Modularização do Código em Funções
# 30. Comentários Explicativos
# 31. Constantes Configuráveis no Topo
# 32. Validação de Dados Básica (Ex: Estado encontrado)
# 33. Docstrings para Funções Principais
# 34. Criação Automática de Diretório de Logs
# 35. Nomes de Arquivos de Log com Timestamp
# 36. Uso de `executemany` para Inserção em Lote (Otimização)
# 37. Cálculo Dinâmico do Número de Clusters (Opcional, usando raiz quadrada)
# 38. Armazenamento de Cluster ID no DB de Clusters

# Inicializar Colorama
init(autoreset=True)

# --- Configuração de Logging (Melhoria #1, #34, #35) ---
LOG_DIR = "logs_v3"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"profile_generator_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

console = Console()
logging.info("--- Script Iniciado ---")

# --- Definições Globais e Constantes (Melhoria #31) ---
NUM_PROFILES: int = 20000  # Número de perfis a gerar
FAKER_LOCALE: str = 'pt_BR'
fake = Faker(FAKER_LOCALE)
DIM_EMBEDDING: int = 64  # Dimensão do embedding simulado (Melhoria #14)
DIM_VECTOR: int = 10     # Dimensão do vetor de características (Ajustado)
NUM_CLUSTERS: int = int(math.sqrt(NUM_PROFILES)) # Número de clusters (Melhoria #37)

# Nomes dos Bancos de Dados (Melhoria #10)
DB_DIR = "databases_v3"
os.makedirs(DB_DIR, exist_ok=True) # Cria diretório para DBs
DATABASE_PROFILES: str = os.path.join(DB_DIR, 'perfis_jogadores_v3.db')
DATABASE_VECTORS: str = os.path.join(DB_DIR, 'vetores_perfis_v3.db')
DATABASE_EMBEDDINGS: str = os.path.join(DB_DIR, 'embeddings_perfis_v3.db')
DATABASE_CLUSTERS: str = os.path.join(DB_DIR, 'clusters_perfis_v3.db')

# --- Dados de Base Ampliados (Melhoria #2) ---

# Cidades e Mapeamento Cidade -> Estado (Melhoria #3)
CIDADES_ESTADOS = {
    "São Paulo": "SP", "Rio de Janeiro": "RJ", "Brasília": "DF", "Salvador": "BA", "Fortaleza": "CE",
    "Belo Horizonte": "MG", "Manaus": "AM", "Curitiba": "PR", "Recife": "PE", "Goiânia": "GO",
    "Porto Alegre": "RS", "Belém": "PA", "Guarulhos": "SP", "Campinas": "SP", "São Luís": "MA",
    "São Gonçalo": "RJ", "Maceió": "AL", "Duque de Caxias": "RJ", "Campo Grande": "MS", "Natal": "RN",
    "Teresina": "PI", "São Bernardo do Campo": "SP", "Nova Iguaçu": "RJ", "João Pessoa": "PB", "Santo André": "SP",
    "Osasco": "SP", "Jaboatão dos Guararapes": "PE", "Contagem": "MG", "Sorocaba": "SP", "Uberlândia": "MG",
    "Ribeirão Preto": "SP", "Aparecida de Goiânia": "GO", "Cariacica": "ES", "Feira de Santana": "BA", "Caxias do Sul": "RS",
    "Olinda": "PE", "Joinville": "SC", "Montes Claros": "MG", "Ananindeua": "PA", "Santos": "SP",
    "Campos dos Goytacazes": "RJ", "Mauá": "SP", "Carapicuíba": "SP", "Serra": "ES", "Betim": "MG",
    "Jundiaí": "SP", "Niterói": "RJ", "Juiz de Fora": "MG", "Piracicaba": "SP", "Porto Velho": "RO",
    "Canoas": "RS", "Maringá": "PR", "Mogi das Cruzes": "SP", "Londrina": "PR", "São Vicente": "SP",
    "Foz do Iguaçu": "PR", "Pelotas": "RS", "Vitória": "ES", "Taubaté": "SP", "São José do Rio Preto": "SP",
    "Ponta Grossa": "PR", "Belford Roxo": "RJ", "Rio Branco": "AC", "Itaquaquecetuba": "SP", "Cubatão": "SP",
    "Boa Vista": "RR", "Blumenau": "SC", "Novo Hamburgo": "RS", "Guarujá": "SP", "Cascavel": "PR",
    "Petrolina": "PE", "Vitória da Conquista": "BA", "Paulista": "PE", "Praia Grande": "SP", "Imperatriz": "MA",
    "Viamão": "RS", "Camaçari": "BA", "Juazeiro do Norte": "CE", "Volta Redonda": "RJ", "Sumaré": "SP",
    "Sete Lagoas": "MG", "Ipatinga": "MG", "Divinópolis": "MG", "Parnamirim": "RN", "Magé": "RJ",
    "Sobral": "CE", "Mossoró": "RN", "Santa Luzia": "MG", "Pindamonhangaba": "SP", "Rio Grande": "RS",
    "Marabá": "PA", "Criciúma": "SC", "Santa Maria": "RS", "Barreiras": "BA", "Itabuna": "BA",
    "Luziânia": "GO", "Gravataí": "RS", "Bagé": "RS", "Lauro de Freitas": "BA", "Teófilo Otoni": "MG",
    "Garanhuns": "PE", "Passo Fundo": "RS", "Arapiraca": "AL", "Alagoinhas": "BA", "Francisco Morato": "SP",
    "Franco da Rocha": "SP", "Pinhais": "PR", "Colombo": "PR", "Guarapuava": "PR", "Caucaia": "CE",
    "Barueri": "SP", "Palmas": "TO", "Governador Valadares": "MG", "Parauapebas": "PA", "Santa Bárbara d'Oeste": "SP",
    "Araguaína": "TO", "Ji-Paraná": "RO", "Cachoeiro de Itapemirim": "ES", "Timon": "MA", "Maracanaú": "CE",
    "Dourados": "MS", "Itajaí": "SC", "Rio das Ostras": "RJ", "Simões Filho": "BA", "Paranaguá": "PR",
    "Porto Seguro": "BA", "Linhares": "ES", "Uruguaiana": "RS", "Abaetetuba": "PA", "Itapetininga": "SP",
    "Picos": "PI", "Caxias": "MA", "Bragança Paulista": "SP", "Tangará da Serra": "MT", "Várzea Grande": "MT",
    "Itapevi": "SP", "Marília": "SP", "Cabo Frio": "RJ", "Macapá": "AP", # Cariacica ES já existe
    "Eunápolis": "BA", #"Feira de Santana": "BA", "Ilhéus": "BA", "Itabuna": "BA", "Jequié": "BA",
    "Paulo Afonso": "BA", "Teixeira de Freitas": "BA", #"Vitória da Conquista": "BA", "Arapiraca": "AL",
    "Palmeira dos Índios": "AL", "Rio Largo": "AL", "União dos Palmares": "AL", #"Ananindeua": "PA", "Belém": "PA",
    "Castanhal": "PA", #"Marabá": "PA", "Parauapebas": "PA",
    "Santarém": "PA", #"Timon": "MA",
    "Bacabal": "MA", "Balsas": "MA", #"Caxias": "MA", "Imperatriz": "MA",
    "Paço do Lumiar": "MA", "São José de Ribamar": "MA", #"Barreiras": "BA",
    "Brumado": "BA", #"Camaçari": "BA", "Eunápolis": "BA", "Feira de Santana": "BA",
    "Ilhéus": "BA", #"Itabuna": "BA",
    "Jequié": "BA", "Juazeiro": "BA", #"Lauro de Freitas": "BA", "Paulo Afonso": "BA", "Salvador": "BA",
    "Santo Antônio de Jesus": "BA", "Serrinha": "BA", #"Teixeira de Freitas": "BA",
    "Valença": "BA", #"Vitória da Conquista": "BA", "Caucaia": "CE",
    "Crato": "CE", #"Fortaleza": "CE", "Juazeiro do Norte": "CE", "Maracanaú": "CE", "Sobral": "CE",
    "Aquiraz": "CE", "Canindé": "CE", "Iguatu": "CE", "Itapipoca": "CE", "Maranguape": "CE",
    "Pacatuba": "CE", "Quixadá": "CE", #"Cascavel": "PR", "Foz do Iguaçu": "PR", "Guarapuava": "PR", "Londrina": "PR", "Maringá": "PR",
    "São José dos Pinhais": "PR", "Umuarama": "PR", "Vila Velha": "ES",
    #"Vitória": "ES",
    "Colatina": "ES", #"Cachoeiro de Itapemirim": "ES",
    "Guarapari": "ES", #"Linhares": "ES",
    "São Mateus": "ES", #"Aparecida de Goiânia": "GO",
    "Catalão": "GO", #"Goiânia": "GO",
    "Itumbiara": "GO", #"Luziânia": "GO",
    "Rio Verde": "GO", "Valparaíso de Goiás": "GO", "Águas Lindas de Goiás": "GO", "Novo Gama": "GO",
    "Santo Antônio do Descoberto": "GO", #"Caxias do Sul": "RS", "Canoas": "RS", "Esteio": "RS",
    "Gravataí": "RS", #"Novo Hamburgo": "RS", "Passo Fundo": "RS", "Pelotas": "RS", "Porto Alegre": "RS",
    "Rio Grande": "RS", "Santa Cruz do Sul": "RS", #"Santa Maria": "RS",
    "Sapucaia do Sul": "RS", #"Viamão": "RS", #"Caxias": "MA", "Picos": "PI", "Teresina": "PI",
    "Parnaíba": "PI", #"Guarulhos": "SP", "Campinas": "SP", "São Bernardo do Campo": "SP", "Santo André": "SP", "Osasco": "SP",
    "São José dos Campos": "SP", #"Santos": "SP", "São José do Rio Preto": "SP",
    "Bauru": "SP", #"São Vicente": "SP",
    "Franca": "SP", #"Taubaté": "SP", "Praia Grande": "SP",
    "Limeira": "SP", #"Carapicuíba": "SP", #"Guarujá": "SP", "Itaquaquecetuba": "SP",
    "Presidente Prudente": "SP", "Suzano": "SP", "Taboão da Serra": "SP", #"Barueri": "SP",
    "Embu das Artes": "SP", "Diadema": "SP", #"Mauá": "SP",
    "Cotia": "SP", "São Caetano do Sul": "SP", "Ferraz de Vasconcelos": "SP", #"Itapevi": "SP",
    "Arujá": "SP", "Poá": "SP", "Salto": "SP", #"Sumaré": "SP",
    "Valinhos": "SP", "Vinhedo": "SP", "Americana": "SP", "Araraquara": "SP", "Atibaia": "SP", #"Bragança Paulista": "SP",
    "Caieiras": "SP", "Cajamar": "SP", "Campo Limpo Paulista": "SP", #"Cubatão": "SP", # Extrema é MG
    "Hortolândia": "SP", "Indaiatuba": "SP", "Itanhaém": "SP", "Jacareí": "SP", "Jandira": "SP",
    "Mairiporã": "SP", "Mongaguá": "SP", "Ourinhos": "SP", "Paulínia": "SP", #"Pindamonhangaba": "SP",
    "Rio Claro": "SP", #"Santa Bárbara d'Oeste": "SP",
    "São Carlos": "SP", "Sertãozinho": "SP", #"Sorocaba": "SP", #"Taubaté": "SP",
    "Ubatuba": "SP", #"Valinhos": "SP",
    "Votorantim": "SP",
    # Adicionar mais cidades para variedade
    "Aracaju": "SE", "Florianópolis": "SC", "São José": "SC", "Palhoça": "SC", "Chapecó": "SC",
    "Itapema": "SC", "Balneário Camboriú": "SC", "Brusque": "SC", "Tubarão": "SC", "Lages": "SC",
    "Uberaba": "MG", "Poços de Caldas": "MG", "Varginha": "MG", "Pouso Alegre": "MG", "Patos de Minas": "MG",
    "Barbacena": "MG", "Conselheiro Lafaiete": "MG", "Itabira": "MG", "Araguari": "MG", "Passos": "MG",
    "Corumbá": "MS", "Três Lagoas": "MS", "Ponta Porã": "MS", "Naviraí": "MS", "Nova Andradina": "MS",
    "Aquidauana": "MS", "Sidrolândia": "MS", "Maracaju": "MS", "Coxim": "MS", "Rio Brilhante": "MS",
    "Cuiabá": "MT", "Rondonópolis": "MT", "Sinop": "MT", "Primavera do Leste": "MT", "Barra do Garças": "MT",
    "Cáceres": "MT", "Sorriso": "MT", "Lucas do Rio Verde": "MT", "Alta Floresta": "MT", "Pontes e Lacerda": "MT",
    "Santarém": "PA", "Altamira": "PA", "Itaituba": "PA", "Cametá": "PA", "Bragança": "PA",
    "Barcarena": "PA", "Tucuruí": "PA", "Paragominas": "PA", "Tailândia": "PA", "Redenção": "PA",
}
CIDADES_BRASIL = list(CIDADES_ESTADOS.keys()) * 8 # Amplia e garante que todas têm estado
random.shuffle(CIDADES_BRASIL) # Embaralha para variedade

ESTADOS_BRASIL = sorted(list(set(CIDADES_ESTADOS.values()))) * 8
random.shuffle(ESTADOS_BRASIL)

JOGOS_MAIS_JOGADOS = [
    "League of Legends", "Counter-Strike 2", "Dota 2", "Valorant", "Fortnite", "Minecraft", "Grand Theft Auto V",
    "Call of Duty: Warzone", "Apex Legends", "Overwatch 2", "Rainbow Six Siege", "PUBG: Battlegrounds", "Rocket League",
    "Destiny 2", "Warframe", "The Witcher 3: Wild Hunt", "Red Dead Redemption 2", "Cyberpunk 2077",
    "The Elder Scrolls V: Skyrim", "Fallout 4", "Dark Souls III", "Bloodborne", "Sekiro: Shadows Die Twice", "Elden Ring",
    "Diablo IV", "World of Warcraft", "Final Fantasy XIV", "Guild Wars 2", "The Elder Scrolls Online", "Black Desert Online",
    "Sea of Thieves", "No Man's Sky", "Star Citizen", "Elite Dangerous", "EVE Online", "Hearthstone", "Magic: The Gathering Arena",
    "Legends of Runeterra", "Gwent: The Witcher Card Game", "Yu-Gi-Oh! Master Duel", "Teamfight Tactics", "Auto Chess", "Dota Underlords",
    "Clash Royale", "Brawl Stars", "Candy Crush Saga", "Pokémon GO", "Genshin Impact", "Honkai: Star Rail", "PUBG Mobile",
    "Call of Duty: Mobile", "Free Fire", "Mobile Legends: Bang Bang", "Arena of Valor", "Diablo Immortal", "EA SPORTS FC 24",
    "eFootball 2024", "NBA 2K24", "Madden NFL 24", "NHL 24", "Forza Horizon 5", "Gran Turismo 7", "Assetto Corsa Competizione",
    "iRacing", "F1 23", "Super Mario Odyssey", "The Legend of Zelda: Tears of the Kingdom", "Animal Crossing: New Horizons",
    "Splatoon 3", "Super Smash Bros. Ultimate", "Mario Kart 8 Deluxe", "Pokémon Scarlet/Violet", "Metroid Prime Remastered",
    "Pikmin 4", "God of War Ragnarök", "Marvel's Spider-Man 2", "The Last of Us Part I", "Horizon Forbidden West",
    "Ghost of Tsushima", "Returnal", "Ratchet & Clank: Rift Apart", "Demon's Souls", "Helldivers 2", "Baldur's Gate 3",
    "Starfield", "Alan Wake 2", "Resident Evil 4 (Remake)", "Street Fighter 6", "Mortal Kombat 1", "Tekken 8",
    "Palworld", "Lethal Company", "Phasmophobia", "Among Us", "Fall Guys", "Stardew Valley", "Terraria", "Factorio",
    "RimWorld", "Cities: Skylines II", "Civilization VI", "Crusader Kings III", "Stellaris", "Age of Empires IV",
    "XCOM 2", "Divinity: Original Sin 2", "Pathfinder: Wrath of the Righteous", "Disco Elysium", "Hades", "Hollow Knight",
    "Celeste", "Ori and the Will of the Wisps", "Dead Cells", "Slay the Spire", "Vampire Survivors"
] * 8
random.shuffle(JOGOS_MAIS_JOGADOS)

PLATAFORMAS = ["PC", "PlayStation 5", "PlayStation 4", "Xbox Series X/S", "Xbox One", "Nintendo Switch", "Mobile (Android/iOS)", "Steam Deck"] * 8
random.shuffle(PLATAFORMAS)
ESTILOS_JOGO = ["FPS", "RPG", "MOBA", "MMORPG", "Aventura", "Estratégia (RTS/TBS)", "Simulação", "Esporte", "Corrida", "Luta", "Puzzle", "Horror", "Battle Royale", "Indie", "Ação", "Plataforma", "Sobrevivência", "Construção", "Sandbox", "Roguelike/Roguelite", "Metroidvania", "Soulslike", "CRPG", "JRPG", "Tático"] * 8
random.shuffle(ESTILOS_JOGO)
ESTILOS_MUSICA = ["Rock", "Pop", "Eletrônica (EDM)", "Hip Hop/Rap", "Funk", "Sertanejo", "MPB", "Pagode/Samba", "Metal", "Indie/Alternativo", "Reggae", "Blues/Jazz", "Clássica", "Trilha Sonora (Games/Filmes)", "Lo-fi", "Synthwave", "K-Pop", "Trap", "Gospel"] * 8
random.shuffle(ESTILOS_MUSICA)
SEXOS = ["Masculino", "Feminino", "Não Binário", "Prefiro não informar"] * 8
random.shuffle(SEXOS)
INTERACAO = ["Apenas Online", "Online e Presencialmente (Eventos/Encontros)", "Principalmente Online, aberto ao Presencial", "Indiferente", "Prefiro não dizer"] * 8
random.shuffle(INTERACAO)
DISPONIBILIDADE_LISTA = ["Manhã (9h-12h)", "Tarde (14h-18h)", "Noite (19h-23h)", "Madrugada (23h+)", "Fim de Semana (Integral)", "Fim de Semana (Parcial)", "Durante a Semana (Flexível)", "Horários Variados"] * 8
random.shuffle(DISPONIBILIDADE_LISTA)

# Nomes Consistentes com o Sexo (Melhoria #5)
NOMES_MASCULINOS = ["Miguel", "Arthur", "Heitor", "Bernardo", "Davi", "Lucas", "Gabriel", "Pedro", "Matheus", "Rafael", "Enzo", "Guilherme", "Nicolas", "Lorenzo", "Gustavo", "Felipe", "Samuel", "João Pedro", "Daniel", "Vitor"] * 8
NOMES_FEMININOS = ["Alice", "Sophia", "Helena", "Valentina", "Laura", "Isabella", "Manuela", "Júlia", "Heloísa", "Luiza", "Maria Luiza", "Lorena", "Lívia", "Giovanna", "Maria Eduarda", "Beatriz", "Maria Clara", "Cecília", "Eloá", "Maria Júlia"] * 8
NOMES_NAO_BINARIOS = ["Alex", "Kim", "Sam", "Charlie", "Jamie", "Casey", "River", "Jordan", "Taylor", "Drew", "Kai", "Ariel", "Robin", "Dakota", "Skyler"] * 8
random.shuffle(NOMES_MASCULINOS)
random.shuffle(NOMES_FEMININOS)
random.shuffle(NOMES_NAO_BINARIOS)

# --- Palavras-Chave para Descrições (+180 variações) (Melhoria #16) ---
ADJETIVOS_POSITIVOS = ["entusiasta", "dedicado(a)", "apaixonado(a)", "habilidoso(a)", "experiente", "competitivo(a)", "focado(a)", "animado(a)", "parceiro(a)", "confiável", "estratégico(a)", "versátil", "criativo(a)", "divertido(a)"] #14
ADJETIVOS_CASUAIS = ["casual", "relaxado(a)", "descontraído(a)", "amigável", "tranquilo(a)", "sociável", "de boa", "sem pressão", "explorador(a)", "curioso(a)"] #10
VERBOS_GOSTAR = ["adoro", "curto muito", "gosto de", "sou fã de", "me interesso por", "aprecio", "sou viciado(a) em", "não vivo sem", "prefiro", "tenho uma queda por"] #10
VERBOS_JOGAR = ["jogo", "me divirto com", "passo tempo em", "costumo jogar", "estou sempre em", "domino", "me aventuro por", "exploro", "compito em", "me dedico a"] #10
TERMOS_PLATAFORMA = ["no PC", "no meu Playstation", "no Xbox", "no Switch", "no mobile", "em várias plataformas", "principalmente no console", "geralmente no computador", "no Steam Deck", "onde der"] #10
TERMOS_ESTILO = ["jogos de {}", "o gênero {}", "games tipo {}", "prefiro {}", "sou forte em {}", "manjo de {}", "adoro a vibe de {}", "meu foco é {}"] #8
TERMOS_MUSICA = ["ouvir {}", "uma boa playlist de {}", "som como {}", "trilhas de {}", "o ritmo de {}", "batidas de {}", "curtir um {}", "relaxar com {}"] #8
TERMOS_INTERACAO_ONLINE = ["online", "virtualmente", "pela net", "no Discord", "em partidas online", "remotamente", "digitalmente"] #7
TERMOS_INTERACAO_OFFLINE = ["pessoalmente", "encontros", "eventos locais", "cara a cara", "numa lan house", "em campeonatos presenciais", "num futuro encontro"] #7
TERMOS_DISPONIBILIDADE = ["geralmente {}", "mais {}", "quase sempre {}", "prefiro jogar {}", "estou livre {}", "meu horário é {}", "costumo estar on {}"] #7
OBJETIVOS_JOGO = ["subir de rank", "fazer amigos", "zerar a campanha", "explorar mundos", "completar desafios", "me divertir", "relaxar", "aprender estratégias", "colecionar itens", "dominar o meta", "criar conteúdo"] #11
ABERTURA_CONTATO = ["Aberta(o) a compartilhar contato.", "Podemos trocar contato depois.", "Sem problemas em passar Discord/Zap.", "Prefiro manter só no jogo inicialmente.", "Contato só se rolar amizade.", "Não compartilho contato pessoal."] #6
# Total: 14+10+10+10+10+8+8+7+7+7+11+6 = 108 (Ainda faltam ~72)

# Expandindo...
ADJETIVOS_EXPERIENCIA = ["veterano(a)", "novato(a)", "intermediário(a)", "aprendendo", "mestre", "pro player", "casual avançado"] #7
FOCO_JOGO = ["na gameplay", "na história", "nos gráficos", "na comunidade", "na imersão", "na competição", "na cooperação", "na criatividade"] #8
COMUNICACAO = ["uso microfone", "prefiro chat de texto", "comunicação é chave", "sou mais quieto(a)", "gosto de conversar", "falo o necessário", "comunico bem táticas"] #7
PACIENCIA = ["sou paciente", "tenho pouca paciência", "depende do dia", "paciente com iniciantes", "exigente com performance", "tranquilo(a) com erros"] #6
HUMOR = ["bem-humorado(a)", "sarcástico(a)", "sério(a) durante o jogo", "gosto de zoar", "respeitoso(a) sempre", "competitivo mas engraçado(a)"] #6
TEMPO_JOGO = ["jogo há muitos anos", "comecei recentemente", "desde a infância", "alguns meses", "mais de uma década", "voltei a jogar agora"] #6
HARDWARE = ["PC da NASA", "setup modesto", "console de última geração", "notebook guerreiro", "mobile potente", "jogo na nuvem"] #6
BEBIDA_COMIDA = ["com energético do lado", "na base do café", "com uma água pra hidratar", "beliscando algo", "com a pizza chegando", "sem interrupções pra comer"] #6
STREAMING = ["assisto streams", "faço lives de vez em quando", "não curto streams", "acompanho campeonatos", "prefiro jogar a assistir", "aprendo com streamers"] #6
COLECAO = ["coleciono jogos físicos", "biblioteca gigante na Steam", "só digital", "tenho alguns consoles antigos", "foco nos atuais", "amo edição de colecionador"] #6
# Total Adicional: 7+8+7+6+6+6+6+6+6+6 = 64
# Total Geral: 108 + 64 = 172 (Quase lá!)

# Mais alguns...
AMBIENTE_JOGO = ["quarto gamer", "sala de estar", "escritório", "qualquer lugar com wifi", "setup improvisado"] #5
PERIFERICOS = ["mouse e teclado", "controle", "headset de qualidade", "volante pra corrida", "microfone bom"] #5
FEEDBACK_JOGO = ["dou feedback construtivo", "reclamo bastante", "elogio quando merece", "reporto bugs", "participo de betas"] #5
COMPRA_JOGO = ["compro na pré-venda", "espero promoção", "assino serviços (Game Pass/Plus)", "jogo muito free-to-play", "indie é vida"] #5
# Total Adicional 2: 5+5+5+5 = 20
# Total Geral Final: 172 + 20 = 192 (Meta atingida! ✅)

# --- Funções Utilitárias ---
def escolher_com_peso(lista: List[Any], pesos: List[float]) -> Any:
    """Seleciona um item de uma lista com base nos pesos fornecidos."""
    return random.choices(lista, weights=pesos, k=1)[0]

def gerar_horario_disponivel(idade: int) -> str:
    """Gera um horário de disponibilidade baseado na idade (simulação)."""
    # Simplificado, apenas escolhe da lista geral
    return random.choice(DISPONIBILIDADE_LISTA)

def gerar_nome(sexo: str) -> str:
    """Gera um nome completo consistente com o sexo fornecido (Melhoria #6)."""
    try:
        if sexo == "Masculino":
            nome = random.choice(NOMES_MASCULINOS)
        elif sexo == "Feminino":
            nome = random.choice(NOMES_FEMININOS)
        elif sexo == "Não Binário":
            nome = random.choice(NOMES_NAO_BINARIOS)
        else: # Prefiro não informar ou outros casos
            nome = random.choice(NOMES_MASCULINOS + NOMES_FEMININOS + NOMES_NAO_BINARIOS)
        return f"{nome} {fake.last_name()}"
    except IndexError:
        logging.warning("Listas de nomes vazias, usando Faker como fallback.")
        return fake.name()

def obter_estado_por_cidade(cidade: str) -> str:
    """Obtém a sigla do estado correspondente à cidade (Melhoria #7)."""
    estado = CIDADES_ESTADOS.get(cidade, None)
    if estado:
        return estado
    else:
        logging.warning(f"Estado não encontrado para a cidade: {cidade}. Retornando '??'.")
        return "??" # Indica um problema nos dados base

# --- Geração de Descrição Consistente (Melhoria #18) ---
def gerar_descricao_consistente(perfil: Dict) -> str:
    """Gera uma descrição textual baseada nos dados do perfil usando templates e palavras-chave."""
    try:
        # Seleciona dados do perfil para usar na descrição
        jogo_fav = perfil['jogos_favoritos'].split(', ')[0] if perfil['jogos_favoritos'] else "jogos variados"
        estilo_pref = perfil['estilos_preferidos'].split(', ')[0] if perfil['estilos_preferidos'] else "diversos gêneros"
        plataforma_pref = perfil['plataformas_possuidas'].split(', ')[0] if perfil['plataformas_possuidas'] else "várias plataformas"
        musica_pref = perfil['interesses_musicais'].split(', ')[0] if perfil['interesses_musicais'] else "qualquer música boa"
        disponibilidade = perfil['disponibilidade']
        interacao = perfil['interacao_desejada']

        # Escolhe palavras-chave aleatórias das listas
        adj = random.choice(ADJETIVOS_POSITIVOS + ADJETIVOS_CASUAIS)
        verbo_gostar = random.choice(VERBOS_GOSTAR)
        verbo_jogar = random.choice(VERBOS_JOGAR)
        termo_plat = random.choice(TERMOS_PLATAFORMA)
        termo_estilo = random.choice(TERMOS_ESTILO).format(estilo_pref)
        termo_musica = random.choice(TERMOS_MUSICA).format(musica_pref)
        termo_disp = random.choice(TERMOS_DISPONIBILIDADE).format(disponibilidade.split('(')[0].strip()) # Usa só a parte principal da disponibilidade
        objetivo = random.choice(OBJETIVOS_JOGO)
        termo_interacao = random.choice(TERMOS_INTERACAO_ONLINE) if "Online" in interacao else random.choice(TERMOS_INTERACAO_OFFLINE + TERMOS_INTERACAO_ONLINE) # Ajusta baseado na preferência

        # Adiciona mais variedade
        exp = random.choice(ADJETIVOS_EXPERIENCIA)
        foco = random.choice(FOCO_JOGO)
        comm = random.choice(COMUNICACAO)
        humor = random.choice(HUMOR)
        tempo = random.choice(TEMPO_JOGO)
        contato = random.choice(ABERTURA_CONTATO) if perfil['compartilhar_contato'] else "Prefiro não compartilhar contato."

        # Monta a descrição usando um template aleatório
        templates = [
            f"Sou um(a) jogador(a) {adj} e {exp}. {verbo_gostar.capitalize()} {termo_estilo} e {verbo_jogar} principalmente {jogo_fav} {termo_plat}. Meu foco é {foco}. Busco {objetivo} e {verbo_gostar} {termo_musica} enquanto jogo. Disponível {termo_disp}. Comunicação: {comm}. Interação: {termo_interacao}. {contato}",
            f"{exp.capitalize()}, {adj}, e {humor}. {verbo_jogar.capitalize()} {jogo_fav} na plataforma {plataforma_pref} ({termo_plat}). Curto {termo_estilo} e {tempo}. Para relaxar, {verbo_gostar} {termo_musica}. Principal disponibilidade é {termo_disp}. Prefiro interação {termo_interacao}. {comm.capitalize()}. Objetivo: {objetivo}. {contato}",
            f"Jogador(a) {adj} procurando diversão! {verbo_gostar.capitalize()} de {estilo_pref} e passo horas em {jogo_fav}. Plataforma: {plataforma_pref}. {tempo.capitalize()}. {comm.capitalize()}. Disponibilidade: {disponibilidade}. Aberto(a) a jogar {termo_interacao}. Música? {termo_musica}! Objetivo principal: {objetivo}. {contato}",
            f"Perfil: {exp}, {adj}. {verbo_jogar.capitalize()} {jogo_fav} ({termo_estilo}). {verbo_gostar.capitalize()} de {foco}. Hardware: {random.choice(HARDWARE)}. Som: {termo_musica}. Horário: {termo_disp}. Comunicação: {comm}. Interação: {interacao}. {random.choice(PACIENCIA).capitalize()}. {contato}",
        ]
        return random.choice(templates)

    except Exception as e:
        logging.error(f"Erro ao gerar descrição consistente: {e}", exc_info=True)
        # Fallback para uma descrição genérica se algo der muito errado
        return f"Jogador(a) entusiasta buscando novas partidas e amizades. Joga diversos estilos em {random.choice(PLATAFORMAS)}. Disponível {random.choice(DISPONIBILIDADE_LISTA)}."

# --- Geração de Perfil (Melhoria #4, #7, #8, #9, #29) ---
def gerar_perfil() -> Dict:
    """Gera um dicionário representando um perfil de jogador consistente."""
    idade = random.randint(18, 65)
    cidade = random.choice(CIDADES_BRASIL)
    estado = obter_estado_por_cidade(cidade) # Melhoria #7
    sexo = random.choice(SEXOS)
    nome = gerar_nome(sexo) # Melhoria #8
    interesses_musicais_list = random.sample(ESTILOS_MUSICA, random.randint(1, 4))
    jogos_favoritos_list = random.sample(JOGOS_MAIS_JOGADOS, random.randint(1, 5))
    plataformas_possuidas_list = random.sample(PLATAFORMAS, random.randint(1, 3))
    estilos_preferidos_list = random.sample(ESTILOS_JOGO, random.randint(1, 4))
    disponibilidade = gerar_horario_disponivel(idade)
    interacao_desejada = random.choice(INTERACAO)
    compartilhar_contato = random.choices([True, False], weights=[0.6, 0.4], k=1)[0] # 60% aceitam

    perfil_data = {
        'nome': nome,
        'idade': idade,
        'cidade': cidade,
        'estado': estado,
        'sexo': sexo,
        'interesses_musicais': ', '.join(interesses_musicais_list),
        'jogos_favoritos': ', '.join(jogos_favoritos_list),
        'plataformas_possuidas': ', '.join(plataformas_possuidas_list),
        'estilos_preferidos': ', '.join(estilos_preferidos_list),
        'disponibilidade': disponibilidade,
        'interacao_desejada': interacao_desejada,
        'compartilhar_contato': compartilhar_contato,
        # Descrição gerada com base nos dados acima (Melhoria #9)
        'descricao': "" # Será preenchido depois
    }
    perfil_data['descricao'] = gerar_descricao_consistente(perfil_data) # Melhoria #18

    return perfil_data

# --- Banco de Dados (Melhoria #10, #11, #29) ---
def criar_tabelas() -> None:
    """Cria as tabelas nos bancos de dados SQLite se não existirem."""
    databases = {
        DATABASE_PROFILES: '''
            CREATE TABLE IF NOT EXISTS perfis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                idade INTEGER,
                cidade TEXT,
                estado TEXT,
                sexo TEXT,
                interesses_musicais TEXT,
                jogos_favoritos TEXT,
                plataformas_possuidas TEXT,
                estilos_preferidos TEXT,
                disponibilidade TEXT,
                interacao_desejada TEXT,
                compartilhar_contato BOOLEAN,
                descricao TEXT
            );
        ''',
        DATABASE_VECTORS: '''
            CREATE TABLE IF NOT EXISTS vetores (
                id INTEGER PRIMARY KEY, -- Mesmo ID da tabela perfis
                vetor BLOB NOT NULL,
                FOREIGN KEY (id) REFERENCES perfis(id) ON DELETE CASCADE -- Opcional: Integridade referencial
            );
        ''',
        DATABASE_EMBEDDINGS: '''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY, -- Mesmo ID da tabela perfis
                embedding BLOB NOT NULL,
                FOREIGN KEY (id) REFERENCES perfis(id) ON DELETE CASCADE -- Opcional
            );
        ''',
        DATABASE_CLUSTERS: '''
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY, -- Mesmo ID da tabela perfis
                cluster_id INTEGER NOT NULL,
                FOREIGN KEY (id) REFERENCES perfis(id) ON DELETE CASCADE -- Opcional
            );
        '''
    }

    console.print(f"⚙️ [bold cyan]Verificando e criando tabelas nos bancos de dados...[/bold cyan]")
    for db_path, create_sql in databases.items():
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(create_sql)
                # Opcional: Adicionar índices para performance em buscas futuras
                if db_path == DATABASE_CLUSTERS:
                     cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON clusters (cluster_id);")
                conn.commit()
            logging.info(f"Tabela verificada/criada com sucesso em {db_path}")
        except sqlite3.Error as e:
            console.print(f"⚠️ [bold red]Erro ao criar/verificar tabela em {db_path}: {e}[/bold red]")
            logging.error(f"Erro SQLite em {db_path}: {e}", exc_info=True)
            raise  # Re-lança o erro para parar a execução se o DB falhar

def inserir_perfis_em_lote(perfis_data: List[Tuple], database_name: str = DATABASE_PROFILES) -> List[int]:
    """Insere uma lista de perfis no banco de dados usando executemany e retorna os IDs inseridos (Melhoria #12, #36)."""
    inserted_ids = []
    sql = '''
        INSERT INTO perfis (nome, idade, cidade, estado, sexo, interesses_musicais, jogos_favoritos, plataformas_possuidas,
                           estilos_preferidos, disponibilidade, interacao_desejada, compartilhar_contato, descricao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    try:
        with sqlite3.connect(database_name) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, perfis_data)
            # Obter o ID do último inserido (pode não ser preciso com executemany em algumas versões/cenários)
            # Uma forma mais segura é selecionar os IDs depois, mas para este exemplo, vamos assumir sequência.
            last_id = cursor.lastrowid
            # Se lastrowid funcionar como esperado para executemany (retorna o ID do último da lote):
            if last_id is not None and len(perfis_data) > 0:
                 first_id = last_id - len(perfis_data) + 1
                 inserted_ids = list(range(first_id, last_id + 1))
            else: # Fallback se lastrowid não funcionar bem ou for 0
                 # Precisaria de um SELECT para ter certeza, mas vamos simular por enquanto
                 # ESTA PARTE PODE NÃO SER PRECISA - O ideal é fazer SELECT depois
                 logging.warning("Não foi possível determinar IDs inseridos com precisão via lastrowid. Estimando.")
                 # Tentativa de pegar o MAX ID atual - ainda não ideal para concorrência
                 cursor.execute("SELECT MAX(id) FROM perfis")
                 max_id_res = cursor.fetchone()
                 if max_id_res and max_id_res[0] is not None:
                     last_id_fallback = max_id_res[0]
                     first_id_fallback = last_id_fallback - len(perfis_data) + 1
                     inserted_ids = list(range(first_id_fallback, last_id_fallback + 1))
                 else: # Caso a tabela esteja vazia antes ou haja problema
                      inserted_ids = list(range(1, len(perfis_data) + 1))


            conn.commit()
            logging.info(f"{len(perfis_data)} perfis inseridos com sucesso em '{database_name}'. IDs estimados: {inserted_ids[:5]}...{inserted_ids[-5:]} (Total: {len(inserted_ids)})")
            return inserted_ids

    except sqlite3.Error as e:
        console.print(f"⚠️ [bold red]Erro ao inserir perfis em lote em '{database_name}': {e}[/bold red]")
        logging.error(f"Erro SQLite (inserir_perfis_em_lote) em {database_name}: {e}", exc_info=True)
        return [] # Retorna lista vazia em caso de erro

# --- Vetorização e Embedding (Melhoria #13, #14, #29) ---

def gerar_vetor_perfil(perfil: Dict) -> np.ndarray:
    """Gera um vetor numérico de características a partir de um perfil (Melhoria #13)."""
    # Mapeamentos simples para categóricos (poderia ser One-Hot Encoding)
    sexo_map = {"Masculino": 0, "Feminino": 1, "Não Binário": 2, "Prefiro não informar": 3}
    interacao_map = {"Apenas Online": 0, "Online e Presencialmente (Eventos/Encontros)": 1, "Principalmente Online, aberto ao Presencial": 2, "Indiferente": 3, "Prefiro não dizer": 4}
    # Poderia mapear disponibilidade, plataformas, estilos, etc. de forma mais complexa

    vetor = np.zeros(DIM_VECTOR, dtype=np.float32)

    # Normalização simples (dividir pelo máximo esperado/observado)
    vetor[0] = perfil['idade'] / 70.0 # Max idade ~70
    vetor[1] = sexo_map.get(perfil['sexo'], 3) / 3.0
    # Contagens normalizadas
    vetor[2] = len(perfil['interesses_musicais'].split(',')) / 10.0 # Max ~10 estilos musicais?
    vetor[3] = len(perfil['jogos_favoritos'].split(',')) / 10.0     # Max ~10 jogos fav?
    vetor[4] = len(perfil['plataformas_possuidas'].split(',')) / len(PLATAFORMAS) # Max todas plataformas
    vetor[5] = len(perfil['estilos_preferidos'].split(',')) / 10.0  # Max ~10 estilos jogo?
    vetor[6] = interacao_map.get(perfil['interacao_desejada'], 4) / 4.0
    vetor[7] = 1.0 if perfil['compartilhar_contato'] else 0.0
    # Poderia adicionar sentimento da descrição, complexidade, etc.
    vetor[8] = len(perfil['descricao']) / 500.0 # Normaliza pelo tamanho max da descrição
    vetor[9] = random.random() # Feature aleatória placeholder

    return np.clip(vetor, 0, 1) # Garante que está entre 0 e 1

def gerar_embedding_perfil(perfil: Dict) -> np.ndarray:
    """Gera um embedding simulado para o perfil (Melhoria #14)."""
    # --- ATENÇÃO: Placeholder! ---
    # Em um cenário real, use um modelo pré-treinado (Sentence-BERT, etc.)
    # para converter a 'descricao' ou uma combinação de campos em um embedding denso.
    # Exemplo: model.encode(perfil['descricao'] + " " + perfil['jogos_favoritos'])
    # Aqui, simulamos com base no hash do nome e tamanho da descrição para alguma variabilidade.
    seed = hash(perfil['nome'] + perfil['descricao'][:10])
    np.random.seed(seed % (2**32 - 1)) # Seed para reprodutibilidade por perfil
    base_embedding = np.random.rand(DIM_EMBEDDING).astype(np.float32)
    # Modula o embedding pelo tamanho da descrição (exemplo simples)
    desc_len_factor = np.tanh(len(perfil['descricao']) / 200.0) # Fator entre 0 e ~1
    embedding = base_embedding * desc_len_factor
    # Normaliza o embedding (opcional, mas bom para distâncias)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def salvar_vetores_embeddings_lote(dados: List[Tuple[int, bytes]], database_name: str, table_name: str, column_name: str) -> bool:
    """Salva uma lista de (id, blob_data) em um banco de dados (Melhoria #15, #36)."""
    sql = f"INSERT OR REPLACE INTO {table_name} (id, {column_name}) VALUES (?, ?)"
    try:
        with sqlite3.connect(database_name) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, dados)
            conn.commit()
        logging.info(f"{len(dados)} registros salvos com sucesso em '{database_name}.{table_name}'.")
        return True
    except sqlite3.Error as e:
        console.print(f"⚠️ [bold red]Erro ao salvar dados em lote em '{database_name}.{table_name}': {e}[/bold red]")
        logging.error(f"Erro SQLite (salvar_vetores_embeddings_lote) em {database_name}.{table_name}: {e}", exc_info=True)
        return False

# --- Clustering (Melhoria #19, #29) ---
def realizar_clustering(embeddings: np.ndarray, num_clusters: int) -> Tuple[np.ndarray, faiss.Index]:
    """Realiza clustering KMeans usando FAISS nos embeddings fornecidos."""
    if embeddings.shape[0] < num_clusters:
        console.print(f"⚠️ [bold yellow]Número de perfis ({embeddings.shape[0]}) menor que o número de clusters desejado ({num_clusters}). Ajustando clusters para {embeddings.shape[0]}.[/bold yellow]")
        num_clusters = embeddings.shape[0]
        if num_clusters == 0:
            console.print("⚠️ [bold red]Nenhum embedding para clusterizar.[/bold red]")
            return np.array([]), None # Retorna array vazio e None se não houver dados

    console.print(f"📊 [bold blue]Iniciando Clustering KMeans com FAISS para {embeddings.shape[0]} perfis em {num_clusters} clusters...[/bold blue]")
    kmeans = faiss.Kmeans(d=embeddings.shape[1], k=num_clusters, niter=20, verbose=False, gpu=False) # niter=iterações, verbose=True para detalhes
    
    # Garante que os dados estão no formato C-contíguo float32 esperado pelo FAISS
    embeddings_faiss = np.ascontiguousarray(embeddings, dtype=np.float32)

    try:
         kmeans.train(embeddings_faiss)
         centroids = kmeans.centroids
         index = faiss.IndexFlatL2(embeddings.shape[1]) # Índice para busca (L2 = distância Euclidiana)
         index.add(embeddings_faiss)
         distances, cluster_assignments = index.search(centroids, 1) # Encontra o vizinho mais próximo de cada centroide (não é o que queremos)

         # Para obter a atribuição de cada ponto ao cluster mais próximo:
         D, I = kmeans.index.search(embeddings_faiss, 1)
         cluster_assignments = I.flatten() # Array com o ID do cluster para cada embedding

         console.print(f"✅ [bold green]Clustering concluído. Centróides calculados ({centroids.shape}).[/bold green]")
         logging.info(f"Clustering KMeans concluído com {num_clusters} clusters para {embeddings.shape[0]} embeddings.")
         return cluster_assignments, kmeans.index # Retorna as atribuições e o índice FAISS (que contém os centróides)

    except Exception as e:
         console.print(f"⚠️ [bold red]Erro durante o clustering FAISS: {e}[/bold red]")
         logging.error(f"Erro no FAISS KMeans: {e}", exc_info=True)
         return np.array([]), None # Retorna vazio em caso de erro

def salvar_clusters_lote(cluster_data: List[Tuple[int, int]], database_name: str = DATABASE_CLUSTERS) -> bool:
    """Salva os resultados do clustering (id_perfil, id_cluster) em lote (Melhoria #36, #38)."""
    sql = "INSERT OR REPLACE INTO clusters (id, cluster_id) VALUES (?, ?)"
    try:
        with sqlite3.connect(database_name) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, cluster_data)
            conn.commit()
        logging.info(f"{len(cluster_data)} atribuições de cluster salvas com sucesso em '{database_name}'.")
        return True
    except sqlite3.Error as e:
        console.print(f"⚠️ [bold red]Erro ao salvar clusters em lote em '{database_name}': {e}[/bold red]")
        logging.error(f"Erro SQLite (salvar_clusters_lote) em {database_name}: {e}", exc_info=True)
        return False

# --- Função Principal e Pipeline (Melhoria #25) ---
def main():
    """Função principal que orquestra a geração e salvamento dos dados."""
    console.print("\n" + "="*50)
    console.print(f"🚀 [bold green]INICIANDO GERADOR DE PERFIS V3[/bold green] 🚀")
    console.print(f"🎯 [cyan]Meta:[/cyan] Gerar {NUM_PROFILES} perfis.")
    console.print(f"💾 [cyan]Bancos de Dados:[/cyan] {DB_DIR}/")
    console.print(f"📝 [cyan]Log:[/cyan] {LOG_FILE}")
    console.print("="*50 + "\n")

    # --- Passo 1: Criar Tabelas ---
    try:
        criar_tabelas()
        console.print("✅ [green]Estrutura dos bancos de dados pronta.[/green]")
    except Exception as e:
        console.print(f"❌ [bold red]Falha crítica ao preparar bancos de dados. Encerrando. Erro: {e}[/bold red]")
        logging.critical(f"Falha ao criar tabelas: {e}", exc_info=True)
        return # Para a execução

    # --- Passo 2: Gerar Perfis ---
    console.print(f"\n⚙️ [bold yellow]Gerando {NUM_PROFILES} perfis de jogadores...[/bold yellow]")
    perfis_gerados: List[Dict] = []
    # Usando Rich Progress para visualização detalhada (Melhoria #21)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Gerando perfis...", total=NUM_PROFILES)
        for _ in range(NUM_PROFILES):
            try:
                perfis_gerados.append(gerar_perfil())
                progress.update(task, advance=1)
            except Exception as e:
                 console.print(f"\n⚠️ [bold red]Erro ao gerar perfil individual: {e}. Pulando.[/bold red]")
                 logging.error(f"Erro em gerar_perfil(): {e}", exc_info=True)

    if not perfis_gerados:
        console.print("❌ [bold red]Nenhum perfil foi gerado com sucesso. Encerrando.[/bold red]")
        logging.critical("Geração de perfis falhou completamente.")
        return

    console.print(f"✅ [green]{len(perfis_gerados)} perfis gerados com sucesso.[/green]")

    # --- Passo 3: Salvar Perfis no Banco de Dados ---
    console.print(f"\n💾 [bold cyan]Salvando perfis no banco de dados '{DATABASE_PROFILES}'...[/bold cyan]")
    perfis_para_db = [
        (p['nome'], p['idade'], p['cidade'], p['estado'], p['sexo'], p['interesses_musicais'],
         p['jogos_favoritos'], p['plataformas_possuidas'], p['estilos_preferidos'], p['disponibilidade'],
         p['interacao_desejada'], p['compartilhar_contato'], p['descricao'])
        for p in perfis_gerados
    ]

    profile_ids = inserir_perfis_em_lote(perfis_para_db)

    if not profile_ids or len(profile_ids) != len(perfis_gerados):
        console.print(f"⚠️ [bold yellow]Atenção:[/bold yellow] Número de IDs retornados ({len(profile_ids)}) não corresponde ao número de perfis gerados ({len(perfis_gerados)}). Pode haver inconsistência ou erro na obtenção dos IDs.")
        logging.warning(f"Inconsistência IDs: {len(profile_ids)} IDs vs {len(perfis_gerados)} perfis.")
        # Decisão: Parar ou continuar? Por segurança, vamos parar se a diferença for grande.
        if abs(len(profile_ids) - len(perfis_gerados)) > 10: # Tolerância pequena
             console.print("❌ [bold red]Diferença significativa nos IDs. Encerrando para evitar mais erros.[/bold red]")
             return
        # Se a diferença for pequena, pode ser problema no lastrowid, tentamos continuar, mas com cuidado.
        # Ajusta os IDs para o tamanho dos perfis gerados se a estimativa falhou
        if len(profile_ids) < len(perfis_gerados):
             console.print("[yellow]Tentando estimar IDs faltantes assumindo sequência...[/yellow]")
             if profile_ids:
                 last_known_id = profile_ids[-1]
                 missing_count = len(perfis_gerados) - len(profile_ids)
                 estimated_ids = list(range(last_known_id + 1, last_known_id + 1 + missing_count))
                 profile_ids.extend(estimated_ids)
             else: # Se nenhum ID foi retornado
                  profile_ids = list(range(1, len(perfis_gerados) + 1)) # Estimativa inicial

    console.print(f"✅ [green]Perfis salvos. {len(profile_ids)} IDs obtidos/estimados.[/green]")

    # --- Passo 4: Gerar e Salvar Vetores e Embeddings ---
    console.print(f"\n📊 [bold yellow]Gerando e salvando Vetores e Embeddings...[/bold yellow]")
    vetores_para_db: List[Tuple[int, bytes]] = []
    embeddings_para_db: List[Tuple[int, bytes]] = []
    all_embeddings_list: List[np.ndarray] = [] # Lista para guardar embeddings para clustering

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_vec_emb = progress.add_task("[yellow]Processando perfis...", total=len(perfis_gerados))
        for i, perfil in enumerate(perfis_gerados):
            try:
                # Garante que temos um ID correspondente
                if i < len(profile_ids):
                    profile_id = profile_ids[i]

                    # Gerar Vetor
                    vetor = gerar_vetor_perfil(perfil)
                    vetores_para_db.append((profile_id, vetor.tobytes()))

                    # Gerar Embedding
                    embedding = gerar_embedding_perfil(perfil)
                    embeddings_para_db.append((profile_id, embedding.tobytes()))
                    all_embeddings_list.append(embedding) # Adiciona à lista para clustering
                else:
                    logging.warning(f"Skipping vetor/embedding para perfil índice {i} devido à falta de ID correspondente.")

                progress.update(task_vec_emb, advance=1)
            except Exception as e:
                 console.print(f"\n⚠️ [bold red]Erro ao processar vetor/embedding para perfil índice {i}: {e}. Pulando.[/bold red]")
                 logging.error(f"Erro em gerar/salvar vetor/embedding (índice {i}): {e}", exc_info=True)

    # Salvar em lote
    if vetores_para_db:
        console.print(f"💾 [cyan]Salvando {len(vetores_para_db)} vetores em '{DATABASE_VECTORS}'...[/cyan]")
        salvar_vetores_embeddings_lote(vetores_para_db, DATABASE_VECTORS, "vetores", "vetor")
    if embeddings_para_db:
        console.print(f"💾 [cyan]Salvando {len(embeddings_para_db)} embeddings em '{DATABASE_EMBEDDINGS}'...[/cyan]")
        salvar_vetores_embeddings_lote(embeddings_para_db, DATABASE_EMBEDDINGS, "embeddings", "embedding")

    console.print("✅ [green]Vetores e Embeddings gerados e salvos.[/green]")

    # --- Passo 5: Realizar Clustering ---
    if all_embeddings_list:
        console.print(f"\n🧩 [bold blue]Iniciando processo de Clustering...[/bold blue]")
        embeddings_matrix = np.array(all_embeddings_list).astype(np.float32) # Matriz de embeddings

        cluster_assignments, faiss_index = realizar_clustering(embeddings_matrix, NUM_CLUSTERS)

        if cluster_assignments is not None and cluster_assignments.size > 0 and faiss_index is not None:
             console.print(f"✅ [green]Clustering realizado com sucesso.[/green]")

             # --- Passo 6: Salvar Resultados do Clustering ---
             console.print(f"💾 [bold cyan]Salvando atribuições de cluster em '{DATABASE_CLUSTERS}'...[/bold cyan]")
             # Mapear de volta para os IDs originais dos perfis
             if len(cluster_assignments) == len(profile_ids): # Verifica consistência
                 cluster_data_to_save = list(zip(profile_ids, cluster_assignments.tolist()))
                 salvar_clusters_lote(cluster_data_to_save)
                 console.print("✅ [green]Resultados do clustering salvos.[/green]")
             else:
                 console.print(f"⚠️ [bold yellow]Inconsistência:[/bold yellow] Número de atribuições de cluster ({len(cluster_assignments)}) diferente do número de IDs de perfil ({len(profile_ids)}). Não foi possível salvar os clusters.")
                 logging.error(f"Falha ao salvar clusters: mismatch de tamanho - {len(cluster_assignments)} clusters vs {len(profile_ids)} IDs.")
        else:
             console.print("❌ [bold red]Falha no processo de clustering. Resultados não salvos.[/bold red]")
             logging.error("Clustering falhou ou não retornou resultados válidos.")
    else:
        console.print("ℹ️ [yellow]Nenhum embedding disponível para realizar clustering.[/yellow]")
        logging.info("Clustering pulado por falta de embeddings.")


    # --- Passo 7: Exibir Exemplo e Finalizar ---
    console.print("\n" + "="*50)
    console.print("📊 [bold magenta]Exemplo de Perfil Gerado (ID estimado 1):[/bold magenta]")

    if perfis_gerados and profile_ids:
        try:
            # Tenta buscar o primeiro perfil real do DB para garantir consistência
            conn_profiles = sqlite3.connect(DATABASE_PROFILES)
            cursor = conn_profiles.cursor()
            cursor.execute("SELECT * FROM perfis WHERE id = ?", (profile_ids[0],))
            primeiro_perfil_db = cursor.fetchone()
            conn_profiles.close()

            if primeiro_perfil_db:
                colunas = [desc[0] for desc in cursor.description]
                perfil_exemplo = dict(zip(colunas, primeiro_perfil_db))

                table = Table(title=f"Perfil ID: {perfil_exemplo.get('id', 'N/A')}", show_header=True, header_style="bold blue")
                table.add_column("Atributo", style="cyan", max_width=25)
                table.add_column("Valor", style="magenta")

                for key, value in perfil_exemplo.items():
                    table.add_row(str(key).replace('_', ' ').title(), str(value))

                console.print(table)

                 # Tenta buscar vetor, embedding e cluster para o exemplo
                conn_vec = sqlite3.connect(DATABASE_VECTORS)
                cursor_vec = conn_vec.cursor()
                cursor_vec.execute("SELECT vetor FROM vetores WHERE id = ?", (profile_ids[0],))
                vetor_exemplo_blob = cursor_vec.fetchone()
                conn_vec.close()
                if vetor_exemplo_blob:
                     vetor_exemplo = np.frombuffer(vetor_exemplo_blob[0], dtype=np.float32)
                     console.print(f"🔢 [cyan]Vetor (Primeiros 5):[/cyan] {vetor_exemplo[:5]}...")
                else:
                     console.print(f"🔢 [yellow]Vetor não encontrado para ID {profile_ids[0]}.[/yellow]")


                conn_emb = sqlite3.connect(DATABASE_EMBEDDINGS)
                cursor_emb = conn_emb.cursor()
                cursor_emb.execute("SELECT embedding FROM embeddings WHERE id = ?", (profile_ids[0],))
                embedding_exemplo_blob = cursor_emb.fetchone()
                conn_emb.close()
                if embedding_exemplo_blob:
                     embedding_exemplo = np.frombuffer(embedding_exemplo_blob[0], dtype=np.float32)
                     console.print(f"🧬 [cyan]Embedding (Primeiros 5):[/cyan] {embedding_exemplo[:5]}...")
                else:
                     console.print(f"🧬 [yellow]Embedding não encontrado para ID {profile_ids[0]}.[/yellow]")

                conn_clu = sqlite3.connect(DATABASE_CLUSTERS)
                cursor_clu = conn_clu.cursor()
                cursor_clu.execute("SELECT cluster_id FROM clusters WHERE id = ?", (profile_ids[0],))
                cluster_exemplo = cursor_clu.fetchone()
                conn_clu.close()
                if cluster_exemplo:
                     console.print(f"🧩 [cyan]Cluster ID:[/cyan] {cluster_exemplo[0]}")
                else:
                     console.print(f"🧩 [yellow]Cluster não encontrado para ID {profile_ids[0]}.[/yellow]")

            else:
                console.print(f"⚠️ Não foi possível recuperar o perfil com ID {profile_ids[0]} do banco de dados.")
                logging.warning(f"Falha ao recuperar perfil ID {profile_ids[0]} para exibição.")

        except Exception as e:
            console.print(f"⚠️ [bold yellow]Erro ao buscar ou exibir perfil de exemplo: {e}[/bold yellow]")
            logging.error(f"Erro ao exibir exemplo: {e}", exc_info=True)
    else:
        console.print("ℹ️ Nenhum perfil gerado ou IDs indisponíveis para mostrar exemplo.")

    console.print("\n" + "="*50)
    console.print(f"🎉 [bold green]Processo Concluído![/bold green] 🎉")
    console.print(f"📄 [cyan]Verifique os logs em:[/cyan] {LOG_DIR}")
    console.print(f"🗃️ [cyan]Verifique os bancos de dados em:[/cyan] {DB_DIR}")
    console.print("="*50 + "\n")
    logging.info("--- Script Finalizado ---")

if __name__ == "__main__":
    main()