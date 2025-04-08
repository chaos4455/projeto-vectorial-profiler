import sqlite3
import numpy as np
import faiss
from faker import Faker
import random
from colorama import init, Fore, Style
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Inicializar Colorama
init(autoreset=True)

console = Console()

# Definições Globais
NUM_PROFILES = 20000
FAKER_LOCALE = 'pt_BR'
fake = Faker(FAKER_LOCALE)

# Definição das 200+ cidades mais populosas do Brasil
CIDADES_BRASIL = [
    "São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza", "Belo Horizonte", "Manaus", "Curitiba", "Recife", "Goiânia",
    "Porto Alegre", "Belém", "Guarulhos", "Campinas", "São Luís", "São Gonçalo", "Maceió", "Duque de Caxias", "Campo Grande", "Natal",
    "Teresina", "São Bernardo do Campo", "Nova Iguaçu", "João Pessoa", "Santo André", "Osasco", "Jaboatão dos Guararapes", "Contagem", "Sorocaba", "Uberlândia",
    "Ribeirão Preto", "Aparecida de Goiânia", "Cariacica", "Feira de Santana", "Caxias do Sul", "Olinda", "Joinville", "Montes Claros", "Ananindeua", "Santos",
    "Campos dos Goytacazes", "Mauá", "Carapicuíba", "Serra", "Betim", "Jundiaí", "Niterói", "Juiz de Fora", "Piracicaba", "Porto Velho",
    "Canoas", "Maringá", "Mogi das Cruzes", "Londrina", "São Vicente", "Foz do Iguaçu", "Pelotas", "Vitória", "Taubaté", "São José do Rio Preto",
    "Ponta Grossa", "Belford Roxo", "Rio Branco", "Itaquaquecetuba", "Cubatão", "Boa Vista", "Blumenau", "Novo Hamburgo", "Guarujá", "Cascavel",
    "Petrolina", "Vitória da Conquista", "Paulista", "Praia Grande", "Imperatriz", "Viamão", "Camaçari", "Juazeiro do Norte", "Volta Redonda", "Sumaré",
    "Sete Lagoas", "Ipatinga", "Divinópolis", "Parnamirim", "Magé", "Sobral", "Mossoró", "Santa Luzia", "Pindamonhangaba", "Rio Grande",
    "Marabá", "Criciúma", "Santa Maria", "Barreiras", "Itabuna", "Luziânia", "Gravataí", "Bagé", "Lauro de Freitas", "Teófilo Otoni",
    "Garanhuns", "Passo Fundo", "Arapiraca", "Alagoinhas", "Francisco Morato", "Franco da Rocha", "Pinhais", "Colombo", "Guarapuava", "Caucaia",
    "Barueri", "Palmas", "Governador Valadares", "Parauapebas", "Santa Bárbara d'Oeste", "Araguaína", "Ji-Paraná", "Cachoeiro de Itapemirim", "Timon", "Maracanaú",
    "Dourados", "Itajaí", "Rio das Ostras", "Simões Filho", "Paranaguá", "Porto Seguro", "Linhares", "Uruguaiana", "Abaetetuba", "Itapetininga",
    "Picos", "Caxias", "Bragança Paulista", "Tangará da Serra", "Várzea Grande", "Itapevi", "Marília", "Cabo Frio", "Macapá", "Cariacica",
    "Eunápolis", "Feira de Santana", "Ilhéus", "Itabuna", "Jequié", "Paulo Afonso", "Teixeira de Freitas", "Vitória da Conquista", "Arapiraca", "Palmeira dos Índios",
    "Rio Largo", "União dos Palmares", "Ananindeua", "Belém", "Castanhal", "Marabá", "Parauapebas", "Santarém", "Timon", "Bacabal",
    "Balsas", "Caxias", "Imperatriz", "Paço do Lumiar", "São José de Ribamar", "Barreiras", "Brumado", "Camaçari", "Eunápolis", "Feira de Santana",
    "Ilhéus", "Itabuna", "Jequié", "Juazeiro", "Lauro de Freitas", "Paulo Afonso", "Salvador", "Santo Antônio de Jesus", "Serrinha", "Teixeira de Freitas",
    "Valença", "Vitória da Conquista", "Caucaia", "Crato", "Fortaleza", "Juazeiro do Norte", "Maracanaú", "Sobral", "Aquiraz", "Canindé",
    "Iguatu", "Itapipoca", "Maranguape", "Pacatuba", "Quixadá", "Cascavel", "Foz do Iguaçu", "Guarapuava", "Londrina", "Maringá",
    "Paranaguá", "Ponta Grossa", "São José dos Pinhais", "Umuarama", "Vila Velha", "Vitória", "Colatina", "Cachoeiro de Itapemirim", "Guarapari", "Linhares",
    "São Mateus", "Aparecida de Goiânia", "Catalão", "Goiânia", "Itumbiara", "Luziânia", "Rio Verde", "Valparaíso de Goiás", "Águas Lindas de Goiás", "Novo Gama",
    "Santo Antônio do Descoberto", "Caxias do Sul", "Canoas", "Caxias do Sul", "Esteio", "Gravataí", "Novo Hamburgo", "Passo Fundo", "Pelotas", "Porto Alegre",
    "Rio Grande", "Santa Cruz do Sul", "Santa Maria", "Sapucaia do Sul", "Viamão", "Caxias", "Picos", "Teresina", "Parnaíba",
    "Guarulhos", "Campinas", "São Bernardo do Campo", "Santo André", "Osasco", "Sorocaba", "Ribeirão Preto", "São José dos Campos", "Santos", "São José do Rio Preto",
    "Mogi das Cruzes", "Jundiaí", "Piracicaba", "Bauru", "São Vicente", "Franca", "Taubaté", "Praia Grande", "Limeira", "Carapicuíba",
    "Guarujá", "Itaquaquecetuba", "Presidente Prudente", "Suzano", "Taboão da Serra", "Barueri", "Embu das Artes", "Diadema", "Mauá", "Cotia",
    "São Caetano do Sul", "Ferraz de Vasconcelos", "Itapevi", "Arujá", "Poá", "Salto", "Sumaré", "Valinhos", "Vinhedo",
    "Americana", "Araraquara", "Atibaia", "Bragança Paulista", "Caieiras", "Cajamar", "Campo Limpo Paulista", "Cubatão", "Extrema", "Hortolândia",
    "Indaiatuba", "Itanhaém", "Jacareí", "Jandira", "Mairiporã", "Mongaguá", "Ourinhos", "Paulínia", "Pindamonhangaba", "Rio Claro",
    "Santa Bárbara d'Oeste", "São Carlos", "Sertãozinho", "Sorocaba", "Taubaté", "Ubatuba", "Valinhos", "Votorantim"
]

ESTADOS_BRASIL = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO"
]

# Jogos, Plataformas, Estilos, etc.
JOGOS_MAIS_JOGADOS = [
    "League of Legends", "Counter-Strike: Global Offensive", "Dota 2", "Valorant", "Fortnite",
    "Minecraft", "Grand Theft Auto V", "Call of Duty: Warzone", "Apex Legends", "Overwatch",
    "Rainbow Six Siege", "PlayerUnknown's Battlegrounds", "Rocket League", "Destiny 2", "Warframe",
    "The Witcher 3: Wild Hunt", "Red Dead Redemption 2", "Cyberpunk 2077", "The Elder Scrolls V: Skyrim", "Fallout 4",
    "Dark Souls III", "Bloodborne", "Sekiro: Shadows Die Twice", "Elden Ring", "Diablo IV",
    "World of Warcraft", "Final Fantasy XIV", "Guild Wars 2", "The Elder Scrolls Online", "Black Desert Online",
    "Sea of Thieves", "No Man's Sky", "Star Citizen", "Elite Dangerous", "EVE Online",
    "Hearthstone", "Magic: The Gathering Arena", "Legends of Runeterra", "Gwent: The Witcher Card Game", "Yu-Gi-Oh! Duel Links",
    "Teamfight Tactics", "Auto Chess", "Dota Underlords", "Clash Royale", "Brawl Stars",
    "Candy Crush Saga", "Pokémon GO", "Genshin Impact", "Honkai: Star Rail", "PUBG Mobile",
    "Call of Duty: Mobile", "Free Fire", "Mobile Legends: Bang Bang", "Arena of Valor", "Diablo Immortal",
    "FIFA", "eFootball", "NBA 2K", "Madden NFL", "NHL",
    "Forza Horizon", "Gran Turismo", "Assetto Corsa", "iRacing", "Project CARS",
    "Super Mario Odyssey", "The Legend of Zelda: Breath of the Wild", "Animal Crossing: New Horizons", "Splatoon 2", "Super Smash Bros. Ultimate",
    "Mario Kart 8 Deluxe", "Pokémon Sword and Shield", "Luigi's Mansion 3", "Metroid Dread", "Bayonetta 3",
    "God of War", "Marvel's Spider-Man", "The Last of Us Part II", "Horizon Zero Dawn", "Ghost of Tsushima",
    "Death Stranding", "Detroit: Become Human", "Uncharted 4: A Thief's End", "Days Gone", "Ratchet & Clank",
    "Stardew Valley", "Terraria", "Don't Starve", "Factorio", "RimWorld",
    "Cities: Skylines", "Civilization VI", "Crusader Kings III", "Stellaris", "Hearts of Iron IV",
    "XCOM 2", "Divinity: Original Sin 2", "Pillars of Eternity", "Wasteland 3", "Disco Elysium",
    "Resident Evil Village", "Silent Hill", "Dead by Daylight", "Friday the 13th: The Game", "Phasmophobia",
    "Among Us", "Fall Guys", "Jackbox Games", "Party Animals", "Pummel Party",
    "Baba Is You", "The Witness", "Portal", "Portal 2", "Superhot",
    "Tetris", "Pac-Man", "Space Invaders", "Donkey Kong", "Street Fighter",
    "Mortal Kombat", "Tekken", "Super Mario Bros.", "The Legend of Zelda", "Final Fantasy"
]

PLATAFORMAS = ["PC", "PlayStation 5", "PlayStation 4", "Xbox Series X/S", "Xbox One", "Nintendo Switch", "Mobile"]
ESTILOS_JOGO = ["FPS", "RPG", "MOBA", "MMORPG", "Aventura", "Estratégia", "Simulação", "Esporte", "Corrida", "Luta", "Puzzle", "Horror", "Battle Royale", "Indie", "Ação"]
ESTILOS_MUSICA = ["Rock", "Pop", "Eletrônica", "Hip Hop", "Funk", "Sertanejo", "MPB", "Clássica", "Metal", "Indie"]
SEXOS = ["Masculino", "Feminino", "Não Binário", "Prefiro não dizer"]
INTERACAO = ["Apenas Online", "Online e Offline", "Prefiro não dizer"]

# Pesos de probabilidade (ajustáveis)
PESO_IDADE = [0.1, 0.2, 0.25, 0.2, 0.15, 0.1]  # Distribuição de idades (18-27, 28-37, ..., 58-67)
PESO_PLATAFORMA = [0.7, 0.5, 0.5, 0.4, 0.4, 0.3, 0.8]  # PC, PS5, PS4, Xbox Series, Xbox One, Switch, Mobile
PESO_HORARIO_JOVEM = [0.2, 0.5, 0.8, 0.6, 0.3]  # Manhã, Tarde, Noite, Madrugada, Fim de Semana
PESO_HORARIO_ADULTO = [0.1, 0.3, 0.7, 0.5, 0.9]  # Manhã, Tarde, Noite, Madrugada, Fim de Semana

# Funções Utilitárias
def escolher_com_peso(lista, pesos):
    return random.choices(lista, weights=pesos, k=1)[0]

def gerar_horario_disponivel(idade):
    if idade <= 25:
        return escolher_com_peso(["Manhã", "Tarde", "Noite", "Madrugada", "Fim de Semana"], PESO_HORARIO_JOVEM)
    else:
        return escolher_com_peso(["Manhã", "Tarde", "Noite", "Madrugada", "Fim de Semana"], PESO_HORARIO_ADULTO)

# Função para gerar um perfil de jogador
def gerar_perfil():
    idade = random.randint(18, 60)
    cidade = random.choice(CIDADES_BRASIL)
    estado = random.choice(ESTADOS_BRASIL) # Escolhe um estado aleatório
    sexo = random.choice(SEXOS)
    interesses_musicais = random.sample(ESTILOS_MUSICA, random.randint(1, 3))
    jogos_favoritos = random.sample(JOGOS_MAIS_JOGADOS, 3)
    plataformas_possuidas = random.sample(PLATAFORMAS, random.randint(1, len(PLATAFORMAS)))
    estilos_preferidos = random.sample(ESTILOS_JOGO, random.randint(1, 3))
    disponibilidade = gerar_horario_disponivel(idade)
    interacao_desejada = random.choice(INTERACAO)
    compartilhar_contato = random.choice([True, False])

    return {
        'nome': fake.name(),
        'idade': idade,
        'cidade': cidade,
        'estado': estado,
        'sexo': sexo,
        'interesses_musicais': ', '.join(interesses_musicais),
        'jogos_favoritos': ', '.join(jogos_favoritos),
        'plataformas_possuidas': ', '.join(plataformas_possuidas),
        'estilos_preferidos': ', '.join(estilos_preferidos),
        'disponibilidade': disponibilidade,
        'interacao_desejada': interacao_desejada,
        'compartilhar_contato': compartilhar_contato,
        'descricao': fake.text(max_nb_chars=200)
    }

# Criar a conexão com o banco de dados SQLite
conn = sqlite3.connect('perfis_jogadores.db')
cursor = conn.cursor()

# Criar a tabela de perfis (se não existir)
cursor.execute('''
    CREATE TABLE IF NOT EXISTS perfis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
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
    )
''')

# Gerar e inserir os perfis no banco de dados
console.print(f"{Fore.GREEN}Gerando {NUM_PROFILES} perfis de jogadores...{Style.RESET_ALL}")
for i in track(range(NUM_PROFILES), description="Processando..."):
    perfil = gerar_perfil()
    cursor.execute('''
        INSERT INTO perfis (nome, idade, cidade, estado, sexo, interesses_musicais, jogos_favoritos, plataformas_possuidas, 
                           estilos_preferidos, disponibilidade, interacao_desejada, compartilhar_contato, descricao)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (perfil['nome'], perfil['idade'], perfil['cidade'], perfil['estado'], perfil['sexo'], perfil['interesses_musicais'],
          perfil['jogos_favoritos'], perfil['plataformas_possuidas'], perfil['estilos_preferidos'], perfil['disponibilidade'],
          perfil['interacao_desejada'], perfil['compartilhar_contato'], perfil['descricao']))

# Commitar as mudanças e fechar a conexão
conn.commit()
conn.close()

console.print(f"{Fore.GREEN}Banco de dados 'perfis_jogadores.db' criado com sucesso!{Style.RESET_ALL}")

# Exemplo de uso do Rich para exibir informações
table = Table(title="Exemplo de Perfil Gerado")
table.add_column("Atributo", style="cyan", no_wrap=True)
table.add_column("Valor", style="magenta")

# Conectar ao banco de dados
conn = sqlite3.connect('perfis_jogadores.db')
cursor = conn.cursor()

# Recuperar o primeiro perfil do banco de dados
cursor.execute("SELECT * FROM perfis LIMIT 1")
primeiro_perfil = cursor.fetchone()

# Fechar a conexão com o banco de dados
conn.close()

# Se o perfil foi recuperado com sucesso, exibir seus dados na tabela
if primeiro_perfil:
    # Descompactar os dados do perfil
    id, nome, idade, cidade, estado, sexo, interesses_musicais, jogos_favoritos, plataformas_possuidas, estilos_preferidos, disponibilidade, interacao_desejada, compartilhar_contato, descricao = primeiro_perfil

    # Adicionar os dados do perfil à tabela
    table.add_row("ID", str(id))
    table.add_row("Nome", nome)
    table.add_row("Idade", str(idade))
    table.add_row("Cidade", cidade)
    table.add_row("Estado", estado)
    table.add_row("Sexo", sexo)
    table.add_row("Interesses Musicais", interesses_musicais)
    table.add_row("Jogos Favoritos", jogos_favoritos)
    table.add_row("Plataformas Possuídas", plataformas_possuidas)
    table.add_row("Estilos Preferidos", estilos_preferidos)
    table.add_row("Disponibilidade", disponibilidade)
    table.add_row("Interação Desejada", interacao_desejada)
    table.add_row("Compartilhar Contato", str(compartilhar_contato))
    table.add_row("Descrição", descricao)

    # Imprimir a tabela no console
    console.print(table)
else:
    console.print("Nenhum perfil encontrado no banco de dados.")