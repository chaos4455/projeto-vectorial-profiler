# 🚀 Vectorial Profiler 🎮

[![Python Version][python-shield]][python-url]
[![License: MIT][license-shield]][license-url]
[![Build Status][build-shield]][build-url]
[![Docker Pulls][docker-pulls-shield]][docker-hub-url]
[![Docker Image Size][docker-size-shield]][docker-hub-url]
[![Code Style: Black][code-style-shield]][code-style-url]
<!-- Add more relevant badges as needed -->

**Autor:** [Elias Andrade (chaos4455)][author-github-url] - Arquiteto de Soluções de IA 🧠

**Repositório:** [chaos4455/vectorial-profiler][repo-url]

---

## 📖 Sumário

1.  [🎯 Visão Geral](#-visão-geral)
2.  [✨ Funcionalidades Chave](#-funcionalidades-chave)
3.  [🛠️ Tecnologias e Arquitetura](#️-tecnologias-e-arquitetura)
    *   [Stack Tecnológico](#stack-tecnológico-)
    *   [Diagrama de Arquitetura (Conceitual)](#diagrama-de-arquitetura-conceitual-)
    *   [Fluxo de Dados](#fluxo-de-dados-)
4.  [📁 Estrutura do Projeto Detalhada](#-estrutura-do-projeto-detalhada)
5.  [💾 Banco de Dados](#-banco-de-dados)
    *   [Esquema (Simplificado)](#esquema-simplificado)
    *   [Observações sobre Versões](#observações-sobre-versões)
6.  [⚙️ Configuração e Instalação](#️-configuração-e-instalação)
    *   [Pré-requisitos](#pré-requisitos)
    *   [Instalação Local (Desenvolvimento)](#instalação-local-desenvolvimento-)
    *   [Instalação via Docker (Recomendado)](#instalação-via-docker-recomendado-)
7.  [▶️ Como Executar](#️-como-executar)
    *   [Execução Local](#execução-local)
    *   [Execução com Docker](#execução-com-docker)
    *   [Acessando o Dashboard](#acessando-o-dashboard-)
8.  [🔬 Explicação dos Scripts Principais](#-explicação-dos-scripts-principais)
    *   [Geração de Perfis (`geraprofiles*.py`)](#geração-de-perfis-geraprofilespy)
    *   [Vetorização (`vectorizerv1.py`)](#vetorização-vectorizerv1py)
    *   [Match Profiler & Dashboard (`match-profiler*.py`)](#match-profiler--dashboard-match-profilerpy)
    *   [Visualização 3D (`data-cubic-viz-v1.py`)](#visualização-3d-data-cubic-viz-v1py)
    *   [Dashboard de Logs (`log-dashboard-real-time*.py`)](#dashboard-de-logs-log-dashboard-real-timepy)
    *   [Geração de Documentação (`docgen*.py`)](#geração-de-documentação-docgenpy)
    *   [Geração de Heatmap (`heathmap-data-gen*.py`)](#geração-de-heatmap-heathmap-data-genpy)
    *   [Scripts Auxiliares (`doc-footer-cleaner.py`)](#scripts-auxiliares-doc-footer-cleanerpy)
9.  [🐳 Dockerização Detalhada](#-dockerização-detalhada)
    *   [Dockerfile Dinâmico](#dockerfile-dinâmico)
    *   [Supervisor (`supervisord.conf`)](#supervisor-supervisordconf)
    *   [Processos Gerenciados](#processos-gerenciados)
10. [🔄 CI/CD com GitHub Actions](#-cicd-com-github-actions)
    *   [Workflow: `⚙️ Vectorial Profiler - Docker Build and Runtime Validation`](#workflow-️-vectorial-profiler---docker-build-and-runtime-validation)
    *   [Etapas do Workflow](#etapas-do-workflow)
11. [🔌 Endpoints da API (Inferido/Exemplo)](#-endpoints-da-api-inferidoexemplo)
12. [⚠️ Pontos de Atenção e Próximos Passos](#️-pontos-de-atenção-e-próximos-passos)
13. [🤝 Contribuição](#-contribuição)
14. [📜 Licença](#-licença)
15. [📞 Contato](#-contato)

---

## 🎯 Visão Geral

> O **Vectorial Profiler** é uma solução avançada de análise e visualização de perfis de usuários (inicialmente focada em jogadores 🎮), concebida e desenvolvida por [Elias Andrade (chaos4455)][author-github-url], Arquiteto de Soluções de IA. O projeto utiliza técnicas de **Inteligência Artificial**, **Processamento de Linguagem Natural (NLP)** e **Aprendizado de Máquina** para transformar dados de perfis em representações vetoriais significativas (embeddings).

**Objetivo Principal:** Facilitar a **descoberta de similaridades**, a **identificação de padrões** e a **formação de comunidades** ou **matches** entre usuários. Isso é alcançado através de:

1.  **Cálculo de Similaridade Vetorial:** Utilizando embeddings para capturar nuances semânticas nos dados dos perfis (interesses, descrições, gostos musicais, jogos, etc.).
2.  **Redução de Dimensionalidade (PCA):** Projetando os vetores de alta dimensionalidade em um espaço 3D para visualização intuitiva.
3.  **Clustering (KMeans + FAISS):** Agrupando perfis semelhantes de forma eficiente, mesmo com grandes volumes de dados.
4.  **Visualizações Interativas:** Oferecendo dashboards web e gráficos 3D (Plotly) para explorar as relações entre os perfis.

Este projeto não é apenas uma ferramenta de análise, mas uma plataforma robusta com foco em **escalabilidade**, **monitoramento** e **automação** (demonstrado pela Dockerização e pipeline de CI/CD). É um exemplo prático de aplicação de técnicas de IA e engenharia de software moderna para resolver problemas complexos de análise de dados e matchmaking.

---

## ✨ Funcionalidades Chave

O Vectorial Profiler oferece um conjunto rico de funcionalidades:

*   ✅ **Análise de Similaridade Customizada:** Calcula scores de similaridade entre perfis considerando múltiplos atributos (interesses textuais via embeddings, preferências categóricas, disponibilidade, etc.) de forma ponderada e configurável.
*   🧬 **Geração de Vetores e Embeddings:**
    *   Transforma dados textuais (descrições, interesses) em vetores densos usando modelos de `sentence-transformers` (NLP).
    *   Combina vetores textuais com representações numéricas de outros atributos para criar um *embedding* completo do perfil.
*   📉 **Redução de Dimensionalidade com PCA:** Aplica Principal Component Analysis (PCA) da `scikit-learn` para reduzir a dimensionalidade dos embeddings, permitindo a visualização em 2D ou 3D sem perda excessiva de informação sobre a variância dos dados.
*   🧩 **Clustering Eficiente com KMeans e FAISS:** Utiliza o algoritmo KMeans para agrupar perfis similares. A integração com `faiss` (Facebook AI Similarity Search) permite otimizar a busca por centróides e vizinhos, tornando o processo eficiente para grandes datasets.
*   📊 **Visualização Interativa em 3D:** Gera gráficos 3D interativos usando `plotly`, onde cada ponto representa um perfil. A posição no espaço 3D reflete a similaridade (após PCA), e os clusters podem ser visualizados com cores distintas. Permite zoom, rotação e hover para inspeção de perfis individuais.
*   🖥️ **Dashboard Web Interativo:** Uma interface web (desenvolvida com `Flask` e `FastAPI` em diferentes versões) para:
    *   Visualizar perfis e clusters.
    *   Buscar perfis específicos.
    *   Encontrar os perfis mais similares (matches) para um dado perfil.
    *   Explorar os dados de forma amigável, com suporte a temas visuais.
*   🖼️ **Geração de Imagens de Similaridade:** Cria representações visuais (heatmaps ou outras visualizações em PNG via `Pillow`) mostrando a matriz de similaridade ou relações específicas entre perfis.
*   ⏱️ **Monitoramento de Logs em Tempo Real:** Implementa um dashboard dedicado (usando `Flask/FastAPI` e possivelmente WebSockets ou Server-Sent Events) para visualizar os logs da aplicação em tempo real, facilitando o debugging e monitoramento da saúde do sistema.
*   👤 **Geração de Perfis Sintéticos:** Utiliza a biblioteca `faker` para criar dados de perfis de jogadores realistas, essenciais para testes, demonstrações e desenvolvimento inicial sem depender de dados reais.
*   🐳 **Containerização com Docker:** O projeto é totalmente containerizado, garantindo um ambiente de execução consistente e facilitando o deploy.
*   🔄 **Automação de Build e Teste com GitHub Actions:** Um pipeline de CI/CD configurado para construir a imagem Docker, testar sua execução e publicá-la no Docker Hub a cada push na branch `main`.

---

## 🛠️ Tecnologias e Arquitetura

Este projeto combina um stack de tecnologias Python robusto com práticas modernas de desenvolvimento e DevOps.

### Stack Tecnológico 📚

*   **Linguagem Principal:**
    *   ![Python][python-shield] (`Python 3.10+`)
*   **Bibliotecas Core de IA/ML/Dados:**
    *   `numpy`: Computação numérica fundamental.
    *   `pandas`: Manipulação e análise de dados tabulares.
    *   `scikit-learn`: Implementação de PCA e outras ferramentas de machine learning.
    *   `faiss-cpu` / `faiss-gpu`: Busca de similaridade em alta velocidade (essencial para clustering e matching em larga escala).
    *   `sentence-transformers`: Geração de embeddings de texto de alta qualidade (estado da arte em NLP).
*   **Visualização:**
    *   `plotly`: Criação de gráficos interativos 2D e 3D para web.
    *   `matplotlib` / `seaborn` (Possivelmente usados nos scripts de heatmap): Visualizações estáticas.
    *   `Pillow (PIL)`: Manipulação e geração de imagens (PNGs de similaridade).
*   **Banco de Dados:**
    *   `sqlite3`: Banco de dados relacional leve, embarcado, usado para armazenamento persistente de perfis, embeddings e resultados de clustering.
*   **Web Frameworks & Servidores:**
    *   `Flask`: Microframework web para criação de APIs e dashboards (usado nas versões iniciais/v2/v3).
    *   `FastAPI`: Framework web moderno e de alta performance para APIs (usado em versões mais recentes, ex: `match-profilerv3-...-fastapi.py`).
    *   `waitress`: Servidor WSGI para produção em ambientes Windows/Linux (alternativa ao Gunicorn/Uvicorn).
    *   `Flask-Cors`: Lida com Cross-Origin Resource Sharing nas APIs Flask.
*   **Utilitários & Outros:**
    *   `rich`: Formatação rica e visualmente agradável de saídas no console.
    *   `faker`: Geração de dados fictícios para testes e população inicial.
    *   `requests`: Realização de requisições HTTP (potencialmente para APIs externas ou comunicação interna).
    *   `colorama`: Adiciona cores à saída do terminal (cross-platform).
    *   `psutil`: Acesso a informações do sistema e processos (útil para monitoramento).
    *   `schedule`: Agendamento de tarefas em Python (potencialmente para jobs periódicos).
*   **DevOps & Automação:**
    *   `Docker`: Containerização da aplicação e suas dependências.
    *   `Supervisor`: Gerenciador de processos para rodar e monitorar os serviços Python dentro do container.
    *   `GitHub Actions`: Automação de CI/CD (build, teste, push da imagem Docker).


vectorial-profiler/
├── .github/
│   └── workflows/
│       └── main.yml             # 🔄 Workflow do GitHub Actions para CI/CD
├── databases_v3/                # 💾 Diretório para bancos de dados SQLite (Versão 3)
│   └── ... (arquivos .db, .db-shm, .db-wal)
├── databases_v6/                # 💾 Diretório para bancos de dados SQLite (Versão 6 - Mais Recente?)
│   ├── clusters_perfis_v6.db      # Provavelmente armazena perfis e clusters
│   ├── clusters_perfis_v6.db-shm  # Arquivo de memória compartilhada do SQLite
│   └── clusters_perfis_v6.db-wal  # Arquivo de write-ahead logging do SQLite
│   └── embeddings_v6.db       # Provavelmente armazena embeddings
│   └── ... (outros possíveis dbs)
├── img-data-outputs/            # 🖼️ Diretório para imagens geradas (heatmaps, etc.)
│   └── ... (arquivos .png)
├── .gitignore                   # Arquivos/diretórios a serem ignorados pelo Git
├── data-cubic-viz-v1.py         # 🧊 Script para gerar visualização 3D interativa (Plotly) - Versão 1
├── doc-footer-cleaner - Copy.py # 🧹 Script auxiliar (cópia) - Provavelmente para limpar HTML gerado
├── doc-footer-cleaner.py         # 🧹 Script auxiliar para limpar HTML gerado
├── docgenv1.py                   # 📄 Geração de documentação (provavelmente interna/MD) - Versão 1
├── docgenv2.py                   # 📄 Geração de documentação - Versão 2
├── docgenv3-webhtmldocgen.py     # 📄 Geração de documentação em formato HTML - Versão 3
├── docgenv3-webhtmldocgenrev.py  # 📄 Geração de documentação HTML (revisada) - Versão 3 Rev
├── Dockerfile                    # 🐳 Instruções para construir a imagem Docker (gerado dinamicamente na Action)
├── geraprofilesv1.py              # 👤 Script para gerar perfis sintéticos - Versão 1 (inicial)
├── geraprofilesv2.py              # 👤 Script para gerar perfis sintéticos - Versão 2 (evolução)
├── geraprofilesv3.py              # 👤 Script para gerar perfis sintéticos - Versão 3 (mais estável/usada no Docker)
├── heathmap-data-gen-v1.py       # 🔥 Script para gerar dados/imagem de heatmap - Versão 1
├── heathmap-data-gen-v2.py       # 🔥 Script para gerar dados/imagem de heatmap - Versão 2
├── log-dashboard-real-time-v1.py # ⏱️ Dashboard de logs em tempo real - Versão 1 (Flask?)
├── log-dashboard-real-time-v2.py # ⏱️ Dashboard de logs em tempo real - Versão 2
├── log-dashboard-real-time-v3.py # ⏱️ Dashboard de logs em tempo real - Versão 3
├── log-dashboard-real-time-v4.py # ⏱️ Dashboard de logs em tempo real - Versão 4 (FastAPI?)
├── match-profilerv1.py           # 🧩 Core do Match Profiler (lógica de similaridade) - Versão 1
├── match-profilerv2-web-dash-full.py # 🖥️ Match Profiler com Dashboard Web completo (Flask?) - V2 Full
├── match-profilerv2-web-dash.py    # 🖥️ Match Profiler com Dashboard Web (Flask?) - V2
├── match-profilerv2.py           # 🧩 Match Profiler (lógica + talvez CLI) - Versão 2
├── match-profilerv3-web-dash-full-themes-fastapi.py # 🚀 Match Profiler + Dashboard Web (FastAPI, Temas) - V3 Avançada
├── match-profilerv3-web-dash-full-themes.py       # 🎨 Match Profiler + Dashboard Web (Flask?, Temas) - V3 com Temas
├── README.md                     # 📄 Este arquivo!
├── requirements.txt              # (Assumido/Necessário) Lista de dependências Python
├── supervisord.conf              # ⚙️ Configuração do Supervisor (gerado dinamicamente na Action)
├── test-v1-match-profilerv3-web-dash-full-themes.py # ✅ Script de teste para o dashboard V3
└── vectorizerv1.py               # 🧬 Script para vetorização/geração de embeddings - Versão 1

[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
pidfile=/tmp/supervisord.pid

[program:generator]
command=python3 /home/vectorial/app/geraprofilesv3.py
directory=/home/vectorial/app
user=vectorial
autostart=true      ; Inicia automaticamente
autorestart=false   ; Não reinicia automaticamente (roda uma vez?)
startsecs=5
exitcodes=0,1       ; Códigos de saída esperados
stopwaitsecs=10
stderr_logfile=/var/log/supervisor/generator.err.log
stdout_logfile=/var/log/supervisor/generator.out.log

[program:profiler]
# Use o script correto aqui (Flask ou FastAPI)
# command=python3 /home/vectorial/app/match-profilerv3-web-dash-full-themes.py
command=python3 /home/vectorial/app/match-profilerv3-web-dash-full-themes-fastapi.py
directory=/home/vectorial/app
user=vectorial
autostart=true      ; Inicia automaticamente
autorestart=true    ; Reinicia se falhar
startsecs=10
stopwaitsecs=60
stderr_logfile=/var/log/supervisor/profiler.err.log
stdout_logfile=/var/log/supervisor/profiler.out.log
# Adicione a porta se necessário (depende de como o script é iniciado)
# environment=FLASK_RUN_PORT=8881

