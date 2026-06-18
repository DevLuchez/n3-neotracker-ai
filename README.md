# ☄️ NEO Tracker AI - v0.1

> Sistema inteligente de detecção, classificação e priorização de **Near-Earth Objects (NEOs)** potencialmente perigosos, desenvolvido como projeto N3 da disciplina de Inteligência Artificial — Engenharia de Software, 7ª Fase.

---

## Visão Geral

O NEO Tracker AI processa o catálogo orbital da NASA/JPL com mais de 958 mil objetos, aplica um pipeline de IA para identificar asteroides potencialmente perigosos (PHAs) e os prioriza por nível de ameaça, calculando a distância mínima de aproximação à Terra (MOID).

### Fluxo de Pipeline

```
CSV NASA/JPL → Ingestão → Classificação NB → Priorização → Projeção A* → MOID
```
<p align="center">
    <img width="916" height="425" alt="image" src="https://github.com/user-attachments/assets/ce24ded3-58e9-4f6b-a845-8bfc52038f45" />
</p>

---

## Algoritmos e Técnicas

| Módulo | Algoritmo | Descrição |
|--------|-----------|-----------|
| `classifier.py` | **Gaussian Naive Bayes** | Classifica NEOs em 3 classes: PHA, Asteroide Inofensivo, Lixo Espacial |
| `orbital_projection.py` | **Algoritmo A\*** | Projeta a trajetória orbital com perturbações gravitacionais (Sol, Júpiter, Vênus) |
| `moid_calculator.py` | **Scipy Brent** | Calcula o MOID via minimização escalar sobre a trajetória interpolada |
| `utility_agent.py` | **Agente de Utilidade** | Prioriza PHAs com fila de prioridade `heapq`: `U = 0.6·diam + 0.4·vel` |
| `ingestion.py` | **Amostragem Estratificada** | Lê o CSV com detecção automática de separador e amostragem proporcional de PHAs |

---

## Interface

Dashboard interativo construído com **Streamlit** e **Plotly**, com 3 abas:

- **Classificação de NEOs** — distribuição de classes, scatter magnitude × diâmetro, histograma de velocidade
- **Ranking de PHAs** — tabela de prioridade, gráfico de barras Top-15, scatter de ameaça
- **Análise Orbital de PHA** — trajetória 3D interativa (A*), gráfico de distância ao longo do tempo, KPIs de MOID

---

## Estrutura do Projeto

```
n3-neotracker-ai/
├── app.py                    # Frontend Streamlit
├── main.py                   # Pipeline CLI (teste end-to-end)
├── requirements.txt
├── data/
│   └── dataset.csv           # NASA/JPL Small Body Catalog (não incluso no repo)
├── models/
│   └── neo_classifier.joblib # Modelo treinado (gerado na primeira execução)
└── src/
    ├── __init__.py
    ├── ingestion.py          # Leitura e normalização do CSV
    ├── classifier.py         # Naive Bayes (treino + classificação + persistência)
    ├── orbital_projection.py # Projeção orbital com algoritmo A*
    ├── moid_calculator.py    # Cálculo do MOID
    └── utility_agent.py      # Agente de utilidade e fila de prioridade
```

---

## Instalação

### Pré-requisitos

- Python 3.11+
- Dataset CSV da NASA/JPL ([download no Kaggle](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset))

### Setup

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/n3-neotracker-ai.git
cd n3-neotracker-ai

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Coloque o dataset em data/dataset.csv

# 4. Execute o dashboard
python -m streamlit run app.py
```

---

## Uso

### Dashboard (recomendado)

```bash
python -m streamlit run app.py
# Acesse: http://localhost:8501
```

**Controles da sidebar:**

| Controle | Descrição |
|----------|-----------|
| **Leitura de dados** | Limita a quantidade de linhas lidas do CSV, útil para analisar diferentes volumes:<br>• **Rápido** — 50 mil linhas<br>• **Balanceado** — 200 mil linhas<br>• **Completo** — 958 mil linhas (dataset inteiro) |
| **Percentual da amostra** | Define qual fração dos registros lidos será usada na análise. O padrão é 100%; valores menores reduzem o volume de trabalho. |

**Observação importante:** o modelo é treinado/atualizado automaticamente sempre que a configuração de leitura ou amostragem é alterada — não há um checkbox manual para "Forçar novo treinamento".

### Pipeline CLI

```bash
# Execução padrão (10.000 registros)
python main.py

# Com amostra menor para teste rápido
python main.py --sample 2000 --nrows 50000

# Forçar re-treinamento no pipeline CLI
python main.py --sample 10000 --retrain
```

---

## Resultados Típicos

Com a configuração padrão (10.000 NEOs amostrados):

| Métrica | Valor |
|---------|-------|
| Acurácia Naive Bayes | ~96–97% |
| Taxa de PHAs detectados | ~4% |
| PHAs priorizados | ~200–400 |
| Threshold de risco MOID | < 0.05 AU |

---

## Dependências

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.35.0
plotly>=5.20.0
```

---

## Referências

- [NASA/JPL Small-Body Database](https://ssd.jpl.nasa.gov/tools/sbdb_query.html)
- [Dataset Kaggle — Asteroid Dataset](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset)
- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach* (algoritmo A*, agentes de utilidade)
- Mitchell, T. — *Machine Learning* (Naive Bayes)

---

## Autores

Desenvolvido por **Laura Heloísa Luchez** e **Daniel Fernando Costa Pereira** como projeto da avaliação N3 de Inteligência Artificial, curso de Engenharia de Software — Católica SC, 2026.
