"""
app.py  —  NEO Tracker AI  |  Frontend Streamlit
-------------------------------------------------
Dashboard interativo para rastreamento e análise de
Near-Earth Objects (NEOs) com classificação Naive Bayes,
projeção orbital A* e cálculo de MOID.

Uso:
    streamlit run app.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Imports do backend (src como pacote)
# ---------------------------------------------------------------------------
from src.classifier import NEOClassifier
from src.ingestion import load_dataset
from src.moid_calculator import calculate_moid
from src.orbital_projection import project_trajectory
from src.utility_agent import UtilityAgent

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)

MODEL_PATH = Path("models") / "neo_classifier.joblib"
DATASET_PATH = "data/dataset.csv"

# ---------------------------------------------------------------------------
# Paleta de cores
# ---------------------------------------------------------------------------
CLR_PHA       = "#FF4B4B"   # vermelho vivo
CLR_SAFE      = "#00D4AA"   # ciano
CLR_JUNK      = "#A0A0B0"   # cinza
CLR_ACCENT    = "#7C6FEC"   # violeta
CLR_BG        = "#0E1117"
CLR_CARD      = "#1A1D27"
CLR_BORDER    = "#2E3250"


# ---------------------------------------------------------------------------
# Helpers de formatação pt-BR  (ponto = milhares · vírgula = decimal)
# ---------------------------------------------------------------------------
def _fmt_int(n) -> str:
    """Inteiro com ponto como separador de milhares. Ex: 1.234.567"""
    return f"{int(n):,}".replace(",", ".")


def _fmt_float(n: float, decimals: int = 2) -> str:
    """Float pt-BR com ponto de milhares e vírgula decimal. Ex: 1.234,56"""
    if decimals == 0:
        return _fmt_int(round(n))
    raw = f"{n:.{decimals}f}"
    int_part, dec_part = raw.split(".")
    int_fmt = f"{int(int_part):,}".replace(",", ".")
    return f"{int_fmt},{dec_part}"


def _fmt_pct(n: float, decimals: int = 1) -> str:
    """Percentual pt-BR. Ex: 87,2%"""
    return f"{n * 100:.{decimals}f}%".replace(".", ",")


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NEO Tracker AI",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS customizado — dark theme premium
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0E1117;
    color: #E8EAF0;
}

/* Cabeçalho hero */
.hero-banner {
    background: linear-gradient(135deg, #1A1D27 0%, #12152B 50%, #0E1117 100%);
    border: 1px solid #2E3250;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    overflow: hidden;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,111,236,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #FFFFFF 0%, #7C6FEC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1rem;
    color: #8A8FAD;
    margin: 0;
    font-weight: 300;
}

/* Cards de métricas */
.metric-card {
    background: #1A1D27;
    border: 1px solid #2E3250;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: #7C6FEC;
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    color: #8A8FAD;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* Badge de status */
.badge-critical {
    background: rgba(255,75,75,0.15);
    color: #FF4B4B;
    border: 1px solid rgba(255,75,75,0.4);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-safe {
    background: rgba(0,212,170,0.12);
    color: #00D4AA;
    border: 1px solid rgba(0,212,170,0.35);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Tabela de ranking */
.rank-table th {
    background: #12152B !important;
    color: #8A8FAD !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.rank-table td {
    background: #1A1D27 !important;
    border-bottom: 1px solid #2E3250 !important;
    font-size: 0.88rem !important;
}

/* Secção de análise */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #C8CADF;
    border-left: 3px solid #7C6FEC;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #12152B;
    border-right: 1px solid #2E3250;
}

/* Abas */
[data-testid="stTab"] {
    font-weight: 500;
}

/* Esconder rodapé padrão */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# FUNÇÕES CACHEADAS
# ===========================================================================

@st.cache_data(show_spinner=False)
def cached_load_dataset(filepath: str, sample_size: int, nrows: int | None, sample_percent: int) -> pd.DataFrame:
    return load_dataset(filepath, sample_size=sample_size, nrows=nrows, sample_percent=sample_percent)


@st.cache_resource(show_spinner=False)
def cached_classifier(df_hash: int, retrain: bool, sample_percent: int) -> NEOClassifier:
    """Carrega ou treina o classificador."""
    if MODEL_PATH.exists() and not retrain:
        return NEOClassifier.load(MODEL_PATH)
    clf = NEOClassifier()
    clf.train(_df_cache)
    clf.save(MODEL_PATH)
    return clf


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configurações")

    nrows_options = {
        "Rápido — 50 mil linhas": 50_000,
        "Balanceado — 200 mil linhas": 200_000,
        "Completo — 958 mil linhas": None,
    }
    nrows_label = st.selectbox(
        "Leitura de dados",
        options=list(nrows_options.keys()),
        index=0,
        help="Limita a quantidade de linhas são lidas do arquivo CSV, sendo útil para analisar diferentes volumes de dados. ",
    )
    nrows = nrows_options[nrows_label]

    sample_percent = st.slider(
        "Percentual da amostra",
        min_value=5, max_value=100, value=100, step=5,
        help="Percentual de registros a processar, a partir da fonte de dados, sendo útil para testar o desempenho e o comportamento com um volume menor de dados",
    )

    sample_size = 1_000_000

    st.markdown("---")
    st.markdown("### 🔬 Sobre o Modelo")
    st.markdown(
        "**Algoritmo:** Gaussian Naive Bayes  \n"
        "**Projeção:** A\\* com perturbações gravitacionais  \n"
        "**MOID:** Scipy `minimize_scalar` (Brent)  \n"
        "**Priorização:** Agente de Utilidade  \n"
        "**Dataset:** NASA/JPL Small Body Catalog"
    )
    st.caption("NEO Tracker AI · Engenharia de Software · 2026")


# ===========================================================================
# HERO BANNER
# ===========================================================================

st.markdown("""
<div class="hero-banner">
    <p class="hero-title">☄️ NEO Tracker AI</p>
    <p class="hero-sub">
        Detecção e priorização de Near-Earth Objects potencialmente perigosos
        com Naive Bayes · Algoritmo A* · Agente de Utilidade
    </p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# CARREGAMENTO DOS DADOS
# ===========================================================================

with st.spinner("Carregando dataset NASA/JPL..."):
    try:
        _df_cache = cached_load_dataset(DATASET_PATH, sample_size, nrows, sample_percent)
        df = _df_cache.copy()
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        st.stop()

with st.spinner("Treinando/atualizando classificador..."):
    try:
        clf = NEOClassifier()
        metrics_train = clf.train(df)
        clf.save(MODEL_PATH)
    except Exception as e:
        st.error(f"Erro ao treinar o classificador: {e}")
        st.stop()

clf = NEOClassifier.load(MODEL_PATH)

with st.spinner("Classificando NEOs..."):
    classified = clf.classify(df)
    phas_df = clf.get_phas(classified)
    agent = UtilityAgent()
    priority_queue = agent.prioritize(phas_df)
    priority_df = agent.to_dataframe(priority_queue)


# ===========================================================================
# KPIs — MÉTRICAS PRINCIPAIS
# ===========================================================================

total_neos  = len(classified)
raw_total   = int(df.attrs.get("records_before_cleanup", len(df)))
n_phas      = (classified["neo_class"] == 2).sum()
n_safe      = (classified["neo_class"] == 1).sum()
n_junk      = (classified["neo_class"] == 0).sum()
top_score   = priority_df["utility_score"].max() if not priority_df.empty else 0
top_neo     = priority_df["neo_id"].iloc[0] if not priority_df.empty else "—"

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#7C6FEC">{_fmt_int(total_neos)}</div>
        <div class="metric-label">NEOs analisados</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{CLR_PHA}">{_fmt_int(n_phas)}</div>
        <div class="metric-label">PHAs detectados</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{CLR_SAFE}">{_fmt_int(n_safe)}</div>
        <div class="metric-label">Inofensivos</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{CLR_JUNK}">{_fmt_int(n_junk)}</div>
        <div class="metric-label">Lixo Espacial</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# ABAS PRINCIPAIS
# ===========================================================================

tab1, tab2, tab3 = st.tabs([
    "Classificação de NEOs",
    "Ranking de PHAs",
    "Análise Orbital de PHA",
])


# ---------------------------------------------------------------------------
# ABA 1 — CLASSIFICAÇÃO
# ---------------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">Distribuição de Classes</p>', unsafe_allow_html=True)

        class_counts = classified["class_label"].value_counts().reset_index()
        class_counts.columns = ["Classe", "Quantidade"]
        color_map = {
            "PHA":                  CLR_PHA,
            "Asteroide Inofensivo": CLR_SAFE,
            "Lixo Espacial":        CLR_JUNK,
        }

        fig_pie = go.Figure(go.Pie(
            labels=class_counts["Classe"],
            values=class_counts["Quantidade"],
            hole=0.55,
            marker=dict(
                colors=[color_map.get(c, "#888") for c in class_counts["Classe"]],
                line=dict(color=CLR_BG, width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="#E8EAF0"),
            hovertemplate="<b>%{label}</b><br>%{value:,} NEOs (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=340,
            annotations=[dict(
                text=f"<b>{_fmt_int(total_neos)}</b><br>NEOs",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#E8EAF0"),
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Probabilidade PHA vs. Features</p>', unsafe_allow_html=True)

        # Scatter: magnitude × diâmetro, colorido por pha_prob
        sample_plot = classified.sample(min(1500, len(classified)), random_state=42)
        fig_scatter = px.scatter(
            sample_plot,
            x="magnitude", y="diameter",
            color="pha_prob",
            color_continuous_scale=[[0, CLR_SAFE], [0.5, "#FFB347"], [1, CLR_PHA]],
            size_max=10,
            opacity=0.75,
            labels={"magnitude": "Magnitude Absoluta (H)", "diameter": "Diâmetro (km)", "pha_prob": "P(PHA)"},
            hover_data={"neo_id": True, "class_label": True, "velocity": ":.1f"},
        )
        fig_scatter.update_traces(marker=dict(size=5))
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12152B",
            font=dict(color="#E8EAF0"),
            coloraxis_colorbar=dict(
                title=dict(text="P(PHA)", font=dict(color="#8A8FAD")),
                tickfont=dict(color="#8A8FAD"),
            ),
            xaxis=dict(gridcolor="#2E3250", zerolinecolor="#2E3250"),
            yaxis=dict(gridcolor="#2E3250", zerolinecolor="#2E3250"),
            margin=dict(t=20, b=40, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Histograma de velocidade por classe ---
    st.markdown('<p class="section-header">Distribuição de Velocidade por Classe</p>', unsafe_allow_html=True)

    fig_hist = go.Figure()
    for label, color in color_map.items():
        subset = classified[classified["class_label"] == label]["velocity"]
        if len(subset) > 0:
            fig_hist.add_trace(go.Histogram(
                x=subset, name=label,
                marker_color=color,
                opacity=0.75,
                nbinsx=60,
                hovertemplate=f"<b>{label}</b><br>Velocidade: %{{x:.1f}} km/s<br>Contagem: %{{y}}<extra></extra>",
            ))
    fig_hist.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12152B",
        font=dict(color="#E8EAF0"),
        xaxis=dict(title="Velocidade (km/s)", gridcolor="#2E3250"),
        yaxis=dict(title="Contagem", gridcolor="#2E3250"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#E8EAF0")),
        margin=dict(t=20, b=40, l=10, r=10),
        height=280,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# ABA 2 — RANKING DE PHAs
# ---------------------------------------------------------------------------
with tab2:
    if priority_df.empty:
        st.info("Nenhum PHA detectado na amostra atual. Tente aumentar o tamanho da amostra.")
    else:
        col_rank_l, col_rank_r = st.columns([1.2, 1], gap="large")

        with col_rank_l:
            st.markdown('<p class="section-header">Fila de Prioridade (Agente de Utilidade)</p>', unsafe_allow_html=True)
            st.caption(f"U = 0.6 × norm(diâmetro) + 0.4 × norm(velocidade) — {len(priority_df)} PHAs priorizados")

            moid_lookup = {}
            for _, row in priority_df.iterrows():
                neo_row = classified[classified["neo_id"] == row["neo_id"]].iloc[0]
                trajectory = project_trajectory(
                    ra_deg=float(neo_row.get("ra", 180.0)),
                    dec_deg=float(neo_row.get("dec", 0.0)),
                    velocity=float(row["velocity_kms"]),
                    diameter=float(row["diameter_km"]),
                    neo_id=str(row["neo_id"]),
                )
                moid = calculate_moid(
                    neo_id=str(row["neo_id"]),
                    trajectory=trajectory,
                    epoch=datetime.now(timezone.utc),
                )
                moid_lookup[str(row["neo_id"])] = float(moid.moid_au)

            priority_df_with_moid = priority_df.copy()
            priority_df_with_moid["moid_au"] = priority_df_with_moid["neo_id"].map(moid_lookup)

            display_df = priority_df_with_moid[["rank", "neo_id", "utility_score", "moid_au", "diameter_km", "velocity_kms", "pha_prob"]].copy()
            display_df.columns = ["#", "NEO ID", "Score U", "MOID (AU)", "Diâm. (km)", "Vel. (km/s)", "P(PHA)"]
            display_df["Score U"] = display_df["Score U"].map(lambda x: _fmt_float(x, 4))
            display_df["MOID (AU)"] = display_df["MOID (AU)"].map(lambda x: _fmt_float(x, 4))
            display_df["Diâm. (km)"] = display_df["Diâm. (km)"].map(lambda x: _fmt_float(x, 3))
            display_df["Vel. (km/s)"] = display_df["Vel. (km/s)"].map(lambda x: _fmt_float(x, 1))
            display_df["P(PHA)"] = display_df["P(PHA)"].map(lambda x: _fmt_pct(x, 1))

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "#": st.column_config.NumberColumn(width="small"),
                    "Score U": st.column_config.TextColumn(width="medium"),
                },
            )

        with col_rank_r:
            st.markdown('<p class="section-header">Top 15 — Score de Utilidade</p>', unsafe_allow_html=True)

            top15 = priority_df.head(15).copy()
            top15["color"] = [
                CLR_PHA if p > 0.7 else (CLR_ACCENT if p > 0.4 else CLR_SAFE)
                for p in top15["pha_prob"]
            ]

            fig_bar = go.Figure(go.Bar(
                x=top15["utility_score"][::-1],
                y=top15["neo_id"][::-1],
                orientation="h",
                marker=dict(
                    color=top15["color"][::-1].tolist(),
                    line=dict(width=0),
                ),
                text=[_fmt_float(s, 4) for s in top15["utility_score"][::-1]],
                textposition="outside",
                textfont=dict(color="#E8EAF0", size=11),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#12152B",
                font=dict(color="#E8EAF0"),
                xaxis=dict(title="Score de Utilidade", gridcolor="#2E3250", range=[0, 1.05]),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
                margin=dict(t=10, b=40, l=10, r=60),
                height=420,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Scatter: diâmetro × velocidade dos PHAs ---
        st.markdown('<p class="section-header">PHAs — Diâmetro × Velocidade (tamanho = Score U)</p>', unsafe_allow_html=True)

        fig_pha_scatter = px.scatter(
            priority_df,
            x="velocity_kms", y="diameter_km",
            size="utility_score",
            color="pha_prob",
            hover_name="neo_id",
            color_continuous_scale=[[0, CLR_SAFE], [0.5, "#FFB347"], [1, CLR_PHA]],
            size_max=30,
            labels={
                "velocity_kms": "Velocidade (km/s)",
                "diameter_km": "Diâmetro (km)",
                "pha_prob": "P(PHA)",
                "utility_score": "Score U",
            },
        )
        fig_pha_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12152B",
            font=dict(color="#E8EAF0"),
            xaxis=dict(gridcolor="#2E3250"),
            yaxis=dict(gridcolor="#2E3250"),
            coloraxis_colorbar=dict(tickfont=dict(color="#8A8FAD"), title=dict(font=dict(color="#8A8FAD"))),
            margin=dict(t=20, b=40, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_pha_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# ABA 3 — ANÁLISE ORBITAL & MOID
# ---------------------------------------------------------------------------
with tab3:
    if priority_df.empty:
        st.info("Nenhum PHA disponível para análise orbital.")
    else:
        neo_options = priority_df["neo_id"].tolist()
        selected_neo = st.selectbox(
            "Selecione o NEO para análise orbital",
            options=neo_options,
            format_func=lambda x: f"{x}  (rank #{priority_df[priority_df['neo_id']==x]['rank'].values[0]})",
        )

        row = priority_df[priority_df["neo_id"] == selected_neo].iloc[0]
        neo_row = classified[classified["neo_id"] == selected_neo].iloc[0]

        ra  = float(neo_row.get("ra",  180.0))
        dec = float(neo_row.get("dec", 0.0))
        vel = float(row["velocity_kms"])
        dia = float(row["diameter_km"])

        with st.spinner(f"Calculando trajetória A* e MOID para {selected_neo}..."):
            trajectory = project_trajectory(
                ra_deg=ra, dec_deg=dec,
                velocity=vel, diameter=dia,
                neo_id=selected_neo,
            )
            epoch = datetime.now(timezone.utc)
            moid  = calculate_moid(neo_id=selected_neo, trajectory=trajectory, epoch=epoch)

        # --- KPIs do NEO selecionado ---
        m1, m2, m3, m4, m5 = st.columns(5)

        moid_color = CLR_PHA if moid.is_critical else CLR_SAFE
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{moid_color}">{_fmt_float(moid.moid_au, 4)}</div>
                <div class="metric-label">MOID (AU)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            moid_km = moid.moid_au * 1.496e8
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{CLR_ACCENT}">{_fmt_float(moid_km, 0)}</div>
                <div class="metric-label">MOID (km)</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#FFB347">{_fmt_float(dia, 3)}</div>
                <div class="metric-label">Diâmetro (km)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{CLR_SAFE}">{_fmt_float(vel, 1)}</div>
                <div class="metric-label">Velocidade (km/s)</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            status_label = "CRITICO" if moid.is_critical else "SEGURO"
            status_color = CLR_PHA if moid.is_critical else CLR_SAFE
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{status_color};font-size:1.4rem">{status_label}</div>
                <div class="metric-label">Status MOID</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Preparar os pontos da trajetória usados em ambos os gráficos
        traj_x = [tp.x for tp in trajectory]
        traj_y = [tp.y for tp in trajectory]
        traj_z = [tp.z for tp in trajectory]
        traj_t = [tp.t_days for tp in trajectory]

        # --- Distância ao longo do tempo ---
        st.markdown('<p class="section-header">Distância à Terra ao Longo da Trajetória</p>', unsafe_allow_html=True)

        import math
        distances_au = []
        for tp in trajectory:
            angle = 2 * math.pi * tp.t_days / 365.25
            ex, ey = math.cos(angle), math.sin(angle)
            d = math.sqrt((tp.x - ex) ** 2 + (tp.y - ey) ** 2 + tp.z ** 2)
            distances_au.append(d)

        fig_dist = go.Figure()
        fig_dist.add_hline(
            y=0.05, line=dict(color=CLR_PHA, width=2, dash="dash"),
            annotation_text="Limiar de risco (0.05 AU)",
            annotation_font=dict(color=CLR_PHA, size=11),
        )
        fig_dist.add_trace(go.Scatter(
            x=traj_t,
            y=distances_au,
            mode="lines",
            line=dict(color=CLR_ACCENT, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba(124,111,236,0.1)",
            name="Distância (AU)",
            hovertemplate="Dia %{x:.0f}: %{y:.4f} AU<extra></extra>",
        ))
        fig_dist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#12152B",
            font=dict(color="#E8EAF0"),
            xaxis=dict(title="Dias desde hoje", gridcolor="#2E3250"),
            yaxis=dict(title="Distância à Terra (AU)", gridcolor="#2E3250"),
            showlegend=False,
            margin=dict(t=20, b=40, l=10, r=10),
            height=260,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # --- Info textual ---
        col_info_l, col_info_r = st.columns(2)
        with col_info_l:
            st.info(
                f"**Data da aproximação máxima:** "
                f"{moid.moid_datetime.strftime('%d/%m/%Y %H:%M UTC')}  \n"
                f"**Pontos na trajetória:** {len(trajectory)}  \n"
                f"**Cruza órbita terrestre:** {'Sim ⚠️' if moid.crosses_orbit else 'Não'}"
            )
        with col_info_r:
            pha_prob_val = float(row["pha_prob"])
            st.info(
                f"**Probabilidade PHA (NB):** {_fmt_pct(pha_prob_val, 2)}  \n"
                f"**Score de utilidade:** {_fmt_float(float(row['utility_score']), 4)}  \n"
                f"**Rank de prioridade:** #{int(row['rank'])}"
            )

        # --- Visualização 3D da trajetória (último elemento da aba) ---
        st.markdown('<p class="section-header">Trajetória Orbital 3D (A*)</p>', unsafe_allow_html=True)

        theta = np.linspace(0, 2 * np.pi, 200)
        earth_x = np.cos(theta)
        earth_y = np.sin(theta)
        earth_z = np.zeros(200)

        fig_3d = go.Figure()

        fig_3d.add_trace(go.Scatter3d(
            x=earth_x, y=earth_y, z=earth_z,
            mode="lines",
            line=dict(color=CLR_SAFE, width=2, dash="dot"),
            name="Órbita Terrestre",
            hoverinfo="skip",
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=[1], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=10, color=CLR_SAFE, symbol="circle"),
            text=["Terra"],
            textfont=dict(color="#E8EAF0", size=11),
            textposition="top center",
            name="Terra",
            hoverinfo="skip",
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=14, color="#FFD700", symbol="circle",
                       line=dict(color="#FFA500", width=2)),
            text=["Sol"],
            textfont=dict(color="#FFD700", size=12),
            textposition="top center",
            name="Sol",
            hoverinfo="skip",
        ))

        fig_3d.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z,
            mode="lines+markers",
            line=dict(color=CLR_PHA, width=3),
            marker=dict(
                size=[3] * len(traj_t),
                color=traj_t,
                colorscale=[[0, "#7C6FEC"], [1, CLR_PHA]],
                opacity=0.8,
            ),
            name=f"NEO {selected_neo}",
            hovertemplate="<b>NEO %{text}</b><br>x=%{x:.3f} AU<br>y=%{y:.3f} AU<br>z=%{z:.3f} AU<extra></extra>",
            text=[f"{selected_neo} (t={t:.0f}d)" for t in traj_t],
        ))

        moid_idx = min(
            range(len(trajectory)),
            key=lambda i: (trajectory[i].x - np.cos(2 * np.pi * trajectory[i].t_days / 365.25)) ** 2
                        + (trajectory[i].y - np.sin(2 * np.pi * trajectory[i].t_days / 365.25)) ** 2
        )
        fig_3d.add_trace(go.Scatter3d(
            x=[traj_x[moid_idx]], y=[traj_y[moid_idx]], z=[traj_z[moid_idx]],
            mode="markers+text",
            marker=dict(size=12, color=moid_color, symbol="diamond",
                       line=dict(color="#FFFFFF", width=1)),
            text=["MOID"],
            textfont=dict(color=moid_color, size=11),
            textposition="top center",
            name=f"MOID ({_fmt_float(moid.moid_au, 4)} AU)",
        ))

        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="#0E1117",
                xaxis=dict(title="X (AU)", gridcolor="#2E3250", showbackground=False, color="#8A8FAD"),
                yaxis=dict(title="Y (AU)", gridcolor="#2E3250", showbackground=False, color="#8A8FAD"),
                zaxis=dict(title="Z (AU)", gridcolor="#2E3250", showbackground=False, color="#8A8FAD"),
                aspectmode="cube",
            ),
            legend=dict(bgcolor="rgba(26,29,39,0.9)", font=dict(color="#E8EAF0"), bordercolor="#2E3250", borderwidth=1),
            margin=dict(t=10, b=10, l=10, r=10),
            height=520,
            font=dict(color="#E8EAF0"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


