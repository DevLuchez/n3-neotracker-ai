"""
src/ingestion.py
Módulo de Ingestão e Filtragem de Dados
----------------------------------------
Lê o dataset CSV da NASA/JPL (Kaggle), detecta automaticamente as colunas
relevantes, aplica amostragem inteligente e retorna um DataFrame padronizado.

Suporta dois formatos principais:
  - Dataset de aproximações (Close Approach): separador vírgula, coluna 'relative_velocity'
  - Dataset de catálogo orbital JPL (Small Body): separador ponto-e-vírgula,
    colunas 'H', 'diameter', 'albedo', 'pha', elementos orbitais (a, e, n)

Colunas padronizadas de saída:
    neo_id       : identificador do objeto
    timestamp    : data/hora de observação
    ra           : ascensão reta (graus)
    dec          : declinação (graus)
    magnitude    : magnitude absoluta
    albedo       : albedo (brilho/refletividade)
    velocity     : velocidade relativa (km/s)
    diameter     : diâmetro estimado médio (km)
    is_hazardous : rótulo booleano (True = PHA conhecido — usado p/ treino)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa de sinônimos de colunas conhecidas nos datasets Kaggle NASA/JPL
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "neo_id":       ["id", "neo_id", "spkid", "pdes", "name", "neo reference id",
                     "Neo Reference ID"],
    "timestamp":    ["close_approach_date", "close_approach_date_full", "timestamp",
                     "epoch_osculation", "close approach date", "epoch_cal"],
    "ra":           ["ra", "ra_deg", "Right Ascension"],
    "dec":          ["dec", "dec_deg", "Declination"],
    "magnitude":    ["absolute_magnitude_h", "absolute_magnitude", "H",
                     "Absolute Magnitude"],
    "albedo":       ["albedo", "Albedo"],
    "velocity":     ["relative_velocity", "v_rel", "velocity_km_per_s",
                     "relative_velocity_km_per_sec",
                     "Miles per hour",
                     "Kilometers per second",
                     "kilometers_per_second"],
    "diameter":     ["estimated_diameter_min", "estimated_diameter_max",
                     "diameter", "Est Dia in KM(min)", "Est Dia in KM(max)",
                     "Kilometers - Min",
                     "estimated_diameter.kilometers.estimated_diameter_min"],
    "is_hazardous": ["is_potentially_hazardous_asteroid", "hazardous",
                     "potentially_hazardous", "Hazardous", "pha"],
}

# Colunas de elementos orbitais usados para estimar velocidade no dataset JPL
_ORBITAL_COLS = {"a", "e", "n"}   # semi-eixo maior (AU), excentricidade, mov. médio (°/dia)


def _find_column(df_columns: list[str], candidates: list[str]) -> str | None:
    """Retorna o primeiro nome de coluna que bater (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _detect_separator(filepath: str, encoding: str) -> str:
    """
    Detecta automaticamente o separador do CSV lendo apenas a primeira linha.
    Retorna ';' se houver mais ponto-e-vírgulas do que vírgulas, senão ','.
    """
    try:
        with open(filepath, encoding=encoding, errors="replace") as f:
            header = f.readline()
        return ";" if header.count(";") > header.count(",") else ","
    except Exception:
        return ","


def load_dataset(
    filepath: str,
    sample_size: int = 10_000,
    random_state: int = 42,
    nrows: int | None = None,
    sample_percent: int | None = None,
) -> pd.DataFrame:
    """
    Carrega o dataset CSV, normaliza colunas e retorna amostra padronizada.

    Parâmetros
    ----------
    filepath : str
        Caminho para o arquivo CSV.
    sample_size : int
        Número máximo de registros a processar (padrão: 10.000).
        Se o CSV tiver menos linhas, usa todos.
    random_state : int
        Semente para reprodutibilidade da amostragem.
    nrows : int | None
        Limite de linhas lidas do CSV (útil para arquivos muito grandes).
        Se None, lê o arquivo inteiro.

    Retorna
    -------
    pd.DataFrame com colunas padronizadas.
    """
    logger.info(f"[Ingestão] Lendo arquivo: {filepath}")

    # --- Leitura com tratamento de encoding, separador e linhas malformadas ---
    raw = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            sep = _detect_separator(filepath, enc)
            raw = pd.read_csv(
                filepath,
                encoding=enc,
                sep=sep,
                low_memory=False,
                on_bad_lines="skip",   # ignora linhas com campos inconsistentes
                nrows=nrows,
            )
            logger.info(
                f"[Ingestão] Encoding={enc!r} | Separador={sep!r} | "
                f"Linhas lidas: {len(raw):,}"
                + (f" (limitado a {nrows:,})" if nrows else "")
            )
            break
        except UnicodeDecodeError:
            continue

    if raw is None:
        raise ValueError("Não foi possível decodificar o CSV com encodings suportados.")

    # --- Mapeamento de colunas ---
    col_map: dict[str, str] = {}
    missing: list[str] = []
    for std_name, aliases in COLUMN_ALIASES.items():
        found = _find_column(list(raw.columns), aliases)
        if found:
            col_map[found] = std_name
        elif std_name not in ("ra", "dec", "timestamp", "albedo", "is_hazardous", "velocity"):
            # ra, dec, timestamp, albedo e velocity são opcionais (serão sintéticos se ausentes)
            missing.append(std_name)

    if missing:
        logger.warning(f"[Ingestão] Colunas não encontradas (serão sintéticas): {missing}")

    df = raw.rename(columns=col_map)

    # --- Garantir colunas obrigatórias com fallback sintético ---
    _ensure_columns(df, raw, len(raw), random_state)

    # --- Selecionar apenas as colunas padronizadas ---
    std_cols = list(COLUMN_ALIASES.keys())
    df = df[[c for c in std_cols if c in df.columns]].copy()

    # Total antes da limpeza/tratamento (importante para a interface)
    df.attrs["records_before_cleanup"] = len(df)

    # --- Conversão numérica ---
    for col in ("velocity", "diameter", "magnitude", "albedo"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Limpeza: remover linhas sem features essenciais ---
    essential = [c for c in ["magnitude", "diameter"] if c in df.columns]
    if essential:
        df = df.dropna(subset=essential)

    # Velocidade: remover NaN somente se não foi gerado sinteticamente
    if "velocity" in df.columns:
        df = df.dropna(subset=["velocity"])

    # --- Estimativas / fallbacks de colunas derivadas ---

    # Albedo: se ausente ou totalmente NaN, estimar a partir da magnitude
    if "albedo" not in df.columns or df["albedo"].isna().all():
        df["albedo"] = _estimate_albedo(df["magnitude"])
    else:
        df["albedo"] = df["albedo"].fillna(_estimate_albedo(df["magnitude"]))

    # is_hazardous: garantir booleano
    if "is_hazardous" in df.columns:
        df["is_hazardous"] = df["is_hazardous"].astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y", "t", "y"]
        )
    else:
        df["is_hazardous"] = False

    # neo_id: fallback
    if "neo_id" not in df.columns:
        df["neo_id"] = [f"NEO-{i:07d}" for i in range(len(df))]
    else:
        df["neo_id"] = df["neo_id"].astype(str).str.strip()

    # timestamp: fallback sintético
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp("2026-01-01")

    # ra / dec: fallback sintético
    if "ra" not in df.columns:
        rng = np.random.default_rng(random_state)
        df["ra"]  = rng.uniform(0, 360, len(df))
        df["dec"] = rng.uniform(-90, 90, len(df))

    # --- Amostragem inteligente ---
    if sample_percent is not None:
        pct = max(1, min(100, int(sample_percent)))
        if pct < 100:
            target_size = max(1, min(len(df), int(len(df) * pct / 100.0)))
            if target_size < len(df):
                n_pha = int(target_size * 0.15)
                n_safe = target_size - n_pha

                pha_df = df[df["is_hazardous"]].sample(
                    n=min(n_pha, df["is_hazardous"].sum()), random_state=random_state
                )
                safe_df = df[~df["is_hazardous"]].sample(
                    n=min(n_safe, (~df["is_hazardous"]).sum()), random_state=random_state
                )
                df = pd.concat([pha_df, safe_df]).sample(frac=1, random_state=random_state)
                logger.info(
                    f"[Ingestão] Amostra percentual aplicada: {pct}% do total lido "
                    f"=> {len(df):,} registros ({len(pha_df)} PHAs + {len(safe_df)} seguros)"
                )
            else:
                logger.info(f"[Ingestão] Usando 100% dos {len(df):,} registros lidos.")
        else:
            logger.info(f"[Ingestão] Usando 100% dos {len(df):,} registros lidos.")
    elif len(df) > sample_size:
        # Estratificada: mantém proporção de PHAs
        n_pha  = int(sample_size * 0.15)   # ~15 % PHAs na amostra
        n_safe = sample_size - n_pha

        pha_df  = df[df["is_hazardous"]].sample(
            n=min(n_pha, df["is_hazardous"].sum()), random_state=random_state
        )
        safe_df = df[~df["is_hazardous"]].sample(
            n=min(n_safe, (~df["is_hazardous"]).sum()), random_state=random_state
        )
        df = pd.concat([pha_df, safe_df]).sample(frac=1, random_state=random_state)
        logger.info(f"[Ingestão] Amostra estratificada: {len(df):,} registros "
                    f"({len(pha_df)} PHAs + {len(safe_df)} seguros)")
    else:
        logger.info(f"[Ingestão] Usando todos os {len(df):,} registros.")

    df = df.reset_index(drop=True)
    df.attrs["records_after_cleanup"] = len(df)
    logger.info(f"[Ingestão] Dataset pronto: {len(df):,} registros, "
                f"{df['is_hazardous'].sum()} PHAs conhecidos.")
    return df


def _ensure_columns(df: pd.DataFrame, raw: pd.DataFrame, n: int, random_state: int = 42) -> None:
    """
    Adiciona colunas ausentes com valores sintéticos ou estimados.
    Tenta calcular a velocidade a partir de elementos orbitais quando disponíveis.
    """
    rng = np.random.default_rng(random_state)

    if "ra" not in df.columns:
        df["ra"]  = rng.uniform(0, 360, n)
    if "dec" not in df.columns:
        df["dec"] = rng.uniform(-90, 90, n)

    # --- Estimativa de velocidade a partir de elementos orbitais (dataset JPL) ---
    if "velocity" not in df.columns:
        # Verifica se as colunas orbitais existem no raw/df (antes do rename, podem ter sido renomeadas)
        has_orbital = _ORBITAL_COLS.issubset(set(raw.columns) | set(df.columns))
        if has_orbital:
            # Velocidade orbital média aproximada usando vis-viva simplificada:
            # v ≈ n * a * (2π/360) * AU_KM   onde n em °/dia, a em AU
            # Convertemos para km/s
            AU_KM   = 1.496e8
            SECS    = 86400.0
            a_col   = _find_column(list(raw.columns), ["a"])
            n_col   = _find_column(list(raw.columns), ["n"])
            e_col   = _find_column(list(raw.columns), ["e"])
            if a_col and n_col and e_col:
                a = pd.to_numeric(raw[a_col], errors="coerce").fillna(1.5)
                n = pd.to_numeric(raw[n_col], errors="coerce").fillna(0.5)
                e = pd.to_numeric(raw[e_col], errors="coerce").fillna(0.1)
                # Velocidade orbital média (km/s) — aproximação circular + correção excentricidade
                v_orb = (np.deg2rad(n) * a * AU_KM / SECS) * np.sqrt(1 + e)
                # Velocidade relativa à Terra ≈ |v_orb - v_terra|; v_terra ≈ 29.78 km/s
                V_EARTH = 29.78
                df["velocity"] = np.abs(v_orb.values - V_EARTH).clip(1.0, 70.0)
                logger.info("[Ingestão] Velocidade estimada a partir de elementos orbitais (n, a, e).")
            else:
                df["velocity"] = rng.uniform(5.0, 35.0, len(df))
                logger.info("[Ingestão] Velocidade gerada sinteticamente (elementos orbitais incompletos).")
        else:
            df["velocity"] = rng.uniform(5.0, 35.0, len(df))
            logger.info("[Ingestão] Velocidade gerada sinteticamente (sem elementos orbitais).")


def _estimate_albedo(magnitude: pd.Series) -> pd.Series:
    """
    Estima o albedo a partir da magnitude absoluta usando relação empírica.
    Objetos mais brilhantes (magnitude menor) tendem a ter maior albedo.
    """
    return np.clip(0.5 - magnitude * 0.015, 0.01, 0.9)
