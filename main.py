"""
main.py
Pipeline Completo - NEO Tracker AI
------------------------------------
Executa o fluxo end-to-end para teste e validação:

    1. Ingestão        -> load_dataset()
    2. Treinamento     -> NEOClassifier.train()
    3. Classificação   -> NEOClassifier.classify()
    4. Priorização     -> UtilityAgent.prioritize()
    5. Proj. Orbital   -> project_trajectory()   (top-5 PHAs)
    6. MOID            -> calculate_moid()        (top-5 PHAs)

Uso:
    python main.py                  # usa data/dataset.csv por padrão
    python main.py --csv outro.csv  # especifica outro arquivo
    python main.py --sample 5000    # tamanho da amostra
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

from src.ingestion import load_dataset
from src.classifier import NEOClassifier
from src.utility_agent import UtilityAgent
from src.orbital_projection import project_trajectory
from src.moid_calculator import calculate_moid

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

MODEL_PATH = Path("models") / "neo_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NEO Tracker AI - Pipeline Completo")
    parser.add_argument("--csv",    default="data/dataset.csv", help="Caminho do CSV")
    parser.add_argument("--sample", type=int, default=10_000,   help="Tamanho da amostra")
    parser.add_argument("--nrows",  type=int, default=None,      help="Limite de linhas lidas do CSV")
    parser.add_argument("--retrain", action="store_true",        help="Força novo treinamento mesmo se modelo existir")
    return parser.parse_args()


def run_pipeline(csv_path: str, sample_size: int, nrows: int | None, retrain: bool) -> None:
    separator = "=" * 60

    # -----------------------------------------------------------------------
    # ETAPA 1 - Ingestão
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("ETAPA 1 - Ingestão de Dados")
    print(separator)
    df = load_dataset(csv_path, sample_size=sample_size, nrows=nrows)
    print(f"âœ“ Dataset carregado: {len(df):,} registros")
    print(f"  Colunas: {list(df.columns)}")
    print(f"  PHAs conhecidos (rÃ³tulo): {df['is_hazardous'].sum()}")

    # -----------------------------------------------------------------------
    # ETAPA 2 - Treinamento (ou carregamento do modelo salvo)
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("ETAPA 2 - Classificador Naive Bayes")
    print(separator)

    clf = NEOClassifier()

    if MODEL_PATH.exists() and not retrain:
        clf = NEOClassifier.load(MODEL_PATH)
        print(f"âœ“ Modelo carregado de: {MODEL_PATH}")
    else:
        metrics = clf.train(df)
        print(f"âœ“ Treinamento concluÃ­do")
        print(f"  AcurÃ¡cia: {metrics['accuracy']:.2%}")
        print(f"  Features: {metrics['features_used']}")
        clf.save(MODEL_PATH)
        print(f"  Modelo salvo em: {MODEL_PATH}")

    # -----------------------------------------------------------------------
    # ETAPA 3 - Classificação
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("ETAPA 3 - Classificação dos NEOs")
    print(separator)

    classified = clf.classify(df)
    dist = classified["class_label"].value_counts().to_dict()
    for label, count in dist.items():
        print(f"  {label}: {count:,}")

    phas = clf.get_phas(classified)
    print(f"\nâœ“ PHAs identificados: {len(phas):,}")

    if phas.empty:
        print("  Nenhum PHA encontrado. Encerrando.")
        return

    # -----------------------------------------------------------------------
    # ETAPA 4 - Priorização (Agente de Utilidade)
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("ETAPA 4 - Priorização (Agente de Utilidade)")
    print(separator)

    agent = UtilityAgent()
    priority_queue = agent.prioritize(phas)
    priority_df = agent.to_dataframe(priority_queue)

    print(f"âœ“ {len(priority_queue)} PHAs priorizados. Top 5:")
    top5 = priority_df.head(5)
    for _, row in top5.iterrows():
        print(f"  #{int(row['rank'])} {row['neo_id']:<15} "
              f"score={row['utility_score']:.4f}  "
              f"diam={row['diameter_km']:.3f} km  "
              f"vel={row['velocity_kms']:.1f} km/s  "
              f"PHA_prob={row['pha_prob']:.2%}")

    # -----------------------------------------------------------------------
    # ETAPA 5 + 6 - Projeção Orbital (A*) + MOID (top-5 PHAs)
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("ETAPA 5+6 - Projeção Orbital (A*) + CÃ¡lculo de MOID")
    print(separator)

    epoch = datetime.now(timezone.utc)
    top5_rows = top5.to_dict("records")

    critical_count = 0
    for row in top5_rows:
        neo_id   = row["neo_id"]
        ra       = float(classified.loc[
            classified["neo_id"] == neo_id, "ra"
        ].iloc[0]) if "ra" in classified.columns else 180.0
        dec      = float(classified.loc[
            classified["neo_id"] == neo_id, "dec"
        ].iloc[0]) if "dec" in classified.columns else 0.0
        velocity = row["velocity_kms"]
        diameter = row["diameter_km"]

        # Projeção A*
        trajectory = project_trajectory(
            ra_deg=ra, dec_deg=dec,
            velocity=velocity, diameter=diameter,
            neo_id=neo_id,
        )

        # MOID
        moid = calculate_moid(neo_id=neo_id, trajectory=trajectory, epoch=epoch)

        status = "ðŸ”´ CRÃTICO" if moid.is_critical else "ðŸŸ¢ Seguro"
        print(f"\n  [{neo_id}]")
        print(f"    Pontos na trajetÃ³ria : {len(trajectory)}")
        print(f"    MOID                 : {moid.moid_au:.4f} AU")
        print(f"    Aproximação mÃ¡xima   : {moid.moid_datetime.strftime('%d/%m/%Y %H:%M UTC')}")
        print(f"    Cruza Ã³rbita terrestre: {'Sim' if moid.crosses_orbit else 'Não'}")
        print(f"    Status               : {status}")

        if moid.is_critical:
            critical_count += 1

    # -----------------------------------------------------------------------
    # RESUMO FINAL
    # -----------------------------------------------------------------------
    print(f"\n{separator}")
    print("RESUMO FINAL")
    print(separator)
    print(f"  Dataset processado   : {len(df):,} NEOs")
    print(f"  PHAs detectados      : {len(phas):,}")
    print(f"  PHAs crÃ­ticos (MOID) : {critical_count} / 5 analisados")
    print(f"  Threshold de risco   : MOID < 0.05 AU")
    print(f"\n{'Pipeline concluÃ­do com sucesso!':^60}")
    print(separator)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        sample_size=args.sample,
        nrows=args.nrows,
        retrain=args.retrain,
    )

