"""
src/utility_agent.py
Agente de Utilidade — Priorização de PHAs
-----------------------------------------
Pondera os PHAs detectados pelo Naive Bayes atribuindo uma pontuação de
utilidade baseada em diâmetro e velocidade angular relativa, e os organiza
em uma fila de prioridade (heapq) do mais urgente ao menos urgente.

Função de utilidade:
    U = w_diam * norm(diameter) + w_vel * norm(velocity)

Pesos padrão (w_diam=0.6, w_vel=0.4) refletem que o tamanho do objeto
determina o potencial de dano catastrófico mais do que a velocidade isolada.
"""

import heapq
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pesos da função de utilidade
W_DIAMETER = 0.6
W_VELOCITY = 0.4


@dataclass(order=True)
class PriorityNEO:
    """
    Representa um NEO na fila de prioridade.
    O heapq é min-heap, então negamos o score para simular max-heap.
    """
    neg_score: float           # -utilidade (para max-heap via min-heap)
    neo_id:    str = field(compare=False)
    diameter:  float = field(compare=False)
    velocity:  float = field(compare=False)
    pha_prob:  float = field(compare=False)
    row_data:  dict  = field(compare=False, default_factory=dict)

    @property
    def utility_score(self) -> float:
        return -self.neg_score


class UtilityAgent:
    """
    Agente baseado em Utilidade para priorização de PHAs.

    Uso:
        agent = UtilityAgent()
        priority_queue = agent.prioritize(pha_df)
        ordered_list   = agent.get_ordered_list(priority_queue)
    """

    def __init__(self, w_diameter: float = W_DIAMETER, w_velocity: float = W_VELOCITY):
        self.w_diameter = w_diameter
        self.w_velocity = w_velocity

    def prioritize(self, pha_df: pd.DataFrame) -> list[PriorityNEO]:
        """
        Recebe o DataFrame de PHAs (saída de NEOClassifier.get_phas) e
        retorna uma lista ordenada por utilidade decrescente.

        Parâmetros
        ----------
        pha_df : pd.DataFrame
            Deve conter ao menos: neo_id, diameter, velocity, pha_prob.

        Retorna
        -------
        list[PriorityNEO] — fila de prioridade ordenada (maior utilidade primeiro).
        """
        if pha_df.empty:
            logger.info("[UtilityAgent] Nenhum PHA para priorizar.")
            return []

        df = pha_df.copy()

        # Normalização min-max
        diam_norm = self._normalize(df["diameter"])
        vel_norm  = self._normalize(df["velocity"])

        # Pontuação de utilidade
        scores = self.w_diameter * diam_norm + self.w_velocity * vel_norm

        # Montar heap
        heap: list[PriorityNEO] = []
        for idx, row in df.iterrows():
            score = float(scores.loc[idx])
            item  = PriorityNEO(
                neg_score = -score,
                neo_id    = str(row.get("neo_id", f"NEO-{idx}")),
                diameter  = float(row.get("diameter", 0.0)),
                velocity  = float(row.get("velocity", 0.0)),
                pha_prob  = float(row.get("pha_prob", 0.0)),
                row_data  = row.to_dict(),
            )
            heapq.heappush(heap, item)

        ordered = []
        while heap:
            ordered.append(heapq.heappop(heap))

        logger.info(f"[UtilityAgent] {len(ordered)} PHAs priorizados. "
                    f"Topo da fila: {ordered[0].neo_id} "
                    f"(score={ordered[0].utility_score:.4f})")
        return ordered

    def to_dataframe(self, priority_queue: list[PriorityNEO]) -> pd.DataFrame:
        """Converte a fila de prioridade em DataFrame para uso posterior."""
        records = []
        for rank, item in enumerate(priority_queue, start=1):
            records.append({
                "rank":          rank,
                "neo_id":        item.neo_id,
                "utility_score": item.utility_score,
                "diameter_km":   item.diameter,
                "velocity_kms":  item.velocity,
                "pha_prob":      item.pha_prob,
                **{k: v for k, v in item.row_data.items()
                   if k not in ("neo_id", "diameter", "velocity", "pha_prob")},
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Normalização min-max segura (evita divisão por zero)."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)
