"""
src/orbital_projection.py
Projeção Orbital — Algoritmo A*
--------------------------------
Discretiza o espaço orbital em um grafo 3D e usa o algoritmo A* para
calcular a trajetória futura de cada NEO, aplicando:

  g(n) = distância percorrida acumulada (integração numérica simplificada)
  h(n) = distância euclidiana até a Terra em linha reta

Correções gravitacionais:
  - Sol (corpo central)
  - Júpiter (maior influência entre planetas)
  - Vênus (influência secundária relevante para NEOs de cruzamento)

Retorna uma lista de pontos (x, y, z, t) em AU (Unidades Astronômicas)
e dias julianos.

NOTA: Esta é uma simulação de 2-corpos perturbada, não uma mecânica orbital
completa de N-corpos. Suficiente para estimativas de MOID (v0.1).
"""

import heapq
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------
AU_KM     = 1.496e8        # 1 UA em km
GM_SUN    = 4 * math.pi**2  # GM do Sol em AU³/ano² (unidades astronômicas)
GM_JUP    = GM_SUN * 9.55e-4
GM_VEN    = GM_SUN * 2.45e-6

# Posições médias aproximadas dos planetas perturbadores (em AU, eclíptica)
# Simplificação: órbitas circulares médias
PERTURBERS = [
    {"name": "Sol",     "GM": GM_SUN,  "pos": np.array([0.0, 0.0, 0.0])},
    {"name": "Júpiter", "GM": GM_JUP,  "pos": np.array([5.2, 0.0, 0.0])},
    {"name": "Vênus",   "GM": GM_VEN,  "pos": np.array([0.72, 0.0, 0.0])},
]

# Parâmetros do grafo A*
STEP_AU     = 0.005          # passo espacial em AU (~750.000 km)
STEP_DAYS   = 1.0            # passo temporal em dias
MAX_STEPS   = 730            # máximo 2 anos de projeção
EARTH_POS   = np.array([1.0, 0.0, 0.0])  # Terra em 1 AU (órbita simplificada)


class TrajectoryPoint(NamedTuple):
    x: float       # AU
    y: float       # AU
    z: float       # AU
    t_days: float  # dias desde a época de referência


@dataclass(order=True)
class _Node:
    f_score: float
    pos:     np.ndarray = field(compare=False)
    t_days:  float      = field(compare=False)
    g_score: float      = field(compare=False)
    path:    list       = field(compare=False, default_factory=list)


def project_trajectory(
    ra_deg:    float,
    dec_deg:   float,
    velocity:  float,          # km/s
    diameter:  float,          # km (usado apenas para logging)
    neo_id:    str = "NEO",
    max_steps: int = MAX_STEPS,
) -> list[TrajectoryPoint]:
    """
    Projeta a trajetória do NEO usando A*.

    Parâmetros
    ----------
    ra_deg   : Ascensão reta (graus) — define direção inicial no plano eclíptico
    dec_deg  : Declinação (graus)    — define inclinação orbital
    velocity : Velocidade relativa à Terra (km/s)
    diameter : Diâmetro do objeto (km)
    neo_id   : Identificador para log
    max_steps: Número máximo de iterações A*

    Retorna
    -------
    list[TrajectoryPoint] — pontos (x, y, z, t) em AU e dias.
    """
    # --- Posição inicial: aproximação simplificada ---
    # NEOs próximos à Terra: inicializar próximo a 1 AU com direção dada por ra/dec
    ra_rad  = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)

    start_pos = np.array([
        math.cos(dec_rad) * math.cos(ra_rad),
        math.cos(dec_rad) * math.sin(ra_rad),
        math.sin(dec_rad),
    ]) * 1.05  # 1.05 AU de distância inicial (próximo à Terra)

    # Velocidade em AU/dia
    vel_au_day = (velocity * 86400) / AU_KM

    # Vetor de velocidade inicial: direção oposta à posição (em direção à Terra)
    vel_dir    = EARTH_POS - start_pos
    vel_mag    = np.linalg.norm(vel_dir)
    vel_unit   = vel_dir / vel_mag if vel_mag > 0 else vel_dir
    vel_vec    = vel_unit * vel_au_day

    # --- Algoritmo A* ---
    start_node = _Node(
        f_score = _heuristic(start_pos),
        pos     = start_pos.copy(),
        t_days  = 0.0,
        g_score = 0.0,
        path    = [TrajectoryPoint(*start_pos, 0.0)],
    )

    heap = [start_node]
    best_trajectory: list[TrajectoryPoint] = [TrajectoryPoint(*start_pos, 0.0)]
    min_dist_to_earth = float("inf")
    current_vel = vel_vec.copy()

    step = 0
    while heap and step < max_steps:
        node = heapq.heappop(heap)
        step += 1

        # Aplicar aceleração gravitacional
        accel = _compute_acceleration(node.pos)
        current_vel = current_vel + accel * STEP_DAYS

        # Nova posição
        new_pos = node.pos + current_vel * STEP_DAYS
        new_t   = node.t_days + STEP_DAYS
        g_new   = node.g_score + np.linalg.norm(new_pos - node.pos)
        h_new   = _heuristic(new_pos)

        tp = TrajectoryPoint(*new_pos, new_t)
        new_path = node.path + [tp]

        dist_earth = np.linalg.norm(new_pos - _earth_pos_at(new_t))
        if dist_earth < min_dist_to_earth:
            min_dist_to_earth = dist_earth
            best_trajectory   = new_path

        # Parar se passou muito longe (> 2 AU) sem se aproximar mais
        if np.linalg.norm(new_pos) > 2.5:
            break

        new_node = _Node(
            f_score = g_new + h_new,
            pos     = new_pos,
            t_days  = new_t,
            g_score = g_new,
            path    = new_path,
        )
        heapq.heappush(heap, new_node)

    logger.debug(f"[A*] {neo_id}: {len(best_trajectory)} pontos | "
                 f"MOID aprox.: {min_dist_to_earth:.4f} AU")
    return best_trajectory


def _heuristic(pos: np.ndarray) -> float:
    """h(n): distância euclidiana até a Terra."""
    return float(np.linalg.norm(pos - EARTH_POS))


def _earth_pos_at(t_days: float) -> np.ndarray:
    """
    Posição aproximada da Terra em AU após t_days dias.
    Órbita circular de 1 AU com período de 365.25 dias.
    """
    angle = 2 * math.pi * t_days / 365.25
    return np.array([math.cos(angle), math.sin(angle), 0.0])


def _compute_acceleration(pos: np.ndarray) -> np.ndarray:
    """
    Calcula a aceleração gravitacional resultante de todos os perturbadores.
    Unidades: AU/dia²
    """
    accel = np.zeros(3)
    # Converter GM de AU³/ano² → AU³/dia²
    day_factor = (1 / 365.25) ** 2
    for body in PERTURBERS:
        r_vec = body["pos"] - pos
        r_mag = np.linalg.norm(r_vec)
        if r_mag < 1e-6:
            continue
        accel += body["GM"] * day_factor * r_vec / r_mag**3
    return accel
