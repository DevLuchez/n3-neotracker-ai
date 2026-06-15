"""
src/moid_calculator.py
Cálculo do MOID — Distância Mínima de Aproximação
---------------------------------------------------
Usa scipy.optimize para encontrar o ponto exato (distância e data/hora)
onde a trajetória do NEO mais se aproxima da órbita da Terra.

MOID = Minimum Orbit Intersection Distance

Se MOID < 0.05 UA → o NEO é classificado como risco crítico.
"""

import math
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

try:
    from .orbital_projection import TrajectoryPoint, _earth_pos_at   # import de pacote
except ImportError:
    from orbital_projection import TrajectoryPoint, _earth_pos_at     # import direto (standalone)

logger = logging.getLogger(__name__)

# Limiar de segurança definido no relatório
SAFETY_THRESHOLD_AU = 0.05


class MOIDResult:
    """Resultado completo do cálculo MOID para um NEO."""

    def __init__(
        self,
        neo_id:       str,
        moid_au:      float,
        moid_datetime: datetime,
        crosses_orbit: bool,
        trajectory:   list[TrajectoryPoint],
        is_critical:  bool,
    ):
        self.neo_id        = neo_id
        self.moid_au       = moid_au
        self.moid_datetime = moid_datetime
        self.crosses_orbit = crosses_orbit
        self.trajectory    = trajectory
        self.is_critical   = is_critical

    def __repr__(self):
        return (f"MOIDResult(neo_id={self.neo_id!r}, "
                f"moid_au={self.moid_au:.4f}, "
                f"critical={self.is_critical}, "
                f"datetime={self.moid_datetime.strftime('%d/%m/%Y %H:%M UTC')})")


def calculate_moid(
    neo_id:      str,
    trajectory:  list[TrajectoryPoint],
    epoch:       datetime | None = None,
) -> MOIDResult:
    """
    Calcula o MOID para um NEO a partir de sua trajetória projetada.

    Parâmetros
    ----------
    neo_id     : Identificador do NEO
    trajectory : Lista de TrajectoryPoint (saída de orbital_projection)
    epoch      : Data/hora de referência (t=0 da trajetória). Padrão: agora UTC.

    Retorna
    -------
    MOIDResult com distância, data/hora, flag de cruzamento e criticidade.
    """
    if epoch is None:
        epoch = datetime.now(timezone.utc)

    if len(trajectory) < 2:
        logger.warning(f"[MOID] {neo_id}: trajetória com menos de 2 pontos.")
        return MOIDResult(
            neo_id=neo_id, moid_au=float("inf"),
            moid_datetime=epoch, crosses_orbit=False,
            trajectory=trajectory, is_critical=False,
        )

    # --- Interpolação da trajetória ---
    t_arr = np.array([tp.t_days for tp in trajectory])
    x_arr = np.array([tp.x for tp in trajectory])
    y_arr = np.array([tp.y for tp in trajectory])
    z_arr = np.array([tp.z for tp in trajectory])

    kind = "linear" if len(t_arr) < 4 else "cubic"
    fx = interp1d(t_arr, x_arr, kind=kind, fill_value="extrapolate")
    fy = interp1d(t_arr, y_arr, kind=kind, fill_value="extrapolate")
    fz = interp1d(t_arr, z_arr, kind=kind, fill_value="extrapolate")

    def distance_at(t: float) -> float:
        """Distância entre NEO e Terra no instante t (dias)."""
        neo_pos   = np.array([float(fx(t)), float(fy(t)), float(fz(t))])
        earth_pos = _earth_pos_at(t)
        return float(np.linalg.norm(neo_pos - earth_pos))

    # --- Minimização com Brent (scipy) ---
    t_min_bound = float(t_arr[0])
    t_max_bound = float(t_arr[-1])

    result = minimize_scalar(
        distance_at,
        bounds=(t_min_bound, t_max_bound),
        method="bounded",
        options={"xatol": 0.1, "maxiter": 500},
    )

    moid_t_days = float(result.x)
    moid_au     = float(result.fun)

    # --- Data/hora do MOID ---
    moid_dt = epoch + timedelta(days=moid_t_days)

    # --- Verifica cruzamento de órbita ---
    # Se em algum momento a distância fica abaixo de 0.01 AU (dentro da órbita)
    sample_distances = [distance_at(t) for t in np.linspace(t_min_bound, t_max_bound, 200)]
    crosses_orbit = any(d < 0.01 for d in sample_distances)

    is_critical = moid_au < SAFETY_THRESHOLD_AU

    logger.info(
        f"[MOID] {neo_id}: MOID={moid_au:.4f} AU | "
        f"Data: {moid_dt.strftime('%d/%m/%Y %H:%M UTC')} | "
        f"Crítico: {is_critical}"
    )

    return MOIDResult(
        neo_id=neo_id,
        moid_au=moid_au,
        moid_datetime=moid_dt,
        crosses_orbit=crosses_orbit,
        trajectory=trajectory,
        is_critical=is_critical,
    )
