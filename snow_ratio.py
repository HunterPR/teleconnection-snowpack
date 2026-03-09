"""
Temperature-dependent snow-to-liquid ratio (Kuchera-style) for snowfall calculations.

Use this instead of a fixed 10:1 when converting liquid precipitation to snowfall
(e.g. in pipeline derived columns, forecast chunks, or display logic).

Reference: Kuchera-style ratio increases when colder (drier snow), decreases when warmer.
  - ~10:1 at 30–32°F
  - ~15:1 at 20°F, ~20:1 at 10°F
  - ~8:1 at 34°F, lower when warmer (wet snow / mix)
"""

from __future__ import annotations

import numpy as np


def snow_ratio_kuchera(temp_f: float | np.ndarray) -> float | np.ndarray:
    """
    Kuchera-style snow-to-liquid ratio from surface temperature (°F).

    ratio = 10 + (32 - T) * 0.5, clamped to [5, 30].
    At 32°F → 10:1; at 22°F → 15:1; at 12°F → 20:1; at 34°F → 9:1.
    Above ~52°F returns 5:1 (wet/mix); below ~-28°F returns 30:1 (very dry).
    """
    scalar = np.isscalar(temp_f)
    t = np.asarray(temp_f, dtype=float)
    r = 10.0 + (32.0 - t) * 0.5
    r = np.clip(r, 5.0, 30.0)
    return float(r) if scalar else r


def liquid_to_snow_inches(
    liquid_in: float | np.ndarray,
    temp_f: float | np.ndarray,
    ratio_func=snow_ratio_kuchera,
) -> float | np.ndarray:
    """
    Convert liquid precipitation (inches) to snowfall (inches) using a temperature-dependent ratio.

    snow_in = liquid_in * ratio(temp_F).
    Use ratio_func=snow_ratio_kuchera (default) for Kuchera-style; or pass a callable (temp_f) -> ratio.
    """
    ratio = ratio_func(temp_f)
    return liquid_in * ratio
