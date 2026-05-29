from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard aspect angles and their names (including minor aspects)
ASPECT_ANGLES = {
    "conjunction": 0,
    "semi_sextile": 30,
    "semi_square": 45,
    "sextile": 60,
    "square": 90,
    "trine": 120,
    "sesquiquadrate": 135,
    "quincunx": 150,
    "opposition": 180,
}

# Default orbs for each aspect type (degrees)
DEFAULT_ORBS = {
    "conjunction": 10,
    "semi_sextile": 3,
    "semi_square": 3,
    "sextile": 8,
    "square": 10,
    "trine": 10,
    "sesquiquadrate": 3,
    "quincunx": 5,
    "opposition": 10,
}

# BTC genesis natal date
BTC_NATAL_DATE = "2009-01-03 18:15:00"

# Bradley Siderograph valency table: +1 = harmonious, -1 = challenging
BRADLEY_VALENCY = {
    ("jupiter", "saturn"): {
        "conjunction": -1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
    ("jupiter", "neptune"): {
        "conjunction": 1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
    ("jupiter", "uranus"): {
        "conjunction": 1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
    ("saturn", "neptune"): {
        "conjunction": -1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
    ("saturn", "uranus"): {
        "conjunction": -1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
    ("neptune", "uranus"): {
        "conjunction": 1,
        "sextile": 1,
        "square": -1,
        "trine": 1,
        "opposition": -1,
    },
}

# Mid-term planet pairs for Bradley
BRADLEY_MIDTERM_PAIRS = [
    ("sun", "mercury"),
    ("sun", "venus"),
    ("sun", "mars"),
    ("mercury", "venus"),
    ("mercury", "mars"),
    ("venus", "mars"),
]

# First-trade dates for commodities/indices (from astro_dates.jpg)
COMMODITY_NATAL_DATES = {
    "Wheat": "1884-05-01",
    "Corn": "1888-07-14",
    "Oats": "1888-07-13",
    "Soybeans": "1936-10-05",
    "Soybean_Oil": "1950-07-17",
    "Soybean_Meal": "1951-08-17",
    "Gold": "1974-12-31",
    "Silver": "1931-06-15",
    "Copper": "1933-07-05",
    "Platinum": "1956-03-04",
    "Palladium": "1968-01-02",
    "Coffee": "1882-03-07",
    "Cocoa": "1925-10-01",
    "Sugar": "1914-12-16",
    "Cotton": "1870-09-01",
    "Orange_Juice": "1966-02-01",
    "Lumber": "1969-10-01",
    "Crude_Oil": "1983-03-30",
    "Natural_Gas": "1990-04-03",
    "Heating_Oil": "1978-11-14",
    "Treasury_Bonds": "1977-08-22",
    "SP500": "1982-04-21",
    "Live_Cattle": "1964-11-30",
    "Lean_Hogs": "1966-02-28",
    "Feeder_Cattle": "1971-11-30",
    "Currencies": "1972-05-16",
    "BTC": "2009-01-03",
    "ETH": "2015-07-30",
}

# Essential dignity table (Compendium 2.3)
DIGNITY_TABLE = {
    "sun": {4: 5, 0: 4, 10: -5, 6: -4},
    "moon": {3: 5, 1: 4, 9: -5, 7: -4},
    "mercury": {2: 5, 5: 5, 8: -5, 11: -5},
    "venus": {1: 5, 6: 5, 11: 4, 7: -5, 0: -5, 5: -4},
    "mars": {0: 5, 7: 5, 9: 4, 6: -5, 1: -5, 3: -4},
    "jupiter": {8: 5, 11: 5, 3: 4, 2: -5, 5: -5, 9: -4},
    "saturn": {9: 5, 10: 5, 6: 4, 3: -5, 4: -5, 0: -4},
    "uranus": {10: 5, 7: 4, 4: -5, 1: -4},
    "neptune": {11: 5, 3: 4, 5: -5, 9: -4},
    "pluto": {7: 5, 0: 4, 1: -5, 6: -4},
}

DEWEY_CYCLES = {
    "41_month": {"period_days": 41 * 30.44, "ref_trough": "1932-06-01"},
    "9_2_year": {"period_days": 9.2 * 365.25, "ref_trough": "1932-06-01"},
    "18_6_year": {"period_days": 18.6 * 365.25, "ref_trough": "1932-06-01"},
    "54_year": {"period_days": 54 * 365.25, "ref_trough": "1932-06-01"},
}

SYNODIC_PAIRS = {
    "jupiter_saturn": ("jupiter", "saturn"),
    "jupiter_uranus": ("jupiter", "uranus"),
    "saturn_neptune": ("saturn", "neptune"),
    "saturn_pluto": ("saturn", "pluto"),
    "jupiter_neptune": ("jupiter", "neptune"),
    "jupiter_pluto": ("jupiter", "pluto"),
}

MCWHIRTER_BULLISH_SIGNS = {11, 0, 1, 2, 3, 4}


def get_dates_planets(ephemeris_df: pd.DataFrame) -> Tuple[pd.Index, List[str]]:
    """Extract sorted date index and available planets from ephemeris input."""
    dates = ephemeris_df.index.get_level_values("date").unique()
    planets = ephemeris_df.index.get_level_values("ticker").unique().tolist()
    return dates, planets


def deg_to_lowest_180(angle: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Reduce an angle to the range -180 to +180 degrees."""
    a = angle % 360
    if isinstance(a, (pd.Series, np.ndarray)):
        mask = a > 180
        if isinstance(a, pd.Series):
            a = a.copy()
            a[mask] = a[mask] - 360
        else:
            a = np.where(mask, a - 360, a)
    else:
        if a > 180:
            a -= 360
    return a


def compute_aspect_distance(
    lon1: Union[float, pd.Series],
    lon2: Union[float, pd.Series],
    aspect_angle: float,
) -> Union[float, pd.Series]:
    """Compute distance from exact aspect angle between two longitudes."""
    separation = abs(deg_to_lowest_180(lon1 - lon2))
    if isinstance(separation, pd.Series):
        separation = separation.abs()
    else:
        separation = abs(separation)
    return abs(separation - aspect_angle)


def aspect_weight(distance: Union[float, pd.Series], orb: float = 15.0) -> Union[float, pd.Series]:
    """Step-interpolated weight based on distance to exact aspect (0-10 scale)."""
    if isinstance(distance, pd.Series):
        w = pd.Series(0.0, index=distance.index)
        w[distance <= orb] = 0.0
        w[distance <= 10] = 2.5
        w[distance <= 5] = 7.5
        w[distance <= 0] = 10.0
        w[distance > orb] = 0.0
        return w

    if distance > orb:
        return 0.0
    if distance > 10:
        return 0.0
    if distance > 5:
        return 2.5
    if distance > 0:
        return 7.5
    return 10.0


def get_zodiac_sign(longitude: Union[float, pd.Series]) -> Union[int, pd.Series]:
    """Convert ecliptic longitude to zodiac sign index (0=Aries..11=Pisces)."""
    if isinstance(longitude, pd.Series):
        return (longitude // 30).astype(int) % 12
    return int(longitude // 30) % 12


def get_planet_longitude(ephemeris_df: pd.DataFrame, planet: str) -> pd.Series:
    """Extract longitude series for one planet from ephemeris."""
    try:
        return ephemeris_df.xs(planet, level="ticker")["longitude"]
    except KeyError:
        logger.warning("Planet '%s' not found in ephemeris data.", planet)
        return pd.Series(dtype="float64")


def get_planet_field(ephemeris_df: pd.DataFrame, planet: str, field: str) -> pd.Series:
    """Extract a field series for one planet from ephemeris."""
    try:
        return ephemeris_df.xs(planet, level="ticker")[field]
    except KeyError:
        logger.warning("Field '%s' for planet '%s' not found.", field, planet)
        return pd.Series(dtype="float64")


def event_impact_kernel(
    binary_signal: pd.Series,
    halflife_forward: int = 7,
    halflife_backward: int = 3,
    max_multiples: int = 4,
) -> pd.Series:
    """Convert a binary event flag into a smooth anticipation/decay kernel."""
    if binary_signal.empty:
        return binary_signal.copy()

    n = len(binary_signal)
    values = binary_signal.values.astype(float)
    result = np.zeros(n)

    fwd_len = halflife_forward * max_multiples
    fwd_kernel = np.exp(-np.arange(fwd_len) * np.log(2) / halflife_forward)

    bwd_len = halflife_backward * max_multiples
    bwd_kernel = np.exp(-np.arange(1, bwd_len + 1) * np.log(2) / halflife_backward)

    event_idx = np.where(values > 0)[0]

    for idx in event_idx:
        fwd_end = min(idx + fwd_len, n)
        fwd_slice = slice(idx, fwd_end)
        result[fwd_slice] = np.maximum(result[fwd_slice], fwd_kernel[: fwd_end - idx])

        bwd_start = max(idx - bwd_len, 0)
        bwd_slice = slice(bwd_start, idx)
        bwd_vals = bwd_kernel[: idx - bwd_start][::-1]
        result[bwd_slice] = np.maximum(result[bwd_slice], bwd_vals)

    return pd.Series(result, index=binary_signal.index, name=binary_signal.name)


def smooth_binary_features(
    binary_df: pd.DataFrame,
    halflives: Optional[List[int]] = None,
    include_density: bool = True,
    density_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Apply multi-scale smoothing to binary event features."""
    if halflives is None:
        halflives = [3, 7, 14, 30, 60, 90]
    if density_windows is None:
        density_windows = [30, 90]

    results = {}
    for col in binary_df.columns:
        series = binary_df[col]
        for hl in halflives:
            bwd_hl = max(1, hl // 3)
            smoothed = event_impact_kernel(series, halflife_forward=hl, halflife_backward=bwd_hl)
            results[f"{col}_smooth_{hl}d"] = smoothed

    if include_density:
        for window in density_windows:
            density = binary_df.rolling(window, min_periods=1).sum()
            density.columns = [f"{c}_density_{window}d" for c in binary_df.columns]
            for col in density.columns:
                results[col] = density[col]

    return pd.DataFrame(results)


def get_natal_positions(natal_date: Union[str, pd.Timestamp], planets: List[str]) -> Dict[str, float]:
    """Compute natal planet longitudes using cryptodatapy Ephemeris when available."""
    try:
        from cryptodatapy.extract.libraries.ephemeris import Ephemeris
    except Exception:
        logger.warning(
            "cryptodatapy ephemeris dependency is unavailable; natal transit features will be empty."
        )
        return {}

    natal_eph = Ephemeris(start_date=natal_date, end_date=natal_date, freq="d")
    positions: Dict[str, float] = {}
    for planet in planets:
        try:
            lon = natal_eph.get_planet_longitude(planet)
            if not lon.empty:
                positions[planet] = float(lon.iloc[0])
        except Exception:
            continue
    return positions
