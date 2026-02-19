from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from factorlab.factors.astrology.base import AstrologyFactor
from factorlab.factors.astrology.common import DEWEY_CYCLES, get_dates_planets


class DeweyOscillators(AstrologyFactor):
    """Idealized Dewey cycle oscillators."""

    def __init__(
        self,
        cycles: Optional[Dict[str, dict]] = None,
        data_driven: bool = False,
        target_series: Optional[pd.Series] = None,
        fit_window: int = 2520,
        **kwargs,
    ):
        super().__init__(
            description="Dewey oscillator features for validated cycle periods.",
            tags=["astrology", "dewey", "cycles"],
            **kwargs,
        )
        self.cycles = cycles
        self.data_driven = data_driven
        self.target_series = target_series
        self.fit_window = fit_window

    def _compute_astrology(self, ephemeris_df: pd.DataFrame) -> pd.DataFrame:
        dates, _ = get_dates_planets(ephemeris_df)
        cycles = self.cycles or DEWEY_CYCLES
        t = (dates - pd.Timestamp("1900-01-01")).days.values.astype(float)

        results = {}
        for name, params in cycles.items():
            period = params["period_days"]
            ref_date = pd.Timestamp(params["ref_trough"])
            t0 = (ref_date - pd.Timestamp("1900-01-01")).days

            osc = np.sin(2 * np.pi * (t - t0) / period)
            osc_cos = np.cos(2 * np.pi * (t - t0) / period)
            results[f"dewey_{name}_sin"] = pd.Series(osc, index=dates)
            results[f"dewey_{name}_cos"] = pd.Series(osc_cos, index=dates)

        if self.data_driven and self.target_series is not None:
            target_aligned = self.target_series.reindex(dates).dropna()
            if len(target_aligned) > self.fit_window:
                for name, params in cycles.items():
                    period = params["period_days"]
                    t_target = (target_aligned.index - pd.Timestamp("1900-01-01")).days.values.astype(float)

                    sin_comp = np.sin(2 * np.pi * t_target / period)
                    cos_comp = np.cos(2 * np.pi * t_target / period)

                    sin_s = pd.Series(sin_comp, index=target_aligned.index)
                    cos_s = pd.Series(cos_comp, index=target_aligned.index)

                    a_coef = target_aligned.rolling(self.fit_window).corr(sin_s)
                    b_coef = target_aligned.rolling(self.fit_window).corr(cos_s)

                    sin_full = np.sin(2 * np.pi * t / period)
                    cos_full = np.cos(2 * np.pi * t / period)
                    a_full = a_coef.reindex(dates).ffill().fillna(0)
                    b_full = b_coef.reindex(dates).ffill().fillna(0)

                    fitted = a_full * sin_full + b_full * cos_full
                    fitted_max = fitted.abs().expanding().max().replace(0, 1)
                    results[f"dewey_{name}_fitted"] = fitted / fitted_max

        return pd.DataFrame(results) if results else pd.DataFrame(index=dates)
