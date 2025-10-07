import pandas as pd
import numpy as np
import warnings

from factorlab.factors.liquidity.base import LiquidityFactor
from factorlab.utils import to_dataframe, grouped


class EDGE(LiquidityFactor):
    """
    Computes the EDGE bid-ask spread estimator from Open/High/Low/Close prices.

    Based on Ardia, Guidotti, & Kroencke (2024), Journal of Financial Economics.
    https://doi.org/10.1016/j.jfineco.2024.103916

    Parameters
    ----------
    sign : bool, default False
        Whether to return the signed root spread.
    """

    def __init__(self,
                 open_col: str = "open",
                 high_col: str = "high",
                 low_col: str = "low",
                 close_col: str = "close",
                 output_col: str = "edge_spread",
                 sign: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.name = "EDGE"
        self.description = "EDGE bid-ask spread estimator from Open/High/Low/Close prices."
        self.category = "Liquidity"
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.output_col = output_col
        self.sign = sign
        if self.window_type == "ewm" or self.window_type == "fixed":
            raise ValueError("EDGE does not support 'ewm' window_type. Use 'rolling' or 'expanding'.")

    @property
    def inputs(self):
        return [self.open_col, self.high_col, self.low_col, self.close_col]

    def _compute_edge_series(self, df: pd.DataFrame) -> float:
        o, h, l, c = df[self.open_col], df[self.high_col], df[self.low_col], df[self.close_col]

        o = np.log(o.to_numpy())
        h = np.log(h.to_numpy())
        l = np.log(l.to_numpy())
        c = np.log(c.to_numpy())
        m = (h + l) / 2.

        h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
        o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

        r1 = m - o
        r2 = o - m1
        r3 = m - c1
        r4 = c1 - m1
        r5 = o - c1

        tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, (h != l) | (l != c1))
        po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
        po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
        pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
        pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pt = np.nanmean(tau)
            po = np.nanmean(po1) + np.nanmean(po2)
            pc = np.nanmean(pc1) + np.nanmean(pc2)

            if np.nansum(tau) < 2 or po == 0 or pc == 0:
                return np.nan

            d1 = r1 - np.nanmean(r1) / pt * tau
            d3 = r3 - np.nanmean(r3) / pt * tau
            d5 = r5 - np.nanmean(r5) / pt * tau

            x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
            x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

            e1 = np.nanmean(x1)
            e2 = np.nanmean(x2)
            v1 = np.nanmean(x1 ** 2) - e1 ** 2
            v2 = np.nanmean(x2 ** 2) - e2 ** 2

        vt = v1 + v2
        s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.
        s = np.sqrt(np.abs(s2))
        return float(s * np.sign(s2)) if self.sign else float(s)

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the EDGE bid-ask spread estimator.

        Parameters
        ----------
        df: pd.DataFrame
            The input DataFrame containing the required high, low, and close price columns.

        Returns
        -------
        pd.Series
            A Series with the computed EDGE spread estimates in the specified output column.

        """

        df = df.copy()

        multiindex = isinstance(df.index, pd.MultiIndex)
        g = grouped(df, axis="ts")

        if multiindex:
            results, out = [], None

            for group, gdf in g:

                if self.window_type == "rolling":
                    idx = gdf.index[self.window_size - 1:]
                    out = pd.Series([
                        self._compute_edge_series(gdf.iloc[i - self.window_size:i])
                        for i in range(self.window_size, len(gdf) + 1)
                    ], index=idx)

                elif self.window_type == "expanding":
                    idx = gdf.index[2:]
                    out = pd.Series([
                        self._compute_edge_series(gdf.iloc[:i])
                        for i in range(3, len(gdf) + 1)
                    ], index=idx)

                results.append(out)

            # Combine all series and sort index
            result = to_dataframe(pd.concat(results).sort_index(), name=self.output_col)

        else:

            if self.window_type == "rolling":
                result = []
                index = g.index[self.window_size - 1:]

                for i in range(self.window_size, len(g) + 1):
                    window = g.iloc[i - self.window_size:i]
                    val = self._compute_edge_series(window)
                    result.append(val)

                result = pd.Series(result, index=index)

            elif self.window_type == "expanding":
                result = []
                index = g.index[2:]

                for i in range(3, len(g) + 1):
                    window = g.iloc[:i]  # expanding window: start from 0 to i
                    val = self._compute_edge_series(window)
                    result.append(val)

                result = pd.Series(result, index=index)

            else:
                raise ValueError(f"Invalid window_type: {self.window_type}")

            result = to_dataframe(result, name=self.output_col).sort_index()

        edge = result[self.output_col]

        return edge
