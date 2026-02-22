import pandas as pd
import numpy as np
from factorlab.core.base_transform import BaseTransform


class Orthogonalize(BaseTransform):
    """
    Orthogonalize factors.

    As described by Klein and Chow (2013) in Orthogonalized Factors and Systematic Risk Decompositions:
    https://www.sciencedirect.com/science/article/abs/pii/S1062976913000185

    They propose an optimal simultaneous orthogonal transformation of factors, following the so-called symmetric
    procedure of Schweinler and Wigner (1970) and Löwdin (1970).  The data transformation allows the identification
    of the underlying uncorrelated components of common factors without changing their correlation with the original
    factors. It also facilitates the systematic risk decomposition by disentangling the coefficient of determination
    (R²) based on factor volatilities, which makes it easier to distinguish the marginal risk contribution of each

    Returns
    -------
    orthogonal_factors: pd.DataFrame
        Orthogonalized factors.
    """
    # convert to array
    if isinstance(df, pd.DataFrame):
        # convert to numpy
        arr = df.to_numpy(dtype=np.float64)
    else:
        arr = df.copy()

    # compute cov matrix
    M = np.cov(arr.T)
    # factorize cov matrix M
    u, s, vh = np.linalg.svd(M)
    # solve for symmetric matrix
    S = u @ np.diag(s ** (-0.5)) @ vh
    # rescale symmetric matrix to original variances
    M[M < 0] = np.nan  # remove negative values
    S_rs = S @ (np.diag(np.sqrt(M)) * np.eye(S.shape[0], S.shape[1]))
    # convert to orthogonal matrix
    orthogonal_arr = arr @ S_rs

    return pd.DataFrame(orthogonal_arr, index=df.index, columns=df.columns)