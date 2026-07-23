import jax
import jax.numpy as jnp

def _corrcoef(x, eps=0.0):
    """Pearson correlation of rows of `x`. eps<=0 -> jnp.corrcoef (bit-exact).
    eps>0 -> stabilised denominator sqrt(var_i*var_j + eps) so zero-variance
    rows give 0 instead of NaN (keeps gradients finite)."""
    if eps <= 0:
        return jnp.corrcoef(x)
    xc = x - jnp.mean(x, axis=1, keepdims=True)
    T = x.shape[1]
    cov = (xc @ xc.T) / (T - 1)
    var = jnp.diag(cov)
    return cov / jnp.sqrt(jnp.outer(var, var) + eps)

def get_fc(bold, eps=0.0, keep_negative=False):
    """
    Extract the functional correlation (pearson correlation) of the input (bold_d.T from running the mpr model).
    eps>0 uses a stabilised correlation (safe for gradients); eps=0 is the original jnp.corrcoef.
    keep_negative=False (default) zeroes negative correlations (the original ReLU);
    True keeps anti-correlations (more info, no zero-variance-feature spike, smoother).
    """
    FC = _corrcoef(bold, eps)
    if not keep_negative:
        FC = FC * (FC > 0)
    FC = FC - jnp.diag(jnp.diag(FC))
    return FC

def stable_corrcoef(bold, eps=1e-6):
    """
    for model with gradients
    """
    # bold: shape (nodes, timepoints)
    x = bold - jnp.mean(bold, axis=1, keepdims=True)
    T = bold.shape[1]
    cov = (x @ x.T) / (T - 1)
    var = jnp.diag(cov)
    denom = jnp.sqrt(jnp.outer(var, var) + eps)
    corr = cov / denom
    corr = corr * (corr > 0)
    corr = corr - jnp.diag(jnp.diag(corr))
    return corr

def extract_FCD(data, wwidth=30, maxNwindows=200, olap=0.94, coldata=False, mode='corr'):
    """
    Extract Functional Connectivity Dynamics (FCD) from time series data.  

    This function computes time-resolved functional connectivity using a sliding window approach and 
    derives the Functional Connectivity Dynamics (FCD) matrix by correlating the vectorized FC matrices across windows. 
    It supports multiple modes of FC computation: correlation, phase synchronization, phase locking, and time-delayed correlation.

    Parameters
    ----------
    data : jax.numpy.ndarray
        Time series data of shape (nodes, timepoints) (e.g. bold_d.T) or (timepoints, nodes) if `coldata=True`.
    wwidth : int, optional
        Width of the sliding window in time points (default is 30).
    maxNwindows : int, optional
        Maximum number of windows to compute (default is 200).
    olap : float, optional
        Fractional overlap between consecutive windows (must be <1, default is 0.94).
    coldata : bool, optional
        If True, assumes data is organized as (timepoints, nodes) and will be transposed internally (default is False).
    mode : str, optional
        Functional connectivity mode to use. Options:
        - 'corr': Pearson correlation (default)
        - 'psync': phase synchronization
        - 'plock': phase locking
        - 'tdcorr': time-delayed correlation

    Returns
    -------
    fcd_matrix : jax.numpy.ndarray
        Functional Connectivity Dynamics (FCD) matrix of shape (Nwindows, Nwindows), representing
        correlations between vectorized FC matrices across all windows.
    corr_vectors : jax.numpy.ndarray
        Array of vectorized FC matrices of shape (Nwindows, nnodes*(nnodes-1)/2).
    shift : int
        Number of timepoints between the start of consecutive windows.
    """
    if olap >= 1.0:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data = data.T

    nnodes, lenseries = data.shape

    est_windows = (lenseries - wwidth * olap) // (wwidth * (1 - olap))
    Nwindows = int(min(est_windows, maxNwindows))
    #print(est_windows,Nwindows)
    if Nwindows < 1:
        raise ValueError("Too few windows")
    shift = int((lenseries - wwidth) // (Nwindows - 1)) if Nwindows > 1 else lenseries - wwidth
    starts = jnp.arange(0, lenseries - wwidth + 1, shift, dtype=int)
    stops = starts + wwidth

    def compute_corr_mat(aux_s):
        if mode == 'corr':
            return jnp.corrcoef(aux_s)
        mat = jnp.zeros((nnodes, nnodes))
        for i in range(nnodes):
            for j in range(i):
                if mode == 'psync':
                    val = jnp.mean(jnp.abs(jnp.mean(jnp.exp(1j * aux_s[[i, j], :]), axis=0)))
                elif mode == 'plock':
                    diff = jnp.diff(aux_s[[i, j], :], axis=0)
                    val = jnp.abs(jnp.mean(jnp.exp(1j * diff)))
                elif mode == 'tdcorr':
                    xi, xj = aux_s[i, :], aux_s[j, :]
                    cross = jnp.correlate(xi, xj, mode='full')
                    mid = cross.shape[0] // 2
                    max_corr = jnp.max(cross[mid : mid + wwidth])
                    norm = jnp.sqrt(jnp.dot(xi, xi) * jnp.dot(xj, xj))
                    val = max_corr / norm
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                mat = mat.at[i, j].set(val)
        return mat

    # Build all FC matrices
    fc_list = []
    for j1, j2 in zip(starts, stops):
        aux_s = data[:, j1:j2]
        fc_list.append(compute_corr_mat(aux_s))
    fc_matrices = jnp.stack(fc_list)

    tril_idx = jnp.tril_indices(nnodes, -1)
    corr_vectors = jax.vmap(lambda m: m[tril_idx])(fc_matrices)

    CV_centered = corr_vectors - jnp.mean(corr_vectors, axis=1, keepdims=True)
    fcd_matrix = jnp.corrcoef(CV_centered)

    return fcd_matrix, corr_vectors, shift



@jax.jit
def extract_FCD_jax_old(data, wwidth=30, olap=0.9, shift=None):
    nnodes, T = data.shape
    shift = jnp.round(wwidth * (1 - olap)).astype(int)
    starts = jnp.arange(0, T - wwidth + 1, shift, dtype=int)

    def compute_fc(start):
        window = data[:, start:start+wwidth]
        fc = jnp.corrcoef(window)
        fc = fc * (fc > 0)
        tril_idx = jnp.tril_indices(nnodes, -1)
        return fc[tril_idx]

    corr_vectors = jax.vmap(compute_fc)(starts)
    CV_centered = corr_vectors - jnp.mean(corr_vectors, axis=1, keepdims=True)
    fcd_matrix = jnp.corrcoef(CV_centered)
    return fcd_matrix


# compute nnodes, T = data.shape before calling the next function

def precompute_shift_and_starts(T, wwidth, olap, stride1=False):
    # stride1=True reproduces vbi's get_fcd windowing (one window per sample,
    # shift=1) for which the hardcoded FCD triu offset k=30 was designed. Default
    # (stride1=False) keeps the overlap-fraction shift = round(wwidth*(1-olap)),
    # i.e. the prior behaviour -> pass stride1 to switch, omit it to revert.
    shift = 1 if stride1 else int((jnp.round(wwidth * (1 - olap)).astype(int)))
    starts = jnp.arange(0, T - wwidth + 1, shift, dtype=int)
    return shift, starts

#@jax.jit(static_argnums=(2, 3))
def extract_FCD_jax(data, starts, nnodes, wwidth=30, olap=0.94, eps=0.0, keep_negative=False):
    def compute_fc(start):
        window = jax.lax.dynamic_slice(data, (0, start), (nnodes, wwidth))
        #print("Window shape:", window.shape)
        fc = _corrcoef(window, eps)
        if not keep_negative:
            fc = fc * (fc > 0)
        tril_idx = jnp.tril_indices(nnodes, -1)
        return fc[tril_idx]

    # Vectorized loop over the starting indices
    corr_vectors = jax.vmap(compute_fc)(starts)

    # Center the correlation vectors
    CV_centered = corr_vectors - jnp.mean(corr_vectors, axis=1, keepdims=True)

    # Compute FCD matrix as the correlation of the centered vectors
    fcd_matrix = _corrcoef(CV_centered, eps)

    return fcd_matrix

extract_FCD_jax_jitted = jax.jit(extract_FCD_jax, static_argnums=(2, 3))

def fluidity(fcd, win_len=30, overlap=0.94):
    k = int((float(win_len) / (float(win_len) - float(overlap))) + 1)
    n = fcd.shape[0]
    idx = jnp.triu_indices(n, k)
    return jnp.var(fcd[idx])