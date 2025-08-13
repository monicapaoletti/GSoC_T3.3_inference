import jax
import jax.numpy as jnp

def get_fc(bold):
    FC = jnp.corrcoef(bold)
    FC = FC * (FC > 0)
    FC = FC - jnp.diag(jnp.diag(FC))
    return FC

def extract_FCD(data, wwidth=1000, maxNwindows=100, olap=0.9, coldata=False, mode='corr'):
    if olap >= 1.0:
        raise ValueError("olap must be lower than 1")
    if coldata:
        data = data.T

    nnodes, lenseries = data.shape

    est_windows = (lenseries - wwidth * olap) // (wwidth * (1 - olap))
    Nwindows = int(min(est_windows, maxNwindows))
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
