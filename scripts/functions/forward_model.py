import numpy as np
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from copy import deepcopy
import mpr_jax
import FCD_jax

@as_op(itypes=[pt.dscalar], otypes=[pt.dvector])
def pytensor_forward_model_matrix(G, params, cut, tr, starts, nn):
    par = deepcopy(params)
    par["G"] = float(G)

    sde = mpr_jax.MPR_sde.create(par)
    data = sde.run({})
    bold_d = data["bold_d"]

    FC_full = FCD_jax.get_fc(bold_d[int(cut):].T)
    tri_idx = np.triu_indices(FC_full.shape[0], k=1)
    return np.asarray(FC_full[tri_idx], dtype=np.float64).flatten()
