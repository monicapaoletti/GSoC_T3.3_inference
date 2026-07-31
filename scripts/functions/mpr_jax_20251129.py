import warnings
import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from flax import struct
from functools import partial
import time

import numpy as np


# is P is a static argument, use @partial(jit, static_argnames=["P"])
#@partial(jax.jit, static_argnames=["nn"])
#@jax.jit
def f_mpr(x, t, P, nn): #, nn, method, output
    """
    MPR model
    x: state vector at time t split into x0 and x1 (r and v), t time variable (time step at which r and v are calculated), P parameter object strucure
    
    """

    #nn = P.nn #cancel if partially jitted
    x0 = jax.lax.dynamic_slice(x, (0,), (nn,))
    x1 = jax.lax.dynamic_slice(x, (nn,), (nn,))

    delta_over_tau_pi = P.delta / (P.tau * jnp.pi)
    J_tau = P.J * P.tau
    pi2 = jnp.pi ** 2
    tau2 = P.tau ** 2
    rtau = 1.0 / P.tau

    coupling = jnp.dot(P.weights, x0)

    dx0 = rtau * (delta_over_tau_pi + 2.0 * x0 * x1)
    dx1 = rtau * (
        x1 ** 2 + P.eta + P.iapp + J_tau * x0 - (pi2 * tau2 * x0 ** 2) + P.G * coupling
    )

    dxdt = jnp.concatenate([dx0, dx1])
    dxdt = jnp.clip(dxdt, -10.0, 10.0)
    #dxdt = 50.0 * jnp.tanh(dxdt/50.0)
    return dxdt

#@jax.jit
def do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P):
    """balloon windkessel model
    r_in: averge firing rate form the Montbriò model (mpr)
    s: vasoactive signal (s[0] is current, s[1] is next)
    f: blood inf low
    f_tilde, v_tilde, q_tilde: log transformed versions of f,v,q for numerical stability
    dtt: time step
    P: parameters (k: decay rate of f, 
        gamma: gain of flow-inducing singal, 
        alpha: grubb's exponent(relates flow to volume), 
        tau: transit time through the venous compartment, 
        E_0: resting oxygen extration fraction)
    """
    kappa = P.kappa
    gamma = P.gamma
    ialpha = 1.0 / P.alpha
    tau = P.tau
    Eo = P.Eo    # Vasodilatory signal update

    s1 = s + dtt * (r_in - kappa * s - gamma * (f - 1))

    f_clipped = jnp.clip(f, 1.0, jnp.inf)

    ftilde1 = ftilde + dtt * (s / f_clipped)

    fv = v ** ialpha  
    vtilde1 = vtilde + dtt * ((f_clipped - fv) / (tau * v))

    q_clipped = jnp.clip(q, 0.01, jnp.inf)
    ff = (1 - (1 - Eo) ** (1 / f_clipped)) / Eo
    qtilde1 = qtilde + dtt * ((f_clipped * ff - fv * q_clipped / v) / (tau * q_clipped))

    f1 = jnp.exp(ftilde1)
    v1 = jnp.exp(vtilde1)
    q1 = jnp.exp(qtilde1)

    return s1, f1, ftilde1, vtilde1, qtilde1, v1, q1

def integrate_fast_noise(nn, P, B, key, record_rv, record_bold, nt, rv_decimate, bold_decimate):
    """
    Optimized integration:
    - Pre-generate all noise increments
    - Single lax.scan over time steps
    - Heun integration for neural state
    - BOLD integration
    - Downsampling of RV and BOLD
    """

    dt = P.dt
    r_period = dt * rv_decimate
    dtt = r_period / 1000.0  # BOLD substep in seconds
    vo = B.vo

    # Precompute BOLD constants
    k1 = 4.3 * B.theta0 * B.Eo * B.TE
    k2 = B.epsilon * B.r0 * B.Eo * B.TE
    k3 = 1 - B.epsilon

    # Pre-generate all neural noise (nt x nn)
    key_r, key_v = jax.random.split(key)
    dW_r_all = P.sigma_r * jax.random.normal(key_r, shape=(nt, nn))
    dW_v_all = P.sigma_v * jax.random.normal(key_v, shape=(nt, nn))

    # Initial neural state
    rv = P.initial_state

    # Initial BOLD state
    s = jnp.ones(nn)
    f = jnp.ones(nn)
    ftilde = jnp.zeros(nn)
    vtilde = jnp.zeros(nn)
    qtilde = jnp.zeros(nn)
    v = jnp.ones(nn)
    q = jnp.ones(nn)

    # Preallocate downsampled arrays
    n_rv = nt // rv_decimate if record_rv else 1
    n_bold = nt // bold_decimate if record_bold else 1
    rv_d = jnp.zeros((n_rv, 2 * nn), dtype=jnp.float32)
    rv_t = jnp.zeros((n_rv,), dtype=jnp.float32)
    vv_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)
    qq_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)

    def step(carry, i):
        rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d = carry
        dWr = dW_r_all[i]
        dWv = dW_v_all[i]

        # Advance neural state
        rv_next = heun_sde_with_noise(rv, 0.0, P, nn, dWr, dWv)
        r_in = rv_next[:nn]

        # Advance BOLD
        s, f, ftilde, vtilde, qtilde, v, q = do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, B)

        # Downsample RV
        rv_idx = i // rv_decimate
        rv_d = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rd: rd.at[rv_idx].set(rv_next),
            lambda rd: rd,
            rv_d
        )
        rv_t = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rt: rt.at[rv_idx].set(i * dt * 10.0),
            lambda rt: rt,
            rv_t
        )

        # Downsample BOLD
        bold_idx = i // bold_decimate
        vv_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda vv_: vv_.at[bold_idx].set(v),
            lambda vv_: vv_,
            vv_d
        )
        qq_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda qq_: qq_.at[bold_idx].set(q),
            lambda qq_: qq_,
            qq_d
        )

        return (rv_next, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d), None

    init_carry = (rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d)
    final_carry, _ = jax.lax.scan(step, init_carry, jnp.arange(nt))

    rv_d, rv_t, vv_d, qq_d = final_carry[8], final_carry[9], final_carry[10], final_carry[11]

    # Compute final BOLD
    if record_bold:
        bold_d = vo * (k1 * (1 - qq_d) + k2 * (1 - qq_d / vv_d) + k3 * (1 - vv_d))
        bold_t = jnp.linspace(0, P.t_end - dt * bold_decimate, len(bold_d)) * 10.0
    else:
        bold_d = jnp.zeros((1,), dtype=jnp.float32)
        bold_t = jnp.zeros((1,), dtype=jnp.float32)

    return {
        "rv_t": rv_t.astype(jnp.float32),
        "rv_d": rv_d.astype(jnp.float32),
        "bold_t": bold_t.astype(jnp.float32),
        "bold_d": bold_d.astype(jnp.float32),
    }


# JIT the optimized integrator
integrate_jitted_fast_noise = jax.jit(
    integrate_fast_noise,
    static_argnames=["nn", "record_rv", "record_bold", "nt", "rv_decimate", "bold_decimate"]
)

def integrate_fast_noise_remat(nn, P, B, key, record_rv, record_bold, nt, rv_decimate, bold_decimate, noise_r=None, noise_v=None):
    """
    Same API as integrate_fast_noise but with remat checkpointing of the per-step function.

    - If you provide noise_r / noise_v (shape (nt, nn)), the integrator will use them (differentiable).
    - If not provided, it will generate noise internally (non-differentiable w.r.t. noise).
    """
    dt = P.dt
    r_period = dt * rv_decimate
    dtt = r_period / 1000.0  # BOLD substep in seconds
    vo = B.vo

    # Precompute BOLD constants
    k1 = 4.3 * B.theta0 * B.Eo * B.TE
    k2 = B.epsilon * B.r0 * B.Eo * B.TE
    k3 = 1 - B.epsilon

    # If noise arrays are not provided, generate them (but for differentiability you should pass them).
    if (noise_r is None) or (noise_v is None):
        key_r, key_v = jax.random.split(key)
        noise_r = P.sigma_r * jax.random.normal(key_r, shape=(nt, nn), dtype=jnp.float32)
        noise_v = P.sigma_v * jax.random.normal(key_v, shape=(nt, nn), dtype=jnp.float32)

    # Initial neural state
    rv = P.initial_state

    # Initial BOLD state
    s = jnp.ones(nn)
    f = jnp.ones(nn)
    ftilde = jnp.zeros(nn)
    vtilde = jnp.zeros(nn)
    qtilde = jnp.zeros(nn)
    v = jnp.ones(nn)
    q = jnp.ones(nn)

    # Preallocate downsampled arrays
    n_rv = nt // rv_decimate if record_rv else 1
    n_bold = nt // bold_decimate if record_bold else 1
    rv_d = jnp.zeros((n_rv, 2 * nn), dtype=jnp.float32)
    rv_t = jnp.zeros((n_rv,), dtype=jnp.float32)
    vv_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)
    qq_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)

    # --- define the per-step function that will be rematted ---
    def step(carry, i):
        rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d = carry

        # fetch precomputed noise for this step (shape (nn,))
        dWr = noise_r[i]
        dWv = noise_v[i]

        # Advance neural state (Heun + precomputed noise)
        rv_next = heun_sde_with_noise(rv, 0.0, P, nn, dWr, dWv)
        r_in = rv_next[:nn]

        # Advance BOLD
        s1, f1, ftilde1, vtilde1, qtilde1, v1, q1 = do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, B)

        # Downsample RV
        rv_idx = i // rv_decimate
        rv_d = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rd: rd.at[rv_idx].set(rv_next),
            lambda rd: rd,
            rv_d
        )
        rv_t = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rt: rt.at[rv_idx].set(i * dt * 10.0),
            lambda rt: rt,
            rv_t
        )

        # Downsample BOLD
        bold_idx = i // bold_decimate
        vv_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda vv_: vv_.at[bold_idx].set(v1),
            lambda vv_: vv_,
            vv_d
        )
        qq_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda qq_: qq_.at[bold_idx].set(q1),
            lambda qq_: qq_,
            qq_d
        )

        return (rv_next, s1, f1, ftilde1, vtilde1, qtilde1, v1, q1, rv_d, rv_t, vv_d, qq_d), None

    # Remat (checkpoint) the per-step function:
    # jax.remat(step) will drop activations for the step during forward and recompute during backward.
    remat_step = jax.remat(step)

    init_carry = (rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d)

    # NOTE: we still use lax.scan; the scanned function is rematted.
    final_carry, _ = jax.lax.scan(remat_step, init_carry, jnp.arange(nt))

    rv_d, rv_t, vv_d, qq_d = final_carry[8], final_carry[9], final_carry[10], final_carry[11]

    # Compute final BOLD
    if record_bold:
        bold_d = vo * (k1 * (1 - qq_d) + k2 * (1 - qq_d / vv_d) + k3 * (1 - vv_d))
        bold_t = jnp.linspace(0, P.t_end - dt * bold_decimate, len(bold_d)) * 10.0
    else:
        bold_d = jnp.zeros((1,), dtype=jnp.float32)
        bold_t = jnp.zeros((1,), dtype=jnp.float32)

    return {
        "rv_t": rv_t.astype(jnp.float32),
        "rv_d": rv_d.astype(jnp.float32),
        "bold_t": bold_t.astype(jnp.float32),
        "bold_d": bold_d.astype(jnp.float32),
        "final_state": final_carry[0],                     # rv_next at end
        "final_bold_state": final_carry[1:8],  
    }

integrate_jitted_fast_noise = jax.jit(
    integrate_fast_noise_remat,
    static_argnames=["nn", "record_rv", "record_bold", "nt", "rv_decimate", "bold_decimate"]
)

def heun_sde_with_noise(x, t, P, nn, dW_r, dW_v):
    dt = P.dt
    x0, x1 = x[:nn], x[nn:]
    
    dx = f_mpr(x, t, P, nn)
    dx0, dx1 = dx[:nn], dx[nn:]
    
    # Heun predictor
    r_pred = x0 + dt * dx0 + dW_r
    v_pred = x1 + dt * dx1 + dW_v
    
    dx_pred = f_mpr(jnp.concatenate([r_pred, v_pred]), t + dt, P, nn)
    dx0_pred, dx1_pred = dx_pred[:nn], dx_pred[nn:]
    
    # Heun corrector
    r_new = x0 + 0.5 * dt * (dx0 + dx0_pred) + dW_r
    v_new = x1 + 0.5 * dt * (dx1 + dx1_pred) + dW_v
    
    # Enforce r >= 0
    r_new = jnp.maximum(r_new, 0.0)
    
    return jnp.concatenate([r_new, v_new])

@struct.dataclass
class ParMPR:
    G: float = 0.5
    dt: float = 0.01
    J: float = 14.5
    eta: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([-4.6]))
    tau: float = 1.0
    weights: jnp.ndarray = struct.field(default_factory=lambda: jnp.zeros((0, 0)))
    delta: float = 0.7
    t_init: float = 0.0
    t_cut: float = 0.0
    t_end: float = 1000.0
    nn: int = 0  
    method: str = struct.field(default="default", pytree_node=False)  
    seed: int = 42
    initial_state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    noise_amp: float = 0.037
    sigma_r: float = 0.0  
    sigma_v: float = 0.0  
    iapp: float = 0.0
    output: str = struct.field(default="output", pytree_node=False)
    RECORD_RV: bool = True
    RECORD_BOLD: bool = True
    rv_decimate: int = 10
    tr: float = 1.0#500.0

    @classmethod
    def create(cls, **kwargs):
        """Helper to compute derived quantities and return an instance."""
        weights = kwargs.get("weights", jnp.zeros((0, 0)))
        dt = kwargs.get("dt", 0.01)
        noise_amp = kwargs.get("noise_amp", 0.037)
        sigma_r = jnp.sqrt(dt) * jnp.sqrt(2 * noise_amp)
        sigma_v = jnp.sqrt(dt) * jnp.sqrt(4 * noise_amp)
        nn = int(weights.shape[0])

        return cls(
            sigma_r=sigma_r,
            sigma_v=sigma_v,
            nn=nn,
            **kwargs
        )


@struct.dataclass
class ParBold:
    kappa: float = 0.65
    gamma: float = 0.41
    tau: float = 0.98
    alpha: float = 0.32
    epsilon: float = 0.34
    Eo: float = 0.4
    TE: float = 0.04
    vo: float = 0.08
    r0: float = 25.0
    theta0: float = 40.3
    t_min: float = 0.0
    rtol: float = 1e-5
    atol: float = 1e-8

@struct.dataclass
class MPR_sde:
    P: ParMPR = struct.field(default_factory=ParMPR.create)
    B: ParBold = struct.field(default_factory=ParBold)
    key: jax.Array = struct.field(default_factory=lambda: jax.random.PRNGKey(0))
    initial_state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    INITIAL_STATE_SET: bool = False

    @staticmethod
    def create(par_mpr: dict = {}, key: jax.Array = None) -> "MPR_sde":

        valid_par = list(ParMPR.__annotations__.keys())
        for dictkey in par_mpr:
            if dictkey not in valid_par:
                raise ValueError(f"Invalid parameter: {dictkey}")

        if "initial_state" in par_mpr:
            par_mpr["initial_state"] = jnp.array(par_mpr["initial_state"])
        if "weights" in par_mpr:
            weights = jnp.array(par_mpr["weights"])
            assert weights.shape[0] == weights.shape[1]
            par_mpr["weights"] = weights

        P = ParMPR.create(**par_mpr)
        if hasattr(P, "nn"):
            P = P.replace(nn=int(P.nn))
        B = ParBold()
        if key is not None:
            set_key = key
        else:
            set_key = jax.random.PRNGKey(P.seed)
        return MPR_sde(P=P, B=B, key=set_key)
    
    def with_initial_state(self) -> "MPR_sde":
        key, subkey = jax.random.split(self.key)
        nn = self.P.nn
        if isinstance(nn, jnp.ndarray):
            nn = int(nn.item())
        init_state = set_initial_state(nn, subkey)
        return self.replace(initial_state=init_state, INITIAL_STATE_SET=True, key=key)

    def check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == 2 * self.P.weights.shape[0]

    def run(self, par: dict = {}, x0: jnp.ndarray = None):
        P = self.P

        # If external x0 not provided, use internal initial state
        if x0 is None:
            sim = self.with_initial_state()
            P = P.replace(initial_state=sim.initial_state)
            key = sim.key
        else:
            P = P.replace(initial_state=x0)
            P = P.replace(nn=(len(x0)//2))
            key = self.key
        # Override parameters
        for key, val in par.items():
            if key not in ParMPR.__annotations__:
                raise ValueError(f"Invalid parameter: {key}")
            P = P.replace(**{key: val})
        # Sanity check
        P = P.replace(
            eta=check_vec_size(P.eta, P.nn),
            t_end=P.t_end / 10,
            t_cut=P.t_cut / 10
        )        

        assert P.weights.shape[0] == P.weights.shape[1]
        assert P.initial_state is not None
        assert len(P.initial_state) == 2 * P.nn

        #key = jax.random.PRNGKey(P.seed)
        new_self = self.replace(P=P)
        new_self.check_input()
        nn = int(new_self.P.nn)
        nt = int(new_self.P.t_end / new_self.P.dt)

        dt = float(new_self.P.dt)
        tr = float(new_self.P.tr)
        rv_decimate = int(new_self.P.rv_decimate)
        r_period = dt * rv_decimate
        bold_decimate = int(np.round(tr / r_period))
        key, new_key = jax.random.split(self.key)
        new_self = self.replace(P=P, key=new_key)

        #result = integrate_jitted_fast_noise(
        #                nn=nn,
        #                P=new_self.P,
        #                B=self.B,
        #                key=new_self.key,
        #                record_rv=new_self.P.RECORD_RV,
        #                record_bold=new_self.P.RECORD_BOLD,
        #                nt=nt,#int(new_self.P.t_end / new_self.P.dt),
        #                rv_decimate=rv_decimate,
        #                bold_decimate=bold_decimate)

        # in run()
        key_r, key_v = jax.random.split(new_self.key)
        noise_r = new_self.P.sigma_r * jax.random.normal(key_r, shape=(nt, nn), dtype=jnp.float32)
        noise_v = new_self.P.sigma_v * jax.random.normal(key_v, shape=(nt, nn), dtype=jnp.float32)

        result = integrate_jitted_fast_noise(
            nn=nn,
            P=new_self.P,
            B=self.B,
            key=new_self.key,
            record_rv=new_self.P.RECORD_RV,
            record_bold=new_self.P.RECORD_BOLD,
            nt=nt,
            rv_decimate=rv_decimate,
            bold_decimate=bold_decimate,
            noise_r=noise_r,
            noise_v=noise_v
        )





        """It's paramount to call block_until_ready(), either here, or during pmap/vmap.
        Otherwise it'd seem that the simulation has been completed, when it fact it hasn't!""" 
        return jax.block_until_ready(result)
    
    def run_chunked(self, par: dict = {}, x0: jnp.ndarray = None, block_size: int = 5000, noise_blocks=None):
        # apply par overrides
        P = self.P
        for k, v in par.items():
            P = P.replace(**{k: v})
        P = P.replace(eta=P.eta, t_end=P.t_end / 10, t_cut=P.t_cut / 10)
        if x0 is not None:
            P = P.replace(initial_state=x0)

        nn = int(P.nn)
        nt_total = int(P.t_end / P.dt)
        nblocks = (nt_total + block_size - 1) // block_size

        # compute bold_decimate as integer
        dt = float(P.dt)
        tr = float(P.tr)
        r_period = dt * int(P.rv_decimate)
        bold_decimate = max(1, int(np.round(tr / r_period)))

        use_external = noise_blocks is not None
        if use_external:
            noise_r_seq, noise_v_seq = noise_blocks
            assert len(noise_r_seq) == nblocks and len(noise_v_seq) == nblocks

        # the per-block kernel; block_len and use_external are static so RNG shape is concrete
        #@partial(jax.jit, static_argnames=("block_len", "use_external"))
        def simulate_block(state, bold_state, key, block_len, P, use_external, noise_r_block, noise_v_block):
            # state: neural state (2*nn,)
            # bold_state: tuple(s,f,ftilde,vtilde,qtilde,v,q)
            key, k_r, k_v = jax.random.split(key, 3)

            def ext(_):
                # external block is already the right shape (block_len, nn)
                return noise_r_block, noise_v_block

            def inte(_):
                # generate internal noise of concrete shape (block_len, nn)
                nr = jax.random.normal(k_r, (block_len, nn), dtype=jnp.float32) * P.sigma_r
                nv = jax.random.normal(k_v, (block_len, nn), dtype=jnp.float32) * P.sigma_v
                return nr, nv

            noise_r, noise_v = jax.lax.cond(use_external, ext, inte, operand=None)

            out = integrate_jitted_fast_noise(
                nn=nn,
                P=P,
                B=self.B,
                key=key,
                record_rv=P.RECORD_RV,
                record_bold=P.RECORD_BOLD,
                nt=block_len,
                rv_decimate=P.rv_decimate,
                bold_decimate=bold_decimate,
                noise_r=noise_r,
                noise_v=noise_v,
            )

            final_state = out["final_state"]   # shape (2*nn,)
            final_bold_state = out["final_bold_state"]  # tuple
            return final_state, final_bold_state, key, out

        # Python-side loop calling the jitted block kernel (keeps memory low)
        state = P.initial_state
        # initialize bold_state to zeros same form as integrator
        bold_state = (jnp.ones(nn), jnp.ones(nn), jnp.zeros(nn), jnp.zeros(nn), jnp.zeros(nn), jnp.ones(nn), jnp.ones(nn))
        key = self.key

        rv_blocks = []
        bold_blocks = []
        remaining = nt_total

        for b in range(nblocks):
            block_len = min(block_size, remaining)
            if use_external:
                nrb = noise_r_seq[b]
                nvb = noise_v_seq[b]
            else:
                # placeholders whose shapes are only used in the ext branch; ext will be unused (use_external False)
                nrb = jnp.zeros((block_len, nn), dtype=jnp.float32)
                nvb = jnp.zeros((block_len, nn), dtype=jnp.float32)

            state, bold_state, key, out = simulate_block(state, bold_state, key, block_len, P, use_external, nrb, nvb)

            if P.RECORD_RV:
                rv_blocks.append(out["rv_d"])
            if P.RECORD_BOLD:
                bold_blocks.append(out["bold_d"])

            remaining -= block_len

        rv = jnp.concatenate(rv_blocks, axis=0) if len(rv_blocks) > 0 else None
        bold = jnp.concatenate(bold_blocks, axis=0) if len(bold_blocks) > 0 else None
        return {"rv": rv, "bold": bold, "final_state": state, "final_bold_state": bold_state}




#@jax.jit(static_argnames=["nn"])
def set_initial_state(nn, key):

    key_r, key_v = jax.random.split(key)

    r = jax.random.uniform(key_r, shape=(nn,), minval=0.0, maxval=1.5)
    v = jax.random.uniform(key_v, shape=(nn,), minval=-2.0, maxval=2.0)

    y0 = jnp.concatenate([r, v])
    return y0


def check_vec_size(x, nn):
    """
    Ensures that `x` is a vector of length `nn`.
    If it's a scalar, broadcast it into a vector of that length.
    """
    x = jnp.asarray(x)
    return jnp.ones(nn) * x if x.ndim == 0 or x.size != nn else x




###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
# MORNING VERSION ABOVE, EVENING VERSION BELOW
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################




import warnings
import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from flax import struct
from functools import partial
import time

import numpy as np


# is P is a static argument, use @partial(jit, static_argnames=["P"])
#@partial(jax.jit, static_argnames=["nn"])
#@jax.jit
def f_mpr(x, t, P, nn): #, nn, method, output
    """
    MPR model
    x: state vector at time t split into x0 and x1 (r and v), t time variable (time step at which r and v are calculated), P parameter object strucure
    
    """

    #nn = P.nn #cancel if partially jitted
    x0 = jax.lax.dynamic_slice(x, (0,), (nn,))
    x1 = jax.lax.dynamic_slice(x, (nn,), (nn,))

    delta_over_tau_pi = P.delta / (P.tau * jnp.pi)
    J_tau = P.J * P.tau
    pi2 = jnp.pi ** 2
    tau2 = P.tau ** 2
    rtau = 1.0 / P.tau

    coupling = jnp.dot(P.weights, x0)

    dx0 = rtau * (delta_over_tau_pi + 2.0 * x0 * x1)
    dx1 = rtau * (
        x1 ** 2 + P.eta + P.iapp + J_tau * x0 - (pi2 * tau2 * x0 ** 2) + P.G * coupling
    )

    dxdt = jnp.concatenate([dx0, dx1])
    dxdt = jnp.clip(dxdt, -10.0, 10.0)
    #dxdt = 50.0 * jnp.tanh(dxdt/50.0)
    return dxdt

def heun_sde_with_noise(x, t, P, nn, dW_r, dW_v):
    dt = P.dt
    x0, x1 = x[:nn], x[nn:]
    
    dx = f_mpr(x, t, P, nn)
    dx0, dx1 = dx[:nn], dx[nn:]
    
    # Heun predictor
    r_pred = x0 + dt * dx0 + dW_r
    v_pred = x1 + dt * dx1 + dW_v
    
    dx_pred = f_mpr(jnp.concatenate([r_pred, v_pred]), t + dt, P, nn)
    dx0_pred, dx1_pred = dx_pred[:nn], dx_pred[nn:]
    
    # Heun corrector
    r_new = x0 + 0.5 * dt * (dx0 + dx0_pred) + dW_r
    v_new = x1 + 0.5 * dt * (dx1 + dx1_pred) + dW_v
    
    # Enforce r >= 0
    r_new = jnp.maximum(r_new, 0.0)
    
    return jnp.concatenate([r_new, v_new])

#@jax.jit
def do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P):
    """balloon windkessel model
    r_in: averge firing rate form the Montbriò model (mpr)
    s: vasoactive signal (s[0] is current, s[1] is next)
    f: blood inf low
    f_tilde, v_tilde, q_tilde: log transformed versions of f,v,q for numerical stability
    dtt: time step
    P: parameters (k: decay rate of f, 
        gamma: gain of flow-inducing singal, 
        alpha: grubb's exponent(relates flow to volume), 
        tau: transit time through the venous compartment, 
        E_0: resting oxygen extration fraction)
    """
    kappa = P.kappa
    gamma = P.gamma
    ialpha = 1.0 / P.alpha
    tau = P.tau
    Eo = P.Eo    # Vasodilatory signal update

    s1 = s + dtt * (r_in - kappa * s - gamma * (f - 1))

    f_clipped = jnp.clip(f, 1.0, jnp.inf)

    ftilde1 = ftilde + dtt * (s / f_clipped)

    fv = v ** ialpha  
    vtilde1 = vtilde + dtt * ((f_clipped - fv) / (tau * v))

    q_clipped = jnp.clip(q, 0.01, jnp.inf)
    ff = (1 - (1 - Eo) ** (1 / f_clipped)) / Eo
    qtilde1 = qtilde + dtt * ((f_clipped * ff - fv * q_clipped / v) / (tau * q_clipped))

    f1 = jnp.exp(ftilde1)
    v1 = jnp.exp(vtilde1)
    q1 = jnp.exp(qtilde1)

    return s1, f1, ftilde1, vtilde1, qtilde1, v1, q1

def integrate_fast_noise_remat(
    nn, P, B, key, record_rv, record_bold, nt, rv_decimate, bold_decimate,
    noise_r=None, noise_v=None, initial_bold_state=None
):
    """
    Same API as integrate_fast_noise but with remat checkpointing.
    Supports carrying over BOLD state between blocks.
    """
    dt = P.dt
    r_period = dt * rv_decimate
    dtt = r_period / 1000.0  # BOLD substep in seconds
    vo = B.vo

    # Precompute BOLD constants
    k1 = 4.3 * B.theta0 * B.Eo * B.TE
    k2 = B.epsilon * B.r0 * B.Eo * B.TE
    k3 = 1 - B.epsilon

    # Generate noise if not provided
    if (noise_r is None) or (noise_v is None):
        key_r, key_v = jax.random.split(key)
        noise_r = P.sigma_r * jax.random.normal(key_r, shape=(nt, nn), dtype=jnp.float32)
        noise_v = P.sigma_v * jax.random.normal(key_v, shape=(nt, nn), dtype=jnp.float32)

    # Initial neural state
    rv = P.initial_state

    # Initial BOLD state
    if initial_bold_state is not None:
        s, f, ftilde, vtilde, qtilde, v, q = initial_bold_state
    else:
        s = jnp.ones(nn)
        f = jnp.ones(nn)
        ftilde = jnp.zeros(nn)
        vtilde = jnp.zeros(nn)
        qtilde = jnp.zeros(nn)
        v = jnp.ones(nn)
        q = jnp.ones(nn)

    # Preallocate downsampled arrays
    n_rv = nt // rv_decimate if record_rv else 1
    n_bold = nt // bold_decimate if record_bold else 1
    rv_d = jnp.zeros((n_rv, 2 * nn), dtype=jnp.float32)
    rv_t = jnp.zeros((n_rv,), dtype=jnp.float32)
    vv_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)
    qq_d = jnp.zeros((n_bold, nn), dtype=jnp.float32)

    # --- define the per-step function ---
    def step(carry, i):
        rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d = carry
        dWr = noise_r[i]
        dWv = noise_v[i]

        # Advance neural state
        rv_next = heun_sde_with_noise(rv, 0.0, P, nn, dWr, dWv)
        r_in = rv_next[:nn]

        # Advance BOLD
        s1, f1, ftilde1, vtilde1, qtilde1, v1, q1 = do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, B)

        # Downsample RV
        rv_idx = i // rv_decimate
        rv_d = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rd: rd.at[rv_idx].set(rv_next),
            lambda rd: rd,
            rv_d
        )
        rv_t = jax.lax.cond(
            record_rv & (i % rv_decimate == 0),
            lambda rt: rt.at[rv_idx].set(i * dt * 10.0),
            lambda rt: rt,
            rv_t
        )

        # Downsample BOLD
        bold_idx = i // bold_decimate
        vv_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda vv_: vv_.at[bold_idx].set(v1),
            lambda vv_: vv_,
            vv_d
        )
        qq_d = jax.lax.cond(
            record_bold & (i % bold_decimate == 0),
            lambda qq_: qq_.at[bold_idx].set(q1),
            lambda qq_: qq_,
            qq_d
        )

        return (rv_next, s1, f1, ftilde1, vtilde1, qtilde1, v1, q1, rv_d, rv_t, vv_d, qq_d), None

    remat_step = jax.remat(step)
    init_carry = (rv, s, f, ftilde, vtilde, qtilde, v, q, rv_d, rv_t, vv_d, qq_d)
    final_carry, _ = jax.lax.scan(remat_step, init_carry, jnp.arange(nt))

    rv_d, rv_t, vv_d, qq_d = final_carry[8], final_carry[9], final_carry[10], final_carry[11]

    # Compute final BOLD signal
    if record_bold:
        bold_d = vo * (k1 * (1 - qq_d) + k2 * (1 - qq_d / vv_d) + k3 * (1 - vv_d))
        bold_t = jnp.linspace(0, P.t_end - dt * bold_decimate, len(bold_d)) * 10.0
    else:
        bold_d = jnp.zeros((1,), dtype=jnp.float32)
        bold_t = jnp.zeros((1,), dtype=jnp.float32)

    return {
        "rv_t": rv_t.astype(jnp.float32),
        "rv_d": rv_d.astype(jnp.float32),
        "bold_t": bold_t.astype(jnp.float32),
        "bold_d": bold_d.astype(jnp.float32),
        "final_state": final_carry[0],
        "final_bold_state": final_carry[1:8],  # s, f, ftilde, vtilde, qtilde, v, q
    }

integrate_jitted_fast_noise = jax.jit(
    integrate_fast_noise_remat,
    static_argnames=["nn", "record_rv", "record_bold", "nt", "rv_decimate", "bold_decimate"]
)

@struct.dataclass
class ParMPR:
    G: float = 0.5
    dt: float = 0.01
    J: float = 14.5
    eta: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([-4.6]))
    tau: float = 1.0
    weights: jnp.ndarray = struct.field(default_factory=lambda: jnp.zeros((0, 0)))
    delta: float = 0.7
    t_init: float = 0.0
    t_cut: float = 0.0
    t_end: float = 1000.0
    nn: int = 0  
    method: str = struct.field(default="default", pytree_node=False)  
    seed: int = 42
    initial_state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    noise_amp: float = 0.037
    sigma_r: float = 0.0  
    sigma_v: float = 0.0  
    iapp: float = 0.0
    output: str = struct.field(default="output", pytree_node=False)
    RECORD_RV: bool = True
    RECORD_BOLD: bool = True
    rv_decimate: int = 10
    tr: float = 300.0#500.0

    @classmethod
    def create(cls, **kwargs):
        """Helper to compute derived quantities and return an instance."""
        weights = kwargs.get("weights", jnp.zeros((0, 0)))
        dt = kwargs.get("dt", 0.01)
        noise_amp = kwargs.get("noise_amp", 0.037)
        sigma_r = jnp.sqrt(dt) * jnp.sqrt(2 * noise_amp)
        sigma_v = jnp.sqrt(dt) * jnp.sqrt(4 * noise_amp)
        nn = int(weights.shape[0])

        return cls(
            sigma_r=sigma_r,
            sigma_v=sigma_v,
            nn=nn,
            **kwargs
        )


@struct.dataclass
class ParBold:
    kappa: float = 0.65
    gamma: float = 0.41
    tau: float = 0.98
    alpha: float = 0.32
    epsilon: float = 0.34
    Eo: float = 0.4
    TE: float = 0.04
    vo: float = 0.08
    r0: float = 25.0
    theta0: float = 40.3
    t_min: float = 0.0
    rtol: float = 1e-5
    atol: float = 1e-8

@struct.dataclass
class MPR_sde:
    P: ParMPR = struct.field(default_factory=ParMPR.create)
    B: ParBold = struct.field(default_factory=ParBold)
    key: jax.Array = struct.field(default_factory=lambda: jax.random.PRNGKey(0))
    initial_state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    INITIAL_STATE_SET: bool = False

    @staticmethod
    def create(par_mpr: dict = {}, key: jax.Array = None) -> "MPR_sde":

        valid_par = list(ParMPR.__annotations__.keys())
        for dictkey in par_mpr:
            if dictkey not in valid_par:
                raise ValueError(f"Invalid parameter: {dictkey}")

        if "initial_state" in par_mpr:
            par_mpr["initial_state"] = jnp.array(par_mpr["initial_state"])
        if "weights" in par_mpr:
            weights = jnp.array(par_mpr["weights"])
            assert weights.shape[0] == weights.shape[1]
            par_mpr["weights"] = weights

        P = ParMPR.create(**par_mpr)
        if hasattr(P, "nn"):
            P = P.replace(nn=int(P.nn))
        B = ParBold()
        if key is not None:
            set_key = key
        else:
            set_key = jax.random.PRNGKey(P.seed)
        return MPR_sde(P=P, B=B, key=set_key)
    
    def with_initial_state(self) -> "MPR_sde":
        key, subkey = jax.random.split(self.key)
        nn = self.P.nn
        if isinstance(nn, jnp.ndarray):
            nn = int(nn.item())
        init_state = set_initial_state(nn, subkey)
        return self.replace(initial_state=init_state, INITIAL_STATE_SET=True, key=key)

    def check_input(self):
        assert self.P.weights is not None
        assert self.P.weights.shape[0] == self.P.weights.shape[1]
        assert self.P.initial_state is not None
        assert len(self.P.initial_state) == 2 * self.P.weights.shape[0]
    
    def run(self, par: dict = {}, x0: jnp.ndarray = None,
            block_size: int = 10_000, noise_blocks=None):

        """FULL replacement for run(), matching old behavior but chunked."""

        P = self.P

        # ✔ Apply external x0 or internal initial state
        if x0 is None:
            sim = self.with_initial_state()
            P = P.replace(initial_state=sim.initial_state)
            key = sim.key
        else:
            P = P.replace(initial_state=x0)
            P = P.replace(nn=(len(x0)//2))
            key = self.key

        # ✔ Override parameters (validate keys exactly like old run)
        for k, val in par.items():
            if k not in ParMPR.__annotations__:
                raise ValueError(f"Invalid parameter: {k}")
            P = P.replace(**{k: val})

        # ✔ Same time scaling as original run
        P = P.replace(
            eta=check_vec_size(P.eta, P.nn),
            t_end=P.t_end / 10,
            t_cut=P.t_cut / 10
        )

        # 🔥 Create updated model instance (critical!)
        new_self = self.replace(P=P, key=key)
        new_self.check_input()

        nn = int(new_self.P.nn)
        nt_total = int(new_self.P.t_end / new_self.P.dt)

        dt = float(new_self.P.dt)
        tr = float(new_self.P.tr)
        rv_decimate = int(new_self.P.rv_decimate)
        r_period = dt * rv_decimate
        bold_decimate = max(1, int(np.round(tr / r_period)))

        # ==========================================================
        #                  CHUNKED INTEGRATION LOOP
        # ==========================================================

        state = new_self.P.initial_state
        bold_state = (
            jnp.ones(nn), jnp.ones(nn), jnp.zeros(nn),
            jnp.zeros(nn), jnp.zeros(nn), jnp.ones(nn), jnp.ones(nn)
        )

        nblocks = (nt_total + block_size - 1) // block_size
        rv_blocks, bold_blocks = [], []

        for i in range(nblocks):
            blen = min(block_size, nt_total - i*block_size)

            key, subkey = jax.random.split(key)

            if noise_blocks is not None:
                noise_r, noise_v = noise_blocks[0][i], noise_blocks[1][i]
            else:
                noise_r = noise_v = None

            P_block = new_self.P.replace(initial_state=state)

            out = integrate_fast_noise_remat(
                nn=nn, P=P_block, B=new_self.B,
                key=subkey, record_rv=P.RECORD_RV, record_bold=P.RECORD_BOLD,
                nt=blen, rv_decimate=rv_decimate, bold_decimate=bold_decimate,
                noise_r=noise_r, noise_v=noise_v,
                initial_bold_state=bold_state                # ✔ carry BOLD forward
            )

            state = out["final_state"]
            bold_state = out["final_bold_state"]            # ✔ persist BOLD

            if P.RECORD_RV:   rv_blocks.append(out["rv_d"])
            if P.RECORD_BOLD: bold_blocks.append(out["bold_d"])

        rv = jnp.concatenate(rv_blocks) if rv_blocks else None
        bold = jnp.concatenate(bold_blocks) if bold_blocks else None

        # ==========================================================
        #                    SAME OUTPUT FORMAT AS OLD
        # ==========================================================
        dt = float(new_self.P.dt)

        if bold is not None:
            bold_t = jnp.arange(bold.shape[0]) * (dt * rv_decimate * bold_decimate) * 10.0
        else:
            bold_t = None

        rv_t = jnp.arange(rv.shape[0]) * (dt * rv_decimate) * 10.0 if rv is not None else None
        
        result = jax.block_until_ready({
            "rv_d": rv,
            "bold_d": bold,
            "final_state": state,
            "final_bold_state": bold_state,
            "rv_t": rv_t,
            "bold_t": bold_t,
        })

        return jax.block_until_ready(result)


#@jax.jit(static_argnames=["nn"])
def set_initial_state(nn, key):

    key_r, key_v = jax.random.split(key)

    r = jax.random.uniform(key_r, shape=(nn,), minval=0.0, maxval=1.5)
    v = jax.random.uniform(key_v, shape=(nn,), minval=-2.0, maxval=2.0)

    y0 = jnp.concatenate([r, v])
    return y0


def check_vec_size(x, nn):
    """
    Ensures that `x` is a vector of length `nn`.
    If it's a scalar, broadcast it into a vector of that length.
    """
    x = jnp.asarray(x)
    return jnp.ones(nn) * x if x.ndim == 0 or x.size != nn else x


