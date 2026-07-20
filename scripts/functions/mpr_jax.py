import warnings
import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from flax import struct
from functools import partial
import time

import numpy as np

_PI2 = float(np.pi ** 2)

# is P is a static argument, use @partial(jit, static_argnames=["P"])
#@partial(jax.jit, static_argnames=["nn"])
#@jax.jit
def f_mpr(r, v, P):
    """
    MPR model right-hand side, operating on the firing rate `r` and membrane
    potential `v` as separate arrays (avoids the slice/concat churn of a stacked
    state). Returns the derivatives (dr, dv) as separate arrays.
    """
    cm = P.clip_mode
    coupling = jnp.dot(P.weights, r)
    if cm == "hard":
        coupling = jnp.clip(coupling, -5.0, 5.0)
    elif cm == "soft":
        coupling = 5.0 * jnp.tanh(coupling / 5.0)

    dr = P.rtau * (P.delta_over_tau_pi + 2.0 * r * v)
    dv = P.rtau * (
        v ** 2 + P.eta + P.iapp + P.J_tau * r - P.pi_tau_sq * r ** 2 + P.G * coupling
    )

    # Bound the derivatives to avoid chaotic blow-up. "hard" = jnp.clip (current
    # default, bit-exact, but zero gradient outside the bounds). "soft" = tanh
    # saturation: same bound, smooth, gradient 1-tanh^2 in (0,1] everywhere ->
    # no dead-zones, finite forward. "none" = unbounded (NaNs at high G).
    if cm == "hard":
        dr = jnp.clip(dr, -10.0, 10.0)
        dv = jnp.clip(dv, -10.0, 10.0)
    elif cm == "soft":
        dr = 10.0 * jnp.tanh(dr / 10.0)
        dv = 10.0 * jnp.tanh(dv / 10.0)
    return dr, dv

def heun_sde_with_noise(r, v, P, dW_r, dW_v):
    dt = P.dt

    dr, dv = f_mpr(r, v, P)

    # Heun predictor
    r_pred = r + dt * dr + dW_r
    v_pred = v + dt * dv + dW_v

    dr_pred, dv_pred = f_mpr(r_pred, v_pred, P)

    # Heun corrector
    r_new = r + 0.5 * dt * (dr + dr_pred) + dW_r
    v_new = v + 0.5 * dt * (dv + dv_pred) + dW_v

    # Enforce r >= 0
    r_new = jnp.maximum(r_new, 0.0)

    return r_new, v_new

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

    #ftilde1 = jnp.clip(ftilde1, -10.0, 10.0)
    #vtilde1 = jnp.clip(vtilde1, -10.0, 10.0)
    #qtilde1 = jnp.clip(qtilde1, -10.0, 10.0)

    f1 = jnp.exp(ftilde1)
    v1 = jnp.exp(vtilde1)
    q1 = jnp.exp(qtilde1)

    return s1, f1, ftilde1, vtilde1, qtilde1, v1, q1

def integrate_fast_noise_remat(
    nn, P, B, key, record_rv, record_bold, nt, rv_decimate, bold_decimate,
    noise_r=None, noise_v=None, initial_bold_state=None, use_remat=False,
    fast_bold=False, grad_horizon=0
):
    """
    Integrates the MPR SDE + Balloon-Windkessel BOLD via jax.lax.scan.
    Supports carrying over BOLD state between blocks.
    use_remat: enable gradient checkpointing (only needed when computing gradients).
    fast_bold: integrate BOLD once per rv_decimate-step block driven by the
        block-mean firing rate, instead of every neural step. ~rv_decimate x
        fewer (expensive) hemodynamic updates. Approximate: not bit-identical to
        fast_bold=False; validate summary features (FC/FCD/fluidity) before use.
    """
    dt = P.dt
    r_period = dt * rv_decimate
    dtt = r_period / 1000.0  # BOLD substep in seconds
    dtt_block = dtt * rv_decimate  # one BOLD Euler step spanning a full block
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

    # Initial neural state (kept as separate r / v arrays in the carry)
    r0 = P.initial_state[:nn]
    v0 = P.initial_state[nn:]

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

    # Number of downsampled samples
    n_rv = nt // rv_decimate if record_rv else 1
    n_bold = nt // bold_decimate if record_bold else 1

    # Nested-scan geometry: the outer scan walks over blocks of `rv_decimate`
    # neural steps, the inner scan advances the sub-steps within a block. The
    # first sub-step of block k is flat step k*rv_decimate, whose post-step
    # value is exactly what the old "record when i % rv_decimate == 0" logic
    # kept -> bit-for-bit identical recording, with no lax.cond and no
    # full-resolution buffer.
    n_blocks = nt // rv_decimate            # outer iterations (loop geometry)
    trailing = nt - n_blocks * rv_decimate  # steps after the last full block

    # BOLD is recorded every `bold_decimate` neural steps; those indices must
    # land on block boundaries so we can pick them out by striding the per-block
    # BOLD samples.
    if record_bold and n_blocks > 0:
        assert bold_decimate % rv_decimate == 0, (
            "nested-scan recording requires bold_decimate to be a multiple of "
            f"rv_decimate (got {bold_decimate} and {rv_decimate})"
        )
        bold_stride = bold_decimate // rv_decimate
    else:
        bold_stride = 1

    def neural_bold_step(carry, i):
        """(exact path) Advance neural + BOLD one step using noise at flat i.

        `r`, `v` are the neural firing rate / membrane potential; `vb`, `q` are
        the BOLD blood volume / deoxyhemoglobin. No output is emitted per step;
        recorded samples are read straight from the carry in `outer_block`, so
        the stacked (r, v) concatenation happens once per recorded sample, not
        every step.
        """
        r, v, s, f, ftilde, vtilde, qtilde, vb, q = carry
        r_new, v_new = heun_sde_with_noise(r, v, P, noise_r[i], noise_v[i])
        s1, f1, ftilde1, vtilde1, qtilde1, vb1, q1 = do_bold_step(
            r_new, s, f, ftilde, vtilde, qtilde, vb, q, dtt, B
        )
        return (r_new, v_new, s1, f1, ftilde1, vtilde1, qtilde1, vb1, q1), None

    def neural_step(carry, i):
        """(fast path) Advance neural state only, accumulating the firing rate."""
        r, v, r_sum = carry
        r_new, v_new = heun_sde_with_noise(r, v, P, noise_r[i], noise_v[i])
        return (r_new, v_new, r_sum + r_new), None

    # --- truncated backprop-through-time (bounds gradient explosion in the
    # chaotic regime). We segment the trajectory and stop_gradient the carry at
    # segment starts, so gradients flow back at most `grad_horizon` neural steps.
    # stop_gradient is the identity in the forward pass -> forward is bit-exact;
    # only the backward pass changes. grad_horizon == 0 disables it entirely.
    seg_blocks = max(1, int(round(grad_horizon / rv_decimate))) if grad_horizon else 0

    def maybe_truncate(carry, k):
        if not grad_horizon:
            return carry
        return jax.lax.cond(
            k % seg_blocks == 0,
            lambda c: jax.tree_util.tree_map(jax.lax.stop_gradient, c),
            lambda c: c,
            carry,
        )

    def outer_block_exact(carry, k):
        carry = maybe_truncate(carry, k)
        base = k * rv_decimate
        # First sub-step of the block -> this is the recorded sample.
        carry, _ = neural_bold_step(carry, base)
        r_rec, v_rec_n = carry[0], carry[1]
        vb_rec, q_rec = carry[7], carry[8]
        out_rv = jnp.concatenate([r_rec, v_rec_n]) if record_rv else None
        out_v = vb_rec if record_bold else None
        out_q = q_rec if record_bold else None
        # Remaining rv_decimate-1 sub-steps advance the state without recording.
        if rv_decimate > 1:
            carry, _ = jax.lax.scan(
                neural_bold_step, carry, base + 1 + jnp.arange(rv_decimate - 1)
            )
        return carry, (out_rv, out_v, out_q)

    def outer_block_fast(carry, k):
        # Neural state advances every step; BOLD is integrated ONCE per block,
        # driven by the block-mean firing rate, with a single Euler step
        # spanning the whole block (dtt_block == rv_decimate * dtt).
        carry = maybe_truncate(carry, k)
        r, v, s, f, ftilde, vtilde, qtilde, vb, q = carry
        base = k * rv_decimate
        # First neural sub-step -> the recorded RV sample.
        r, v = heun_sde_with_noise(r, v, P, noise_r[base], noise_v[base])
        out_rv = jnp.concatenate([r, v]) if record_rv else None
        r_sum = r
        # Remaining neural sub-steps, accumulating the firing rate.
        if rv_decimate > 1:
            (r, v, r_sum), _ = jax.lax.scan(
                neural_step, (r, v, r_sum), base + 1 + jnp.arange(rv_decimate - 1)
            )
        r_mean = r_sum / rv_decimate
        s, f, ftilde, vtilde, qtilde, vb, q = do_bold_step(
            r_mean, s, f, ftilde, vtilde, qtilde, vb, q, dtt_block, B
        )
        out_v = vb if record_bold else None
        out_q = q if record_bold else None
        return (r, v, s, f, ftilde, vtilde, qtilde, vb, q), (out_rv, out_v, out_q)

    outer_block = outer_block_fast if fast_bold else outer_block_exact
    init_carry = (r0, v0, s, f, ftilde, vtilde, qtilde, v, q)
    outer_fn = jax.remat(outer_block) if use_remat else outer_block

    if n_blocks > 0:
        final_carry, (rv_stack, vv_stack, qq_stack) = jax.lax.scan(
            outer_fn, init_carry, jnp.arange(n_blocks)
        )
    else:
        final_carry = init_carry
        rv_stack = vv_stack = qq_stack = None

    # Trailing steps (nt not a multiple of rv_decimate): integrate, no recording.
    if trailing > 0:
        trail_idx = n_blocks * rv_decimate + jnp.arange(trailing)
        if fast_bold:
            r, v = final_carry[0], final_carry[1]
            bold_state = final_carry[2:9]
            (r, v, r_sum), _ = jax.lax.scan(
                neural_step, (r, v, jnp.zeros(nn)), trail_idx
            )
            s, f, ftilde, vtilde, qtilde, vb, q = do_bold_step(
                r_sum / trailing, *bold_state, dtt * trailing, B
            )
            final_carry = (r, v, s, f, ftilde, vtilde, qtilde, vb, q)
        else:
            final_carry, _ = jax.lax.scan(neural_bold_step, final_carry, trail_idx)

    # --- assemble recorded outputs ---
    if record_rv and rv_stack is not None:
        rv_d = rv_stack[:n_rv]
        rv_t = jnp.arange(n_rv, dtype=jnp.float32) * (rv_decimate * dt * 10.0)
    else:
        rv_d = jnp.zeros((1, 2 * nn), dtype=jnp.float32)
        rv_t = jnp.zeros((1,), dtype=jnp.float32)

    if record_bold and vv_stack is not None:
        vv_d = vv_stack[::bold_stride][:n_bold]
        qq_d = qq_stack[::bold_stride][:n_bold]
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
        "final_state": jnp.concatenate([final_carry[0], final_carry[1]]),
        "final_bold_state": final_carry[2:9],  # s, f, ftilde, vtilde, qtilde, v, q
    }

integrate_jitted_fast_noise = jax.jit(
    integrate_fast_noise_remat,
    static_argnames=["nn", "record_rv", "record_bold", "nt", "rv_decimate", "bold_decimate", "use_remat", "fast_bold", "grad_horizon"]
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
    # Derivative/coupling bounding in f_mpr: "hard" = jnp.clip (default, bit-exact),
    # "soft" = tanh saturation (smooth, gradient-friendly), "none" = unbounded.
    # Static (non-pytree) so f_mpr can branch on it at trace time.
    clip_mode: str = struct.field(default="hard", pytree_node=False)
    # Derived constants — precomputed once in create(), read every f_mpr call
    delta_over_tau_pi: float = 0.0
    J_tau: float = 0.0
    rtau: float = 1.0
    pi_tau_sq: float = _PI2

    @classmethod
    def create(cls, **kwargs):
        """Helper to compute derived quantities and return an instance."""
        weights = kwargs.get("weights", jnp.zeros((0, 0)))
        dt = kwargs.get("dt", 0.01)
        noise_amp = kwargs.get("noise_amp", 0.037)
        tau = kwargs.get("tau", 1.0)
        J = kwargs.get("J", 14.5)
        delta = kwargs.get("delta", 0.7)
        sigma_r = jnp.sqrt(dt) * jnp.sqrt(2 * noise_amp)
        sigma_v = jnp.sqrt(dt) * jnp.sqrt(4 * noise_amp)
        nn = int(weights.shape[0])

        return cls(
            sigma_r=sigma_r,
            sigma_v=sigma_v,
            nn=nn,
            delta_over_tau_pi=float(delta / (tau * np.pi)),
            J_tau=float(J * tau),
            rtau=float(1.0 / tau),
            pi_tau_sq=float((np.pi * tau) ** 2),
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
    
    def run(
        self,
        par: dict = {},
        x0: jnp.ndarray = None,
        block_size: int = 10_000,
        record_rv: bool = True,
        noise_blocks=None,
        use_remat: bool = False,
        fast_bold: bool = False,
        grad_horizon: int = 0
    ):
        """
        Chunked, memory-friendly run() for MPR+Balloon simulation.
        Supports optional recording of RV to save memory.
        Accepts dict parameter overrides like old API.
        """
        # ------------------ parameters ------------------
        P = self.P
        for k, v in par.items():
            if k not in ParMPR.__annotations__:
                raise ValueError(f"Invalid parameter: {k}")
            P = P.replace(**{k: v})

        P = P.replace(
            eta=check_vec_size(P.eta, P.nn),
            t_end=P.t_end / 10,
            t_cut=P.t_cut / 10
        )

        nn = int(P.nn)
        dt = float(P.dt)
        tr = float(P.tr)
        rv_decimate = int(P.rv_decimate)
        r_period = dt * rv_decimate
        bold_decimate = max(1, int(np.round(tr / r_period)))
        nt_total = int(P.t_end / P.dt)

        # ------------------ initial state ------------------
        if x0 is not None:
            state = x0
            key = self.key
        elif P.initial_state.size == 2 * nn:
            state = P.initial_state
            key = self.key
        else:
            sim = self.with_initial_state()
            P = P.replace(initial_state=sim.initial_state)
            state = sim.initial_state
            key = sim.key
        assert state.size == 2 * nn

        # ------------------ blocks decomposition ------------------
        n_full = nt_total // block_size
        remainder = nt_total - n_full * block_size
        use_external_noise = noise_blocks is not None

        if use_external_noise:
            noise_r_seq, noise_v_seq = noise_blocks
            assert len(noise_r_seq) == (n_full + (1 if remainder else 0))
            assert len(noise_v_seq) == (n_full + (1 if remainder else 0))

        # ------------------ initial bold state ------------------
        bold_state = (
            jnp.ones(nn), jnp.ones(nn), jnp.zeros(nn),
            jnp.zeros(nn), jnp.zeros(nn), jnp.ones(nn), jnp.ones(nn)
        )

        # ------------------ storage ------------------
        rv_blocks, bold_blocks = [], []

        # ------------------ JITed scan for full blocks ------------------
        if n_full > 0:
            def scan_full_blocks(state, bold_state, key):
                def body(carry, block_idx):
                    key, state, bold_state = carry
                    key, subkey = jax.random.split(key)

                    # noise
                    if use_external_noise:
                        noise_r = jax.lax.dynamic_index_in_dim(noise_r_seq, block_idx, False)
                        noise_v = jax.lax.dynamic_index_in_dim(noise_v_seq, block_idx, False)
                    else:
                        k1, k2 = jax.random.split(subkey)
                        noise_r = jax.random.normal(k1, (block_size, nn)) * P.sigma_r
                        noise_v = jax.random.normal(k2, (block_size, nn)) * P.sigma_v

                    # integrate
                    out = integrate_jitted_fast_noise(
                        nn=nn,
                        P=P.replace(initial_state=state),
                        B=self.B,
                        key=subkey,
                        record_rv=record_rv,
                        record_bold=P.RECORD_BOLD,
                        nt=block_size,
                        rv_decimate=rv_decimate,
                        bold_decimate=bold_decimate,
                        noise_r=noise_r,
                        noise_v=noise_v,
                        initial_bold_state=bold_state,
                        use_remat=use_remat,
                        fast_bold=fast_bold,
                        grad_horizon=grad_horizon
                    )

                    new_state = out["final_state"]
                    new_bold_state = out["final_bold_state"]

                    rv_chunk = out["rv_d"] if record_rv else jnp.zeros((0, 2*nn))
                    bold_chunk = out["bold_d"]

                    return (key, new_state, new_bold_state), (rv_chunk, bold_chunk)

                (key_out, state_out, bold_state_out), (rv_chunks, bold_chunks) = jax.lax.scan(
                    body,
                    (key, state, bold_state),
                    jnp.arange(n_full)
                )
                return key_out, state_out, bold_state_out, rv_chunks, bold_chunks

            key, state, bold_state, rv_chunks, bold_chunks = scan_full_blocks(state, bold_state, key)
            rv_blocks.append(rv_chunks)
            bold_blocks.append(bold_chunks)

        # ------------------ remainder block ------------------
        if remainder:
            key, subkey = jax.random.split(key)
            if use_external_noise:
                noise_r_last = noise_r_seq[n_full]
                noise_v_last = noise_v_seq[n_full]
            else:
                k1, k2 = jax.random.split(subkey)
                noise_r_last = jax.random.normal(k1, (remainder, nn)) * P.sigma_r
                noise_v_last = jax.random.normal(k2, (remainder, nn)) * P.sigma_v

            out_last = integrate_jitted_fast_noise(
                nn=nn,
                P=P.replace(initial_state=state),
                B=self.B,
                key=subkey,
                record_rv=record_rv,
                record_bold=P.RECORD_BOLD,
                nt=remainder,
                rv_decimate=rv_decimate,
                bold_decimate=bold_decimate,
                noise_r=noise_r_last,
                noise_v=noise_v_last,
                initial_bold_state=bold_state,
                use_remat=use_remat,
                fast_bold=fast_bold,
                grad_horizon=grad_horizon
            )

            state = out_last["final_state"]
            bold_state = out_last["final_bold_state"]

            if record_rv:
                rv_blocks.append(jnp.expand_dims(out_last["rv_d"], axis=0))
            bold_blocks.append(jnp.expand_dims(out_last["bold_d"], axis=0))

        # ------------------ stitch blocks ------------------
        if record_rv:
            rv = jnp.concatenate([jnp.concatenate(rb, axis=0) for rb in rv_blocks], axis=0)
            rv_t = jnp.arange(rv.shape[0]) * (dt * rv_decimate) * 10.0
        else:
            rv = None
            rv_t = None

        bold = jnp.concatenate([jnp.concatenate(bb, axis=0) for bb in bold_blocks], axis=0)
        bold_t = jnp.arange(bold.shape[0]) * (dt * rv_decimate * bold_decimate) * 10.0 if P.RECORD_BOLD else None

        result = {
            "rv_d": rv,
            "bold_d": bold,
            "final_state": state,
            "final_bold_state": bold_state,
            "rv_t": rv_t,
            "bold_t": bold_t
        }

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


