# Implemented by: Abolfazl Ziaeemehr
# Original code comes from Virtual Brain inference (VBI) package
# https://github.com/ins-amu/vbi
# Date: 2025-01-01

import warnings
import jax 
import jax.numpy as jnp
from dataclasses import dataclass
from flax import struct
from functools import partial

import numpy as np


# is P is a static argument, use @partial(jit, static_argnames=["P"])
#@partial(jax.jit, static_argnames=["nn", "method", "output"])
def f_mpr(x, t, P, nn): #, nn, method, output
    """
    MPR model
    x: state vector at time t split into x0 and x1 (r and v), t time variable, P parameter object strucure

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
    return dxdt


def heun_sde(x, t, P, key, nn):
    """
    heun method, to numerically integrate stochastic differential equations (see paper for the formula)
    x: state vector split into x0, x1. t: current time, P paramter vector
    """
    #nn = P.nn
    dt = P.dt

    key_r, key_v = jax.random.split(key)

    dW_r = P.sigma_r * jax.random.normal(key_r, shape=(nn,))
    dW_v = P.sigma_v * jax.random.normal(key_v, shape=(nn,))

    k1 = f_mpr(x, t, P, nn)

    x1 = x + dt * k1
    x1 = x1.at[:nn].add(dW_r)
    x1 = x1.at[nn:].add(dW_v)

    k2 = f_mpr(x1, t + dt, P, nn)

    x_new = x + 0.5 * dt * (k1 + k2)
    x_new = x_new.at[:nn].add(dW_r)
    x_new = x_new.at[nn:].add(dW_v)

    x_new = x_new.at[:nn].set(jnp.maximum(x_new[:nn], 0.0))
    return x_new


#@jax.jit
def do_bold_step(r_in, s, f, ftilde, vtilde, qtilde, v, q, dtt, P):
    """balloon windkessel model
    r_in: averge firing rate form the Montbriò model
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


def integrate(nn, P, B, method, key, record_rv, record_bold, nt, rv_decimate, bold_decimate):
    """
    Simulates neural dynamics + BOLD using the Montbrió + Balloon-Windkessel model, in JAX.

    Inputs:
    - P: parameters for the neural model and simulation
    - B: parameters for the BOLD (Balloon-Windkessel) model
    - heun_sde: pure function to update neural state
    """

    #nn = P.nn
    #tr = P.tr
    #rv_decimate = P.rv_decimate
    dt = P.dt
    r_period = dt * rv_decimate
    #bold_decimate = int(jnp.round(tr / r_period))

    dtt = r_period / 1000.0  # BOLD model step in seconds
    k1 = 4.3 * B.theta0 * B.Eo * B.TE
    k2 = B.epsilon * B.r0 * B.Eo * B.TE
    k3 = 1 - B.epsilon
    vo = B.vo
    
    t_end = P.t_end
    #nt = int(jnp.round(t_end / dt))

    rv_current = P.initial_state

    #RECORD_RV = P.RECORD_RV
    #RECORD_BOLD = P.RECORD_BOLD

    s = jnp.ones(nn)
    f = jnp.ones(nn)
    ftilde = jnp.zeros(nn)
    vtilde = jnp.zeros(nn)
    qtilde = jnp.zeros(nn)
    v = jnp.ones(nn)
    q = jnp.ones(nn)

    #if RECORD_RV:
    if record_rv:
        rv_d = jnp.zeros((nt // rv_decimate, 2 * nn), dtype=jnp.float32)
        rv_t = jnp.zeros((nt // rv_decimate,), dtype=jnp.float32)
    else:
        rv_d = jnp.zeros((1, 2 * nn))
        rv_t = jnp.zeros((1,))

    #if RECORD_BOLD:
    if record_bold:
        vv = jnp.zeros((nt // bold_decimate, nn))
        qq = jnp.zeros((nt // bold_decimate, nn))
    else:
        vv = jnp.zeros((1, nn))
        qq = jnp.zeros((1, nn))

    def loop_body(carry, i):
        (
            rv_current, s, f, ftilde, vtilde, qtilde, v, q,
            rv_d, rv_t, vv, qq, key
        ) = carry

        t_current = i * dt
        key, subkey = jax.random.split(key)
        rv_current = method(rv_current, t_current, P, subkey, nn)

        def save_rv(i, rv_current, rv_d, rv_t):
            idx = i // rv_decimate
            rv_d = rv_d.at[idx].set(rv_current)
            rv_t = rv_t.at[idx].set(i * dt)
            return rv_d, rv_t

        #if RECORD_RV:
        if record_rv:
            do_record_rv = (i % rv_decimate == 0)
            rv_d, rv_t = jax.lax.cond(
                do_record_rv,
                lambda carry: save_rv(i, rv_current, *carry),
                lambda carry: carry,
                (rv_d, rv_t)
            )


        def save_bold(i, v, q, vv, qq):
            idx = i // bold_decimate
            vv = vv.at[idx].set(v)
            qq = qq.at[idx].set(q)
            return vv, qq

        #if RECORD_BOLD:
        if record_bold:
            s, f, ftilde, vtilde, qtilde, v, q = do_bold_step(
                rv_current[:nn], s, f, ftilde, vtilde, qtilde, v, q, dtt, B
            )
            do_record_bold = (i % bold_decimate == 0)
            vv, qq = jax.lax.cond(
                do_record_bold,
                lambda carry: save_bold(i, v, q, *carry),
                lambda carry: carry,
                (vv, qq)
            )



        new_carry = (
            rv_current, s, f, ftilde, vtilde, qtilde, v, q,
            rv_d, rv_t, vv, qq, key
        )
        return new_carry, None

    init_carry = (
        rv_current, s, f, ftilde, vtilde, qtilde, v, q,
        rv_d, rv_t, vv, qq, key
    )
    final_carry, _ = jax.lax.scan(loop_body, init_carry, jnp.arange(nt - 1))

    (
        rv_current, s, f, ftilde, vtilde, qtilde, v, q,
        rv_d, rv_t, vv, qq, key
    ) = final_carry


    #if RECORD_BOLD:
    if record_bold:
        bold_d = vo * (k1 * (1 - qq) + k2 * (1 - qq / vv) + k3 * (1 - vv))
        bold_t = jnp.linspace(0, t_end - dt * bold_decimate, len(bold_d))
    else:
        bold_d = jnp.zeros((1,))
        bold_t = jnp.zeros((1,))

    return {
        "rv_t": rv_t * 10,  # ms
        "rv_d": rv_d,
        "bold_t": bold_t.astype(jnp.float32)* 10,  # ms
        "bold_d": bold_d.astype(jnp.float32),
    }

integrate = jax.jit(integrate, static_argnames=["nn","method", "record_rv", "record_bold", "nt", "rv_decimate", "bold_decimate"])

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
    seed: int = 0
    initial_state: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([]))
    noise_amp: float = 0.037
    sigma_r: float = 0.0  
    sigma_v: float = 0.0  
    iapp: float = 0.0
    output: str = struct.field(default="output", pytree_node=False)
    RECORD_RV: bool = True
    RECORD_BOLD: bool = True
    rv_decimate: int = 10
    tr: float = 500.0

    @classmethod
    def create(cls, **kwargs):
        """Helper to compute derived quantities and return an instance."""
        weights = kwargs.get("weights", jnp.zeros((0, 0)))
        dt = kwargs.get("dt", 0.01)
        noise_amp = kwargs.get("noise_amp", 0.037)
        sigma_r = jnp.sqrt(dt) * jnp.sqrt(2 * noise_amp)
        sigma_v = jnp.sqrt(dt) * jnp.sqrt(4 * noise_amp)
        nn = weights.shape[0]

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
    def create(par_mpr: dict = {}) -> "MPR_sde":

        valid_par = list(ParMPR.__annotations__.keys())
        for key in par_mpr:
            if key not in valid_par:
                raise ValueError(f"Invalid parameter: {key}")

        if "initial_state" in par_mpr:
            par_mpr["initial_state"] = jnp.array(par_mpr["initial_state"])
        if "weights" in par_mpr:
            weights = jnp.array(par_mpr["weights"])
            assert weights.shape[0] == weights.shape[1]
            par_mpr["weights"] = weights

        P = ParMPR.create(**par_mpr)
        B = ParBold()
        key = jax.random.PRNGKey(P.seed)
        return MPR_sde(P=P, B=B, key=key)

    def with_initial_state(self) -> "MPR_sde":
        key, subkey = jax.random.split(self.key)
        init_state = set_initial_state(self.P.nn, subkey)
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
        rv_decimate = new_self.P.rv_decimate 
        r_period = new_self.P.dt * rv_decimate
        bold_decimate = int(jnp.round(new_self.P.tr / r_period))

        return integrate(nn, 
                         new_self.P, 
                         self.B, 
                         method=heun_sde, 
                         key=key, 
                         record_rv=new_self.P.RECORD_RV, 
                         record_bold=new_self.P.RECORD_BOLD,
                         nt=nt,
                         rv_decimate=rv_decimate,
                         bold_decimate=bold_decimate)



#@jax.jit
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


