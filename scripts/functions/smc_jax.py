"""JAX-native, particle-vmapped Sequential Monte Carlo for the MPR G-inference.

Why hand-rolled (not blackjax.smc): (1) the ABC variant is likelihood-free and
blackjax has no built-in ABC kernel; (2) writing the particle loop ourselves
guarantees that every forward-model evaluation is a `jax.vmap` over the N
particles, which is exactly the on-device batching that makes the GPU win
(measured: forward throughput is flat in wall-time up to batch 1024). The whole
SMC run is one `jax.lax.scan`, so it compiles to a single GPU program.

Two flavors, one framework (fixed tempering schedule -> predictable cost, good
for benchmarking):
  * run_tempered_smc  -- LIKELIHOOD tempering, exponent lambda: 0 -> 1.
                         Target = exact posterior (analog of pymc `smclik`,
                         same target as NUTS/BlackJAX).
  * run_abc_smc       -- ABC epsilon tempering, eps: eps0 -> eps_target with a
                         Gaussian ABC kernel pseudo-likelihood -0.5*(dist/eps)^2
                         (analog of pymc `smcabc`, likelihood-free).

Both operate on a scalar POSITIVE parameter G with a multiplicative
(log-space) random-walk MH move, so they stay in the constrained support and
need no unconstraining transform. `loglik_fn` / `distance_fn` take a scalar G;
we vmap them over the particle cloud internally.
"""
import jax
import numpy as np
import jax.numpy as jnp


def _bcast(mask, parts):
    """Broadcast a per-particle (N,) boolean over a possibly (N, D) particle array."""
    return mask if parts.ndim == 1 else mask[:, None]


def systematic_resample(key, w):
    """Systematic resampling indices for normalized weights w (shape (N,))."""
    N = w.shape[0]
    u = (jax.random.uniform(key) + jnp.arange(N)) / N
    return jnp.clip(jnp.searchsorted(jnp.cumsum(w), u), 0, N - 1)


def _ess(logw):
    w = jax.nn.softmax(logw)
    return 1.0 / jnp.sum(w ** 2)


def _make_proposal(move, rw_step, demc_gamma, demc_jitter, d=1):
    """Within-stage MCMC proposal on the particle cloud, in log-G space.

    move="rw"   : independent multiplicative random walk (one particle at a time).
    move="demc" : DE-MC ensemble-difference proposal -- z_i' = z_i + gamma*(z_r1 - z_r2)
                  + jitter, with r1,r2 drawn from the CURRENT cloud. This is the
                  run_demc proposal from mcmc_jax, applied INSIDE the tempering ladder:
                  the SMC particle cloud already *is* the population DE-MC needs, so
                  tempering (which is what lets ABC succeed at G=0.7, RMSE 0.0099,
                  where plain DE-MC stalls at 0.155) combines with a proposal that
                  adapts to the cloud's own scale instead of a fixed rw_step.

    Both are symmetric in z=log(G), so the Hastings correction stays log(prop/parts)
    exactly as in the plain-RW case -- callers need no other change."""
    # DE-MC scaling is 2.38/sqrt(2*d). `d` is the PARAMETER dimension, passed in by the
    # caller because it must be resolved here, outside the traced scan -- float() on a
    # jnp value inside the proposal raises ConcretizationTypeError. At d=1 this is the
    # original expression, so scalar runs stay bit-for-bit unchanged.
    gamma = float(2.38 / jnp.sqrt(2.0 * d)) if demc_gamma is None else demc_gamma

    def rw(key, parts):
        return parts * jnp.exp(rw_step * jax.random.normal(key, parts.shape))

    def demc(key, parts):
        N = parts.shape[0]
        k1, k2, k3 = jax.random.split(key, 3)
        z = jnp.log(parts)
        r1 = jax.random.randint(k1, (N,), 0, N)
        r2 = jax.random.randint(k2, (N,), 0, N)
        zp = z + gamma * (z[r1] - z[r2]) + demc_jitter * jax.random.normal(k3, z.shape)
        return jnp.exp(zp)

    if move == "demc":
        return demc
    if move == "rw":
        return rw
    raise ValueError(f"move must be 'rw' or 'demc', got {move!r}")


def run_tempered_smc(key, init_particles, logprior_fn, loglik_fn,
                     n_stages=50, n_mcmc=5, rw_step=0.1,
                     move="rw", demc_gamma=None, demc_jitter=1e-4):
    """Likelihood-tempered SMC. `loglik_fn(G)`/`logprior_fn(G)` are scalar->scalar
    (vmapped here). Returns (particles (N,), info dict with per-stage ess/accept).

    `move` selects the within-stage kernel: "rw" (default, unchanged behaviour) or
    "demc" for tempered DE-MC. See _make_proposal."""
    _d = 1 if init_particles.ndim == 1 else init_particles.shape[-1]
    propose = _make_proposal(move, rw_step, demc_gamma, demc_jitter, _d)
    vprior = jax.vmap(logprior_fn)
    vloglik = jax.vmap(loglik_fn)
    lambdas = jnp.linspace(0.0, 1.0, n_stages + 1)

    def mh_scan(key, parts, lp, ll, lmbda):
        def step(carry, k):
            parts, lp, ll = carry
            k1, k2 = jax.random.split(k)
            prop = propose(k1, parts)
            lp_p, ll_p = vprior(prop), vloglik(prop)   # N forward evals, vmapped
            # tempered target logprior + lambda*loglik; multiplicative RW is
            # symmetric in log-space -> Hastings correction is log(prop/parts).
            # Hastings correction for the multiplicative RW, log(prop/parts). With a
            # PARAMETER AXIS this is a per-coordinate array and must be summed: the
            # proposal is a product of independent per-coordinate moves, so its
            # log-Jacobian is the sum. Leaving it unsummed would compare an (N,D)
            # quantity against an (N,) log-density below.
            hast = jnp.log(prop) - jnp.log(parts)
            if parts.ndim > 1:
                hast = hast.sum(-1)
            logr = (lp_p + lmbda * ll_p) - (lp + lmbda * ll) + hast
            # Accept/reject is PER PARTICLE, never per coordinate. Drawing the uniform
            # with parts.shape would accept G and reject eta independently, which is not
            # Metropolis and yields a plausible-looking but wrong posterior. For D=1 this
            # draws the identical numbers from the identical key, so scalar runs are
            # bit-for-bit unchanged.
            acc = jnp.log(jax.random.uniform(k2, (parts.shape[0],))) < logr
            parts = jnp.where(_bcast(acc, parts), prop, parts)
            lp = jnp.where(acc, lp_p, lp)
            ll = jnp.where(acc, ll_p, ll)
            return (parts, lp, ll), acc.mean()
        keys = jax.random.split(key, n_mcmc)
        (parts, lp, ll), accs = jax.lax.scan(step, (parts, lp, ll), keys)
        return parts, lp, ll, accs.mean()

    def stage(carry, lam_pair):
        parts, lp, ll, key = carry
        lam_prev, lam = lam_pair
        key, kr, km = jax.random.split(key, 3)
        logw = (lam - lam_prev) * ll               # incremental importance weight
        w = jax.nn.softmax(logw)
        ess = 1.0 / jnp.sum(w ** 2)
        idx = systematic_resample(kr, w)
        parts, lp, ll = parts[idx], lp[idx], ll[idx]
        parts, lp, ll, acc = mh_scan(km, parts, lp, ll, lam)
        return (parts, lp, ll, key), (ess, acc)

    lp0, ll0 = vprior(init_particles), vloglik(init_particles)
    lam_pairs = (lambdas[:-1], lambdas[1:])
    (parts, lp, ll, _), (ess, acc) = jax.lax.scan(
        stage, (init_particles, lp0, ll0, key), lam_pairs)
    return parts, {"ess": ess, "accept": acc}


def run_abc_smc(key, init_particles, logprior_fn, distance_fn, eps_schedule,
                n_mcmc=5, rw_step=0.1,
                move="rw", demc_gamma=None, demc_jitter=1e-4):
    """ABC epsilon-tempered SMC (likelihood-free). `distance_fn(G)` -> scalar
    distance between simulated and observed features. `eps_schedule` is a
    decreasing (n_stages+1,) array eps0 -> eps_target. Gaussian ABC kernel
    pseudo-loglik = -0.5*(dist/eps)^2. Returns (particles, info).

    `move` selects the within-stage kernel: "rw" (default, unchanged behaviour) or
    "demc" for tempered DE-MC. See _make_proposal."""
    _d = 1 if init_particles.ndim == 1 else init_particles.shape[-1]
    propose = _make_proposal(move, rw_step, demc_gamma, demc_jitter, _d)
    vprior = jax.vmap(logprior_fn)
    vdist = jax.vmap(distance_fn)

    def pseudo_ll(dist, eps):
        return -0.5 * (dist / eps) ** 2

    def mh_scan(key, parts, lp, dist, eps):
        def step(carry, k):
            parts, lp, dist = carry
            k1, k2 = jax.random.split(k)
            prop = propose(k1, parts)
            lp_p, dist_p = vprior(prop), vdist(prop)   # N forward evals, vmapped
            hast = jnp.log(prop) - jnp.log(parts)
            if parts.ndim > 1:
                hast = hast.sum(-1)          # see run_tempered_smc: per-coordinate
            logr = (lp_p + pseudo_ll(dist_p, eps)) - (lp + pseudo_ll(dist, eps)) + hast
            acc = jnp.log(jax.random.uniform(k2, (parts.shape[0],))) < logr
            parts = jnp.where(_bcast(acc, parts), prop, parts)
            lp = jnp.where(acc, lp_p, lp)
            dist = jnp.where(acc, dist_p, dist)
            return (parts, lp, dist), acc.mean()
        keys = jax.random.split(key, n_mcmc)
        (parts, lp, dist), accs = jax.lax.scan(step, (parts, lp, dist), keys)
        return parts, lp, dist, accs.mean()

    def stage(carry, eps_pair):
        parts, lp, dist, key = carry
        eps_prev, eps = eps_pair
        key, kr, km = jax.random.split(key, 3)
        logw = pseudo_ll(dist, eps) - pseudo_ll(dist, eps_prev)
        w = jax.nn.softmax(logw)
        ess = 1.0 / jnp.sum(w ** 2)
        idx = systematic_resample(kr, w)
        parts, lp, dist = parts[idx], lp[idx], dist[idx]
        parts, lp, dist, acc = mh_scan(km, parts, lp, dist, eps)
        return (parts, lp, dist, key), (ess, acc)

    lp0, dist0 = vprior(init_particles), vdist(init_particles)
    eps_pairs = (eps_schedule[:-1], eps_schedule[1:])
    (parts, lp, dist, _), (ess, acc) = jax.lax.scan(
        stage, (init_particles, lp0, dist0, key), eps_pairs)
    return parts, {"ess": ess, "accept": acc, "final_dist": dist}
