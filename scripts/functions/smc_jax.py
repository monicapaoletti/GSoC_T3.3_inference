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


RW_TARGET_ACCEPT = 0.234   # optimal acceptance for a random-walk Metropolis kernel


def _adapt_log_step(log_step, acc, rate, target=RW_TARGET_ACCEPT):
    """Robbins-Monro update of the log RW scale toward `target` acceptance.

    Feedback, not a schedule: acceptance falls monotonically as the step grows, so
    nudging log(step) by (acc - target) is a stable loop that finds the scale the stage
    actually needs. Clamped to [1e-4, 10] so a stage that accepts nothing (or everything)
    cannot drive the step to 0 or infinity before the loop recovers.
    """
    return jnp.clip(log_step + rate * (acc - target), jnp.log(1e-4), jnp.log(10.0))


def _make_proposal(move, demc_gamma, demc_jitter, d=1):
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

    def rw(key, parts, step):
        return parts * jnp.exp(step * jax.random.normal(key, parts.shape))

    def demc(key, parts, step):
        # DE-MC draws its scale from the cloud itself, so the adaptive step does not
        # apply; it is accepted and ignored to keep one proposal signature.
        del step
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
                     move="rw", demc_gamma=None, demc_jitter=1e-4,
                     adapt_rw=False, adapt_rate=1.0):
    """Likelihood-tempered SMC. `loglik_fn(G)`/`logprior_fn(G)` are scalar->scalar
    (vmapped here). Returns (particles (N,), info dict with per-stage ess/accept).

    `move` selects the within-stage kernel: "rw" (default, unchanged behaviour) or
    "demc" for tempered DE-MC. See _make_proposal.

    `adapt_rw=True` tunes the RW scale toward RW_TARGET_ACCEPT after every MH move
    instead of holding `rw_step` fixed (which is then only the starting value). Default
    False so existing runs stay bit-for-bit reproducible -- see the note in run_abc_smc."""
    _d = 1 if init_particles.ndim == 1 else init_particles.shape[-1]
    propose = _make_proposal(move, demc_gamma, demc_jitter, _d)
    # adapt_rw is static, so this resolves at trace time: with adaptation off the
    # proposal receives the original Python float, not exp(log(rw_step)) -- that
    # round-trip is inexact in float32 and would silently perturb every fixed-step run
    # away from the results already in the paper.
    _step_of = (lambda ls: jnp.exp(ls)) if adapt_rw else (lambda ls: rw_step)
    vprior = jax.vmap(logprior_fn)
    vloglik = jax.vmap(loglik_fn)
    lambdas = jnp.linspace(0.0, 1.0, n_stages + 1)

    def mh_scan(key, parts, lp, ll, lmbda, log_step):
        def step(carry, k):
            parts, lp, ll, log_step = carry
            k1, k2 = jax.random.split(k)
            prop = propose(k1, parts, _step_of(log_step))
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
            a = acc.mean()
            if adapt_rw:
                log_step = _adapt_log_step(log_step, a, adapt_rate)
            return (parts, lp, ll, log_step), a
        keys = jax.random.split(key, n_mcmc)
        (parts, lp, ll, log_step), accs = jax.lax.scan(
            step, (parts, lp, ll, log_step), keys)
        return parts, lp, ll, accs.mean(), log_step

    def stage(carry, lam_pair):
        parts, lp, ll, key, log_step = carry
        lam_prev, lam = lam_pair
        key, kr, km = jax.random.split(key, 3)
        logw = (lam - lam_prev) * ll               # incremental importance weight
        w = jax.nn.softmax(logw)
        ess = 1.0 / jnp.sum(w ** 2)
        idx = systematic_resample(kr, w)
        parts, lp, ll = parts[idx], lp[idx], ll[idx]
        parts, lp, ll, acc, log_step = mh_scan(km, parts, lp, ll, lam, log_step)
        return (parts, lp, ll, key, log_step), (ess, acc, jnp.exp(log_step))

    lp0, ll0 = vprior(init_particles), vloglik(init_particles)
    lam_pairs = (lambdas[:-1], lambdas[1:])
    (parts, lp, ll, _, _), (ess, acc, step_hist) = jax.lax.scan(
        stage, (init_particles, lp0, ll0, key, jnp.log(rw_step)), lam_pairs)
    return parts, {"ess": ess, "accept": acc, "step": step_hist}


def run_abc_smc(key, init_particles, logprior_fn, distance_fn, eps_schedule,
                n_mcmc=5, rw_step=0.1,
                move="rw", demc_gamma=None, demc_jitter=1e-4,
                adapt_rw=False, adapt_rate=1.0, fresh_noise=False):
    """ABC epsilon-tempered SMC (likelihood-free).

    fresh_noise=True: distance_fn(G, key) -> (K,) distances from K NEW noise draws, and
    the pseudo-loglik is log mean_k exp(-0.5 (d_k/eps)^2), an unbiased estimate of the
    ABC likelihood. Each particle keeps the distances it was simulated with (they are
    re-weighted, never re-simulated, between stages), which is what keeps the
    pseudo-marginal MH move exact. With False the code path is the historical one.

    Frozen noise (default): `distance_fn(G)` -> scalar
    distance between simulated and observed features. `eps_schedule` is a
    decreasing (n_stages+1,) array eps0 -> eps_target. Gaussian ABC kernel
    pseudo-loglik = -0.5*(dist/eps)^2. Returns (particles, info).

    `move` selects the within-stage kernel: "rw" (default, unchanged behaviour) or
    "demc" for tempered DE-MC. See _make_proposal.

    ADAPT_RW EXISTS BECAUSE THE FIXED STEP COLLAPSES AT SCALE. At 88 nodes with 4096
    particles the fixed rw_step=0.1 accepted 0.2% of moves for smc_lik in the 2-D run, so
    the 250 rejuvenation moves separated nothing and the cloud held 254-419 unique values
    out of 4096. The weight ESS still read ~4000, which is why this hid: resampled
    DUPLICATES carry healthy weights. Adapting the scale to RW_TARGET_ACCEPT fixes the
    kernel rather than the diagnostic. Default False so the paper's existing matrix stays
    bit-for-bit reproducible; every run that turns it on must be compared only against
    other runs that did."""
    _d = 1 if init_particles.ndim == 1 else init_particles.shape[-1]
    propose = _make_proposal(move, demc_gamma, demc_jitter, _d)
    # adapt_rw is static, so this resolves at trace time: with adaptation off the
    # proposal receives the original Python float, not exp(log(rw_step)) -- that
    # round-trip is inexact in float32 and would silently perturb every fixed-step run
    # away from the results already in the paper.
    _step_of = (lambda ls: jnp.exp(ls)) if adapt_rw else (lambda ls: rw_step)
    vprior = jax.vmap(logprior_fn)
    if fresh_noise:
        _vd = jax.vmap(distance_fn)
        def vdist(parts, key):
            return _vd(parts, jax.random.split(key, parts.shape[0]))      # (N, K)

        def pseudo_ll(dist, eps):
            k = -0.5 * (dist / eps) ** 2
            return jax.scipy.special.logsumexp(k, axis=-1) - jnp.log(k.shape[-1])
    else:
        _vd = jax.vmap(distance_fn)
        vdist = lambda parts, key: _vd(parts)          # key unused: frozen noise

        def pseudo_ll(dist, eps):
            return -0.5 * (dist / eps) ** 2

    def mh_scan(key, parts, lp, dist, eps, log_step):
        def step(carry, k):
            parts, lp, dist, log_step = carry
            # the frozen path keeps its historical 2-way split, so its random stream --
            # and every result already produced with it -- is unchanged
            if fresh_noise:
                k1, k2, k3 = jax.random.split(k, 3)
            else:
                (k1, k2), k3 = jax.random.split(k), None
            prop = propose(k1, parts, _step_of(log_step))
            lp_p, dist_p = vprior(prop), vdist(prop, k3)   # N forward evals, vmapped
            hast = jnp.log(prop) - jnp.log(parts)
            if parts.ndim > 1:
                hast = hast.sum(-1)          # see run_tempered_smc: per-coordinate
            logr = (lp_p + pseudo_ll(dist_p, eps)) - (lp + pseudo_ll(dist, eps)) + hast
            acc = jnp.log(jax.random.uniform(k2, (parts.shape[0],))) < logr
            parts = jnp.where(_bcast(acc, parts), prop, parts)
            lp = jnp.where(acc, lp_p, lp)
            dist = jnp.where(_bcast(acc, dist), dist_p, dist)
            a = acc.mean()
            if adapt_rw:
                log_step = _adapt_log_step(log_step, a, adapt_rate)
            return (parts, lp, dist, log_step), a
        keys = jax.random.split(key, n_mcmc)
        (parts, lp, dist, log_step), accs = jax.lax.scan(
            step, (parts, lp, dist, log_step), keys)
        return parts, lp, dist, accs.mean(), log_step

    def stage(carry, eps_pair):
        parts, lp, dist, key, log_step = carry
        eps_prev, eps = eps_pair
        key, kr, km = jax.random.split(key, 3)
        logw = pseudo_ll(dist, eps) - pseudo_ll(dist, eps_prev)
        w = jax.nn.softmax(logw)
        ess = 1.0 / jnp.sum(w ** 2)
        idx = systematic_resample(kr, w)
        parts, lp, dist = parts[idx], lp[idx], dist[idx]
        parts, lp, dist, acc, log_step = mh_scan(km, parts, lp, dist, eps, log_step)
        return (parts, lp, dist, key, log_step), (ess, acc, jnp.exp(log_step))

    if fresh_noise:
        key, k0 = jax.random.split(key)
    else:
        k0 = None
    lp0, dist0 = vprior(init_particles), vdist(init_particles, k0)
    eps_pairs = (eps_schedule[:-1], eps_schedule[1:])
    (parts, lp, dist, _, _), (ess, acc, step_hist) = jax.lax.scan(
        stage, (init_particles, lp0, dist0, key, jnp.log(rw_step)), eps_pairs)
    return parts, {"ess": ess, "accept": acc, "step": step_hist, "final_dist": dist}
