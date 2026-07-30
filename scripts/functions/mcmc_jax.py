"""JAX-native, chain-vmapped gradient-free MCMC samplers for the MPR G-inference.

Companion to smc_jax.py. Where SMC batches a PARTICLE cloud on-device, these batch
N independent MCMC CHAINS on-device: a single chain is sequential, but running
many chains at once makes every forward-model evaluation a jax.vmap over the
chains -> the same GPU-batching win we measured for the forward model (throughput
flat in batch size), with NO gradient (so none of the FD-NUTS step-size pathology).

These are the GPU counterparts of the pymc host-side samplers (which stay CPU-only):
  * run_parallel_rwmh  -- Random-Walk Metropolis (analog of pymc Metropolis)
  * run_demc           -- Differential-Evolution Metropolis, a POPULATION sampler
                          whose ensemble-difference proposal is naturally vmappable.
                          Analog of pymc DEMetropolis ONLY: the difference vector is
                          drawn from the CURRENT ensemble. There is deliberately no
                          DEMetropolisZ analog here -- Z substitutes a per-chain history
                          archive for a large population, which is precisely what
                          on-device chain batching (n_chains up to 4096) already provides.
  * run_slice          -- univariate Slice sampling with stepping-out + shrinkage
                          (analog of pymc Slice); bounded fori_loops keep it vmap-safe

All operate on a scalar POSITIVE parameter G via z = log(G) (unconstrained), so the
target in z-space is  logpost_z(z) = logprior_G(exp z) + z + loglik_G(exp z)
(the +z is the log-Jacobian of G = exp(z)). `logprior_G` / `loglik_G` take a scalar
G; we vmap them over the chain axis internally. Each sampler returns G-space draws
shaped (n_chains, n_draws) so ArviZ gets real chains -> split-R-hat/ESS available.
"""
import jax
import jax.numpy as jnp
from functools import partial


def _make_logpost_z(logprior_G, loglik_G):
    vlp = jax.vmap(logprior_G)
    vll = jax.vmap(loglik_G)

    def logpost_z(z):                      # z: (N,)
        G = jnp.exp(z)
        return vlp(G) + z + vll(G)         # +z = log|dG/dz| Jacobian
    return logpost_z


# ---------------------------------------------------------------- Metropolis
def run_parallel_rwmh(key, init_G, logprior_G, loglik_G, n_tune, n_draws,
                      step_size=0.4):
    """N parallel random-walk Metropolis chains (Gaussian RW on z=log G).
    Returns (G_draws (N, n_draws), info)."""
    logpost_z = _make_logpost_z(logprior_G, loglik_G)
    z0 = jnp.log(init_G)
    lp0 = logpost_z(z0)

    def step(carry, key):
        z, lp = carry
        k1, k2 = jax.random.split(key)
        zp = z + step_size * jax.random.normal(k1, z.shape)      # symmetric RW
        lpp = logpost_z(zp)                                       # N forward evals, vmapped
        acc = jnp.log(jax.random.uniform(k2, z.shape)) < (lpp - lp)
        z = jnp.where(acc, zp, z)
        lp = jnp.where(acc, lpp, lp)
        return (z, lp), (z, acc)

    keys = jax.random.split(key, n_tune + n_draws)
    _, (zs, accs) = jax.lax.scan(step, (z0, lp0), keys)          # zs: (n_tune+n_draws, N)
    G_draws = jnp.exp(zs[n_tune:]).T                             # (N, n_draws)
    return G_draws, {"accept": accs[n_tune:].mean()}


# ------------------------------------------------------------- DE-Metropolis
def run_demc(key, init_G, logprior_G, loglik_G, n_tune, n_draws,
             gamma=None, jitter=1e-4):
    """Differential-Evolution Metropolis over an ensemble of N chains. Proposal for
    chain i: z_i' = z_i + gamma*(z_r1 - z_r2) + N(0, jitter), with r1,r2 random other
    chains -> the ensemble-difference vector is independent of z_i, so the proposal
    is symmetric and the MH ratio is just the target difference. Returns (G_draws, info)."""
    N = init_G.shape[0]
    gamma = float(2.38 / jnp.sqrt(2.0)) if gamma is None else gamma   # d=1 default
    logpost_z = _make_logpost_z(logprior_G, loglik_G)
    z0 = jnp.log(init_G)
    lp0 = logpost_z(z0)

    def step(carry, key):
        z, lp = carry
        k1, k2, k3, k4 = jax.random.split(key, 4)
        r1 = jax.random.randint(k1, (N,), 0, N)
        r2 = jax.random.randint(k2, (N,), 0, N)
        zp = z + gamma * (z[r1] - z[r2]) + jitter * jax.random.normal(k3, z.shape)
        lpp = logpost_z(zp)                                      # N forward evals, vmapped
        acc = jnp.log(jax.random.uniform(k4, z.shape)) < (lpp - lp)
        z = jnp.where(acc, zp, z)
        lp = jnp.where(acc, lpp, lp)
        return (z, lp), (z, acc)

    keys = jax.random.split(key, n_tune + n_draws)
    _, (zs, accs) = jax.lax.scan(step, (z0, lp0), keys)
    G_draws = jnp.exp(zs[n_tune:]).T
    return G_draws, {"accept": accs[n_tune:].mean()}


# -------------------------------------------------------------------- Slice
def run_slice(key, init_G, logprior_G, loglik_G, n_tune, n_draws,
              w=1.0, max_expand=4, max_shrink=10):
    """N parallel univariate slice-sampling chains on z=log G (Neal 2003:
    stepping-out + shrinkage). Uses fixed-iteration fori_loops with masking instead
    of while_loops so it is efficient and safe under vmap. Returns (G_draws, info).

    COST NOTE: under vmap every chain pays the WORST-CASE trip count, because lanes
    cannot exit early independently -- so each step costs exactly
    2*max_expand + max_shrink likelihood evaluations regardless of how quickly the
    bracket would have converged. On CPU slice is cheap because it exits early; that
    advantage is structurally unavailable here. Defaults were 10/20 (=40 evals/step,
    ~40x rwmh, measured 122 s/step) which forced a 15-tune/30-draw budget and left
    R-hat ~= 2.95 (unconverged). 4/10 (=18 evals/step) is ~2.2x cheaper, buying ~2.2x
    more draws for the same wall clock. For a 1-D log-G posterior, 10 doublings of a
    w=1.0 bracket is far more than needed."""
    logpost_z = _make_logpost_z(logprior_G, loglik_G)
    z0 = jnp.log(init_G)

    def slice_step(carry, key):
        z, lp = carry                                  # lp = logpost_z(z), (N,)
        k1, k2, k3 = jax.random.split(key, 3)
        logy = lp + jnp.log(jax.random.uniform(k1, z.shape))   # slice level < lp

        # --- stepping-out: place [L,R] of width w randomly around z, then expand ---
        u = jax.random.uniform(k2, z.shape)
        L = z - w * u
        R = L + w

        def expand_L(_, L):
            grow = logpost_z(L) > logy                  # still inside slice -> keep expanding
            return jnp.where(grow, L - w, L)
        def expand_R(_, R):
            grow = logpost_z(R) > logy
            return jnp.where(grow, R + w, R)
        L = jax.lax.fori_loop(0, max_expand, expand_L, L)
        R = jax.lax.fori_loop(0, max_expand, expand_R, R)

        # --- shrinkage: sample in [L,R], accept first inside slice, else shrink ---
        keys = jax.random.split(k3, max_shrink)
        def shrink(i, state):
            L, R, znew, done = state
            up = jax.random.uniform(keys[i], z.shape)
            zp = L + up * (R - L)
            inside = logpost_z(zp) > logy
            take = inside & (~done)
            znew = jnp.where(take, zp, znew)
            done = done | inside
            # shrink the side of the interval that zp fell on (only if not done)
            shrinkL = (zp < z) & (~done)
            shrinkR = (zp >= z) & (~done)
            L = jnp.where(shrinkL, zp, L)
            R = jnp.where(shrinkR, zp, R)
            return (L, R, znew, done)
        _, _, znew, done = jax.lax.fori_loop(
            0, max_shrink, shrink, (L, R, z, jnp.zeros_like(z, dtype=bool)))
        znew = jnp.where(done, znew, z)                 # fallback: stay put if never accepted
        return (znew, logpost_z(znew)), (znew, done)

    keys = jax.random.split(key, n_tune + n_draws)
    lp0 = logpost_z(z0)
    _, (zs, dones) = jax.lax.scan(slice_step, (z0, lp0), keys)
    G_draws = jnp.exp(zs[n_tune:]).T
    return G_draws, {"accept": dones[n_tune:].mean()}   # fraction of steps that moved
