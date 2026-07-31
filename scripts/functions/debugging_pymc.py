import numpy as np
import pytensor
import pytensor.tensor as pt
import pymc as pm
# 🔹 Import YOUR module
import mpr_jax_pymc as mpr


def main():

    print("\n=== IMPORT SUCCESSFUL ===")
    print("Module loaded:", mpr)

    # ----------------------------------------------------
    # 1️⃣ Rebuild the exact FIXED inputs you use in PyMC
    # ----------------------------------------------------

    G_test = np.array(0.33, dtype=np.float32)

    SC = np.zeros((6, 6), dtype=np.float32)   # adjust if needed
    t_end = np.float32(30000.0)
    cut = np.float32(30.0)

    print("\n=== INPUTS ===")
    print("G:", G_test)
    print("SC:", SC.shape)
    print("t_end:", t_end)
    print("cut:", cut)

    # ----------------------------------------------------
    # 2️⃣ Call the RAW JAX / NUMPY WRAPPER DIRECTLY
    # ----------------------------------------------------
    print("\n=== TESTING RAW WRAPPER (NO OP) ===")

    try:
        raw_out = mpr.wrapper_fcd(
            G_test,
            SC,
            t_end,
            cut
        )

        raw_out = np.asarray(raw_out)

        print("Wrapper output shape:", raw_out.shape)
        print("Wrapper output dtype:", raw_out.dtype)
        print("NaNs in wrapper:", np.isnan(raw_out).any())
        print("Infs in wrapper:", np.isinf(raw_out).any())
        print("First 10 values:", raw_out[:10])

    except Exception as e:
        print("❌ WRAPPER CRASHED:")
        raise e


    # ----------------------------------------------------
    # 3️⃣ Test the PYTENSOR OP DIRECTLY
    # ----------------------------------------------------
    print("\n=== TESTING PYTENSOR OP ===")

    # Rebuild symbolic variables
    G_sym = pt.scalar("G")
    SC_sym = pt.matrix("SC")
    t_end_sym = pt.scalar("t_end")
    cut_sym = pt.scalar("cut")

    # Recreate the Op EXACTLY how your model does
    op_out = mpr.pytensor_forward_model_matrix(
        G_sym, SC_sym, t_end_sym, cut_sym
    )

    # Compile a pure pytensor function
    f = pytensor.function(
        inputs=[G_sym, SC_sym, t_end_sym, cut_sym],
        outputs=op_out,
        on_unused_input="ignore"
    )

    print("\nCompiling OP -> function successful")

    # ----------------------------------------------------
    # 4️⃣ Execute the Op Safely
    # ----------------------------------------------------
    try:
        op_result = f(G_test, SC, t_end, cut)
        op_result = np.asarray(op_result)

        print("\n=== OP OUTPUT ===")
        print("Shape:", op_result.shape)
        print("Dtype:", op_result.dtype)
        print("NaNs in OP:", np.isnan(op_result).any())
        print("Infs in OP:", np.isinf(op_result).any())
        print("First 10 values:", op_result[:10])

    except Exception as e:
        print("\n❌ OP EXECUTION FAILED")
        raise e

    # ----------------------------------------------------
    # 5️⃣ Direct Comparison
    # ----------------------------------------------------
    print("\n=== RAW vs OP COMPARISON ===")

    if raw_out.shape == op_result.shape:
        diff = np.abs(raw_out - op_result)
        print("Max abs difference:", np.nanmax(diff))
    else:
        print("⚠️ Shape mismatch:",
              raw_out.shape, "vs", op_result.shape)
        

    print("\n=== TESTING PURE LIKELIHOOD NUMERIC STABILITY (PyMC v5 SAFE) ===")

    FC_sim = raw_out.astype(np.float64)
    FC_obs = FC_sim + 0.05  # small mismatch

    for sigma in [0.1, 0.01, 0.001]:
        with pm.Model() as m:
            x = pm.Normal("x", mu=FC_sim, sigma=sigma, observed=FC_obs)

            try:
                # ✅ This is the correct way in PyMC v5
                logp_val = m.point_logps()

                all_logp_vals = np.array(list(logp_val.values()), dtype=float)

                print(
                    f"sigma={sigma} → "
                    f"min={np.min(all_logp_vals):.2e}, "
                    f"max={np.max(all_logp_vals):.2e}, "
                    f"finite={np.all(np.isfinite(all_logp_vals))}"
                )

            except Exception as e:
                print(f"sigma={sigma} FAILED:", e)




if __name__ == "__main__":
    main()
