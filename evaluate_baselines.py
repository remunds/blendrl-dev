from statistics import median_grouped
from evaluate import main as evaluate
import numpy as np

# NUDGE
evals = {
    # "kangaroo_jax_0": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_0",
#     "kangaroo_jax_1": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_1_20251118_113300",
    # "kangaroo_jax_2": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "seaquest_jax_0": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_0",
#     "seaquest_jax_1": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_1_20251118_113234",
#     "seaquest_jax_2": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",
}

# BLENDRL
evals = {
    # "kangaroo_jax_0": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0",
    # "kangaroo_jax_1": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__1_20251118_140348",
    "kangaroo_jax_2": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__2_20251118_222638",
    # "seaquest_jax_0": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0",
    # "seaquest_jax_1": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__1_20251118_113132",
    # "seaquest_jax_2": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__2_20251118_140442"
}

scores = []
aligned_scores = []
mod_scores = []
aligned_mod_scores = []

for run, path in evals.items():
    print("Evaluating run:", run)
    env_name = "_".join(run.split("_")[0:-1])
    seed = int(run.split("_")[-1])
    score, _, _, _, aligned_score, _ = evaluate(env_name, path, episodes=3, seed=seed, modified_env=False)
    mod_score, _, _, _, mod_aligned_score, _ = evaluate(env_name, path, episodes=3, seed=seed, modified_env=True)
    scores.append(score)
    aligned_scores.append(aligned_score)
    mod_scores.append(mod_score)
    aligned_mod_scores.append(mod_aligned_score)

mean_score = np.mean(scores)
std_score = np.std(scores)
mean_aligned_score = np.mean(aligned_scores)
std_aligned_score = np.std(aligned_scores)
mean_mod_score = np.mean(mod_scores)
std_mod_score = np.std(mod_scores)
mean_aligned_mod_score = np.mean(aligned_mod_scores)
std_aligned_mod_score = np.std(aligned_mod_scores)
print("Results over different seeds:")
print("Standard Env Score:", mean_score, "+-", std_score)
print("Standard Env Aligned Score:", mean_aligned_score, "+-", std_aligned_score) 
print("Modified Env Score:", mean_mod_score, "+-", std_mod_score)
print("Modified Env Aligned Score:", mean_aligned_mod_score, "+-", std_aligned_mod_score)
print(mean_score, std_score, mean_mod_score, std_mod_score, mean_aligned_score, std_aligned_score, mean_aligned_mod_score, std_aligned_mod_score)