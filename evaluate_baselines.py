from aiofiles.os import path
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
    # "kangaroo_jax_0": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_1": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__1_20251118_140348",
    # "kangaroo_jax_2": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__2_20251118_222638",
    # "seaquest_jax_0": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753"
    # "seaquest_jax_1": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__1_20251118_113132",
    # "seaquest_jax_2": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__2_20251118_140442"
}

# For ASTRA
# Need:
# - nudge and blendrl: 
#   - kangaroo: default, center_ladders, four_ladders, flame_trap, cactus_trap, danger_trap, tanks, snakes, dragons, replace_coconut_fireball, replace_coconut_honey_bee, replace_coconut_wasp
#   - seaquest: default, fireballs, mines, no_divers
# BLENDRL:
evals = {
    # "seaquest_jax_0_fireballs": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753",
    # "seaquest_jax_0_mines": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753",
    # "seaquest_jax_0_mines-detect": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753",
    # "seaquest_jax_0_no-divers": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753",
    # "seaquest_jax_0_None": "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753",

    "kangaroo_jax_0_None": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    "kangaroo_jax_0_no-coconut": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_center-ladders": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_four-ladders": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_flame-trap": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_cactus-trap": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_danger-trap": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_tanks": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_snakes": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_dragons": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_replace-coconut-fireball": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_replace-coconut-honey-bee": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
    # "kangaroo_jax_0_replace-coconut-wasp": "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001",
}

# NUDGE
# evals = {
#     "seaquest_jax_0_fireballs": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",
#     "seaquest_jax_0_mines": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",
#     "seaquest_jax_0_mines-detect": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",
#     "seaquest_jax_0_no-divers": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",
#     "seaquest_jax_0_None": "out_nudge/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_140233",

#     "kangaroo_jax_0_center-ladders": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_four-ladders": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_flame-trap": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_cactus-trap": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_danger-trap": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_tanks": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_snakes": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_dragons": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_replace-coconut-fireball": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_replace-coconut-honey-bee": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_replace-coconut-wasp": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
#     "kangaroo_jax_0_None": "out_nudge/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_512_steps_128_2_20251118_132417",
# }

# NLRL
# evals = {
#     "seaquest_jax_0_fireballs": "out_nlrl/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260124_003447",
#     "seaquest_jax_0_mines": "out_nlrl/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260124_003447",
#     "seaquest_jax_0_mines-detect": "out_nlrl/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260124_003447",
#     "seaquest_jax_0_no-divers": "out_nlrl/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260124_003447",
#     "seaquest_jax_0_None": "out_nlrl/runs/seaquest_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260124_003447",

#     "kangaroo_jax_0_center-ladders": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_four-ladders": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_flame-trap": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_cactus-trap": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_danger-trap": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_tanks": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_snakes": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_dragons": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_replace-coconut-fireball": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_replace-coconut-honey-bee": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
#     "kangaroo_jax_0_replace-coconut-wasp": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
    # "kangaroo_jax_0_None": "out_nlrl/runs/kangaroo_jax_softmax_lr_0.00025_llr_0.00025_gamma_0.99_numenvs_256_steps_128_0_20260123_184901",
# }


scores = []
aligned_scores = []
mod_scores = []
aligned_mod_scores = []

for run, paths in evals.items():
    detect_all_enemies = False #standard setting: blendrl does not detect objects as enemies
    print("Evaluating run:", run)
    env_name = "_".join(run.split("_")[0:-2])
    seed = int(run.split("_")[-2])
    modification = run.split("_")[-1]
    # replace - with _ in modification
    modification = modification.replace("-", "_")
    if modification == "mines_detect":
        modification = "mines"
        detect_all_enemies=True

    score, _, _, _, aligned_score, _ = evaluate(env_name, paths, episodes=3, seed=seed, modified_env=modification, device="cpu", detect_all_enemies=detect_all_enemies)
    # mod_score, _, _, _, mod_aligned_score, _ = evaluate(env_name, path, episodes=3, seed=seed, modified_env=True)
    scores.append(score)
    aligned_scores.append(aligned_score)
    # mod_scores.append(mod_score)
    # aligned_mod_scores.append(mod_aligned_score)

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