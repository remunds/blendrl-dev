import tyro
from nudge.evaluator import Evaluator

from dataclasses import dataclass
import tyro


def main(
    env_name: str = "seaquest",
    agent_path: str = "out/runs/models/kangaroo_demo",
    fps: int = 5,
    episodes: int = 2,
    model: str = "blendrl",
    device: str = "cuda:0",
    seed: int = 0,
    modified_env: bool = False,
) -> None:
    """
    Evaluation script. This script evaluates the performance of the blendrl on new episodes.
    """
    evaluator = Evaluator(
        episodes=episodes,
        agent_path=agent_path,
        env_name=env_name,
        fps=fps,
        deterministic=False,
        device=device,
        # env_kwargs=dict(render_oc_overlay=True),
        seed=seed,
        env_kwargs=dict(render_oc_overlay=False, modified_env=modified_env),
        render_predicate_probs=True,
    )
    return evaluator.run()


if __name__ == "__main__":
    tyro.cli(main)
