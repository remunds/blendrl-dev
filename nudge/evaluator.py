from datetime import datetime
from typing import Union

import numpy as np
import torch as th

# import pygame
# import vidmaker

from nudge.agents.logic_agent import NsfrActorCritic
from nudge.agents.neural_agent import ActorCritic
from nudge.utils import load_model, yellow
from nudge.env import NudgeBaseEnv

SCREENSHOTS_BASE_PATH = "out/screenshots/"
PREDICATE_PROBS_COL_WIDTH = 500 * 2
FACT_PROBS_COL_WIDTH = 1000
CELL_BACKGROUND_DEFAULT = np.array([40, 40, 40])
CELL_BACKGROUND_HIGHLIGHT = np.array([40, 150, 255])
CELL_BACKGROUND_HIGHLIGHT_POLICY = np.array([234, 145, 152])
CELL_BACKGROUND_SELECTED = np.array([80, 80, 80])

import torch

torch.set_num_threads(5)


class Evaluator:
    model: Union[NsfrActorCritic, ActorCritic]
    # window: pygame.Surface
    # clock: pygame.time.Clock

    def __init__(
        self,
        agent_path: str = None,
        env_name: str = "seaquest",
        device: str = "cpu",
        fps: int = None,
        deterministic=True,
        env_kwargs: dict = None,
        render_predicate_probs=True,
        episodes: int = 2,
        seed=0,
    ):

        self.fps = fps
        self.deterministic = deterministic
        self.render_predicate_probs = render_predicate_probs
        self.episodes = episodes
        self.agent_path = agent_path
        self.env_name = env_name

        # Load model and environment
        self.model = load_model(
            agent_path, env_kwargs_override=env_kwargs, device=device
        )
        self.env = NudgeBaseEnv.from_name(
            env_name, mode="deictic", seed=seed, **env_kwargs
        )
        # self.env = self.model.env
        self.env.reset()

        print(self.model._print())

        print(
            f"Playing '{self.model.env.name}' with {'' if deterministic else 'non-'}deterministic policy."
        )

        if fps is None:
            fps = 15
        self.fps = fps

        try:
            self.action_meanings = self.env.env.get_action_meanings()
            self.keys2actions = self.env.env.unwrapped.get_keys_to_action()
        except Exception:
            print(
                yellow(
                    "Info: No key-to-action mapping found for this env. No manual user control possible."
                )
            )
            self.action_meanings = None
            self.keys2actions = {}
        self.current_keys_down = set()

        self.predicates = self.model.logic_actor.prednames

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False

    def run(self):
        length = 0
        ret = 0

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.model.device)
        obs = obs.to(self.model.device)
        # print(obs_nn.shape)

        episode_count = 0
        step_count = 0
        blend_entropies = []

        runs = range(self.episodes)
        while self.running:
            self.reset = False
            # self._handle_user_input()
            if not self.paused:
                if not self.running:
                    break  # outer game loop

                if self.takeover:  # human plays game manually
                    # assert False, "Unimplemented."
                    action = self._get_action()
                else:  # AI plays the game
                    obs_nn = obs_nn.to(th.float32)
                    obs = obs.to(th.float32)
                    action, logprob = self.model.act(
                        obs_nn, obs
                    )  # update the model's internals
                    value = self.model.get_value(obs_nn, obs)
                    # get blend entropy
                    _, newlogprob, entropy, blend_entropy, newvalue = (
                        self.model.get_action_and_value(obs_nn, obs, action)
                    )
                    blend_entropies.append(blend_entropy.detach().item())
                    step_count += 1
                    # if step_count > 1000:
                    # break

                (new_obs, new_obs_nn), reward, done, terminations, infos = (
                    self.env.step(action, is_mapped=self.takeover)
                )
                if reward > 0:
                    print(f"Reward: {reward:.2f} at Step {step_count}")
                new_obs_nn = th.tensor(new_obs_nn, device=self.model.device)

                # self._render()

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                if self.reset:
                    done = True
                    new_obs = self.env.reset()
                    # self._render()

                obs = new_obs
                obs = obs.to(self.model.device)
                obs_nn = new_obs_nn
                obs_nn = obs_nn.to(self.model.device)
                length += 1

                if terminations:
                    print("Episode terminated.")
                    game_return = infos["returned_episode_env_returns"].mean().item()
                    blendrl_return = infos["returned_episode_returns_0"].mean().item()
                    length = infos["returned_episode_lengths"].mean().item()
                    print(
                        f"Game Return: {game_return:.2f}, BlendRL Return: {blendrl_return:.2f}, Length: {length}"
                    )
                    episode_count += 1
                    if episode_count >= self.episodes:
                        break
                    self.env.reset()
                    # for self tracking
                    print("Terminate episode at time step: ", step_count)

                if step_count > 10000:
                    print("Terminate episode at time step: ", step_count)
                    self.env.reset()
                    if episode_count >= self.episodes:
                        break

        # compute mean and std

        # pygame.quit()

    def _get_action(self):
        if self.keys2actions is None:
            return 0  # NOOP
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP
