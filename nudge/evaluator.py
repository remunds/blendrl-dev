from datetime import datetime
from typing import Union
import os
import numpy as np
import torch as th

# import pygame
# import vidmaker
from moviepy import *

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

        # Turn off episodic life for evaluation
        env_kwargs["episodic_life"] = False

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

        # self.predicates = self.model.logic_actor.prednames

        self.running = True
        self.paused = False
        self.fast_forward = False
        self.reset = False
        self.takeover = False

    def run(self):

        obs, obs_nn = self.env.reset()
        obs_nn = th.tensor(obs_nn, device=self.model.device)
        obs = obs.to(self.model.device)
        # print(obs_nn.shape)

        episode_count = 0
        step_count = 0

        game_returns = []
        blendrl_returns = []
        aligned_scores = []
        frames = []

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
                    # action, logprob = self.model.act(
                    #     obs_nn, obs
                    if isinstance(self.model, NsfrActorCritic):
                        action, _, _, _ = self.model.get_action_and_value(
                            obs_nn, obs
                        )
                    else:
                        action, _, _, _, _ = self.model.get_action_and_value(
                            obs_nn, obs
                        )
                        # action, logprob = self.model.act(
                        #     obs_nn, obs
                        # )  # update the model's internals
                    step_count += 1
                    # if step_count > 1000:
                    # break

                (new_obs, new_obs_nn), reward, truncations, dones, infos = (
                    self.env.step(action, is_mapped=self.takeover)
                )
                if reward > 0:
                    print(f"Reward: {reward:.2f} at Step {step_count}")
                new_obs_nn = th.tensor(new_obs_nn, device=self.model.device)

                frames.append(self.env.render())
                # self._render()

                if self.takeover and float(reward) != 0:
                    print(f"Reward {reward:.2f}")

                obs = new_obs
                obs = obs.to(self.model.device)
                obs_nn = new_obs_nn
                obs_nn = obs_nn.to(self.model.device)

                if dones.any():
                    print("Episode done.: ", dones)
                    game_return = infos["returned_episode_env_returns"].mean().item()
                    game_returns.append(game_return)
                    blendrl_return = infos["returned_episode_returns_0"].mean().item()
                    blendrl_returns.append(blendrl_return)
                    ep_length = infos["returned_episode_lengths"].mean().item()
                    if "returned_episode_returns_1" in infos:
                        aligned_score = infos["returned_episode_returns_1"].mean().item()
                        aligned_scores.append(aligned_score)
                        print(
                            f"Game Return: {game_return:.2f}, BlendRL Return: {blendrl_return:.2f}, Aligned Score: {aligned_score:.2f}, Length: {ep_length}"
                        )
                    else:
                        print(
                            f"Game Return: {game_return:.2f}, BlendRL Return: {blendrl_return:.2f}, Length: {ep_length}"
                        )
                    episode_count += 1

                    # frames = np.array(frames)
                    # frames = np.transpose(frames, (0, 3, 1, 2))
                    video = ImageSequenceClip(frames, fps=30)
                    dir_path = "out/videos/"
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    path = dir_path + f"eval_video_{episode_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"


                    video.write_videofile(path, fps=30)
                    frames = []
                    # for self tracking
                    obs, obs_nn = self.env.reset()
                    obs_nn = th.tensor(obs_nn, device=self.model.device)
                    obs = obs.to(self.model.device)

                    if step_count >= 9_999:
                        print("Terminated episode at time step: ", step_count)
                    else:
                        print("Done at time step: ", step_count)
                    step_count = 0
                    if episode_count >= self.episodes:
                        break

        # compute mean and std
        mean_return = np.mean(game_returns)
        std_return = np.std(game_returns)
        mean_blendrl_return = np.mean(blendrl_returns)
        std_blendrl_return = np.std(blendrl_returns)
        aligned_mean_return = np.mean(aligned_scores) if len(aligned_scores) > 0 else 0
        aligned_std_return = np.std(aligned_scores) if len(aligned_scores) > 0 else 0
        return mean_return, std_return, mean_blendrl_return, std_blendrl_return, aligned_mean_return, aligned_std_return 
        # game_ret, game_std, aligned_ret, aligned_std, mod_game_ret, mod_game_std, mod_aligned_ret, mod_aligned_std
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
