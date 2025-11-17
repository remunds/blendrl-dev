import functools
import time
from turtle import position
from typing import Sequence
import torch
# from HackAtari.hackatari.games import kangaroo
from nudge.env import NudgeBaseEnv
from blendrl.env_utils import make_env
from hackatari.core import HackAtari
import torch as th
from ocatari.ram.kangaroo import MAX_ESSENTIAL_OBJECTS
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jaxatari.games.jax_kangaroo import JaxKangaroo
from jaxatari.wrappers import AtariWrapper, MultiRewardLogWrapper

import time

from blendrl.env_utils import kangaroo_modifs

def blendrl_reward_function(prev_state, state) -> float:
    org_reward = state.score - prev_state.score
    
    cond = jnp.logical_and(state.player.y <= 10, prev_state.player.y > 4) # near the top
    reward = jnp.where(cond, 20.0, jnp.where(org_reward >= 1.0, 1.0, 0.0))
    return reward

class NudgeEnv(NudgeBaseEnv):
    """
    NUDGE environment for Kangaroo.

    Args:
        mode (str): Mode of the environment. Possible values are "train" and "eval".
        n_envs (int): Number of environments.
        render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
        render_oc_overlay (bool): Whether to render the overlay of OC.
        seed (int): Seed for the environment.
    """

    name = "kangaroo_jax"
    pred2action = {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
    }
    pred_names: Sequence

    def __init__(
        self,
        mode: str,
        render_mode="rgb_array",
        render_oc_overlay=False,
        seed=0,
    ):
        """
        Constructor for the VectorizedNudgeEnv class.

        Args:
            mode (str): Mode of the environment. Possible values are "train" and "eval".
            n_envs (int): Number of environments.
            render_mode (str): Mode of rendering. Possible values are "rgb_array" and "human".
            render_oc_overlay (bool): Whether to render the overlay of OC.
            seed (int): Seed for the environment.
        """
        super().__init__(mode)
        # set up multiple envs

        env = JaxKangaroo(reward_funcs=[blendrl_reward_function])
    
        #TODO: For actual BlendRL style, we should use ObjectCentricAndPixelObsWrapper
        # then feed pixel as neural state and oc as logic state
        # But: for fair comparison with NEXUS, we keep only OC observations
        env = AtariWrapper(
            env,
            episodic_life=True, # explicitly set in cleanRL-envpool
            clip_reward=False, 
            max_episode_length=108000,
            frame_stack_size=4,
            max_pooling=True,
            frame_skip=4,
            noop_reset=30,
            sticky_actions=False, 
            first_fire=True,
        )
        self.env = MultiRewardLogWrapper(env)

        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 49
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        orig_key = jax.random.PRNGKey(seed)
        self.key = orig_key 

        # observation space: (4, 180)
        # logic observation space: (180,)
        # self.single_observation_space = self.env.reset(self.keys[0])[0].shape
        self.single_observation_space = jax.vmap(self._kangaroo_observation_to_array)(self.env.reset(self.key)[0]).shape
        self.single_logic_observation_space = tuple(list(self.single_observation_space)[1:])
        print("Single obs space:", self.single_observation_space)
        print("Single logic obs space:", self.single_logic_observation_space)

        self.state = None

    @functools.partial(jax.jit, static_argnums=(0,))
    def _kangaroo_observation_to_array(self, obs):

        def _position_arr_to_nudge(position_array):
            # array has shape (N, 2), but need (N, 4)
            # inactive, if any is -1
            n_dims = position_array.ndim
            if n_dims == 1:
                position_array = position_array.reshape(1, -1)
            n_objects = position_array.shape[0]
            final_array = jnp.zeros((n_objects, 4), dtype=jnp.int32)
            final_array = final_array.at[:, 0].set(jnp.where(position_array[:, 0] == -1, 0, 1))  # active
            final_array = final_array.at[:, 1].set(position_array[:, 0])
            final_array = final_array.at[:, 2].set(position_array[:, 1])
            # orientation is not available
            final_array = final_array.at[:, 3].set(0)
            if n_dims == 1:
                return final_array[0]
            return final_array 


        final_obs = jnp.zeros((self.n_objects, self.n_features), dtype=np.int32)
        # player
        player = jnp.stack([1, obs.player_x, obs.player_y, obs.player_o])
        final_obs = final_obs.at[0].set(jnp.array(player, dtype=jnp.int32))
        # child
        final_obs = final_obs.at[1].set(jnp.array([1, obs.child_position[0], obs.child_position[1], 0], dtype=jnp.int32))
        # monkeys
        final_obs = final_obs.at[2:6].set(_position_arr_to_nudge(obs.monkey_positions))
        
        # falling_coco
        final_obs = final_obs.at[6].set(_position_arr_to_nudge(obs.falling_coco_position))
        
        # thrown_coco
        # Note: only first 3 coconuts are kept for now (there are max 4 in jaxatari)
        final_obs = final_obs.at[7:10].set(_position_arr_to_nudge(obs.coco_positions)[:3])
        
        # fruit
        final_obs = final_obs.at[10:13].set(_position_arr_to_nudge(obs.fruit_positions))

        # bell
        final_obs = final_obs.at[13].set(_position_arr_to_nudge(obs.bell_position).squeeze()) 

        # ladder
        # Note: only first 6 ladders are kept for now (there are max 20 in jaxatari, all levels...)
        final_obs = final_obs.at[14:20].set(_position_arr_to_nudge(obs.ladder_positions)[:6])
        
        # platform
        final_obs = final_obs.at[20:40].set(_position_arr_to_nudge(obs.platform_positions))

        # platforms
        return final_obs


    def reset(self):
        obs, state = self.env.reset(self.key)
        self.state = state
        self.key, _ = jax.random.split(self.key) 
        # prob: obs arrays have shape (n_envs, frame_stack)
        obs = jax.vmap(self._kangaroo_observation_to_array)(obs)
        # add batch_dim in front
        obs = obs[jnp.newaxis, ...]
        # for logic_obs, we take only the last frame (no frame stack)
        return torch.tensor(np.array(obs[:, -1])), np.array(obs) # Use jaxatari OC-obs directly for both logic and neural state

    def step(self, action, is_mapped: bool = False):
        action = jnp.array(action).squeeze()
        obs, state, rewards, dones, infos = self.env.step(self.state, action)
        truncations = jnp.zeros_like(dones).astype(bool)  # jaxatari does not yet support truncations separately
        self.state = state
        obs = jax.vmap(self._kangaroo_observation_to_array)(obs)
        # add batch_dim in front
        obs = obs[jnp.newaxis, ...]
        logic_obs = torch.tensor(np.array(obs[:, -1])) 
        obs = np.array(obs)
        all_rewards = infos.pop("all_rewards")
        rewards = np.array(all_rewards[0])
        dones = np.array(dones)
        truncations = np.array(truncations)

        return (
            (logic_obs, obs),
            rewards,
            truncations,
            dones,
            infos,
        )

    def close(self):
        pass