import functools
from typing import Sequence
from nudge.env import NudgeBaseEnv
from blendrl.env_utils import make_env
import torch
from ocatari.ram.seaquest import MAX_NB_OBJECTS
import gymnasium as gym
from hackatari.core import HackAtari
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer, SeaquestState
from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, MultiRewardLogWrapper, PixelObsWrapper
from nsfr.nsfr.fol import logic

def blendrl_reward_function(prev_state, state) -> float:
    org_reward = state.score - prev_state.score
    cond = jnp.logical_and(org_reward > 0.0, jnp.logical_not(jnp.isin(state.player_y, jnp.array([44, 45, 46, 47]))))
    cond2 = jnp.logical_and(org_reward > 0.0, jnp.isin(state.player_y, jnp.array([44, 45, 46, 47])))
    reward = jnp.where(cond, 0.5, jnp.where(cond2, 1.0, 0.0))
    return reward

@jax.jit
def total_collected(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if player is at surface with 6 divers
    reward = jnp.where(
        state.divers_collected > prev_state.divers_collected,
        1.0,
        0.0
    )
    return reward 

class NudgeEnv(NudgeBaseEnv):
    name = "seaquest_jax"
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
        modified_env=False,
        episodic_life=True
    ):
        super().__init__(mode)
        # set up multiple envs
        env = JaxSeaquest(reward_funcs=[blendrl_reward_function, total_collected])
        self.renderer = SeaquestRenderer()
        if modified_env:
            env = DisableEnemiesWrapper(env)
    
        #TODO: For actual BlendRL style, we should use ObjectCentricAndPixelObsWrapper
        # then feed pixel as neural state and oc as logic state
        # But: for fair comparison with NEXUS, we keep only OC observations
        env = AtariWrapper(
            env,
            episodic_life=episodic_life, # explicitly set in cleanRL-envpool
            clip_reward=False, 
            max_episode_length=10_000, 
            frame_stack_size=4,
            # max_pooling=True,
            frame_skip=4,
            noop_reset=0,
            sticky_actions=False, 
            first_fire=False,
        )
        self.env = MultiRewardLogWrapper(env)

        # for learning script from cleanrl
        self.n_actions = 6
        self.n_raw_actions = 18
        # TODO: Important: n_objects and n_features needs to be correct
        # self.n_objects = 43
        self.n_objects = 37
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.key = jax.random.PRNGKey(seed)
        # observation space: (4, 180)
        # logic observation space: (180,)
        # self.single_observation_space = self.env.reset(self.keys[0])[0].shape
        self.single_observation_space = jax.vmap(self._seaquest_observation_to_array)(self.env.reset(self.key)[0]).shape
        self.single_logic_observation_space = tuple(list(self.single_observation_space)[1:])
        print("Single obs space:", self.single_observation_space)
        print("Single logic obs space:", self.single_logic_observation_space)

        self.state = None

    @functools.partial(jax.jit, static_argnums=(0,))
    def _seaquest_observation_to_array(self, obs):
        def entity_array_to_nudge(array):
            # array: (b, N, 4)
            # else to 1
            active = array[..., 4]
            x = array[..., 0]
            y = array[..., 1]
            # orientation is not available for most entities in jaxatari seaquest
            orientation = jnp.zeros_like(active)
            return jnp.stack([active, x, y, orientation], axis=-1)  # (b, 4)

        final_obs = jnp.zeros((self.n_objects, self.n_features), dtype=np.int32)
        # player
        player = jnp.stack([obs.player.active, obs.player.x, obs.player.y, obs.player.o])
        final_obs = final_obs.at[0].set(jnp.array(player, dtype=jnp.int32))
        # sharks
        final_obs = final_obs.at[1:13].set(entity_array_to_nudge(obs.sharks))
        
        # submarines
        final_obs = final_obs.at[13:25].set(entity_array_to_nudge(obs.submarines))
        
        # divers
        final_obs = final_obs.at[25:29].set(entity_array_to_nudge(obs.divers))
        
        # enemy missiles
        final_obs = final_obs.at[29:33].set(entity_array_to_nudge(obs.enemy_missiles)) 
        # surface submarine
        final_obs = final_obs.at[33].set(jnp.array([obs.surface_submarine.active, obs.surface_submarine.x, obs.surface_submarine.y, 0]))
        # player missile
        final_obs = final_obs.at[34].set(jnp.array([obs.player_missile.active, obs.player_missile.x, obs.player_missile.y, 0]))
        # oxygen bar
        final_obs = final_obs.at[35].set(jnp.array([1, obs.oxygen_level, 0, 0]))
        # NOTE: this is different to ocatari 
        final_obs = final_obs.at[36].set(jnp.array([1, obs.collected_divers, 0, 0]))
        return final_obs 


    def reset(self):
        obs, state = self.env.reset(self.key)
        self.state = state
        self.key, _ = jax.random.split(self.key)
        # prob: obs arrays have shape (n_envs, frame_stack)
        obs = jax.vmap(self._seaquest_observation_to_array)(obs)
        obs = obs[jnp.newaxis, ...]
        # for logic_obs, we take only the last frame (no frame stack)
        return torch.tensor(np.array(obs[:, -1])), np.array(obs) # Use jaxatari OC-obs directly for both logic and neural state

    def step(self, action, is_mapped: bool = False):
        # need to vmap over both
        action = jnp.array(action).squeeze()
        obs, state, rewards, dones, infos = self.env.step(self.state, action)
        truncations = jnp.zeros_like(dones).astype(bool)  # jaxatari does not yet support truncations separately
        self.state = state
        obs = jax.vmap(self._seaquest_observation_to_array)(obs)
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
    
    def render(self ):
        return self.renderer.render(self.state.atari_state.env_state)


    def close(self):
        pass
