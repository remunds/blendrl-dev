from collections import OrderedDict

import torch
import gymnasium as gym
from nsfr.common import get_nsfr_model, get_blender_nsfr_model
from nsfr.utils.common import load_module
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from stable_baselines3 import PPO
# from huggingface_sb3 import load_from_hub, push_to_hub

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NeuralBlenderActor(nn.Module):
    """
    Neural Blender Actor; 
    a neural network that takes an image as input and outputs a probability distribution over policies.
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 2), std=0.01)
        
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs
    
    
class CNNActor(nn.Module):
    """
    Neural Blender Actor;
    a neural network that takes an image as input and outputs a probability distribution over actions.
    """
    def __init__(self, n_actions=18, ):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs
    
class MLPActor(nn.Module):
    """
    An MLP Actor-Critic model that takes a flat array (1D vector) as input.
    
    Designed to have a parameter count similar to the CNNActor.
    """
    def __init__(self, input_dim, n_actions=18):
        super().__init__()
        
        # This MLP structure approximates the parameter count of the 
        # CNNActor's large Linear(3136, 512) layer.
        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )
        
        # The actor and critic heads are identical to the original
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        # Note: Removed x / 255.0, as the input is a flat array,
        # not assumed to be pixel data.
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def forward(self, x):
        """Used for getting action probabilities during inference."""
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

def get_blender(env, blender_rules, device, train=True, blender_mode='logic', reasoner='nsfr', explain=False):
    """
    Load a Blender model. 
    Args:
        env (gym.Env): Environment.
        blender_rules (str): Path to Blender rules.
        device (torch.device): Device.
        train (bool): Whether to train the model.
        blender_mode (str): Mode of Blender. Possible values are "logic" and "neural".
        reasoner (str): Reasoner. Possible values are "nsfr" and "neumann".
        explain (bool): Whether to explain the model.
    Returns:
        Blender: Blender model.
    """
    assert blender_mode in ['logic', 'neural']
    if blender_mode == 'logic':
        if reasoner == 'nsfr':
            return get_blender_nsfr_model(env.name, blender_rules, device, train=train, explain=explain)
        elif reasoner == 'neumann':
            from neumann.common import get_neumann_model, get_blender_neumann_model
            return get_blender_neumann_model(env.name, blender_rules, device, train=train, explain=explain)
    if blender_mode == 'neural':
        net = NeuralBlenderActor()
        net.to(device)
        return net
    
    
def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from cleanrl.cleanrl.ppo_atari import make_env
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs
    
def load_cleanrl_agent(pretrained, device, cnn=False, input_dim=None):
    # from cleanrl.cleanrl.ppo_atari import Agent
    if cnn:
        agent = CNNActor(n_actions=18) #, device=device, verbose=1)
    else:
        agent = MLPActor(input_dim=input_dim, n_actions=18)
    if pretrained:
        try:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
            agent.to(device)
        except RuntimeError:
            agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth", map_location=torch.device('cpu')))
    else:
        agent.to(device)
    return agent


def load_logic_ppo(agent, path):
    new_actor_dic = OrderedDict()
    new_critic_dic = OrderedDict()
    dic = torch.load(path)
    for name, value in dic.items():
        if 'actor.' in name:
            new_name = name.replace('actor.', '') 
            new_actor_dic[new_name] = value
        if 'critic.' in name:
            new_name = name.replace('critic.', '') 
            new_critic_dic[new_name] = value
    agent.logic_actor.load_state_dict(new_actor_dic)
    agent.logic_critic.load_state_dict(new_critic_dic)
    return agent