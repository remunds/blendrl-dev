import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Sequence, Union

from jaxatari.environment import JAXAtariAction

# Nudge Imports
from nudge.utils import load_model, load_neuralppo_model
from nudge.env import NudgeBaseEnv

class BlendrlRLAgent:
    def __init__(self, env, checkpoint_path, env_name="kangaroo", device="cuda"):
        """
        Adapter for BlendRL / Nudge agents (Logic, Neural, Hybrid).
        
        Args:
            env: The environment instance (e.g. JaxEnvironment). Used by PixelRLAgent pattern, 
                 but BlendRL agents typically manage their own env wrapping or state extraction.
            checkpoint_path: Path to the checkpoint file or model directory.
            env_name: Name of the environment (e.g. 'kangaroo').
            device: 'cuda' or 'cpu'.
        """
        print(f"[BlendRL] Initializing Agent for {env_name} from {checkpoint_path}...")
        self.device = torch.device(device)
        self.env_name = env_name
        self.external_env = env # Keep reference but might not use if we can't extract state
        
        # Parse Path
        path = Path(checkpoint_path)
        self.step = None
        if path.is_file():
            # e.g. .../checkpoints/step_10.pth
            if path.parent.name == "checkpoints":
                self.model_dir = path.parent.parent
                if "step_" in path.stem:
                    self.step = int(path.stem.split('_')[-1])
            else:
                self.model_dir = path.parent
        else:
            self.model_dir = path

        # Attempt to Load Model
        # We try generic load_model first (handles Logic/Blend/Neural-PPO-Logic variants)
        # If that fails or if config suggests otherwise, we might try specific loaders
        try:
            # We redirect stdout to avoid clutter during load
            # self.model, _ = load_model(self.model_dir, steps=self.step, device=self.device)
            # Actually, standard load_model might instantiate a Vectorized Env which we want to avoid if possible,
            # but it is coupled.
            # We will use it for now as it reconstructs the model architecture correctly.
            model_res = load_model(self.model_dir, steps=self.step, device=self.device)
            if isinstance(model_res, tuple):
                self.model = model_res[0]
            else:
                self.model = model_res
            print("[BlendRL] Loaded using standard load_model.")
        except Exception as e:
            print(f"[BlendRL] Standard load_model failed ({e}), trying load_neuralppo_model...")
            try:
                self.model = load_neuralppo_model(self.model_dir, steps=self.step, device=self.device)
                print("[BlendRL] Loaded using load_neuralppo_model.")
            except Exception as e2:
                raise RuntimeError(f"Could not load BlendRL model: {e} / {e2}")

        self.model.eval()

        # Determine Agent Type
        # NsfrActorCritic is purely logic and returns Predicate Index
        # BlenderActorCritic is hybrid but returns Env Action Index (internally agg)
        model_class_name = self.model.__class__.__name__
        
        self.is_pure_logic = "NsfrActorCritic" in model_class_name
        
        # We need logic state if it's pure logic OR hybrid (Blender)
        # We assume if it's not CNN, it likely needs logic.
        # But we need to distinguish for mapping.
        
        self.is_cnn = model_class_name == "CNNActor"
        
        # Check for MLP Neural Actor in Blender (common in kangaroo_jax)
        self.is_blender_mlp = False
        if hasattr(self.model, "visual_neural_actor"):
             # It might be MLPActor or CNNActor loaded via load_cleanrl_agent
             actor_name = self.model.visual_neural_actor.__class__.__name__
             if "MLPActor" in actor_name:
                 self.is_blender_mlp = True
                 print(f"[BlendRL] Detected MLP Neural Actor ({actor_name}): Will flatten inputs without resizing.")
        
        # Load Helper Env for Action Mapping & Metadata
        # We use mode='eval' to avoid training overheads
        try:
            # Don't want to use episodic_life during eval 
            self.helper_env = NudgeBaseEnv.from_name(env_name, mode='eval', episodic_life=False)
        except Exception as e:
            print(f"[BlendRL] Warning: Could not instantiate helper NudgeEnv: {e}")
            self.helper_env = None

    def predict(self, observation):
        """
        Predict action from observations.
        
        Args:
            observation: A tuple `(logic_state, neural_state)` as returned by BlendRL envs.
        
        Returns:
            Action string compatible with ThinkRL/JAXAtari.
        """
        logic_state = None
        neural_state = None

        # 1. Unpack Observation if Tuple
        if isinstance(observation, (tuple, list)) and len(observation) == 2:
            # Assumes format based on kangaroo_jax env: (logic, neural)
            logic_state, neural_state = observation
        else:
            # Assume just pixels
            neural_state = observation

        neural_state = torch.tensor(np.array(neural_state), device=self.device)
        # 2. Logic / Hybrid Execution
        if not self.is_cnn:
            if logic_state is None:
                raise NotImplementedError(
                    "BlendRL Logic/Hybrid Agents require Logic State. "
                    "Ensure 'predict' is called with (logic_state, neural_state) tuple."
                )
            
            # Ensure Tensors
            if not isinstance(logic_state, torch.Tensor):
                # kangaroo_jax might return numpy or jax array, convert:
                logic_state = torch.as_tensor(np.array(logic_state), device=self.device)
            else:
                logic_state = logic_state.to(self.device)

            # Ensure Neural State (Hybrid needs it, Logic ignores it effectively)
            if neural_state is not None:
                # Special Handling for Blender MLP Agents (e.g. kangaroo_jax)
                if self.is_blender_mlp:
                    if isinstance(neural_state, (np.ndarray, list)):
                         neural_state = torch.as_tensor(np.array(neural_state), device=self.device, dtype=torch.float32)
                    elif isinstance(neural_state, torch.Tensor):
                         neural_state = neural_state.to(self.device, dtype=torch.float32)
                    
                    # Flatten: (Batch, ...) -> (Batch, Flat)
                    # If No Batch (e.g. Stack, H, W, C), Add Batch then Flatten
                    if neural_state.ndim >= 1:
                         # Heuristic: If first dim is batch size (usually 1 for inference), keep it.
                         # But input might be (4, 210, 160, 3) which is effectively a single sample.
                         # We assume single sample inference here as predict usually processes one.
                         # If batch dimension given (e.g. (1, 4, ...)), we flatten from dim 1.
                         if neural_state.shape[0] == 1 and neural_state.ndim >= 2: # Likely batch=1
                             neural_state = neural_state.view(1, -1)
                         else:
                             # Assume entire input is one sample, add batch dim then flatten
                             neural_state = neural_state.view(1, -1)
                
                else:
                    # Standard CNN or Unused Neural State
                    if isinstance(neural_state, (np.ndarray, list)):
                         neural_state = np.array(neural_state)
                    elif isinstance(neural_state, torch.Tensor):
                         neural_state = neural_state.to(self.device)

            with torch.no_grad():
                # Call act(neural, logic)
                if logic_state.ndim == 2: # (Objects, Features) -> Add Batch
                    logic_state = logic_state.unsqueeze(0)
                
                # Verify Neural Shape for CNN (if not MLP)
                if not self.is_blender_mlp and neural_state is not None and neural_state.ndim == 3: # (C, H, W) -> Add Batch
                    neural_state = neural_state.unsqueeze(0)
                action, _ = self.model.act(neural_state, logic_state)
                # action is tensor, usually single value if batch=1
                action_idx = action.item() if action.numel() == 1 else action[0].item()

            if self.is_pure_logic and self.helper_env:
                 # Logic agents return Predicate Index.
                 # We must map Predicate Index -> Predicate Name -> Action Index
                 if hasattr(self.model, "prednames"):
                      pred_idx = action_idx
                      if 0 <= pred_idx < len(self.model.prednames):
                          pred_name = self.model.prednames[pred_idx]
                          # NudgeEnv.pred2action: Name -> Int
                          if pred_name in self.helper_env.pred2action:
                               action_idx = self.helper_env.pred2action[pred_name]
                          else:
                               # Fallback: maybe just name matching?
                               print(f"[BlendRL] Warning: Predicate {pred_name} not in env pred2action.")
                      else:
                          print(f"[BlendRL] Warning: Predicate Index {pred_idx} out of bounds.")

            return self._map_to_thinkrl(action_idx)

        # 3. CNN Execution
        else: # self.is_cnn
            # neural_state is the input
            obs = neural_state.to(self.device)

            with torch.no_grad():
                logits = self.model(obs)
                action_idx = torch.argmax(logits, dim=1).item()
                
            return self._map_to_thinkrl(action_idx)

    def _map_to_thinkrl(self, action_idx):
        # 1. Map Model Action -> Env Action
        env_action = action_idx
        if self.helper_env:
            # Logic models might use different internal indices (pred2action)
            # But standard PPO often matches Env.
            # Logic agent mode='logic' in NudgeEnv uses map_action.
            try:
                # If helper_env has map_action, use it?
                # But map_action is for 'logic' mode or 'ppo' mode logic.
                # If is_cnn, we likely use 'ppo' mapping (action + 0 or similar).
                pass
            except:
                pass

        # 2. Map Env Action Index -> String
        mapping = {
            JAXAtariAction.NOOP: "NOOP",
            JAXAtariAction.FIRE: "FIRE",
            JAXAtariAction.UP: "UP",
            JAXAtariAction.RIGHT: "RIGHT",
            JAXAtariAction.LEFT: "LEFT",
            JAXAtariAction.DOWN: "DOWN",
            JAXAtariAction.UPRIGHT: "UPRIGHT",
            JAXAtariAction.UPLEFT: "UPLEFT",
            JAXAtariAction.DOWNRIGHT: "DOWNRIGHT",
            JAXAtariAction.DOWNLEFT: "DOWNLEFT",
            JAXAtariAction.UPFIRE: "UPFIRE",
            JAXAtariAction.RIGHTFIRE: "RIGHTFIRE",
            JAXAtariAction.LEFTFIRE: "LEFTFIRE",
            JAXAtariAction.DOWNFIRE: "DOWNFIRE",
            JAXAtariAction.UPRIGHTFIRE: "UPRIGHTFIRE",
            JAXAtariAction.UPLEFTFIRE: "UPLEFTFIRE",
            JAXAtariAction.DOWNRIGHTFIRE: "DOWNRIGHTFIRE",
            JAXAtariAction.DOWNLEFTFIRE: "DOWNLEFTFIRE"
        }
        
        # JAXAtariAction values are usually 0..N matching ALE.
        # But to be safe, we look up by value (IntEnum).
        # We can also just use the name if we trust the loop.
        
        return mapping.get(env_action, "WAIT")
