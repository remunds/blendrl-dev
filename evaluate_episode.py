import sys
import os
import torch
import numpy as np
from pathlib import Path
import re
import cv2

# Add workspace to sys.path
sys.path.append(os.getcwd())

# Import Adapter
try:
    from blendrl_rl_adapter import BlendrlRLAgent
except ImportError:
    print("Error: Could not import BlendrlRLAgent. Make sure blendrl_rl_adapter.py is in the current directory.")
    sys.exit(1)

# Import NudgeEnv and JAXAtari for Action Enum
try:
    from nudge.env import NudgeBaseEnv
    from jaxatari.environment import JAXAtariAction
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def get_latest_checkpoint(run_dir):
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        # Maybe the run_dir is the parent?
        # Check if run_dir itself has .pth files?
        files = list(Path(run_dir).glob("*.pth"))
        if files:
            # Sort by modification time if no step info? or use step regex
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            return str(latest_file)
        raise FileNotFoundError(f"No checkpoints folder found in {run_dir}")
    
    files = list(ckpt_dir.glob("step_*.pth"))
    if not files:
        raise FileNotFoundError("No checkpoint files found in checkpoints/.")
    
    # Sort by step number
    # step_X.pth
    def extract_step(p):
        match = re.search(r"step_(\d+).pth", p.name)
        return int(match.group(1)) if match else -1
    
    latest_file = max(files, key=extract_step)
    return str(latest_file)

def main():
    # run_dir = "out/runs/kangaroo_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_161001"
    run_dir = "out/runs/seaquest_jax_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_512_steps_128__0_20260119_171753"
    
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return

    print(f"Searching for latest checkpoint in: {run_dir}")
    try:
        checkpoint_path = get_latest_checkpoint(run_dir)
        print(f"Found checkpoint: {checkpoint_path}")
    except Exception as e:
        print(e)
        return

    env_name = "seaquest_jax"
    print(f"Creating Environment: {env_name}")
    try:
        # mode='eval' is critical as per adapter logic (loads helper env with eval)
        # Here we use it as the main env
        env = NudgeBaseEnv.from_name(env_name, mode='eval', episodic_life=False)
    except Exception as e:
        print(f"Failed to create env: {e}")
        return
    
    print("Initializing Agent...")
    try:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu" # Force CPU to avoid OOM
        # The adapter loads its own internal helper env, but expects 'env' arg too.
        # We pass our env, though it might not be used heavily by the adapter itself 
        # (adapter uses it mostly for reference, except internal logic helpers).
        agent = BlendrlRLAgent(env, checkpoint_path, env_name=env_name, device=device)
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return
    
    print("Starting Episode...")
    try:
        obs = env.reset()
        # Ensure obs is tuple (logic, neural)
        if not isinstance(obs, tuple):
             print(f"Warning: env.reset() returned {type(obs)}, expected tuple (logic, neural).")
    except Exception as e:
        print(f"Error during reset: {e}")
        return
    
    done = False
    total_reward = 0.0
    steps = 0
    
    print(f"Agent Type: PureLogic={getattr(agent, 'is_pure_logic', False)}, CNN={agent.is_cnn}, MLP={agent.is_blender_mlp}")

    while not done:
        try:
            # Predict
            action_str = agent.predict(obs)

            # Map String -> Int for NudgeEnv
            # NudgeEnv/kangaroo_jax likely expects an integer index.
            # We use JAXAtariAction to reverse the mapping.
            try:
                # JAXAtariAction members are strings? No, IntEnum usually.
                # The adapter used `mapping.get(env_action, "WAIT")` where env_action was int.
                # So the values in mapping were strings "LEFT", "RIGHT".
                # We need to find the Key (Int) for the Value (String).
                
                # Reconstruct mapping (same as adapter)
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
                
                # Check for reversed
                action_int = None
                for k, v in mapping.items():
                    if v == action_str:
                        # JAXAtariAction members are ints in this env
                        action_int = k 
                        break
                
                if action_int is None:
                    # Fallback or error
                    print(f"Warning: Could not map action string '{action_str}' back to int. Using NOOP (0).")
                    action_int = 0

            except Exception as e:
                print(f"Mapping error: {e}")
                action_int = 0

            # Step
            step_result = env.step(action_int)
            # print(f"DEBUG: step_result len={len(step_result)}")
            
            # Unpack step result
            if len(step_result) == 3:
                obs, reward, done = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                # Gymnasium: obs, reward, terminated, truncated, info
                obs, reward, term, trunc, info = step_result
                done = term or trunc
            else:
                 raise ValueError(f"Unexpected step result length: {len(step_result)}")
            
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"Step {steps}, Reward so far: {total_reward}, Last Action: {action_str}")
                
            if steps > 2000: # Safety break
                print("Breaking after 2000 steps (timeout).")
                break
        
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"Error in stepping: {e}")
            import traceback
            traceback.print_exc()
            break
            
    print(f"Episode Finished. Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
