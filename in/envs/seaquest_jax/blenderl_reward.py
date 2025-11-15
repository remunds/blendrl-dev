import numpy as np


# def reward_function(self) -> float:
#     for obj in self.objects:
#         if 'player' in str(obj).lower():
#             player = obj
#             break

#     # BUG ↓ with multi envs, rewards collected repeatedlydd
#     if self.org_reward == 1.0 and player.y != 45:
#         # e.g. eliminate a shark
#         reward = 0.5
#     elif self.org_reward == 1.0 and player.y == 45:
#         # when rescuesd 6 divers
#         reward = 1.0
#     else:
#         reward = 0.0
#     return reward

# def reward_function(prev_state, state) -> float:
#     # BUG ↓ with multi envs, rewards collected repeatedlydd
#     org_reward = state.score - prev_state.score
#     if org_reward > 0.0 and state.player_y not in [44, 45, 46, 47]:
#         # e.g. eliminate a shark
#         reward = 0.5
#     if org_reward > 0.0 and state.player_y in [44, 45, 46, 47]:
#         # when rescuesd 6 divers
#         reward = 1.0
#     else:
#         reward = 0.0
#     return reward