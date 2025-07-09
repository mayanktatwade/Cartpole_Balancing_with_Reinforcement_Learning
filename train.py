from model import DQN
from utils import select_action, train_step, REPLAY_MEMORY
from config import *

import gym
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize networks
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

epsilon = INITIAL_EPSILON
reward_history = []

print('Start training...')
for ep in range(EPISODES):
    state = env.reset()
    cart_pos = np.random.uniform(-0.5, 0.5)
    pole_angle = np.random.uniform(-0.2, 0.2)
    env.unwrapped.state = np.array([cart_pos, 0.0, pole_angle, 0.0])
    state = env.unwrapped.state

    total_reward = 0
    done = False
    step_count = 0

    while not done:
        action = select_action(state, epsilon, policy_net, action_dim, device)
        next_state, reward, done, _ = env.step(action)

        # Reward shaping
        pos_penalty = -0.5 * abs(next_state[0]) / env.x_threshold
        angle_penalty = -2.0 * abs(next_state[2]) / env.theta_threshold_radians
        vel_penalty = -0.1 * abs(next_state[1])
        ang_vel_penalty = -0.1 * abs(next_state[3])
        alive_bonus = 0.5
        end_penalty = -10.0 if done and step_count < 195 else 0.0

        shaped_reward = alive_bonus + pos_penalty + angle_penalty + vel_penalty + ang_vel_penalty + end_penalty

        REPLAY_MEMORY.append((state, action, shaped_reward, next_state, float(done)))
        state = next_state
        total_reward += shaped_reward
        step_count += 1

        train_step(policy_net, target_net, optimizer, device)

    reward_history.append(total_reward)

    if ep % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

env.close()
print('Training complete.')

# Plot
plt.figure(figsize=(8, 4))
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Shaped Total Reward")
plt.title("Improved DQN on CartPole")
plt.grid(True)
plt.tight_layout()
plt.savefig("docs/reward_plot.png")
plt.show()
