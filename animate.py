import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import DQN
from utils import select_action
from config import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
env = gym.make("CartPole-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load("trained_model.pth", map_location=device))
policy_net.eval()

# Capture frames
state = env.reset()
frames = []
done = False
step_count = 0

print('Generating animation...')

while not done and step_count < 600:
    img = env.render()
    if isinstance(img, list) and len(img) > 0:
        img = img[0]
    elif isinstance(img, np.ndarray):
        pass
    else:
        step_count += 1
        continue

    frames.append(img)

    action = select_action(state, epsilon=0.0, policy_net=policy_net, action_dim=action_dim, device=device)
    state, reward, done, _ = env.step(action)
    step_count += 1

env.close()

# Animate
fig = plt.figure(figsize=(6, 4))
img_plot = plt.imshow(frames[0])
plt.axis('off')
plt.title("Trained DQN on CartPole")

def animate_frame(i):
    img_plot.set_data(frames[i])
    return [img_plot]

ani = animation.FuncAnimation(fig, animate_frame, frames=len(frames), interval=40, blit=True)

ani.save("dqn_cartpole.gif", writer="pillow", fps=25)
ani.save("dqn_cartpole.mp4", writer="ffmpeg", fps=25)
plt.show()
