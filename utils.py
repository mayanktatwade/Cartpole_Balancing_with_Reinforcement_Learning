import torch
import random
import numpy as np
from collections import deque
from config import *

REPLAY_MEMORY = deque(maxlen=REPLAY_MEMORY_SIZE)

def select_action(state, epsilon, policy_net, action_dim, device):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state_tensor).argmax().item()

def train_step(policy_net, target_net, optimizer, device):
    if len(REPLAY_MEMORY) < BATCH_SIZE:
        return

    batch = random.sample(REPLAY_MEMORY, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q_values = target_net(next_states).max(1)[0]
    target_q = rewards + GAMMA * next_q_values * (1 - dones)

    loss = torch.nn.MSELoss()(q_values, target_q.detach())

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
