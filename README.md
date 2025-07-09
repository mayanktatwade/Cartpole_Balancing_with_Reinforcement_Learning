# Cartpole_Balancing_with_Reinforcement_Learning
This project implements an enhanced Deep Q-Network (DQN) using PyTorch to solve the CartPole-v1 environment from OpenAI Gym. The model incorporates several improvements over the vanilla DQN setup, including:

- **Reward Shaping**
- **Gradient Clipping**
- **Layer Normalization**
- **Target Network Updates**
- **Animated Visualization of Agent's Performance**


 Project Structure

```
rl-dqn-cartpole/
│
├── train.py         # Main training script
├── model.py         # DQN model definition
├── utils.py         # Action selection, replay memory
├── animate.py       # Generate GIF/MP4 of trained agent
├── config.py        # Hyperparameters
├── requirements.txt # Python dependencies
└── README.md        # You're here!
```


## 📊 Training Graph

The agent is trained for 1200 episodes with shaped rewards. The graph below shows the shaped total reward per episode.
![reward_plot](https://github.com/user-attachments/assets/f5c38f9e-03b0-4ee6-b348-028f089be357)

## 🧠 How It Works

1. **Environment**: Uses `CartPole-v1` from OpenAI Gym.
2. **Network**: 3-layer fully connected neural network with `LayerNorm` and `ReLU`.
3. **Exploration**: Epsilon-greedy strategy with decay.
4. **Experience Replay**: Random sampling from a deque buffer.
5. **Training**:
    - Loss: MSE between predicted and target Q-values
    - Optimizer: Adam
    - Gradient Clipping to stabilize updates
6. **Evaluation**: The trained agent is evaluated and animated using `matplotlib`.

## 📦 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Requirements include:**

- gym[classic-control]
- pygame
- torch
- matplotlib
- numpy
- ffmpeg-python
- pillow


## 🏁 Run Training

Start training the agent:

```bash
python train.py
```

To save and visualize the agent’s performance:

```bash
python animate.py
```


## 📽 Output Example

Saved outputs:

- `dqn_cartpole_colab.gif`
- `dqn_cartpole_colab.mp4`
- `output: `
![dqn_cartpole_colab](https://github.com/user-attachments/assets/c8d9f6c8-4ddf-4a8e-8190-e21417241ea3)

## 🚀


## 📬 Contact

For questions or suggestions, feel free to raise an issue or connect via GitHub.

⭐ Star the repo if you like it!

