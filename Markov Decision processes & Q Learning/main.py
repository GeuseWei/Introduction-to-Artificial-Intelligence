"""
Solution of Open AI gym environment "Cartpole-v0" (https://gym.openai.com/envs/CartPole-v0) using DQN and Pytorch.
This is a modified version of Pytorch DQN tutorial from https://github.com/mahakal001/reinforcement-learning/tree/master/cartpole-dqn
"""

from dqn_agent import DQN_Agent
import gym
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np



# Install the following:
# Python (3.6 recommended, but should work with later versions as well)
# pip install tqdm
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch (To install pytorch read: https://pytorch.org/)
# pip install 'gym==0.10.11'
# pip install matplotlib


env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
batch_size  = 16
episodes = 20000
epsilon = 1

# sync_freq values
sync_freqs = [1, 10, 30, 100]
runs = [123, 456, 789]
results = []

for sync_freq in sync_freqs:
    run_rewards = []
    for run in runs:
        agent = DQN_Agent(seed=run, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=sync_freq, exp_replay_size=exp_replay_size)
        reward_list = []

        # initialize experience replay
        index = 0
        for i in range(exp_replay_size):
            obs = env.reset()
            done = False
            while not done:
                A = agent.get_action(obs, env.action_space.n, epsilon=1)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])
                obs = obs_next
                index += 1
                if index > exp_replay_size:
                    break

        # main training loop
        index = 128
        for i in tqdm(range(episodes)):
            obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
            while not done:
                ep_len += 1
                A = agent.get_action(obs, env.action_space.n, epsilon)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])

                obs = obs_next
                rew += reward
                index += 1

                if index > 128:
                    index = 0
                    for j in range(4):
                        loss = agent.train(batch_size)
                        losses += loss
            if epsilon > 0.05:
                epsilon -= (1 / 5000)

            reward_list.append(rew)

        run_rewards.append(reward_list)
    results.append(run_rewards)

# Plotting
plt.figure()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Running Average Reward')
for i, sync_freq in enumerate(sync_freqs):
    rewards = np.mean(results[i], axis=0)
    moving_averages= []
    for i in range(len(rewards) - 100 + 1):
        this_window = rewards[i : i + 100]
        window_average = sum(this_window) / 100
        moving_averages.append(window_average)
    plt.plot(moving_averages, label=f"Sync Freq = {sync_freq}")

plt.legend()
plt.savefig('./cartpole.png')
print("Saving trained model")
agent.save_trained_model("cartpole-dqn.pth")


env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
target_update = 10
episodes = 20000
epsilon = 1

# batch_size values
batch_sizes = [1, 16, 30, 200]
runs = 3
results = []

for batch_size in batch_sizes:
    run_rewards = []
    for run in range(runs):
        agent = DQN_Agent(seed=run, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=target_update, exp_replay_size=exp_replay_size)
        reward_list = []

        # initialize experience replay
        index = 0
        for i in range(exp_replay_size):
            obs = env.reset()
            done = False
            while not done:
                A = agent.get_action(obs, env.action_space.n, epsilon=1)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])
                obs = obs_next
                index += 1
                if index > exp_replay_size:
                    break

        # main training loop
        index = 128
        for i in tqdm(range(episodes)):
            obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
            while not done:
                ep_len += 1
                A = agent.get_action(obs, env.action_space.n, epsilon)
                obs_next, reward, done, _ = env.step(A.item())
                agent.collect_experience([obs, A.item(), reward, obs_next])

                obs = obs_next
                rew += reward
                index += 1

                if index > 128:
                    index = 0
                    for j in range(4):
                        loss = agent.train(batch_size)
                        losses += loss
            if epsilon > 0.05:
                epsilon -= (1 / 5000)

            reward_list.append(rew)

        run_rewards.append(reward_list)
    results.append(run_rewards)

# Plotting
plt.figure()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Running Average Reward')
for i, batch_size in enumerate(batch_sizes):
    rewards = np.mean(results[i], axis=0)
    moving_averages= []
    for i in range(len(rewards) - 100 + 1):
        this_window = rewards[i : i + 100]
        window_average = sum(this_window) / 100
        moving_averages.append(window_average)
    plt.plot(moving_averages, label=f"Batch Size = {batch_size}")

plt.legend()
plt.savefig('./cartpole.png')
print("Saving trained model")
agent.save_trained_model("cartpole-dqn.pth")