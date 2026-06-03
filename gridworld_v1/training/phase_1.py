#!/usr/bin/env python
# coding: utf-8

# # Phase 1
# Training the teacher to avoid obstacles and get to the goal efficiently/quickly. I will likely convert this to python so that I can import it into another file where I run phase 1, then phase 2 and 3, rather than having to run each separately, but I'm not sure yet. 

# In[1]:


get_ipython().run_line_magic('cd', '..')
# need to be able to see environment
# this isn't working for some reason


# In[2]:


get_ipython().run_line_magic('pip', 'install torch')


# In[3]:


get_ipython().run_line_magic('pip', 'install tqdm')
get_ipython().run_line_magic('pip', 'install matplotlib')


# In[4]:


import os
# os.chdir("gridworld_v1")


# In[5]:


# imports
from environment.custom_environment import GridWorldBase, TeacherWrapper
from agents.agents import TeacherAgent, ExperienceReplay
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import re


# In[ ]:


# initialization
grid_size = 5
learning_rate = 0.001
initial_epsilon = 1
epsilon_decay = 0.995
final_epsilon = 0.01
num_filters_first_layer = 8
discount_factor = 0.99
episodes = 1000
max_steps = 25 # 25 tiles in whole gridworld
target_update_freq = 500
visibility = None

goal_reward = 10
num_special_regions = 0
special_region_rewards = []

experience_capacity = 2000
batch_size = 128

base_env = GridWorldBase(grid_size, num_special_regions, goal_reward)
env = TeacherWrapper(base_env, max_steps, visibility, special_region_rewards)
agent = TeacherAgent(base_env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, num_filters_first_layer, target_update_freq, discount_factor)

experience_replay = ExperienceReplay(capacity=experience_capacity, batch_size=batch_size)


# In[ ]:


# training
# TODO: switch to epsilon decay per episode
episode_total_rewards = []
losses = []
lengths = []

pbar = tqdm.tqdm(range(episodes), desc="Training")
pbar.set_postfix(epsilon=f"{agent.epsilon:.3f}", steps=0) # I think not having this hear was leading to 2 progress bars (1 stationary)

for episode in pbar:
    episode_losses = [] 
    obs, info = env.reset()
    state = base_env.make_one_agent_grid("teacher")
    episode_reward = 0

    for step in range(max_steps): 
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = base_env.make_one_agent_grid("teacher")

        experience_replay.add_experience(state, action, reward, next_state, terminated)

        if terminated or truncated: 
            lengths.append(step + 1)
            break

        if experience_replay.can_provide_sample(): 
            experiences = experience_replay.sample_batch()
            loss = agent.learn(experiences)
            episode_losses.append(loss)
            pbar.set_postfix(epsilon=f"{agent.epsilon:.3f}", reward=f"{episode_reward:.1f}", steps=step, loss=f"{loss:.3f}")

        episode_reward += reward
        state = next_state

    if episode % 100 == 0:
        pass # in the future I want to record some extra info here or smth

    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
    pbar.set_postfix(epsilon=f"{agent.epsilon:.3f}", reward=f"{episode_reward:.1f}", steps=step + 1, loss=f"{avg_loss:.3f}", refresh=False)

    losses.extend(episode_losses)
    agent.decay_epsilon()

    episode_total_rewards.append(episode_reward)

pbar.close()


# In[ ]:


import sys
sys.stdout = sys.__stdout__

# works no matter what random directory it ends up running from
if os.path.exists("gridworld_v1/models"):
    base_dir = "gridworld_v1/models"
else:
    base_dir = "models/"

dirs = glob.glob(os.path.join(base_dir, "tm_*"))
highest_num = 0
for d in dirs: 
    number = re.search(r'tm_(\d+)$', d)
    if number: 
        highest_num = max(highest_num, int(number.group(1)))

next_num = highest_num + 1
run_dir = os.path.join(base_dir, f"tm_{next_num}")
os.makedirs(run_dir, exist_ok=True)

hyperparameters = {
    "grid_size": grid_size, 
    "learning_rate": learning_rate, 
    "initial_epsilon": initial_epsilon, 
    "epsilon_decay": epsilon_decay, 
    "final_epsilon": final_epsilon, 
    "num_filters_first_layer": num_filters_first_layer, 
    "discount_factor": discount_factor, 
    "episodes": episodes, 
    "max_steps": max_steps, 
    "target_update_freq": target_update_freq, 
    "goal_reward": goal_reward, 
    "num_special_regions": num_special_regions, 
    "special_region_rewards": special_region_rewards, 
    "experience_capacity": experience_capacity, 
    "batch_size": batch_size, 
    "visibility": visibility, 
}

checkpoint = {
    "hyperparameters": hyperparameters,
    "model_state_dict": agent.model.state_dict(),
}

torch.save(checkpoint, f'{run_dir}/model_info.pt')

human_readable = f""
for param in hyperparameters: 
    human_readable += f"{str(param)} = {hyperparameters[param]}\n"

human_readable_time = pbar.format_interval(pbar.format_dict['elapsed'])
human_readable += f"\n\nTraining Time: {human_readable_time}\n"
notes = input("Notes on this training run (consider what won't be recorded, like changes to envs or agents): ")
if len(notes) > 0: 
    human_readable += "Notes: " + notes

with open(f'{run_dir}/human_info.txt', 'w') as f: 
    f.write(human_readable)

print(f"base_dir: {base_dir}")
print(f"run_dir: {run_dir}")


# In[ ]:


import sys
sys.stdout = sys.__stdout__ # sometimes when I run cells out of order (running the bottom one before others, or re-running other cells after running run all) nothing will print, so I have to reset w/ this

# https://gymnasium.farama.org/introduction/train_agent/
def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Smooth over this window
rolling_length = episodes//20
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    episode_total_rewards,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    lengths,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")


# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    losses,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.savefig(f'{run_dir}/plots.png')
print(f"Saved to {run_dir}")
plt.close()


# In[ ]:


# load agent
'''
run_dir = "models/t_m_6"

model_path = os.path.join(run_dir, "model.pt")
checkpoint = torch.load(model_path)

# hyperparameters = checkpoint["hyperparameters"]

agent.model.load_state_dict(checkpoint)
'''


# In[ ]:


# test
print(run_dir)

agent.epsilon = 0

def run_test(): 
    obs, info = env.reset()
    base_env.render()
    total_reward = 0
    state = base_env.make_one_agent_grid("teacher")
    for step in range(max_steps):
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = base_env.make_one_agent_grid("teacher")
        total_reward += reward
        print(f"Step {step} | Action: {action} | Reward: {reward}")
        base_env.render()
        state = next_state
        if terminated or truncated:
            print(f"Done in {step+1} steps | Total reward: {total_reward}")
            break

with open(f'{run_dir}/test_result.txt', 'w') as f:
    sys.stdout = f
    try:
        print("Test 1:\n")
        run_test()
        print("\nTest 2:")
        run_test()
        print("\nTest 3:")
        run_test()
    finally:
        sys.stdout = sys.__stdout__

