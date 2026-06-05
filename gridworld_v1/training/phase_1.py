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
import random
from collections import defaultdict


# In[ ]:


# initialization
grid_size = 6
learning_rate = 0.005
initial_epsilon = 1
epsilon_decay = 0.998
final_epsilon = 0.05
discount_factor = 0.99

episodes = 8000
max_steps = 25 # 25 tiles in whole gridworld
target_update_freq = 200 # for target CNN, in steps
visibility = 7

goal_reward = 10
step_penalty = -0.1

num_special_regions = 1
special_region_rewards = [-2.0]

experience_capacity = 8000
batch_size = 64
seq_len = 8 # should be enough to discourage going in a circle (at least a tight one)

grid_sizes = [4, 8, 12, 16] # training on different sizes for better generalizability

num_filters_first_layer = 8
final_conv_filters = num_filters_first_layer * 2
target_spatial_size = 3 # needs to evenly divide visibility * 2 + 1

base_env = GridWorldBase(grid_size, num_special_regions, goal_reward, step_penalty)
env = TeacherWrapper(base_env, visibility, max_steps, special_region_rewards)
agent = TeacherAgent(num_special_regions, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, num_filters_first_layer, final_conv_filters, target_spatial_size, target_update_freq, discount_factor)

experience_replays = {
    g: ExperienceReplay(capacity=experience_capacity, batch_size=batch_size, seq_len=seq_len) for g in grid_sizes
}


# In[ ]:


# training
# TODO: switch to epsilon decay per episode
episode_total_rewards = []
losses = []
lengths = defaultdict(list)

pbar = tqdm.tqdm(range(episodes), desc="Training")

for episode in pbar:
    grid_size = random.choice(grid_sizes)  # each episode
    max_steps = grid_size * 4 # scaling up
    base_env = GridWorldBase(grid_size, num_special_regions, goal_reward, step_penalty)
    env = TeacherWrapper(base_env, visibility, max_steps, special_region_rewards)
    replay = experience_replays[grid_size]

    agent.reset_hidden_state() # clears lstm (long short term memory)

    episode_losses = [] 
    obs, info = env.reset()
    state = base_env.make_one_agent_grid_relative("teacher", visibility)
    episode_reward = 0

    for step in range(max_steps): 
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = base_env.make_one_agent_grid_relative("teacher", visibility)

        replay.add_experience(state, action, reward, next_state, terminated or truncated) # used to just be terminated, but added truncated to stop bleeding btw episodes which happens when done isn't triggered

        episode_reward += reward
        state = next_state

        if replay.can_provide_sample(): 
            experiences = replay.sample_batch()
            loss = agent.learn(experiences)
            episode_losses.append(loss)
            pbar.set_postfix(epsilon=f"{agent.epsilon:.3f}", reward=f"{episode_reward:.1f}", steps=step, loss=f"{loss:.3f}")

        if terminated or truncated: 
            lengths[grid_size].append(step + 1) # this is very variable depending on grid size, so I'm storing with the grid size as a key
            break

    if episode % 100 == 0:
        pass # in the future I want to record some extra info here or smth

    avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
    pbar.set_postfix(epsilon=f"{agent.epsilon:.3f}", reward=f"{episode_reward:.1f}", steps=step + 1, loss=f"{avg_loss:.3f}", refresh=False)

    losses.extend(episode_losses)
    agent.decay_epsilon()

    episode_total_rewards.append(episode_reward)

pbar.close()


# In[8]:


import sys
sys.stdout = sys.__stdout__

# works no matter what random directory it ends up running from
if os.path.exists("gridworld_v1/all_models"):
    base_dir = "gridworld_v1/all_models"
else:
    base_dir = "all_models/"

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
    "discount_factor": discount_factor, 
    "episodes": episodes, 
    "max_steps": max_steps, 
    "target_update_freq": target_update_freq, 
    "visibility": visibility, 
    "goal_reward": goal_reward, 
    "num_special_regions": num_special_regions, 
    "special_region_rewards": special_region_rewards, 
    "experience_capacity": experience_capacity, 
    "batch_size": batch_size, 
    "grid_sizes": grid_sizes, 
    "num_filters_first_layer": num_filters_first_layer, 
    "final_conv_filters": final_conv_filters, 
    "target_spatial_size": target_spatial_size, 
    "step_penalty": step_penalty, 
    "seq_len": seq_len, 
}

checkpoint = {
    "hyperparameters": hyperparameters,
    "model_state_dict": agent.model.state_dict(),
}

torch.save(checkpoint, f'{run_dir}/model_info.pt')

print(f"base_dir: {base_dir}")
print(f"run_dir: {run_dir}")


# In[9]:


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
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

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


# Training error (how much we're still learning)
axs[1].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    losses,
    rolling_length,
    "same"
)
axs[1].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[1].set_ylabel("Temporal Difference Error")
axs[1].set_xlabel("Step")

plt.tight_layout()
plt.savefig(f'{run_dir}/plots.png')
print(f"Saved to {run_dir}")
plt.close()


# In[10]:


import sys
sys.stdout = sys.__stdout__ # sometimes when I run cells out of order (running the bottom one before others, or re-running other cells after running run all) nothing will print, so I have to reset w/ this

rolling_length = episodes // (len(grid_sizes) * 10) # bc split across diff grid sizes
fig, axs = plt.subplots(ncols=len(grid_sizes), figsize=(12, 5))

for i, key in enumerate(grid_sizes): 
    # Episode lengths (num steps, to reach goal or to truncation)
    axs[i].set_title(f"Episode lengths: {key}")
    if grid_size not in lengths or len(lengths[key]) < rolling_length: 
        continue # not enough data for grid size
    length_moving_average = get_moving_avgs(
        lengths[key],
        rolling_length,
        "valid"
    )
    axs[i].plot(range(len(length_moving_average)), length_moving_average)
    axs[i].set_ylabel("Average Episode Length")
    axs[i].set_xlabel("Episode")


plt.tight_layout()
plt.savefig(f'{run_dir}/length_plots.png')
print(f"Saved to {run_dir}")
plt.close()


# In[11]:


# load agent
'''
run_dir = "all_models/tm_6"

model_path = os.path.join(run_dir, "model.pt")
checkpoint = torch.load(model_path)

# hyperparameters = checkpoint["hyperparameters"]

agent.model.load_state_dict(checkpoint)
'''


# In[12]:


# test
print(run_dir)

agent.epsilon = 0

def run_test(world_size): 
    base_env = GridWorldBase(world_size, num_special_regions) # same as before, but allows different grid sizes
    env = TeacherWrapper(base_env, visibility, max_steps, special_region_rewards)

    obs, info = env.reset()
    base_env.render()
    total_reward = 0
    state = base_env.make_one_agent_grid_relative("teacher", visibility)

    went_through_special_region = False # default

    for step in range(max_steps):
        action = agent.get_action(state)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = base_env.make_one_agent_grid_relative("teacher", visibility)
        # very patchwork-y solution, not good in long term but good enough for now
        if reward in special_region_rewards: 
            went_through_special_region = True
        total_reward += reward
        print(f"Step {step} | Action: {action} | Reward: {reward}")
        base_env.render()
        state = next_state
        if terminated or truncated:
            print(f"Done in {step+1} steps | Total reward: {total_reward}")
            print(f"Went through special region" if went_through_special_region else "")
            break

with open(f'{run_dir}/test_result.txt', 'w') as f:
    sys.stdout = f
    try:
        print("Same size tests: ")
        print("Test 1:\n")
        run_test(grid_size)
        print("\nTest 2:\n")
        run_test(grid_size)
        print("\nTest 3:\n")
        run_test(grid_size)

        print("\n\n2x size tests: ") # tests generalizability to different gridsizes
        print("Test 1:\n")
        run_test(grid_size*2)
        print("\nTest 2:\n")
        run_test(grid_size*2)
        print("\nTest 3:\n")
        run_test(grid_size*2)
    finally:
        sys.stdout = sys.__stdout__


# In[13]:


# this runs at the end so that I can see graphs/results before writing my note
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

