import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v3") # how the environment should be visualized, none is fastest for training and is default, but there's also rgb array (see here: https://gymnasium.farama.org/api/env/#gymnasium.Env.render)

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

# wrappers make environment more modular by separating the environment itself from reward, observations (ex. walking through world is main environment, but need an image that eyes would see), action (interprets agent action and passes to environment; ex. if agent chose action 1, will convert to what that actually means), and general (allows you to manipulate episode flow by resetting environment automatically or enforcing max step limits)
# for reference, there are some useful ones listed here: https://gymnasium.farama.org/introduction/basic_usage/

wrapped_env = FlattenObservation(env)


# always reset environment after making and before making actions
observation, info = wrapped_env.env.reset(seed=42) # first observation + additional info (starting new episode) - if you used the same seed every time you started a new episode, you're limiting the agent's ability to generalize to new starting conditions 

episode_over = False
total_reward = 0

while not episode_over: 
    action = wrapped_env.action_space.sample() # action space: what your agent can do

    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    # observation: what the agent sees after taking the action
    # reward: immediate feedback for action (what to do if there is none?)
    # terminated: whether episode ended naturally (car crashed, hand finished)
    # truncated: whether the episode was cut short (can have continual RL where it doesn't have to stop, I want to look into this more)
    # info: additional debugging info (can usually be ignored, so often _)


    '''
    - observation space: what your agent can see - both action space and observation space are instances of Space python class
    - types of spaces in gymnasium (https://gymnasium.farama.org/introduction/basic_usage/): 
        - Box (continuous control or image pixels, bounded space w/ upper and lower limits of an n-dimensional shape)
        - Discrete
        - Multi-Binary (like on-off switches)
        - Multi-Discrete (different numbers, like a discrete value dial)
        - Text
        - Dict (dictionary of simpler spaces, like GridWorld)
        - Tuple (tuple of simple spaces), Graph (mathematical graph/network)
        - Sequence (variable length of simpler space elements)
    - terminated means environment ended, if timesteps are up then truncated; then restart with env.reset() to begin a new episode
    '''
    
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
