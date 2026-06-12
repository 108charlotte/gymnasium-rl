import numpy as np
import gymnasium as gym
import math
import pandas as pd

class FixedSizeGridworldBase(gym.Env): 
    def __init__(self, num_types_special_regions: int = 0, goal_reward: int = 10, step_penalty: float = -0.1, world_size: int = 10): 
        self._num_types_special_regions = num_types_special_regions
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.world_size = world_size

        self.coords_to_default() # set coordinates to - values to show that they are uninitialized
        num_channels = 1+num_types_special_regions+1+2 # agent + one per region type + target + dx + dy
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_channels, world_size, world_size), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)
        # observation space defined in wrapper, where visibility known
        self._action_to_direction = {
            0: np.array([0, 1]), # right (col + 1)
            1: np.array([-1, 0]), # up (row - 1)
            2: np.array([0, -1]), # left (col - 1)
            3: np.array([1, 0]), # down (row + 1)
        }
    
    def coords_to_default(self): 
        self._teacher_agent_location = np.array([-1, -1], dtype=np.int32)
        self._student_agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._special_regions = [[] for special_region in range(self._num_types_special_regions)]  # each index corresponds to a different type of region, and the values in the list are each tuples of coordinates where the region is
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # randomly sets teacher, sets student to same start location as teacher
        self._teacher_agent_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)
        self._student_agent_location = self._teacher_agent_location.copy()

        self._target_location = self._teacher_agent_location.copy()
        while np.array_equal(self._target_location, self._teacher_agent_location):
            self._target_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)

        special_regions_index_order = self.np_random.permutation(self._num_types_special_regions)
        for index in special_regions_index_order:
            num_to_place = self.np_random.integers(low=0, high=(self.world_size ** 2) // 2)
            self._special_regions[index] = [
                tuple(self.np_random.integers(0, self.world_size, size=2, dtype=int))
                for _ in range(num_to_place)
            ]
        
        return np.array([]), {} # empty info and empty obs
    
    def get_reward_no_shaping(self, agent_loc): 
        return self.goal_reward if np.array_equal(agent_loc, self._target_location) else self.step_penalty
    
    def step_teacher_only(self, action): 
        direction = self._action_to_direction[action]
        # can't use func I wrote to get loc bc I need to update too
        self._teacher_agent_location += direction
        
        terminated = np.array_equal(self._teacher_agent_location, self._target_location)
        truncated = False # will be overridden in wrappers, since wrappers keep track of steps for each agent (not part of world environment)
        # unless overridden later by special area rules for teacher, small penalties for all areas except for goal to encourage going to goal
        reward = self.get_reward_no_shaping(self._teacher_agent_location)
        obs = [] # filled in in wrapper
        # placeholder to not crash
        infos = {}

        return obs, reward, terminated, truncated, infos

    def get_curr_special_region(self, agent): 
        agent_location = self._student_agent_location
        if agent == "teacher": 
            agent_location = self._teacher_agent_location
        for i, region in enumerate(self._special_regions): 
            for coords in region: 
                if coords[0] == agent_location[0] and coords[1] == agent_location[1]: 
                    return i # regions identified by where they appear in the list
        return None
    
    def get_reward_student(self, student_loc, student_action, teacher_action): 
        base_reward = self.get_reward_no_shaping(student_loc)
        action_diff = abs(teacher_action - student_action)
        max_alignment_diff = self.action_space.n // 2
        teacher_alignment_reward = self.teacher_student_multiplier * (1-2*action_diff/max_alignment_diff)
        return teacher_alignment_reward + base_reward

    def step_student(self, action, teacher_action): # teacher action included for calculating reward
        direction = self._action_to_direction[action]
        self._student_agent_location += direction

        terminated = np.array_equal(self._student_agent_location, self._target_location)
        truncated = False

        reward = self.get_reward_student(self._student_agent_location, action, teacher_action)
        obs = []
        infos = {}

        return obs, reward, terminated, truncated, infos
    
    def is_in_visibility_region(self, agent_location, coords, visibility_range = None): 
        if visibility_range is None: 
            return True
        x_y_visibility_range = [agent_location[0] - visibility_range, agent_location[0] + visibility_range], [agent_location[1] - visibility_range, agent_location[1] + visibility_range]
        return x_y_visibility_range[0][0] <= coords[0] <= x_y_visibility_range[0][1] and x_y_visibility_range[1][0] <= coords[1] <= x_y_visibility_range[1][1]

    def get_agent_loc_for_name(self, name): 
        agent_location = self._student_agent_location
        if name == "teacher": 
            agent_location = self._teacher_agent_location
        return agent_location

    # sped up for quality of life improvement
    def get_regions_in_visibility(self, agent, visibility_range): 
        agent_location = self.get_agent_loc_for_name(agent)
        
        visible = []
        for region in self._special_regions: 
            if len(region) == 0: # so program doesn't crash doing math on an empty array 
                visible.append(np.array([]))
                continue
            all_region_coords = np.array(region)
            dists = np.abs(all_region_coords - agent_location)
            in_visibility = np.all(dists <= visibility_range, axis=1) # axis=1 checks row-wise, I'm not sure why it wouldn't be axis=0 but this appears to work and that didn't work so I'm using this
            visible.append(all_region_coords[in_visibility])
        return visible
    
    def make_one_agent_grid(self, agent_name): 
        agent_loc = self.get_agent_loc_for_name(agent_name)

        channels = []
        # this purposefully only shows the agent the function was called for, to avoid noise
        agent_grid = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        agent_grid[agent_loc[0]][agent_loc[1]] = 1
        channels.append(agent_grid)

        target_grid = np.zeros((self.world_size, self.world_size), dtype=np.float32)
        target_grid[self._target_location[0]][self._target_location[1]] = 1
        channels.append(target_grid)

        for special_region in self._special_regions: 
            region_grid = np.zeros((self.world_size, self.world_size), dtype=np.float32)
            for coords in special_region: 
                region_grid[coords[0]][coords[1]] = 1
            channels.append(region_grid)
        
        return np.array(channels)
   
    def make_empty_grid(self, size): # this isn't worth a function really, it used to have a check for None to set to full grid but now there isn't really a full grid
        grid = np.zeros((size, size), dtype=np.float32)
        # grid = [[0 for row in range(self.size)] for col in range(self.size)]
        return grid
    
    def render(self, output_file=None): 
        all_coords = np.array([self._teacher_agent_location, self._student_agent_location, self._target_location] + [np.array(col) for row in self._special_regions for col in row])
        row_min,col_min = all_coords.min(axis=0)
        row_max,col_max = all_coords.max(axis=0)
        # build grid 2d list
        grid = [["  " for row in range(col_max - col_min + 1)] for col in range(row_max - row_min + 1)] # empty but correct size

        for i, region in enumerate(self._special_regions): 
            for coords in region: 
                grid[coords[0]-row_min][coords[1]-col_min] = f"{i} "

        # these override region visibilities
        # make sure student doesn't override teacher
        if not np.array_equal(self._teacher_agent_location, self._student_agent_location): 
            grid[self._teacher_agent_location[0]-row_min][self._teacher_agent_location[1]-col_min] =  "🟦 " # updated to emoji so more visible in bigger gridworlds
            grid[self._student_agent_location[0]-row_min][self._student_agent_location[1]-col_min] = "🟨 " # same as above
        else: 
            grid[self._teacher_agent_location[0]-row_min][self._teacher_agent_location[1]-col_min] = "🟫 " # same as above
        
        # show when goal reached
        if np.array_equal(self._teacher_agent_location, self._target_location): 
            grid[self._target_location[0]-row_min][self._target_location[1]-col_min] = "🟦🎉"
        elif np.array_equal(self._student_agent_location, self._target_location): 
            grid[self._target_location[0]-row_min][self._target_location[1]-col_min] = "🟨🎉"
        else: 
            grid[self._target_location[0]-row_min][self._target_location[1]-col_min] = "🟩 "

        for row in grid:
            print(row) if output_file is None else print(row, file = output_file)
        print('') if output_file is None else print('', file=output_file) # To add some space between renders for each step



class TeacherWrapper(gym.Wrapper): # keeps track of steps for truncation, and updates reward
    def __init__(self, env, max_steps: int = 50, special_region_rewards: list[float] = []): # if visibility not passed just sees whole world, defaults to ignoring regions
        super().__init__(env)
        self.env = env
        
        self.max_steps = max_steps if max_steps is not None else float('inf') # if none, then no max
        self.num_steps = 0 # init value
        
        # if too short, pad with zeros (no effect) to avoid indexing error in step; otherwise, keep as-is
        self.special_region_rewards =  list(special_region_rewards) # should create a copy I can work with so I don't mutate the parameter list
        if self.env._num_types_special_regions - len(special_region_rewards) > 0: 
            self.special_region_rewards += [0] * (self.env._num_types_special_regions - len(special_region_rewards))
        
        self.most_recent_action = -1 # default un-initialized value, should always be updated before student reads but if not there was a bug somewhere
        # actually instead, I might need to store what the teacher did in each state, since if the student deviates from the teacher for one action then everything is kind of lost
    
    def step(self, action): 
        self.most_recent_action = action
        obs, reward, terminated, _, info = self.env.step_teacher_only(action)
        obs = self.make_one_agent_grid("teacher") # will just exclude student, since teacher doesn't need to see student (and student doesn't need to see teacher, only needs to receive its actions)
        self.num_steps += 1
        truncated = self.max_steps <= self.num_steps
        # update reward based on special_region_rewards values
        curr_region = self.env.get_curr_special_region("teacher")
        if curr_region is not None: # in some special region
            reward = self.special_region_rewards[curr_region]

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None): 
        obs, info = self.env.reset(seed=seed, options=options) # trigger base env reset (makes sure I have valid coords for teacher agent start)
        self.num_steps = 0
        return obs, info

''' may implement this later, since the teacherwrapper and studentwrapper both count steps
class AgentWrapper(gym.Wrapper): 
    def __init__(self, env, agent_name, max_steps:int = 50): 
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps if max_steps is not None else float('inf')
        self.num_steps = 0

    def step(self, action, teacher_action=None):
'''
''' temporarily (possibly) commented bc I think this might be a supervised learning problem
class StudentWrapper(gym.Wrapper): 
    def __init__(self, env, max_steps: int = 50): # needs teacher agent to get its most recent ac
        super().__init__(env)
        self.env = env
        self.max_steps = max_steps if max_steps is not None else float('inf') # if none, then no max
        self.num_steps = 0 # init value
    
    def step(self, action, teacher_action): 
        obs, reward, terminated, _, info = self.env.step_student(action, teacher_action)
        obs = self.env.make_one_agent_grid("student")
        self.num_steps += 1
        truncated = self.max_steps <= self.num_steps
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None): 
        obs, info = self.env.reset(seed=seed, options=options) # trigger base env reset (makes sure I have valid coords for teacher agent start)
        self.num_steps = 0
        return obs, info
'''