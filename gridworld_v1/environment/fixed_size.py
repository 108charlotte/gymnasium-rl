import numpy as np
import gymnasium as gym
import math
import pandas as pd
from collections import defaultdict
from pathfinding.core.grid import Grid
from pathfinding.finder.breadth_first import BreadthFirstFinder

class FixedSizeGridworldBase(gym.Env): 
    def __init__(self, num_types_special_regions: int = 0, goal_reward: int = 10, step_penalty: float = -0.1, world_size: int = 10, static_world = False): 
        self._num_types_special_regions = num_types_special_regions
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.world_size = world_size
        self.static_world = static_world
        self.coords = defaultdict(list) # if world is static, then this will store the agent location, the goal location, and the special regions locations

        self.coords_to_default() # set coordinates to - values to show that they are uninitialized
        num_channels = 1+num_types_special_regions+1 # agent + one per region type + target
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_channels, world_size, world_size), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)
        # observation space defined in wrapper, where visibility known
        self._action_to_direction = {
            0: np.array([0, 1]), # right (col + 1)
            1: np.array([-1, 0]), # up (row - 1)
            2: np.array([0, -1]), # left (col - 1)
            3: np.array([1, 0]), # down (row + 1)
        }
    
    def _init_special_regions(self): 
        special_regions_index_order = self.np_random.permutation(self._num_types_special_regions)
        for index in special_regions_index_order:
            num_to_place = self.np_random.integers(low=0, high=(self.world_size ** 2) // 2) # maybe add a saturation parameter later
            self._special_regions[index] = []
            for _ in range(num_to_place): 
                coords = tuple(self.np_random.integers(0, self.world_size, size=2, dtype=int))
                while np.array_equal(coords, self._target_location) or np.array_equal(coords, self._teacher_agent_location): 
                    coords = tuple(self.np_random.integers(0, self.world_size, size=2, dtype=int))
                self._special_regions[index].append(coords)

    def coords_to_default(self): 
        self._teacher_agent_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)
        self._student_agent_location = np.array([-1, -1])
        self._target_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)
        self._special_regions = [[] for _ in range(self._num_types_special_regions)]
        self._init_special_regions()

        self.coords["teacher_agent_loc"] = [self._teacher_agent_location.copy()]
        self.coords["target_loc"] = [self._target_location.copy()]
        self.coords["special_regions"] = self._special_regions

    def shortest_path_to_goal(self): # must be called after all locations are initialized
        # https://github.com/brean/python-pathfinding/blob/main/docs/01_basic_usage.md
        matrix = np.full(shape=(self.world_size, self.world_size), fill_value=1)
        matrix[self._special_regions] = 0 # indicates obstacles, assumes all special regions have a more negative penalty than step penalty (overly simplistic, but the environment doesn't have access to the rewards, only the teacher wrapper does)
        grid = Grid(matrix=matrix)
        start = grid.node(self._teacher_agent_location[1], self._teacher_agent_location[0])
        end = grid.node(self._target_location[1], self._target_location[0])
        finder = BreadthFirstFinder()
        path, runs = finder.find_path(start, end, grid)
        return len(path)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.static_world:
            self._teacher_agent_location = (self.coords["teacher_agent_loc"][0].copy())
            self._target_location = (self.coords["target_loc"][0].copy())
            self._special_regions = self.coords["special_regions"]
        else:    
            # randomly sets teacher, sets student to same start location as teacher
            self._teacher_agent_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)
            self._student_agent_location = self._teacher_agent_location.copy()

            self._target_location = self._teacher_agent_location.copy()
            while np.array_equal(self._target_location, self._teacher_agent_location):
                self._target_location = self.np_random.integers(0, self.world_size, size=2, dtype=int)

            self._init_special_regions()
        
        return np.array([]), {} # empty info and empty obs
    
    def get_reward_no_shaping(self, agent_loc): 
        return self.goal_reward if np.array_equal(agent_loc, self._target_location) else self.step_penalty
    
    def step_teacher_only(self, action): 
        direction = self._action_to_direction[action]
        # can't use func I wrote to get loc bc I need to update too
        self._teacher_agent_location += direction
        # clip so agent stays within world bounds: 
        self._teacher_agent_location = np.clip(self._teacher_agent_location, 0, self.world_size - 1)
        terminated = np.array_equal(self._teacher_agent_location, self._target_location)
        truncated = False # will be overridden in wrappers, since wrappers keep track of steps for each agent (not part of world environment)
        # unless overridden later by special area rules for teacher, small penalties for all areas except for goal to encourage going to goal
        reward = self.get_reward_no_shaping(self._teacher_agent_location)
        obs = self.make_one_agent_grid("teacher")
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
   
    def render(self, output_file=None): 
        # build grid 2d list
        grid = [["  " for row in range(self.world_size)] for col in range(self.world_size)]

        for i, region in enumerate(self._special_regions): 
            for coords in region: 
                grid[coords[0]][coords[1]] = f"{i} "

        # these override region visibilities
        # make sure student doesn't override teacher
        if not np.array_equal(self._teacher_agent_location, self._student_agent_location): 
            grid[self._teacher_agent_location[0]][self._teacher_agent_location[1]] =  "🟦 " # updated to emoji so more visible in bigger gridworlds
            grid[self._student_agent_location[0]][self._student_agent_location[1]] = "🟨 " # same as above
        else: 
            grid[self._teacher_agent_location[0]][self._teacher_agent_location[1]] = "🟫 " # same as above
        
        # show when goal reached
        if np.array_equal(self._teacher_agent_location, self._target_location): 
            grid[self._target_location[0]][self._target_location[1]] = "🟦🎉"
        elif np.array_equal(self._student_agent_location, self._target_location): 
            grid[self._target_location[0]][self._target_location[1]] = "🟨🎉"
        else: 
            grid[self._target_location[0]][self._target_location[1]] = "🟩 "

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
        self.state_action_pairs = defaultdict(int) # state as key, action as value (int since action is numerically encoded)
    
    def step(self, action): 
        # tobytes bc needs to be hashable to be dict key
        self.state_action_pairs[self.env.make_one_agent_grid("teacher").tobytes()] = action # needs to happen before state is updated; I want the student to learn which actions to take in which situations, not which actions caused which situations
        obs, reward, terminated, _, info = self.env.step_teacher_only(action)
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