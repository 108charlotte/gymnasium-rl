import numpy as np
import gymnasium as gym
from copy import copy
import functools

class GridWorldBase(gym.Env): 
    def __init__(self, size=10, num_types_special_regions=0): 
        self.size = size
        self._num_types_special_regions = num_types_special_regions
        self.coords_to_default()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({ # leaving special_regions out of this, will be dealt with in wrappers seperately
            "teacher_agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int32), 
            "student_agent": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int32), 
            "target": gym.spaces.Box(0, size-1, shape=(2,), dtype=np.int32)
        })
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
        self._special_regions = np.array([[()] for special_region in range(self._num_types_special_regions)], dtype=object)  # each index corresponds to a different type of region, and the values in the list are each tuples of coordinates where the region is
    
    def _get_obs(self): 
        return {
            "teacher_agent": self._teacher_agent_location.copy(),
            "student_agent": self._student_agent_location.copy(),
            "target": self._target_location.copy(),
        }

    # copies so that I don't send the np arrays themselves, don't want them to get manipulated at any point by anyone other than this class
    def get_full_world_state(self): 
        obs = self._get_obs()
        obs["special_regions"] = self._special_regions.copy()
        return obs

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        # randomly sets teacher, sets student to same start location as teacher
        self._teacher_agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._student_agent_location = self._teacher_agent_location.copy()

        self._target_location = self._teacher_agent_location.copy()
        while np.array_equal(self._target_location, self._teacher_agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        special_regions_index_order = self.np_random.permutation(self._num_types_special_regions)
        for index in special_regions_index_order:
            num_to_place = self.np_random.integers(low=0, high=(self.size ** 2) // 2)
            self._special_regions[index] = [
                tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
                for _ in range(num_to_place)
            ]
        
        return self._get_obs(), {} # empty info
    
    def step_location(self, action, agent:str = "teacher"): 
        direction = self._action_to_direction[action]

        if agent == "teacher": 
            return np.clip(self._teacher_agent_location + direction, 0, self.size-1)
        else: 
            return np.clip(self._student_agent_location + direction, 0, self.size-1)
    
    def step_one_agent(self, action, agent:str = "teacher"): 
        if agent == "teacher": 
            self._teacher_agent_location = self.step_location(action)
        else: 
            self._student_agent_location = self.step_location(action)
        
        terminations = {"teacher": np.array_equal(self._teacher_agent_location, self._target_location), "student": np.array_equal(self._student_agent_location, self._target_location)}
        truncated = False
        rewards = {"teacher": 1 if terminations["teacher"] else 0, "student": 1 if terminations["student"] else 0}
        full_observations = self.get_full_world_state()
        # check for truncation and terminations
        infos = {"teacher": {}, "student": {}}

        return full_observations, rewards, terminations, truncated, infos


    def get_curr_special_region(self, agent): 
        agent_location = self._student_agent_location
        if agent == "teacher": 
            agent_location = self._teacher_agent_location
        for i, region in enumerate(self._special_regions): 
            for coords in region: 
                if coords[0] == agent_location[0] and coords[1] == agent_location[1]: 
                    return i # regions identified by where they appear in the list
        
        return None

    def get_regions_in_visibility(self, agent, visibility_range): 
        agent_location = self._student_agent_location
        if agent == "teacher": 
            agent_location = self._teacher_agent_location
        x_y_visibility_range = [agent_location[0] - visibility_range, agent_location[0] + visibility_range], [agent_location[1] - visibility_range, agent_location[1] + visibility_range]
        visible = []
        for region in self._special_regions: 
            to_append = [coords for coords in region
                         if x_y_visibility_range[0][0] <= coords[0] < x_y_visibility_range[0][1]
                         and x_y_visibility_range[1][0] <= coords[1] < x_y_visibility_range[1][1]
            ]
            visible.append(to_append)
        
        return visible

class TeacherWrapper(gym.Wrapper): 
    def __init__(self, env, visibility, special_region_rewards: list[float] = []): # defaults to ignoring regions
        super().__init__(env)
        self.env = env
        self.visibility = visibility
        
        # if too short, pad with zeros (no effect) to avoid indexing error in step; otherwise, keep as-is
        self.special_region_rewards =  list(special_region_rewards) # should create a copy I can work with so I don't mutate the parameter list
        if self.env._num_types_special_regions - len(special_region_rewards) > 0: 
            self.special_region_rewards += [0] * (self.env._num_types_special_regions - len(special_region_rewards))
        self.path = [] # teacher needs to record coordinates visited (path) so that student can access it for training (only used in phase 2, where student trains from teacher example)
    
    def step(self, action): 
        full_obs, rewards, terminations, truncated, info = self.env.step_one_agent(action)
        
        reward = rewards["teacher"]
        terminated = terminations["teacher"]

        # restrict obs to limited visibility area, and don't need to know location of student
        obs = {
            "teacher_agent": full_obs["teacher_agent"], # doesn't need to know location of student agent
            "target": full_obs["target"], 
            "special_regions": self.env.get_regions_in_visibility("teacher", self.visibility),
        }

        # update reward based on special_region_rewards values
        curr_region = self.env.get_curr_special_region("teacher")
        if curr_region is not None: # in some special region
            reward = self.special_region_rewards[curr_region]
        
        # record new coordinates for student training
        self.path.append(self.env._teacher_agent_location.copy())

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> tuple[dict, dict]: 
        self.path = []
        return super().reset(seed=seed, options=options)