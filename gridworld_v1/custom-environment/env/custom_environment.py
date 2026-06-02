from pettingzoo import ParallelEnv
import numpy as np
import gymnasium as gym
from copy import copy
import functools

class GridWorld(ParallelEnv): 
    metadata = {
        "name": "gridworld_v0",
    }

    def __init__(self, 
                 size: int = 10,
                 num_types_special_regions: int = 0, 
                 visible_region_sizes: tuple[int, int] = (-1, -1)): 
        self.size = size
        self.timestep = None
        self._possible_agents = ["teacher", "student"]
        self._num_types_special_regions = num_types_special_regions
        # store visible region sizes for teacher and student if not negative (either default or invalid input), if invalid then set to size of gridworld (full worldview)
        # index 0 is teacher, index 1 is student
        self._visible_region_sizes = visible_region_sizes if visible_region_sizes[0] > 0 and visible_region_sizes[1] > 0 else (size, size)

        # default un-initialized locations
        self.coords_to_default()

        # full agent action -> environment interpretation wrapper would be overkill, so just simple lookup table from tutorial
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
        self._special_regions = np.array([[] for special_region in range(self._num_types_special_regions)], dtype=object)  # each index corresponds to a different type of region, and the values in the list are each tuples of coordinates where the region is
    
    def reset(self, seed=None, options=None): 
        super().reset(seed=seed)

        self.agents = copy(self._possible_agents)
        self.timestep = 0

        self.coords_to_default() # set coords to uninitialized values

        # place agents randomly on grid (both at same position though)
        self._teacher_agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._student_agent_location = self._teacher_agent_location

        # randomly place target anywhere other than agent
        self._target_location = self._teacher_agent_location
        while np.array_equal(self._target_location,self._teacher_agent_location): 
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # randomly place a random number (between 0 and 1/2 of full grid size) of each special region (goes through region numbers in random order, if overlap then overrides; there's no bias for earlier or later indices because the order is randomized)
        rng = np.random.default_rng()
        special_regions_index_order = rng.permutation(range(self._num_types_special_regions)) # permutation returns scrambled array (shuffle doesn't return anything)

        for index in special_regions_index_order: 
            num_to_place = rng.integers(low=0, high=(self.size**2)//2) # floor division to force high to be integer (floored, so never rounds up)
            list_of_coords = []
            for location in range(num_to_place): 
                coords = self.np_random.integers(0, self.size, size=2, dtype=int)
                coords = tuple(coords) # I don't want to accidentally change these, so they shouldn't be mutable - they should only ever be replaced with new coords in a new reset (also, its confusing to have a list with lists in it with lists in it, having a tuple at the base level is nice because it lets me know that that is the base level)
                list_of_coords.append(coords)
            
            self._special_regions[index] = list_of_coords

        observations = self._get_obs()

        # dummy infos; "Necessary for proper parallel_to_aec conversion" - https://pettingzoo.farama.org/tutorials/custom_environment/2-environment-logic/
        infos = {a: {} for a in self.agents}

        return observations, infos

    def _get_obs(self): 
        # get special regions in area visible to each agent
        teacher_visibility_range = [self._teacher_agent_location[0] - self._visible_region_sizes[0], self._teacher_agent_location[0] + self._visible_region_sizes[0]], [self._teacher_agent_location[1] - self._visible_region_sizes[0], self._teacher_agent_location[1] + self._visible_region_sizes[0]]
        student_visibility_range = [self._student_agent_location[0] - self._visible_region_sizes[1], self._student_agent_location[0] + self._visible_region_sizes[1]], [self._student_agent_location[1] - self._visible_region_sizes[1], self._student_agent_location[1] + self._visible_region_sizes[1]]]
        
        special_regions_visible_to_teacher = [
            coords for coords_list in self._special_regions for coords in coords_list
            if coords[0] in range(teacher_visibility_range[0][0], teacher_visibility_range[0][1])
            and coords[1] in range(teacher_visibility_range[1][0], teacher_visibility_range[1][1])
        ]

        special_regions_visible_to_student = [
            coords for coords_list in self._special_regions for coords in coords_list
            if coords[0] in range(student_visibility_range[0][0], student_visibility_range[0][1])
            and coords[1] in range(student_visibility_range[1][0], student_visibility_range[1][1])
        ]

        # observations for each agent (what they get to see)
        observations = {
            "teacher": {
                "teacher_agent": self._teacher_agent_location, # doesn't need to know location of student agent
                "target": self._target_location, 
                "special_regions": special_regions_visible_to_teacher,  # agent decides how to deal with these
            }, 
            "student": {
                "teacher_agent": self._teacher_agent_location,
                "student_agent": self._student_agent_location, 
                "target": self._target_location, 
                "special_regions": special_regions_visible_to_student,
            }
        }

        return observations

    def step(self, actions): 
        student_direction = self._action_to_direction[actions[0]]
        teacher_direction = self._action_to_direction[actions[1]]

        self._student_agent_location = np.clip(self._student_agent_location + student_direction, 0, self.size-1)
        self._teacher_agent_location = np.clip(self._teacher_agent_location + teacher_direction, 0, self.size-1)

        terminations = {"teacher": np.array_equal(self._teacher_agent_location, self._target_location), "student": np.array_equal(self._student_agent_location, self._target_location)}
        rewards = 1 if terminations else 0
        observations = self._get_obs()
        # check for truncation and terminations
        infos = {a: {} for a in self.agents} # dummy infos

        return observations, rewards, terminations, truncations, infos

    def render(self): 
        pass

    def observation_space(self, agent): 
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self): 
        return gym.spaces.Discrete(4) # up, down, left, right