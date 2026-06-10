# TODO; not yet implemented, just old code tho

import numpy as np
import gymnasium as gym
import math
import pandas as pd

class MapNavGridworldBase(gym.Env): 
    def __init__(self, num_types_special_regions: int = 0, goal_reward: int = 10, step_penalty: float = -0.1, spawn_width:int = 10, num_directions:int = 16, reward_shaping=False, compass_penalty_multiplier:float=0.05): 
        self._num_types_special_regions = num_types_special_regions
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.spawn_width = spawn_width
        self.num_directions = num_directions
        self.reward_shaping = reward_shaping
        self.compass_penalty_multiplier = compass_penalty_multiplier

        self.curr_world_bounds = np.array([(0,0),(spawn_width,spawn_width)]) # (smallest x, smallest y), (greatest x, greatest y)

        self.coords_to_default() # set coordinates to - values to show that they are uninitialized

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # randomly sets teacher, sets student to same start location as teacher
        self._teacher_agent_location = self.np_random.integers(0, self.spawn_width, size=2, dtype=int)
        self._student_agent_location = self._teacher_agent_location.copy()

        self._target_location = self._teacher_agent_location.copy()
        while np.array_equal(self._target_location, self._teacher_agent_location):
            self._target_location = self.np_random.integers(0, self.spawn_width, size=2, dtype=int)

        special_regions_index_order = self.np_random.permutation(self._num_types_special_regions)
        for index in special_regions_index_order:
            num_to_place = self.np_random.integers(low=0, high=(self.spawn_width ** 2) // 2)
            self._special_regions[index] = [
                tuple(self.np_random.integers(0, self.spawn_width, size=2, dtype=int))
                for _ in range(num_to_place)
            ]
        
        return self._get_obs(), {} # empty info

    def get_reward_no_shaping(self): 
        return self.goal_reward if np.array_equal(self._teacher_agent_location, self._target_location) else self.step_penalty
    
    def get_reward_with_shaping(self, old_loc, agent_loc): # uses bucketed angles
        move_angle = self.get_bucketed_angle(self.get_angle_for_coords(old_loc, agent_loc))
        target_angle = self.get_bucketed_angle(self.get_angle_for_coords(self._target_location, old_loc))
        bin_diff = abs(move_angle - target_angle)
        bin_diff = min(bin_diff, self.num_directions - bin_diff) # removes negatives
        max_bin_diff = self.num_directions // 2 # furthest apart that 2 bins can be is half of total num of divisions of the circle
        dir_reward = self.compass_penalty_multiplier * (1- 2*bin_diff/max_bin_diff) # when bin_diff = max_bin_diff, this will give -1 bc they're going in the opposite direction; when bin_diff = max_bin_diff/2, they're perpendicular, and when bin_diff = 0 will return 1; current formula from claude, may change later
        # TODO: fix this so its not hard-coded for the teacher, maybe use agent loc
        return self.goal_reward + dir_reward if np.array_equal(self._teacher_agent_location, self._target_location) else self.step_penalty + dir_reward

    def get_angle_for_dx_dy(self, dx, dy): # returns on scale of [0, 2pi)
        return (math.atan2(-dx, dy) + 2*math.pi) % (2*math.pi)
    
    def get_bucketed_angle(self, angle): 
        angle_bin = int(angle / (2*math.pi) * self.num_directions) % self.num_directions
        return angle_bin

    def get_angle_for_coords(self, coord1, coord2): 
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        return self.get_angle_for_dx_dy(dx, dy)

    def step_one_agent(self, action, agent:str = "teacher"): 
        direction = self._action_to_direction[action]
        # can't use func I wrote to get loc bc I need to update too
        if agent == "teacher": 
            old_loc = self._teacher_agent_location.copy()
            self._teacher_agent_location += direction
            agent_loc = self._teacher_agent_location
        else: 
            old_loc = self._student_agent_location.copy()
            self._student_agent_location += direction
            agent_loc = self._student_agent_location


        terminations = {"teacher": np.array_equal(self._teacher_agent_location, self._target_location), "student": np.array_equal(self._student_agent_location, self._target_location)}
        truncated = False # will be overridden in wrappers, since wrappers keep track of steps for each agent (not part of world environment)
        # unless overridden later by special area rules for teacher, small penalties for all areas except for goal to encourage going to goal
        rewards = {"teacher": self.get_reward_with_shaping(old_loc, agent_loc) if self.reward_shaping else self.get_reward_no_shaping(), "student": self.get_reward_with_shaping(old_loc, agent_loc) if self.reward_shaping else self.get_reward_no_shaping()}
        full_observations = self.get_full_world_state()
        # placeholder to not crash
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
    
    def is_in_visibility_region(self, agent_location, coords, visibility_range = None): 
        if visibility_range is None: 
            return True
        x_y_visibility_range = [agent_location[0] - visibility_range, agent_location[0] + visibility_range], [agent_location[1] - visibility_range, agent_location[1] + visibility_range]
        return x_y_visibility_range[0][0] <= coords[0] < x_y_visibility_range[0][1] and x_y_visibility_range[1][0] <= coords[1] < x_y_visibility_range[1][1]
    
    def get_x_y_ranges_for_coords(self, coords1, coords2): 
        xs = (min(coords1[0], coords2[0]), max(coords1[0], coords2[0]))
        ys = (min(coords1[1], coords2[1]), max(coords1[1], coords2[1]))
        return (xs, ys) # list of tuples
    
    def update_world_dims(self, agent_loc, visibility_range): 
        x_min = agent_loc[0] - visibility_range
        x_max = agent_loc[0] + visibility_range
        y_min = agent_loc[1] - visibility_range
        y_max = agent_loc[1] + visibility_range
        Y, X = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
        agent_visible_regions = np.column_stack((X.ravel(), Y.ravel())) # flattened + stacked
        outside_explored = ((agent_visible_regions[:, 0] < self.curr_world_bounds[0][0]) | # less than min x
                             (agent_visible_regions[:, 0] > self.curr_world_bounds[1][0]) |  # greater than max x
                             (agent_visible_regions[:, 1] < self.curr_world_bounds[0][1]) | # less than min y
                             (agent_visible_regions[:, 1] > self.curr_world_bounds[1][1]))  # greater than max y
        
        # not sure if those were the right coords
        if outside_explored.any(): 
            new_coords = agent_visible_regions[outside_explored]
            for coord in new_coords: 
                # set new min/max coords
                # I know there must be a better way to do this but I'm not sure what that is
                if coord[0] < self.curr_world_bounds[0][0]: self.curr_world_bounds[0][0] = coord[0] # less than x min
                if coord[0] > self.curr_world_bounds[1][0]: self.curr_world_bounds[1][0] = coord[0] # greater than x max
                if coord[1] < self.curr_world_bounds[0][1]: self.curr_world_bounds[0][1] = coord[1] # less than y min
                if coord[1] > self.curr_world_bounds[1][1]: self.curr_world_bounds[1][1] = coord[1] # greater than y max
            
            new_visible = agent_visible_regions[outside_explored]
            if len(new_visible) > 0: self.generate_necessary_special_regions(new_visible)
    
    def generate_necessary_special_regions(self, coords): 
        special_regions_index_order = self.np_random.permutation(self._num_types_special_regions)
        for index in special_regions_index_order:
            num_to_place = self.np_random.integers(low=0, high=len(coords) // 2)
            if num_to_place == 0: continue
            selected_coords = coords[self.np_random.choice(len(coords), size=num_to_place, replace=False)]
            self._special_regions[index].extend(map(tuple, selected_coords)) # coords need to be tuples, not np arrays

    def get_agent_loc_for_name(self, name): 
        agent_location = self._student_agent_location
        if name == "teacher": 
            agent_location = self._teacher_agent_location
        return agent_location

    # sped up for quality of life improvement
    def get_regions_in_visibility(self, agent, visibility_range): 
        agent_location = self.get_agent_loc_for_name(agent)
        self.update_world_dims(agent_location, visibility_range) # make sure up-to-date before checking special regions
        
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
    
    # don't want to pollute with location of other agent, in format for CNN to take in (different channel for agent, target, and each region)
    def make_one_agent_grid_relative(self, agent, visibility, extra_empty_layers=0): # extra empty layers is for adding empty layers onto an output in situations like curriculum learning where I want to introduce new regions later, but the agent needs to take in a certain number of layers as input so I need to pad earlier inputs with an extra layer
        # scale down to fixed size based on visibility
        channels = []

        agent_grid = np.zeros((2*visibility + 1, 2*visibility + 1), dtype=np.float32) # visibility on both sides + agent cell
        agent_grid[visibility, visibility] = 1 # agent-centric
        agent_coords = self.get_agent_loc_for_name(agent)
        channels.append(agent_grid)

        target = self._target_location
        target_grid = self.make_empty_grid(2*visibility + 1)
        if self.is_in_visibility_region(agent_coords, target, visibility): 
            target_grid[self._target_location[0] - agent_coords[0] + visibility, self._target_location[1] - agent_coords[1] + visibility] = 1
        channels.append(target_grid)

        special_regions = self.get_regions_in_visibility(agent, visibility)
        for special_region in special_regions: 
            region_grid = self.make_empty_grid(2*visibility + 1)
            for coords in special_region: 
                rel_coords = (coords[0] - agent_coords[0] + visibility, coords[1] - agent_coords[1] + visibility)
                region_grid[rel_coords[0], rel_coords[1]] = 1
            channels.append(region_grid)

        # target col and row channels
        dist_to_target_row = (self._target_location[0] - agent_coords[0]) / visibility # normalizing by visibility so model can generalize to different sizes easier (normalizing by size = confusing bc values scaled differently)
        dist_to_target_col = (self._target_location[1] - agent_coords[1]) / visibility

        delta_row_grid = np.zeros((2*visibility + 1, 2*visibility + 1), dtype=np.float32)
        delta_row_grid[0][0] = dist_to_target_row

        delta_col_grid = np.zeros((2*visibility + 1, 2*visibility + 1), dtype=np.float32)
        delta_col_grid[0][0] = dist_to_target_col

        channels.append(delta_row_grid)
        channels.append(delta_col_grid)

        for _ in range(extra_empty_layers): 
            channels.append(self.make_empty_grid(2*visibility + 1))

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



class TeacherWrapper(gym.Wrapper): 
    def __init__(self, env, visibility : int, max_steps: int = 50, special_region_rewards: list[float] = []): # if visibility not passed just sees whole world, defaults to ignoring regions
        super().__init__(env)
        self.env = env
        self.visibility = visibility
        num_channels = 1+env._num_types_special_regions+1+2 # agent + one per region type + target + dx + dy
        grid_size = 2*visibility + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_channels, grid_size, grid_size), dtype=np.float32)
        
        self.max_steps = max_steps if max_steps is not None else float('inf') # if none, then no max
        self.num_steps = 0 # init value
        
        # if too short, pad with zeros (no effect) to avoid indexing error in step; otherwise, keep as-is
        self.special_region_rewards =  list(special_region_rewards) # should create a copy I can work with so I don't mutate the parameter list
        if self.env._num_types_special_regions - len(special_region_rewards) > 0: 
            self.special_region_rewards += [0] * (self.env._num_types_special_regions - len(special_region_rewards))
        self.path = [] # start with starting location; teacher needs to record coordinates visited (path) so that student can access it for training (only used in phase 2, where student trains from teacher example)
    
    def step(self, action): 
        full_obs, rewards, terminations, _, info = self.env.step_one_agent(action)

        self.num_steps += 1
        
        reward = rewards["teacher"]
        terminated = terminations["teacher"]
        truncated = self.max_steps <= self.num_steps

        obs = self.env.make_one_agent_grid_relative("teacher", self.visibility)

        # update reward based on special_region_rewards values
        curr_region = self.env.get_curr_special_region("teacher")
        if curr_region is not None: # in some special region
            reward = self.special_region_rewards[curr_region]
        
        # record new coordinates for student training
        self.path.append(self.env._teacher_agent_location.copy())

        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> tuple[dict, dict]: 
        base_obs, info = self.env.reset(seed=seed, options=options) # trigger base env reset (makes sure I have valid coords for teacher agent start)

        self.path = [self.env._teacher_agent_location.copy()]
        self.num_steps = 0
        
        teacher_obs = self.env.make_one_agent_grid_relative("teacher", self.visibility)
        
        return teacher_obs, info