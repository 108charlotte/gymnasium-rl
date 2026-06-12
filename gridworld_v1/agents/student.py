from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

# TODO: figure out how to make this any different than the teacher's DQN in a way that's actually meaningful

class StudentCNN(torch.nn.Module): # CNN because should have spatial reasoning
    def __init__(self, num_types_special_regions, num_directions, num_filters_first_layer, final_conv_filters, target_spatial_size):
        super().__init__()
        self.spatial_reasoning = nn.Sequential(
            nn.Conv2d(2 + num_types_special_regions, num_filters_first_layer, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters_first_layer, final_conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((target_spatial_size, target_spatial_size)),
            nn.Flatten()
        )

        self.teacher_action_processing = nn.Sequential(
            nn.Linear(num_directions, 16), 
            nn.ReLU()
        )

        self.to_action = nn.Sequential(
            nn.Linear(final_conv_filters * target_spatial_size**2 + 16, 64), # spatial out + linear out
            nn.ReLU(), 
            nn.Linear(64, 4)
        )

        self.num_types_special_regions = num_types_special_regions
        self.num_spatial_channels = 2 + num_types_special_regions
    
    def forward(self, x):
        # x is shape: (batch, 1 + num_types_special_regions + 1 + num_directions, h, w)
        result = self.spatial_reasoning(x[:, :self.num_spatial_channels]) # slice channel dimension (1st is batch dimension) last two are col and row for compass nav
        return self.to_action(result)