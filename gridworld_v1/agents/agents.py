from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

class CNN(torch.nn.Module): # CNN because should have spatial reasoning
    def __init__(self, num_types_special_regions, num_filters_first_layer, final_conv_filters, target_spatial_size): 
        super().__init__()

        if final_conv_filters is None: 
            final_conv_filters = num_filters_first_layer * 2

        self.model = nn.Sequential(
            nn.Conv2d(2 + num_types_special_regions, num_filters_first_layer, kernel_size=3, padding=1), # sees agent, target, and extra regions (not other agent)
            nn.ReLU(), 
            nn.Conv2d(num_filters_first_layer, final_conv_filters, kernel_size=3, padding=1), # outputs 4D tensor
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((target_spatial_size, target_spatial_size)), # squash spatial layout, keep features seperate - need to review this, because having trouble making sense of it conceptually
            nn.Flatten(), 
            nn.Linear(final_conv_filters * target_spatial_size**2, final_conv_filters // 2), 
            nn.ReLU(), 
            nn.Linear(final_conv_filters // 2, 4) # up, down, left, right (space has 4 discrete actions)
        )
    
    def forward(self, input): 
        # num channels = 3 + num types of regions
        # (batch, channels, size, size)
        return self.model(input)

# for now, this is heavily modeled after https://gymnasium.farama.org/v1.1.1/introduction/train_agent/
# and the DQN is this: https://medium.com/data-science/develop-your-first-ai-agent-deep-q-learning-375876ee2472#b396
class TeacherAgent: 
    def __init__(self, 
                 env: gym.Env, 
                 learning_rate: float, 
                 initial_epsilon: float, 
                 epsilon_decay: float, 
                 final_epsilon: float, 
                 num_filters_first_layer = 16, 
                 final_conv_filters = 32, 
                 target_spatial_size = 3, 
                 target_update_freq: int = 1000, 
                 discount_factor: float = 0.95,
                ): 
        self.env = env
        self.learning_rate = learning_rate

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.discount_factor = discount_factor

        # https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.num_filters_first_layer = num_filters_first_layer
        self.final_conv_filters = final_conv_filters
        self.target_spatial_size = target_spatial_size
        self.model = self.build_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.HuberLoss()
        self.target_update_freq = target_update_freq
    
    def build_model(self): 
        print(f"Using {self.device} device")
        model = CNN(self.env._num_types_special_regions, self.num_filters_first_layer, self.final_conv_filters, self.target_spatial_size).to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.steps_done = 0
        return model
    
    def get_action(self, state): 
        if np.random.rand() <= self.epsilon: 
            action = np.random.randint(0, 4) # last num is exclusive
        else: 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # needs to be (1, channels, size, size), but was (channels, size, size)
            with torch.no_grad(): 
                q_values = self.model(state)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def decay_epsilon(self): 
        if self.epsilon > self.final_epsilon: 
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.final_epsilon) # never go below final
    
    def learn(self, experiences): 
        # float32 bc that's what the ANN expects, by default with torch.tensor would be float64
        states = torch.tensor(np.array([experience.state for experience in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([experience.action for experience in experiences]), dtype=torch.long, device=self.device) # needs to be long for .gather
        rewards = torch.tensor(np.array([experience.reward for experience in experiences]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([experience.next_state for experience in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([experience.done for experience in experiences]), dtype=torch.float32, device=self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1) # gather 1 is gather on dimension 1 (cols), in the output of model(states) each row is an experience and each col is the q val for a possible action, so by using gather with actions you're getting the q values for only the actions that were taken in each experience
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1).values

        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # gradient clipping
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item() # for visualization
    
    def load_model_from_saved(self, path_to_save):
        checkpoint = torch.load(path_to_save, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["model_state_dict"])
        
        print("loaded weights and hyperparams")



# https://medium.com/data-science/develop-your-first-ai-agent-deep-q-learning-375876ee2472#b396
# store past experiences so that the agent can use them to improve
class ExperienceReplay: 
    def __init__(self, capacity, batch_size): 
        self.memory = deque(maxlen=capacity) # deque = double-ended queue
        self.batch_size = batch_size
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def add_experience(self, state, action, reward, next_state, done): 
        experience = self.Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def sample_batch(self): 
        batch = random.sample(self.memory, self.batch_size)
        return batch
    
    # checks if enough experiences are in memory to sample a batch
    def can_provide_sample(self): 
        return len(self.memory) >= self.batch_size