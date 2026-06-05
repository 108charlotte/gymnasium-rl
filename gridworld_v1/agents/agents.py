from collections import deque, namedtuple
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

class DQRN(torch.nn.Module): # CNN because should have spatial reasoning
    def __init__(self, num_types_special_regions, num_filters_first_layer, final_conv_filters, target_spatial_size): 
        super().__init__()

        if final_conv_filters is None: 
            final_conv_filters = num_filters_first_layer * 2

        # CNN
        self.spatial_reasoning = nn.Sequential(
            # 4: agent + target + target column delta + target row delta
            nn.Conv2d(4 + num_types_special_regions, num_filters_first_layer, kernel_size=3, padding=1), # sees agent, target, and extra regions (not other agent)
            nn.ReLU(), 
            nn.Conv2d(num_filters_first_layer, final_conv_filters, kernel_size=3, padding=1), # outputs 4D tensor
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((target_spatial_size, target_spatial_size)), # squash spatial layout, keep features seperate - need to review this, because having trouble making sense of it conceptually, and not sure if I want to keep it
            nn.Flatten()
        )

        # lstm for memory
        self.lstm = nn.LSTM(final_conv_filters * target_spatial_size**2, final_conv_filters // 2, batch_first=True)

        self.to_action = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(final_conv_filters // 2, 4) # up, down, left, right (space has 4 discrete actions)
        )
    
    def forward(self, x, hidden_state=None): # BROKEN RN, TODO: FIX SO VALID, LSTM needs 5 dims, spatial needs 4
        # num channels = 3 + num types of regions -> now that this is an rnn, 
        # x: (batch size, sequence length, channels, height, width)
        batch_size, seq_len, channels, h, w = x.size()
        
        # flatten to 4d for CNN by combining batch size and sequence length
        for_CNN = x.view(batch_size * seq_len, channels, h, w)
        spatial_features = self.spatial_reasoning(for_CNN) # returns (batch_size * seq_len, features)

        # de-compress back to 5d for LSTM
        output_features = spatial_features.size(1)
        for_LSTM = spatial_features.view(batch_size, seq_len, output_features)
        hidden_states, new_hidden = self.lstm(for_LSTM, hidden_state) # _ is tuple of just final info @ end of sequence, hidden_states is the hidden state for every timestep

        q_values = self.to_action(hidden_states)
        return q_values, new_hidden

# for now, this is heavily modeled after https://gymnasium.farama.org/v1.1.1/introduction/train_agent/
# and the DQN is this: https://medium.com/data-science/develop-your-first-ai-agent-deep-q-learning-375876ee2472#b396
class TeacherAgent: 
    def __init__(self, 
                 num_types_special_regions_in_env: int, 
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
        self.num_types_special_regions_in_env = num_types_special_regions_in_env
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

        self.hidden_state = None # for lstm

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.HuberLoss()
        self.target_update_freq = target_update_freq
    
    def build_model(self): 
        print(f"Using {self.device} device")
        model = DQRN(self.num_types_special_regions_in_env, self.num_filters_first_layer, self.final_conv_filters, self.target_spatial_size).to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.steps_done = 0
        return model
    
    def get_action(self, state): 
        if np.random.rand() <= self.epsilon: 
            action = np.random.randint(0, 4) # last num is exclusive
        else: 
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0) # needs to be (1, 1, channels, size, size), but was (channels, size, size); the first 1 is batch_size, the next 1 is sequence_length
            with torch.no_grad(): 
                q_values, self.hidden_state = self.model(state, self.hidden_state)
            action = q_values[0, -1, :].argmax(dim=0).item()
        
        return action
    
    def reset_hidden_state(self): # called at end/beginning of episode
        self.hidden_state = None
    
    def decay_epsilon(self): 
        if self.epsilon > self.final_epsilon: 
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.final_epsilon) # never go below final
    
    def learn(self, experiences): 
        # float32 bc that's what the ANN expects, by default with torch.tensor would be float64
        states = torch.tensor(np.array([[step.state for step in seq] for seq in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([[step.action for step in seq] for seq in experiences]), dtype=torch.long, device=self.device) # needs to be long for .gather
        rewards = torch.tensor(np.array([[step.reward for step in seq] for seq in experiences]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([[step.next_state for step in seq] for seq in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([[step.done for step in seq] for seq in experiences]), dtype=torch.float32, device=self.device)

        # gather 1 is gather on dimension 1 (cols), in the output of model(states)
        # each row is an experience and each col is the q val for a possible action, 
        # so by using gather with actions you're getting the q values for only the actions that were taken in each experience

        # self.model(states) outputs a 3d tensor shape (batch_size, seq_len, num_actions)
        # I want to get all q values for an action, so I'm unsqueezing actions at col 1
        current_q_values = self.model(states, None)[0].gather(2, actions.unsqueeze(2)).squeeze(2) # passes hidden = None
        with torch.no_grad():
            next_q_values = self.target_model(next_states, None)[0].max(dim=2).values # the maximum that could happen, assuming the agent takes the best action

        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # gradient clipping
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0: # update target model
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
    def __init__(self, capacity, batch_size, seq_len): 
        self.memory = deque(maxlen=capacity) # deque = double-ended queue
        self.current_episode = []
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def add_experience(self, state, action, reward, next_state, done): 
        self.current_episode.append(self.Experience(state, action, reward, next_state, done))
        if done: # makes sure that shots spanning multiple episodes aren't recorded as one
            if len(self.current_episode) >= self.seq_len: 
                self.memory.append(list(self.current_episode))
            self.current_episode = []
    
    def sample_batch(self): 
        batch = []
        episodes_at_or_over_seq_len = [e for e in self.memory if len(e) >= self.seq_len]
        sampled_episodes = random.sample(episodes_at_or_over_seq_len, min(self.batch_size, len(episodes_at_or_over_seq_len))) # makes sure the model gets enough episodes
        for episode in sampled_episodes: 
            start = random.randint(0, len(episode) - self.seq_len) # starts between the first timestamp and the furthest one to still get a full sequence
            batch.append(episode[start:start+self.seq_len])
        return batch
    
    # checks if enough experiences are in memory to sample a batch
    def can_provide_sample(self): 
        valid_episodes = [e for e in self.memory if len(e) >= self.seq_len]
        return len(valid_episodes) >= self.batch_size # enough valid episodes