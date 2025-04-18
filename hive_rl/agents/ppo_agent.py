import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
import torch.nn.functional as F
from torch.distributions import Categorical

class PPONetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super().__init__()
        
        # CNN layers for board state
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate CNN output size
        conv_out_size = 64 * 5 * 5  # Since we maintain the input dimensions with padding
        
        # MLP layers for other state components
        self.fc1 = nn.Linear(conv_out_size + 7, 512)  # +7 for current_player and remaining_pieces
        self.fc2 = nn.Linear(512, 256)
        
        # Policy head
        self.policy = nn.Linear(256, n_actions)
        
        # Value head
        self.value = nn.Linear(256, 1)
        
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process board state through CNN
        x = F.relu(self.conv1(state_dict['board']))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Concatenate with other state components
        other_state = torch.cat([
            state_dict['current_player'].float().unsqueeze(1),
            state_dict['remaining_pieces'].float()
        ], dim=1)
        
        x = torch.cat([x, other_state], dim=1)
        
        # Process through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get policy and value
        policy = F.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        
        return policy, value

class PPOAgent:
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        batch_size: int = 64
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        self.memory = []
        
    def _process_state(self, state: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy state dict to tensor state dict"""
        return {
            'board': torch.FloatTensor(state['board']).unsqueeze(0).to(self.device),
            'current_player': torch.FloatTensor([state['current_player']]).to(self.device),
            'remaining_pieces': torch.FloatTensor(state['remaining_pieces']).unsqueeze(0).to(self.device)
        }
        
    def get_action(self, state: Dict[str, np.ndarray], action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Get action from policy network with action masking."""
        # Convert state to tensor dict
        state_dict = self._process_state(state)
        action_mask_tensor = torch.FloatTensor(action_mask).to(self.device)
        #print(f"[Turn Log] Valid actions: {int(action_mask_tensor.sum().item())}")


        # Check if any actions are valid
        if not torch.any(action_mask_tensor):
            print("[Warning] No valid actions in mask.")
            return 0, 0.0, 0.0

        with torch.no_grad():
            action_probs, value = self.network(state_dict)

        # Remove batch dimension if needed
        if action_probs.dim() == 2:
            action_probs = action_probs[0]

        # Apply action mask
        masked_probs = action_probs * action_mask_tensor
        masked_probs_sum = masked_probs.sum()

        if masked_probs_sum.item() == 0:
            print("[Warning] No valid actions (masked_probs sum == 0)")
            valid_indices = torch.nonzero(action_mask_tensor, as_tuple=True)[0]
            action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            log_prob = torch.tensor(0.0)
        else:
            masked_probs = masked_probs / masked_probs_sum  # Renormalize
            dist = Categorical(probs=masked_probs)
            action = dist.sample()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device))

        return action, value.item(), log_prob.item()
        
    def remember(self, state: Dict[str, np.ndarray], action: int, reward: float, 
                next_state: Dict[str, np.ndarray], done: bool, action_mask: np.ndarray):
        """Store experience in memory"""
        self.memory.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'action_mask': action_mask.copy()
        })
        
    def update(self):
        """Update policy using PPO"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch
        batch = self.memory[-self.batch_size:]
        self.memory = self.memory[:-self.batch_size]
        
        # Convert to tensors
        states = [self._process_state(item['state']) for item in batch]
        actions = torch.LongTensor([item['action'] for item in batch]).to(self.device)
        rewards = torch.FloatTensor([item['reward'] for item in batch]).to(self.device)
        next_states = [self._process_state(item['next_state']) for item in batch]
        dones = torch.FloatTensor([item['done'] for item in batch]).to(self.device)
        action_masks = torch.FloatTensor([item['action_mask'] for item in batch]).to(self.device)
        
        # Get old action probabilities and values
        old_probs = []
        old_values = []
        with torch.no_grad():
            for state_dict in states:
                probs, value = self.network(state_dict)
                old_probs.append(probs)
                old_values.append(value)
        old_probs = torch.stack(old_probs).squeeze()
        old_values = torch.stack(old_values).squeeze()
        
        # Get new action probabilities and values
        new_probs = []
        new_values = []
        for state_dict in states:
            probs, value = self.network(state_dict)
            new_probs.append(probs)
            new_values.append(value)
        new_probs = torch.stack(new_probs).squeeze()
        new_values = torch.stack(new_values).squeeze()
        
        # Calculate advantages
        advantages = rewards - old_values
        
        # Calculate ratio
        ratio = torch.exp(torch.log(new_probs + 1e-10) - torch.log(old_probs + 1e-10))
        
        # Calculate surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        
        # Calculate policy and value losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, rewards)
        
        # Update network
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, path: str):
        """Save model to path"""
        torch.save(self.network.state_dict(), path)
        
    def load(self, path: str):
        """Load model from path"""
        self.network.load_state_dict(torch.load(path)) 