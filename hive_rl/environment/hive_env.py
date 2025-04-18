import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List
from .hive_rules import HiveRules

class HiveEnv(gym.Env):
    """
    A simplified Hive environment following the Gymnasium interface.
    The environment is designed as a proper MDP for PPO implementation.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action dimensions
        self.action_dims = (3, 2, 5, 5, 5, 5)  # piece_type, action_type, x, y, from_x, from_y
        self.action_size = np.prod(self.action_dims)
        
        # Define action space as a single integer
        self.action_space = spaces.Discrete(self.action_size)
        
        # Define observation space
        # Board state: 3x5x5 array (piece_type, x, y)
        # Current player: 1
        # Turn count: 1
        # Remaining pieces: [p1_queen, p1_beetle, p1_ant, p2_queen, p2_beetle, p2_ant]
        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=2, shape=(3, 5, 5), dtype=np.int8),
            'current_player': spaces.Discrete(2),
            'turn_count': spaces.Discrete(100),  # Reasonable upper limit
            'remaining_pieces': spaces.Box(low=0, high=3, shape=(6,), dtype=np.int8)  # Max 3 ants per player
        })
        
        # Initialize game state
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        Returns:
            observation: The initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset game state
        self.state = np.zeros((3, 5, 5), dtype=np.int8)
        self.current_player = 1
        self.turn_count = 0
        # Initialize remaining pieces: [p1_queen, p1_beetle, p1_ant, p2_queen, p2_beetle, p2_ant]
        self.remaining_pieces = np.array([1, 2, 3, 1, 2, 3], dtype=np.int8)
        
        return self._get_obs(), self._get_info()
    
    def encode_action(self, piece_type: int, action_type: int, x: int, y: int, from_x: int, from_y: int) -> int:
        """Encode action components into a single integer."""
        return np.ravel_multi_index((piece_type, action_type, x, y, from_x, from_y), self.action_dims)
        
    def decode_action(self, action_id: int) -> Tuple[int, int, int, int, int, int]:
        """Decode a single integer into action components."""
        return np.unravel_index(action_id, self.action_dims)
        
    def get_action_mask(self) -> np.ndarray:
        """
        Generate a mask for valid actions.
        Returns a boolean array of shape (action_size,) where:
        - True means the action is valid
        - False means the action is invalid
        """
        # First get the 6D mask
        mask_6d = np.zeros(self.action_dims, dtype=bool)
        
        # For each piece type
        for piece_type in range(3):
            # Check if player has this piece type available
            piece_idx = piece_type + (3 if self.current_player == 2 else 0)
            has_piece = self.remaining_pieces[piece_idx] > 0
            
            # Check placement actions
            if has_piece:
                for x in range(5):
                    for y in range(5):
                        if HiveRules.is_valid_placement(
                            self.state, piece_type, x, y, self.current_player, self.turn_count
                        ):
                            mask_6d[piece_type, 0, x, y, 0, 0] = True  # from_x, from_y don't matter for placements
            
            # Check movement actions
            # First find all pieces of this type that belong to current player
            piece_positions = np.where(self.state[piece_type] == self.current_player)
            for from_x, from_y in zip(*piece_positions):
                # Check all possible moves from this position
                for to_x in range(5):
                    for to_y in range(5):
                        if HiveRules.is_valid_move(
                            self.state, piece_type, (from_x, from_y), (to_x, to_y), self.current_player
                        ):
                            mask_6d[piece_type, 1, to_x, to_y, from_x, from_y] = True
        
        # Flatten the mask
        return mask_6d.reshape(-1)
        
    def _is_valid_action(self, action_id: int) -> bool:
        """Check if an action is valid using the action mask."""
        return self.get_action_mask()[action_id]
    
    def _calculate_reward(self, action_id: int) -> float:
        """Calculate the reward for the current action."""
        reward = 0.0
        
        # Decode the action
        piece_type, action_type, x, y, from_x, from_y = self.decode_action(action_id)
        
        # Check win condition
        game_over, winner = HiveRules.check_win_condition(self.state)
        if game_over:
            return 1.0 if winner == self.current_player else -1.0
            
        # Progressive rewards
        if action_type == 0:  # Placement
            if piece_type == 0:  # Queen placement
                reward += 0.1
                
            # Check if this placement surrounds opponent's queen
            opponent_queen_pos = np.where(self.state[0] == (3 - self.current_player))
            if len(opponent_queen_pos[0]) > 0:
                queen_x, queen_y = opponent_queen_pos[0][0], opponent_queen_pos[1][0]
                if self._is_adjacent((x, y), (queen_x, queen_y)):
                    reward += 0.2
                    
        else:  # Movement
            # Check if movement maintains hive connectivity
            if HiveRules.maintains_hive_connectivity(self.state, piece_type, (from_x, from_y), (x, y)):
                reward += 0.02
                
            # Piece-specific rewards
            if piece_type == 1:  # Beetle
                if self.state[piece_type, x, y] > 0:  # Climbing
                    reward += 0.05
            elif piece_type == 2:  # Ant
                distance = abs(x - from_x) + abs(y - from_y)
                if distance > 2:  # Long-distance movement
                    reward += 0.03
                    
        # Temporal rewards
        if self.turn_count < 20:
            reward += 0.01
        else:
            reward -= 0.01
            
        return reward
        
    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and (x1 != x2 or y1 != y2)
        
    def step(self, action_id: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        Args:
            action_id: The action to take
        Returns:
            observation: The new state
            reward: The reward for the action
            terminated: Whether the episode is finished
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Decode the action
        piece_type, action_type, x, y, from_x, from_y = self.decode_action(action_id)
        
        # Validate action using mask
        if not self._is_valid_action(action_id):
            return self._get_obs(), -0.1, True, False, {'error': 'Invalid action'}
            
        # Execute action
        if action_type == 0:  # Place piece
            self.state[piece_type, x, y] = self.current_player
            # Update remaining pieces
            piece_idx = piece_type + (3 if self.current_player == 2 else 0)
            self.remaining_pieces[piece_idx] -= 1
        else:  # Move piece
            self.state[piece_type, from_x, from_y] = 0
            self.state[piece_type, x, y] = self.current_player
            
        # Calculate reward
        reward = self._calculate_reward(action_id)
        
        # Check win condition
        game_over, winner = HiveRules.check_win_condition(self.state)
        if game_over:
            reward = 1.0 if winner == self.current_player else -1.0
            return self._get_obs(), reward, True, False, {'winner': winner}
            
        # Switch players
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        self.turn_count += 1
        
        return self._get_obs(), reward, False, False, self._get_info()
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        return {
            'board': self.state.copy(),
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'remaining_pieces': self.remaining_pieces.copy()
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'game_over': HiveRules.check_win_condition(self.state)[0],
            'winner': HiveRules.check_win_condition(self.state)[1],
            'remaining_pieces': self.remaining_pieces.copy()
        } 