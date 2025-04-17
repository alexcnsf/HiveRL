import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, List
from .hive_rules import HiveRules

class HiveEnv(gym.Env):
    """
    A simplified Hive environment following the Gymnasium interface.
    The environment is designed as a proper MDP for PPO implementation.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action space
        # Each action is a tuple of (piece_type, position, move_type)
        # piece_type: 0=Queen, 1=Beetle, 2=Ant
        # position: (x, y) coordinates on 5x5 grid
        # move_type: 0=place, 1=move
        self.action_space = gym.spaces.MultiDiscrete([
            3,  # piece_type
            5,  # x coordinate
            5,  # y coordinate
            2   # move_type
        ])
        
        # Define observation space
        # The state is represented as a 3D array:
        # - First dimension: piece types (3)
        # - Second dimension: x coordinates (5)
        # - Third dimension: y coordinates (5)
        # Value at each position: 0=empty, 1=player1, 2=player2
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(3, 5, 5),
            dtype=np.int8
        )
        
        # Initialize game state
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        Returns:
            observation: The initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Initialize empty board
        self.state = np.zeros((3, 5, 5), dtype=np.int8)
        
        # Initialize game variables
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        
        # Track placed pieces
        self.player1_pieces = {
            'queen': 1,
            'beetle': 3,
            'ant': 3
        }
        self.player2_pieces = {
            'queen': 1,
            'beetle': 3,
            'ant': 3
        }
        
        # Track piece positions
        self.piece_positions = {
            1: [],  # List of (piece_type, x, y) for player 1
            2: []   # List of (piece_type, x, y) for player 2
        }
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        Args:
            action: The action to take
        Returns:
            observation: The new state
            reward: The reward for the action
            terminated: Whether the episode is finished
            truncated: Whether the episode was truncated
            info: Additional information
        """
        piece_type, x, y, move_type = action
        
        # Check if action is valid
        if not self._is_valid_action(action):
            return self._get_observation(), -0.1, False, False, self._get_info()
        
        # Execute the action
        self._execute_action(action)
        
        # Check for game termination
        terminated = self._check_termination()
        
        # Calculate reward
        reward = self._calculate_reward(terminated)
        
        # Switch players if game is not over
        if not terminated:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            self.turn_count += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _is_valid_action(self, action: np.ndarray) -> bool:
        """Check if the action is valid given the current state."""
        piece_type, x, y, move_type = action
        
        if move_type == 0:  # Placement
            # Check if player has the piece available
            if piece_type == 0:  # Queen
                if self.current_player == 1 and self.player1_pieces['queen'] <= 0:
                    return False
                if self.current_player == 2 and self.player2_pieces['queen'] <= 0:
                    return False
            elif piece_type == 1:  # Beetle
                if self.current_player == 1 and self.player1_pieces['beetle'] <= 0:
                    return False
                if self.current_player == 2 and self.player2_pieces['beetle'] <= 0:
                    return False
            elif piece_type == 2:  # Ant
                if self.current_player == 1 and self.player1_pieces['ant'] <= 0:
                    return False
                if self.current_player == 2 and self.player2_pieces['ant'] <= 0:
                    return False
            
            return HiveRules.is_valid_placement(
                self.state, piece_type, x, y, self.current_player, self.turn_count
            )
        else:  # Movement
            # Find the piece to move
            for p_type, p_x, p_y in self.piece_positions[self.current_player]:
                if p_type == piece_type:
                    return HiveRules.is_valid_move(
                        self.state, piece_type, (p_x, p_y), (x, y), self.current_player
                    )
            return False
    
    def _execute_action(self, action: np.ndarray) -> None:
        """Execute the action and update the state."""
        piece_type, x, y, move_type = action
        
        if move_type == 0:  # Place
            # Update state
            self.state[piece_type, x, y] = self.current_player
            
            # Update piece positions
            self.piece_positions[self.current_player].append((piece_type, x, y))
            
            # Update piece counts
            if piece_type == 0:  # Queen
                if self.current_player == 1:
                    self.player1_pieces['queen'] -= 1
                else:
                    self.player2_pieces['queen'] -= 1
            elif piece_type == 1:  # Beetle
                if self.current_player == 1:
                    self.player1_pieces['beetle'] -= 1
                else:
                    self.player2_pieces['beetle'] -= 1
            elif piece_type == 2:  # Ant
                if self.current_player == 1:
                    self.player1_pieces['ant'] -= 1
                else:
                    self.player2_pieces['ant'] -= 1
        else:  # Move
            # Find and update the piece position
            for i, (p_type, p_x, p_y) in enumerate(self.piece_positions[self.current_player]):
                if p_type == piece_type:
                    # Update state
                    self.state[p_type, p_x, p_y] = 0
                    self.state[p_type, x, y] = self.current_player
                    
                    # Update piece position
                    self.piece_positions[self.current_player][i] = (p_type, x, y)
                    break
    
    def _check_termination(self) -> bool:
        """Check if the game is over."""
        # Check if either queen is surrounded
        if HiveRules.is_queen_surrounded(self.state, 1):
            self.winner = 2
            self.game_over = True
            return True
        elif HiveRules.is_queen_surrounded(self.state, 2):
            self.winner = 1
            self.game_over = True
            return True
        return False
    
    def _calculate_reward(self, terminated: bool) -> float:
        """Calculate the reward for the current state."""
        if terminated:
            return 1.0 if self.winner == self.current_player else -1.0
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get the current state observation."""
        return self.state.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            'current_player': self.current_player,
            'turn_count': self.turn_count,
            'game_over': self.game_over,
            'winner': self.winner,
            'player1_pieces': self.player1_pieces,
            'player2_pieces': self.player2_pieces
        } 