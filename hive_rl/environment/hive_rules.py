import numpy as np
from typing import Tuple, List, Dict, Set

class HiveRules:
    """Class containing all Hive game rules and validation logic."""
    
    @staticmethod
    def is_valid_placement(state: np.ndarray, piece_type: int, x: int, y: int, 
                         current_player: int, turn_count: int) -> bool:
        """
        Check if a piece placement is valid.
        Rules:
        1. First piece must be placed at center (2,2)
        2. Subsequent pieces must be adjacent to at least one existing piece
           (except for first move of each player)
        3. Cannot place on top of existing pieces
        4. Queen must be placed by turn 4 (must place on turn 4 if not placed yet)
        """
        # Check if position is within bounds
        if not (0 <= x < 5 and 0 <= y < 5):
            return False
            
        # Check if position is already occupied
        if np.any(state[:, x, y] > 0):
            return False
            
        # Queen placement rule - check this first
        if piece_type == 0 and turn_count >= 5:  # Queen cannot be placed after turn 4
            return False
            
        # Special rule for first piece
        if np.sum(state) == 0:
            return x == 2 and y == 2  # Must start at center
            
        # Must place queen by turn 4
        if turn_count == 4 and state[0, :, :].sum() == 0 and piece_type != 0:
            return False
            
        # Check adjacency rules
        adjacent_positions = HiveRules.get_adjacent_positions(x, y)
        has_adjacent = False
        
        # Check if this is player 2's first move
        is_player2_first_move = current_player == 2 and np.sum(state == 2) == 0
        
        for adj_x, adj_y in adjacent_positions:
            if 0 <= adj_x < 5 and 0 <= adj_y < 5:
                # Check for opponent pieces - if found and not player 2's first move, placement is invalid
                if not is_player2_first_move:
                    for piece_layer in range(3):
                        if state[piece_layer, adj_x, adj_y] == 3 - current_player:
                            return False
                # Check for any adjacent pieces
                for piece_layer in range(3):
                    if state[piece_layer, adj_x, adj_y] > 0:
                        has_adjacent = True
                        
        # For player 2's first move, must be adjacent to player 1's piece
        if is_player2_first_move:
            return has_adjacent
            
        # For other moves, must have at least one adjacent piece (except first piece)
        if np.sum(state) > 0 and not has_adjacent:
            return False
            
        return True
    
    @staticmethod
    def is_valid_move(state: np.ndarray, piece_type: int, from_pos: Tuple[int, int], 
                     to_pos: Tuple[int, int], current_player: int) -> bool:
        """
        Check if a piece movement is valid.
        Rules:
        1. Piece must exist at from_pos and belong to current player
        2. to_pos must be empty (except for beetles)
        3. Movement must follow piece-specific rules
        4. Movement must maintain hive connectivity (except for beetles moving on top)
        5. Cannot move a piece that has another piece on top of it
        """
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Basic validation
        if not (0 <= to_x < 5 and 0 <= to_y < 5):
            return False
            
        # Check if piece exists at from_pos and belongs to current player
        if state[piece_type, from_x, from_y] != current_player:
            return False
            
        # Check if there's another piece on top of this piece
        # Only check pieces with higher indices (they would be on top)
        for other_type in range(piece_type + 1, 3):
            if state[other_type, from_x, from_y] > 0:
                return False
            
        # Check if to_pos is empty (except for beetles)
        # Ants (type 2) and Queens (type 0) cannot move onto occupied spaces
        if (piece_type == 0 or piece_type == 2) and np.any(state[:, to_x, to_y] > 0):
            return False
            
        # Check piece-specific movement rules
        if not HiveRules.is_valid_piece_movement(state, piece_type, from_pos, to_pos):
            return False
            
        # Special handling for beetles
        if piece_type == 1:
            # Check if beetle is moving to an adjacent position
            dx = abs(to_x - from_x)
            dy = abs(to_y - from_y)
            if not (dx <= 1 and dy <= 1 and (dx + dy > 0)):
                return False
                
            # If beetle is moving onto another piece or is already on top of a piece,
            # we only need to check adjacency
            if np.any(state[:, to_x, to_y] > 0) or np.any(state[:, from_x, from_y] > 1):
                return True
                
        # Check hive connectivity
        if not HiveRules.maintains_hive_connectivity(state, from_pos, to_pos):
            return False
            
        return True
    
    @staticmethod
    def maintains_hive_connectivity(state: np.ndarray, from_pos: Tuple[int, int], 
                                  to_pos: Tuple[int, int]) -> bool:
        """Check if moving a piece maintains hive connectivity."""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Create temporary state without the moving piece
        temp_state = state.copy()
        piece_type = np.argmax(temp_state[:, from_x, from_y])
        temp_state[piece_type, from_x, from_y] = 0
        
        # For beetles, if moving onto another piece, it's always connected
        if piece_type == 1 and np.any(state[:, to_x, to_y] > 0):
            return True
        
        # Get all pieces
        pieces = set()
        for x in range(5):
            for y in range(5):
                if np.any(temp_state[:, x, y] > 0):
                    pieces.add((x, y))
                    
        if not pieces:
            return True
            
        # Check if the new position is adjacent to any piece
        has_adjacent = False
        for x, y in pieces:
            if abs(to_x - x) <= 1 and abs(to_y - y) <= 1:
                has_adjacent = True
                break
                
        if not has_adjacent:
            return False
            
        # Start DFS from any piece
        start = next(iter(pieces))
        visited = set()
        stack = [start]
        
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            
            # Check all adjacent positions
            for adj_x, adj_y in HiveRules.get_adjacent_positions(x, y):
                if (0 <= adj_x < 5 and 0 <= adj_y < 5 and 
                    np.any(temp_state[:, adj_x, adj_y] > 0) and 
                    (adj_x, adj_y) not in visited):
                    stack.append((adj_x, adj_y))
        
        # All pieces should be connected
        return len(visited) == len(pieces)
    
    @staticmethod
    def is_valid_piece_movement(state: np.ndarray, piece_type: int, 
                              from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """Check if movement follows piece-specific rules."""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        # Calculate distance
        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)
        
        if piece_type == 0:  # Queen
            # Queen moves exactly one space in any direction
            # Cannot move on top of other pieces
            return (dx <= 1 and dy <= 1) and (dx + dy > 0)
            
        elif piece_type == 1:  # Beetle
            # Beetle moves exactly one space in any direction
            # Can move on top of other pieces
            return (dx <= 1 and dy <= 1) and (dx + dy > 0)
            
        elif piece_type == 2:  # Ant
            # Ant can move any number of spaces around the hive
            # Cannot move on top of other pieces
            # Must stay adjacent to at least one piece
            if np.any(state[:, to_x, to_y] > 0):  # Cannot move on top of other pieces
                return False
                
            adjacent_positions = HiveRules.get_adjacent_positions(to_x, to_y)
            for adj_x, adj_y in adjacent_positions:
                if 0 <= adj_x < 5 and 0 <= adj_y < 5:
                    if np.any(state[:, adj_x, adj_y] > 0):
                        return True
            return False
            
        return False
    
    @staticmethod
    def get_adjacent_positions(x: int, y: int) -> List[Tuple[int, int]]:
        """Get all adjacent positions in a hexagonal grid."""
        # For a 5x5 grid, we use offset coordinates
        # This gives us 6 adjacent positions
        return [
            (x-1, y),    # Left
            (x+1, y),    # Right
            (x, y-1),    # Up
            (x, y+1),    # Down
            (x-1, y+1),  # Down-left
            (x+1, y-1)   # Up-right
        ]
    
    @staticmethod
    def is_queen_surrounded(state: np.ndarray, player: int) -> bool:
        """Check if a player's queen is completely surrounded."""
        # Find queen position
        queen_pos = np.where(state[0] == player)
        if len(queen_pos[0]) == 0:
            return False
            
        x, y = queen_pos[0][0], queen_pos[1][0]
        
        # Check all adjacent positions
        adjacent_positions = HiveRules.get_adjacent_positions(x, y)
        for adj_x, adj_y in adjacent_positions:
            if not (0 <= adj_x < 5 and 0 <= adj_y < 5):
                return False
            if not np.any(state[:, adj_x, adj_y] > 0):
                return False
                
        return True 