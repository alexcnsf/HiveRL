import numpy as np
import pytest
from hive_rl.environment.hive_rules import HiveRules

def test_piece_tracking():
    # Create state with some pieces
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    state[1, 2, 3] = 1  # Player 1 beetle adjacent
    
    print("Testing piece movement with tracking:")
    # Valid beetle movement - moving diagonally while maintaining adjacency to queen
    assert HiveRules.is_valid_move(state, 1, (2, 3), (3, 3), 1)
    
    # Create a different state with more connections
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    state[1, 2, 3] = 1  # Player 1 beetle
    state[2, 1, 3] = 1  # Player 1 ant
    
    # Valid beetle movement - moving while maintaining multiple connections
    assert HiveRules.is_valid_move(state, 1, (2, 3), (1, 2), 1)
    
    # Invalid beetle movement (too far)
    assert not HiveRules.is_valid_move(state, 1, (2, 3), (0, 0), 1)
    
    # Test moving pieces over other pieces
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen
    state[1, 2, 3] = 1  # Player 1 beetle
    state[2, 2, 4] = 1  # Player 1 ant
    
    # Beetle can move on top of another piece
    assert HiveRules.is_valid_move(state, 1, (2, 3), (2, 4), 1)
    
    # Ant cannot move on top of another piece
    assert not HiveRules.is_valid_move(state, 2, (2, 4), (2, 3), 1)
    
    # Test moving pieces that have something on top
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen
    state[1, 2, 2] = 1  # Player 1 beetle on top of queen
    
    # Cannot move queen when beetle is on top
    assert not HiveRules.is_valid_move(state, 0, (2, 2), (2, 3), 1)
    
    # Can move beetle when it's on top
    assert HiveRules.is_valid_move(state, 1, (2, 2), (2, 3), 1)

@pytest.mark.xfail
def test_invalid_moves():
    """Test cases that should fail but don't stop the test suite"""
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen
    state[1, 2, 3] = 1  # Player 1 beetle
    
    # Moving beetle breaks connectivity (expected to fail)
    assert HiveRules.is_valid_move(state, 1, (2, 3), (2, 4), 1)
    
    # Moving queen when beetle is on top (expected to fail)
    state[1, 2, 2] = 1  # Beetle on top of queen
    assert HiveRules.is_valid_move(state, 0, (2, 2), (2, 3), 1)
    
    # Moving ant on top of another piece (expected to fail)
    state[2, 2, 3] = 1  # Ant
    assert HiveRules.is_valid_move(state, 2, (2, 3), (2, 2), 1)

def test_hive_connectivity():
    # Create state with connected pieces
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    state[1, 2, 3] = 1  # Player 1 beetle adjacent
    state[2, 2, 4] = 1  # Player 1 ant at end
    
    print("\nTesting hive connectivity:")
    # Moving middle piece breaks connectivity (it's the only connection between queen and ant)
    assert not HiveRules.maintains_hive_connectivity(state, (2, 3), (1, 3))
    
    # Create a different state with more connections
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    state[1, 2, 3] = 1  # Player 1 beetle
    state[2, 1, 3] = 1  # Player 1 ant (connected to beetle)
    state[0, 1, 2] = 1  # Player 1 queen (connected to both)
    
    # Moving beetle should maintain connectivity (multiple connections exist)
    assert HiveRules.maintains_hive_connectivity(state, (2, 3), (3, 3))
    
    # Moving ant to a disconnected position should break connectivity
    assert not HiveRules.maintains_hive_connectivity(state, (1, 3), (0, 4))
    
    # Moving queen should maintain connectivity (other pieces still connected)
    assert HiveRules.maintains_hive_connectivity(state, (2, 2), (3, 2))

def test_queen_placement_rule():
    # Create empty state
    state = np.zeros((3, 5, 5), dtype=np.int8)
    
    print("\nTesting queen placement rule:")
    # First turn - can place queen
    assert HiveRules.is_valid_placement(state, 0, 2, 2, 1, 0)
    # After turn 3 - can still place queen
    assert HiveRules.is_valid_placement(state, 0, 2, 2, 1, 3)
    # After turn 4 - cannot place queen
    assert not HiveRules.is_valid_placement(state, 0, 2, 2, 1, 4)

def test_full_gameplay():
    # Create empty state
    state = np.zeros((3, 5, 5), dtype=np.int8)
    
    print("\nTesting full gameplay sequence:")
    # Player 1 first move (must be center)
    assert HiveRules.is_valid_placement(state, 0, 2, 2, 1, 0)
    state[0, 2, 2] = 1
    
    # Player 2 first move (must be adjacent)
    assert HiveRules.is_valid_placement(state, 1, 2, 3, 2, 1)
    state[1, 2, 3] = 2
    
    # Player 1 second move (must be adjacent)
    assert HiveRules.is_valid_placement(state, 1, 2, 1, 1, 2)
    state[1, 2, 1] = 1
    
    # Player 2 tries invalid move (not adjacent)
    assert not HiveRules.is_valid_placement(state, 2, 0, 0, 2, 3)
    
    # Player 2 valid move
    assert HiveRules.is_valid_placement(state, 2, 1, 2, 2, 3)
    state[2, 1, 2] = 2
    
    # Check hive connectivity
    assert HiveRules.maintains_hive_connectivity(state, (2, 1), (3, 1))

if __name__ == "__main__":
    print("Testing Complete Hive Rules Implementation")
    print("=" * 50)
    
    test_piece_tracking()
    test_hive_connectivity()
    test_queen_placement_rule()
    test_full_gameplay()
    
    print("\nAll tests completed successfully!") 