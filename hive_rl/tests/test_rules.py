import numpy as np
from hive_rl.environment.hive_rules import HiveRules

def test_placement_rules():
    # Create empty state
    state = np.zeros((3, 5, 5), dtype=np.int8)
    
    # Test first piece placement
    print("Testing first piece placement:")
    # Should only be valid at center
    assert HiveRules.is_valid_placement(state, 0, 2, 2, 1)  # Valid
    assert not HiveRules.is_valid_placement(state, 0, 0, 0, 1)  # Invalid
    
    # Place first piece
    state[0, 2, 2] = 1
    
    # Test subsequent placements
    print("\nTesting subsequent placements:")
    # Should be valid adjacent to existing piece
    assert HiveRules.is_valid_placement(state, 1, 2, 3, 1)  # Valid
    assert not HiveRules.is_valid_placement(state, 1, 0, 0, 1)  # Invalid
    
    # Test occupied position
    print("\nTesting occupied positions:")
    assert not HiveRules.is_valid_placement(state, 1, 2, 2, 1)  # Invalid

def test_movement_rules():
    # Create state with some pieces
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    state[1, 2, 3] = 1  # Player 1 beetle adjacent
    
    print("\nTesting queen movement:")
    # Queen should move one space
    assert HiveRules.is_valid_piece_movement(state, 0, (2, 2), (2, 1))  # Valid
    assert not HiveRules.is_valid_piece_movement(state, 0, (2, 2), (0, 0))  # Invalid
    
    print("\nTesting beetle movement:")
    # Beetle should move one space
    assert HiveRules.is_valid_piece_movement(state, 1, (2, 3), (2, 4))  # Valid
    assert not HiveRules.is_valid_piece_movement(state, 1, (2, 3), (0, 0))  # Invalid

def test_queen_surrounded():
    # Create state with surrounded queen
    state = np.zeros((3, 5, 5), dtype=np.int8)
    state[0, 2, 2] = 1  # Player 1 queen at center
    # Surround the queen
    state[1, 2, 1] = 2  # Player 2 pieces
    state[1, 2, 3] = 2
    state[1, 1, 2] = 2
    state[1, 3, 2] = 2
    state[1, 1, 3] = 2
    state[1, 3, 1] = 2
    
    print("\nTesting queen surrounded condition:")
    assert HiveRules.is_queen_surrounded(state, 1)  # Should be surrounded

if __name__ == "__main__":
    print("Testing Hive Rules Implementation")
    print("=" * 50)
    
    test_placement_rules()
    test_movement_rules()
    test_queen_surrounded()
    
    print("\nAll tests completed successfully!") 