from hive_rl.environment.hive_env import HiveEnv

def test_environment():
    # Create environment
    env = HiveEnv()
    
    # Test reset
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial info:", info)
    
    # Test action space
    print("\nAction space:", env.action_space)
    
    # Test observation space
    print("\nObservation space:", env.observation_space)
    
    # Test a few random actions
    print("\nTesting random actions:")
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Info: {info}")
        print("---")

if __name__ == "__main__":
    test_environment() 