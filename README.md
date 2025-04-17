# Hive RL Agent

A reinforcement learning agent trained to play a simplified version of the board game Hive using Proximal Policy Optimization (PPO).

## Project Overview

This project implements a simplified version of Hive with the following characteristics:
- 5x5 hexagonal grid
- 7 pieces per player (1 Queen, 3 Beetles, 3 Ants)
- Core Hive rules maintained (one hive, freedom of movement, queen placement)
- RL agent trained against heuristic bots and through self-play

## Project Structure

```
hive_rl/
├── environment/          # Game environment implementation
│   ├── __init__.py
│   ├── hive_game.py     # Core game logic
│   ├── board.py         # Board representation
│   └── pieces.py        # Piece classes
├── agents/              # Agent implementations
│   ├── __init__.py
│   ├── heuristic_bots.py # Basic bots
│   └── rl_agent.py      # PPO implementation
├── training/            # Training infrastructure
│   ├── __init__.py
│   ├── ppo.py           # PPO algorithm
│   └── trainer.py       # Training loop
└── utils/               # Utility functions
    ├── __init__.py
    ├── visualization.py # Board visualization
    └── metrics.py       # Evaluation metrics
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

[To be added as development progresses]

## Development Status

- [ ] Core game environment
- [ ] Heuristic bots
- [ ] RL implementation
- [ ] Training and evaluation
- [ ] Results analysis 