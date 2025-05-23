project/
├── environments/
│   ├── __init__.py
│   ├── pipetting_env.py          # Interface to partner's MuJoCo env
│   └── reward_functions.py       # Your custom reward function
├── algorithms/
│   ├── __init__.py
│   ├── ppo_agent.py             # Adapted from your ac.py
│   ├── networks.py              # Actor-Critic networks
│   └── buffer.py                # Experience buffer for PPO
├── training/
│   ├── __init__.py
│   ├── train_pipetting.py       # Adapted from your train.py
│   ├── config.py                # Hyperparameters
│   └── evaluation.py            # Testing/evaluation
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Adapted from your logger.py
│   └── visualization.py         # Plot training curves, etc.
└── experiments/
    ├── configs/                 # Different experiment configs
    └── results/                 # Saved models and logs