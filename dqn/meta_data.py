from dataclasses import dataclass


@dataclass
class ExperimentData:
    env_id: str
    buffer_size: int = 100000
    learning_rate: float = 1e-4
    total_timesteps: int = 1000000
    start_e: int = 1
    end_e: int = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4
    gamma: float = 0.99
    target_network_frequency: int = 1000
    tau: float = 1.0
    batch_size: int = 32
    seed: int = 42
    num_envs: int = 1
    eval_episodes: int = 10
    eval_frequency: int = 50000
    bayesian_log: int = 0
