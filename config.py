# config.py
CONFIG = {
    "ENV_SIZE": 21,
    "FORBIDDEN_FRAC": 0.12,
    "MAX_STEPS": 50,
    "EPISODES": 400,
    "MANAGER_INTERVAL": 5,
    "WORKER_LR": 0.16,
    "GAMMA": 0.95,
    "EPS_START": 0.4,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.9995,
    "REPLAY_CAP": 5000,
    "PRIO_CAP": 4000,
    "HER_K": 6,
    "PRIO_ALPHA": 0.6,
    "PRIO_BETA": 0.4,
    "SEED": 1,
    "ONLINE_BATCH_SIZE": 4,
    "PRIO_BATCH_SIZE": 8,
    "MANAGER_EPS": 0.2,
    "MANAGER_LR": 0.1,
    "MANAGER_STATE_QUANTIZATION": 5
}
