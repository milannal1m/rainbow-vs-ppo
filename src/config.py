import os

DATE_FORMAT          = "%Y-%m-%d_%H-%M-%S"
RUNS_DIR             = "runs"
CHECKPOINT_EVERY     = 1000
REPLAY_MEMORY_SEED   = 23
GRAPH_UPDATE_SECONDS = 10
HEADLESS             = os.environ.get("HEADLESS", "0") == "1"
