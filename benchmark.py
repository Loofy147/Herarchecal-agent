import time
import numpy as np
from production_run import train_production_agent

def run_benchmark():
    """Runs a benchmark of the training loop and reports the results."""

    start_time = time.time()

    # Run a shortened training loop for the benchmark
    agent = train_production_agent()

    end_time = time.time()

    duration = end_time - start_time

    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS")
    print("="*70)
    print(f"Total training time for 5000 episodes: {duration:.2f} seconds")
    print(f"Average time per episode: {duration / 5000:.4f} seconds")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
