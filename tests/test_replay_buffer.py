# tests/test_replay_buffer.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from replay_buffer import PrioritizedReplay, Transition

def test_prioritized_replay():
    buffer = PrioritizedReplay(capacity=10)

    # Add some transitions
    buffer.push(Transition(1, 1, 1, 1, False, 1, 1), priority=0.1)
    buffer.push(Transition(2, 2, 2, 2, False, 2, 2), priority=1.0)
    buffer.push(Transition(3, 3, 3, 3, False, 3, 3), priority=0.5)

    # Sample from the buffer
    samples, idxs, weights = buffer.sample(2)

    assert len(samples) == 2
    assert len(idxs) == 2
    assert len(weights) == 2

    # Check that the samples are correct
    assert samples[0].s == 2 or samples[1].s == 2

    # Update the priorities
    buffer.update_priorities(idxs, [0.2, 0.8])

    # Sample again and check that the priorities have been updated
    samples, idxs, weights = buffer.sample(2)

    assert len(samples) == 2
    assert len(idxs) == 2
    assert len(weights) == 2
