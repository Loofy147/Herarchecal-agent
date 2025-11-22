import pytest
import numpy as np

from replay_buffer import Replay, PrioritizedReplay, Transition

def test_replay_buffer():
    buffer = Replay(capacity=10)
    for i in range(5):
        buffer.push(Transition((i,), i, i, (i,), False, i, i))
    assert len(buffer) == 5
    sample = buffer.sample(3)
    assert len(sample) == 3

def test_prioritized_replay_buffer():
    buffer = PrioritizedReplay(capacity=10, alpha=1.0, beta=1.0)
    t1 = Transition((1,), 1, 1, (1,), False, 1, 1)
    t2 = Transition((2,), 2, 2, (2,), False, 2, 2)
    t3 = Transition((3,), 3, 3, (3,), False, 3, 3)

    buffer.push(t1, priority=10.0)
    buffer.push(t2, priority=1.0)
    buffer.push(t3, priority=5.0)

    assert len(buffer) == 3

    samples, idxs, weights = buffer.sample(1)
    assert len(samples) == 1
    assert samples[0] == t1

    buffer.update_priorities(idxs, [0.1])
    samples, _, _ = buffer.sample(1)
    assert samples[0] != t1

def test_prioritized_replay_overwrite_bug():
    buffer = PrioritizedReplay(capacity=3)
    buffer.push(Transition(s=(0,), a=0, r=0, s2=(0,), done=False, goal=0, pos2=0), priority=1)
    buffer.push(Transition(s=(1,), a=1, r=1, s2=(1,), done=False, goal=1, pos2=1), priority=1)
    buffer.push(Transition(s=(2,), a=2, r=2, s2=(2,), done=False, goal=2, pos2=2), priority=1)
    # Buffer is now [T(0), T(1), T(2)]
    buffer.push(Transition(s=(3,), a=3, r=3, s2=(3,), done=False, goal=3, pos2=3), priority=1)
    # Due to the bug, the buffer is now [T(3), T(1), T(2)]. T(0) was correctly replaced.
    buffer.push(Transition(s=(4,), a=4, r=4, s2=(4,), done=False, goal=4, pos2=4), priority=1)
    # Due to the bug, the buffer is now [T(4), T(1), T(2)]. T(1) should have been replaced, but wasn't.

    states_in_buffer = [t.s[0] for t in buffer.buffer]
    assert 1 not in states_in_buffer
