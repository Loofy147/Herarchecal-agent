import numpy as np
import pytest

from ..core import identify_phase, get_hierarchical_state_representation, decompose_target, get_shaped_reward

def test_identify_phase():
    assert identify_phase(101, 200) == 0
    assert identify_phase(50, 200) == 1
    assert identify_phase(5, 200) == 2

def test_get_hierarchical_state_representation():
    state = get_hierarchical_state_representation(100, 200, 10, 100, {150}, [1, 5, 10])
    assert isinstance(state, np.ndarray)
    assert state.shape == (12,)

def test_decompose_target():
    assert decompose_target(0, 100) == [50, 100]
    assert decompose_target(0, 40) == [40]

def test_get_shaped_reward():
    assert get_shaped_reward(10, 20, 100, 1, False, {}) > 0
    assert get_shaped_reward(20, 10, 100, 1, False, {}) < 0
    assert get_shaped_reward(100, 100, 100, 1, True, {'status': 'SUCCESS'}) > 0
    assert get_shaped_reward(100, 100, 100, 1, True, {'status': 'FORBIDDEN'}) < 0
