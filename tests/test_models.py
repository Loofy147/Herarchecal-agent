# tests/test_models.py
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import encode_relational

def test_encode_relational():
    agent = 5
    goal = 15
    forbidden = {10}
    size = 21

    feat = encode_relational(agent, goal, forbidden, size)

    assert isinstance(feat, np.ndarray)
    assert feat.shape == (2 * size + 4,)
    assert np.isclose(np.linalg.norm(feat), 1.0)
