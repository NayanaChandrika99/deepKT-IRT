import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from docs.scripts.exporters.export_ucb_gauge import export_ucb_gauge_for_testing

def test_ucb_gauge_schema():
    """Test UCB gauge export has required fields"""
    # Create minimal test data
    bandit_state = {
        'user_counts': np.array([[5, 3, 10]]),  # 1 user, 3 items
        'user_sum_rewards': np.array([[4.2, 2.1, 8.5]]),
        'user_id_map': {0: '1006OOQBE9'}
    }

    result = export_ucb_gauge_for_testing(bandit_state, alpha=1.0)

    # Assertions
    assert 'data' in result
    assert len(result['data']) >= 1

    student_data = result['data'][0]
    assert 'user_id' in student_data
    assert 'top_recommendation' in student_data

    top_rec = student_data['top_recommendation']
    assert 'item_id' in top_rec
    assert 'expected_reward' in top_rec
    assert 'uncertainty' in top_rec
    assert 'ucb_score' in top_rec
    assert 'confidence_pct' in top_rec
    assert 'mode' in top_rec
    assert top_rec['mode'] in ['explore', 'exploit']

    # Validate ranges
    assert 0 <= top_rec['expected_reward'] <= 1
    assert top_rec['uncertainty'] >= 0
    assert 0 <= top_rec['confidence_pct'] <= 100

def test_ucb_exploration_vs_exploitation():
    """Test mode determination based on uncertainty"""
    # High uncertainty -> explore
    bandit_high_unc = {
        'user_counts': np.array([[2]]),  # Low count = high uncertainty
        'user_sum_rewards': np.array([[1.5]]),
        'user_id_map': {0: 'test_user'}
    }

    result = export_ucb_gauge_for_testing(bandit_high_unc, alpha=2.0)
    assert result['data'][0]['top_recommendation']['mode'] == 'explore'

    # Low uncertainty -> exploit
    bandit_low_unc = {
        'user_counts': np.array([[100]]),  # High count = low uncertainty
        'user_sum_rewards': np.array([[85.0]]),
        'user_id_map': {0: 'test_user'}
    }

    result = export_ucb_gauge_for_testing(bandit_low_unc, alpha=2.0)
    assert result['data'][0]['top_recommendation']['mode'] == 'exploit'

def test_ucb_metadata():
    """Test metadata includes exploration ratio"""
    bandit_state = {
        'user_counts': np.array([[10, 5, 20]]),
        'user_sum_rewards': np.array([[8.0, 3.0, 18.0]]),
        'user_id_map': {0: 'test_user'}
    }

    result = export_ucb_gauge_for_testing(bandit_state, alpha=1.5)

    assert 'metadata' in result
    assert 'exploration_ratio' in result['metadata']
    assert 'mock' in result['metadata']
    assert result['metadata']['mock'] is False
