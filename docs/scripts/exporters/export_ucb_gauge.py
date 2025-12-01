# docs/scripts/exporters/export_ucb_gauge.py
# ABOUTME: UCB confidence gauge exporter for LinUCB bandit recommendations
# ABOUTME: Computes expected rewards, uncertainty, and exploration/exploitation mode

import numpy as np
from pathlib import Path
from typing import Dict, Optional

def compute_ucb_scores(counts: np.ndarray, sum_rewards: np.ndarray, alpha: float = 1.0) -> Dict:
    """
    Compute UCB scores for LinUCB bandit.

    Args:
        counts: Array of shape (n_users, n_items) with interaction counts
        sum_rewards: Array of shape (n_users, n_items) with sum of rewards
        alpha: Exploration parameter (higher = more exploration)

    Returns:
        Dict with expected, uncertainty, ucb_score, mode for each item
    """
    # Avoid division by zero
    counts_safe = np.maximum(counts, 1)

    # Expected reward = average reward
    expected = sum_rewards / counts_safe

    # Uncertainty = sqrt(alpha / count) - UCB1 formula
    uncertainty = np.sqrt(alpha / counts_safe)

    # UCB score = expected + uncertainty
    ucb_score = expected + uncertainty

    return {
        'expected': expected,
        'uncertainty': uncertainty,
        'ucb_score': ucb_score
    }

def determine_mode(expected: float, uncertainty: float, threshold: float = 0.2) -> str:
    """
    Determine if recommendation is explore or exploit.

    Explore if uncertainty is significant compared to expected reward.
    """
    if expected == 0:
        return 'explore'

    uncertainty_ratio = uncertainty / (expected + 1e-6)
    return 'explore' if uncertainty_ratio > threshold else 'exploit'

def export_ucb_gauge_for_testing(bandit_state: Dict, alpha: float = 1.0) -> Dict:
    """
    Export UCB gauge data from in-memory bandit state (for testing).

    Args:
        bandit_state: Dict with keys:
            - user_counts: np.ndarray of shape (n_users, n_items)
            - user_sum_rewards: np.ndarray of shape (n_users, n_items)
            - user_id_map: Dict mapping indices to user IDs
        alpha: Exploration parameter

    Returns:
        Dict with data and metadata
    """
    counts = bandit_state['user_counts']
    sum_rewards = bandit_state['user_sum_rewards']
    user_id_map = bandit_state['user_id_map']

    ucb_data = compute_ucb_scores(counts, sum_rewards, alpha)

    data = []
    explore_count = 0
    total_count = 0

    for user_idx, user_id in user_id_map.items():
        # Get top recommendation for this user
        user_ucb_scores = ucb_data['ucb_score'][user_idx]
        top_item_idx = int(np.argmax(user_ucb_scores))

        expected = float(ucb_data['expected'][user_idx, top_item_idx])
        uncertainty = float(ucb_data['uncertainty'][user_idx, top_item_idx])
        ucb = float(user_ucb_scores[top_item_idx])

        mode = determine_mode(expected, uncertainty)

        # Confidence = expected reward minus uncertainty (clamped to 0-100%)
        confidence_pct = max(0, min(100, (expected - uncertainty) * 100))

        if mode == 'explore':
            explore_count += 1
        total_count += 1

        data.append({
            'user_id': str(user_id),
            'top_recommendation': {
                'item_id': f'item_{top_item_idx}',
                'expected_reward': expected,
                'uncertainty': uncertainty,
                'ucb_score': ucb,
                'confidence_pct': confidence_pct,
                'mode': mode
            }
        })

    exploration_ratio = explore_count / total_count if total_count > 0 else 0.0

    return {
        'data': data,
        'metadata': {
            'alpha': alpha,
            'exploration_ratio': exploration_ratio,
            'mock': False
        }
    }

def export_ucb_gauge(reports_dir: Path, alpha: float = 1.0) -> Dict:
    """
    Export UCB gauge data from NPZ files.

    Args:
        reports_dir: Path to reports directory containing bandit_state.npz
        alpha: Exploration parameter

    Returns:
        Dict with data and metadata for JSON export
    """
    bandit_path = reports_dir / "bandit_state.npz"

    if not bandit_path.exists():
        raise FileNotFoundError(f"Bandit state not found: {bandit_path}")

    # Load NPZ file
    bandit_data = np.load(bandit_path)

    # Extract arrays (adjust keys based on actual NPZ structure)
    bandit_state = {
        'user_counts': bandit_data['user_counts'],
        'user_sum_rewards': bandit_data['user_sum_rewards'],
        'user_id_map': bandit_data.get('user_id_map', {}).item() if 'user_id_map' in bandit_data else {}
    }

    # If no user_id_map, create default mapping
    if not bandit_state['user_id_map']:
        n_users = bandit_state['user_counts'].shape[0]
        bandit_state['user_id_map'] = {i: f'user_{i}' for i in range(min(n_users, 5))}

    return export_ucb_gauge_for_testing(bandit_state, alpha)
