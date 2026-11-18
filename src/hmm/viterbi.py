"""Viterbi decoding for candidate road arcs with direction-aware continuity."""
from __future__ import annotations

import numpy as np

def decode(candidates, probabilities, max_gap_s=5.0):
    """Return candidate-row indices, one per observation.

    Emissions are classifier probabilities. Transitions favour staying on an arc or
    moving head-to-tail to a connected directed arc. A long GPS gap softens continuity.
    """
    if len(candidates) == 0:
        return np.empty(0, dtype=int)
    obs = candidates.obs_idx.to_numpy(dtype=int)
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    groups = [np.arange(a, b) for a, b in zip(starts, ends)]
    logp = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0))
    dp = logp[groups[0]].copy()
    back = []
    for k in range(1, len(groups)):
        prev, cur = groups[k - 1], groups[k]
        dt = float(candidates.iloc[cur[0]].get("dt_prev_s", 1.0))
        continuity_weight = 0.35 if dt > max_gap_s else 2.4
        p_to = candidates.iloc[prev].arc_to.to_numpy()
        p_arc = candidates.iloc[prev].arc_id.to_numpy()
        c_from = candidates.iloc[cur].arc_from.to_numpy()
        c_arc = candidates.iloc[cur].arc_id.to_numpy()
        same_arc = p_arc[:, None] == c_arc[None, :]
        connected = p_to[:, None] == c_from[None, :]
        # Stay is best, directed head-to-tail movement is next, disconnected jumps cost more.
        transition = np.full((len(prev), len(cur)), -continuity_weight * 2.0)
        transition[same_arc] = 0.0
        transition[connected] = -continuity_weight * 0.15
        prev_along = candidates.iloc[prev].along_m.to_numpy()
        prev_len = candidates.iloc[prev].arc_len_m.to_numpy()
        cur_along = candidates.iloc[cur].along_m.to_numpy()
        observed = float(candidates.iloc[cur[0]].get("speed_mps", 0.0)) * max(dt, 0.0)
        expected_same = cur_along[None, :] - prev_along[:, None]
        expected_connected = (prev_len - prev_along)[:, None] + cur_along[None, :]
        expected = np.where(same_arc, expected_same, expected_connected)
        plausible = same_arc | connected
        mismatch = np.maximum(expected - observed, 0.0) + np.maximum(observed - expected, 0.0)
        transition[plausible] -= 0.2 * np.minimum(mismatch[plausible] / max(20.0, observed * 0.5), 5.0)
        score = dp[:, None] + transition
        parent = score.argmax(axis=0)
        back.append(parent)
        dp = score[parent, np.arange(len(cur))] + logp[cur]
    choice = int(dp.argmax())
    decoded = np.empty(len(groups), dtype=int)
    decoded[-1] = groups[-1][choice]
    for k in range(len(groups) - 2, -1, -1):
        choice = int(back[k][choice])
        decoded[k] = groups[k][choice]
    return decoded
