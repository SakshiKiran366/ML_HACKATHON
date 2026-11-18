"""HMM + Viterbi route reconstruction on top of the classifier's candidate probabilities.

Hidden state      : the candidate arc chosen for a GPS observation (<=16 states per observation;
                    the candidate-to-arc correspondence is kept: state k of observation t IS arc_id[t][k]).
Emission          : log p_k / sum_j p_j  (classifier probability normalised over the observation's
                    candidates), scaled by `alpha`; floored at 1e-6.
Transition t-1->t : Newson-Krumm style. d_net = road-network distance travelled from the previous
                    candidate (at its projected position) to the current candidate (at its projected
                    position); delta = straight-line distance between the two GPS points.
                       cost = |d_net - delta| / beta
                    + backward   penalty  if d_net < -tol (moving backwards on the same arc)
                    + unreachable penalty if the arc cannot be reached within REACH_M metres
                    + u-turn     penalty  if the arcs are exact reverses of each other.
Gaps              : if the time gap exceeds GAP_S seconds the transition costs are zeroed (restart:
                    the decoder trusts emissions only for that step).
Missing candidates: observations with no candidate row are skipped (no state); the next observation
                    is then treated as a gap.
Viterbi recurrence: delta_t(b) = alpha*log e_t(b) + max_a [ delta_{t-1}(a) - cost(a, b) ],
                    with back-pointers; the path is recovered from argmax delta_T.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np

GAP_S = 5.0
REACH_M = 600.0
BACK_TOL_M = 8.0
DEFAULT_PARAMS = dict(alpha=1.0, beta=30.0, p_unreach=12.0, p_back=8.0, p_uturn=4.0)


class TripGraph:
    """Directed road graph of one trip with bounded Dijkstra (cached per start node)."""

    def __init__(self, arcs: np.ndarray, L: np.ndarray):
        self.af, self.at, self.L = arcs[:, 0], arcs[:, 1], L
        self.out = defaultdict(list)
        for a, (u, v) in enumerate(arcs):
            self.out[int(u)].append((int(v), float(L[a])))
        self._cache: dict[int, dict] = {}

    def reach(self, node: int) -> dict:
        r = self._cache.get(node)
        if r is None:
            r, heap = {node: 0.0}, [(0.0, node)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > r.get(u, np.inf):
                    continue
                for v, w in self.out.get(u, ()):
                    nd = d + w
                    if nd <= REACH_M and nd < r.get(v, np.inf):
                        r[v] = nd
                        heapq.heappush(heap, (nd, v))
            self._cache[node] = r
        return r


class Trellis:
    """Parameter-independent part of the HMM for one trip (built once, decoded many times)."""

    def __init__(self, graph: TripGraph, P, T, obs_idx, cand_arcs, cand_frac, cand_prob):
        """obs_idx: sorted observation indices that have candidates; cand_*: list of arrays per obs."""
        self.obs_idx, self.arcs, self.prob = obs_idx, cand_arcs, cand_prob
        self.D, self.U, self.delta, self.gap = [None], [None], [0.0], [False]
        g = graph
        for t in range(1, len(obs_idx)):
            ia, ib = obs_idx[t - 1], obs_idx[t]
            a_arc, b_arc = cand_arcs[t - 1], cand_arcs[t]
            fa, fb = cand_frac[t - 1], cand_frac[t]
            D = np.full((len(a_arc), len(b_arc)), np.inf, np.float32)
            U = np.zeros((len(a_arc), len(b_arc)), bool)
            for i, a in enumerate(a_arc):
                reach = g.reach(int(g.at[a]))
                la = (1.0 - fa[i]) * g.L[a]
                for j, b in enumerate(b_arc):
                    if a == b:
                        D[i, j] = (fb[j] - fa[i]) * g.L[a]
                    else:
                        r = reach.get(int(g.af[b]))
                        if r is not None:
                            D[i, j] = la + r + fb[j] * g.L[b]
                            U[i, j] = g.af[b] == g.at[a] and g.at[b] == g.af[a]
            self.D.append(D); self.U.append(U)
            self.delta.append(float(np.linalg.norm(P[ib] - P[ia])))
            self.gap.append(bool(T[ib] - T[ia] > GAP_S))

    def decode(self, alpha, beta, p_unreach, p_back, p_uturn):
        n = len(self.obs_idx)
        if n == 0:
            return np.empty(0, int), []
        emis = [alpha * np.log(np.maximum(p / max(p.sum(), 1e-12), 1e-6)) for p in self.prob]
        score = emis[0].copy()
        back = [None]
        for t in range(1, n):
            if self.gap[t]:
                cost = np.zeros_like(self.D[t])
            else:
                D = self.D[t]
                cost = np.abs(D - self.delta[t]) / beta
                cost = np.where(D < -BACK_TOL_M, p_back, cost)
                cost = np.where(np.isinf(D), p_unreach, cost)
                cost = cost + self.U[t] * p_uturn
            tot = score[:, None] - cost
            k = tot.argmax(0)
            score = tot[k, np.arange(tot.shape[1])] + emis[t]
            back.append(k)
        path = np.empty(n, int)
        path[-1] = int(score.argmax())
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t][path[t]]
        return path, [self.arcs[t][path[t]] for t in range(n)]

    def classifier_only(self):
        return [self.arcs[t][int(np.argmax(self.prob[t]))] for t in range(len(self.obs_idx))]

    def reachable_fraction(self, chosen):
        """Share of consecutive chosen arcs that are reachable on the network (continuity, relaxed)."""
        ok, tot = 0, 0
        for t in range(1, len(chosen)):
            if self.gap[t]:
                continue
            a = np.where(self.arcs[t - 1] == chosen[t - 1])[0][0]
            b = np.where(self.arcs[t] == chosen[t])[0][0]
            d = self.D[t][a, b]
            ok += bool(np.isfinite(d) and d >= -BACK_TOL_M); tot += 1
        return ok / tot if tot else float("nan")


def strict_continuity(chosen, graph: TripGraph):
    """Share of consecutive chosen arcs that are identical or directly head-to-tail connected."""
    c = np.asarray(chosen)
    if len(c) < 2:
        return float("nan")
    same = c[1:] == c[:-1]
    linked = graph.at[c[:-1]] == graph.af[c[1:]]
    return float((same | linked).mean())


def trellis_from_table(graph, P, T, rows, prob):
    """rows: candidate DataFrame of ONE trip with obs_idx, arc_id, proj_frac; prob aligned to rows."""
    obs, arcs, frac, pr = [], [], [], []
    r_obs = rows["obs_idx"].to_numpy()
    order = np.argsort(r_obs, kind="stable")
    r_obs = r_obs[order]
    a_all = rows["arc_id"].to_numpy()[order]; f_all = rows["proj_frac"].to_numpy(float)[order]
    p_all = np.asarray(prob)[order]
    bounds = np.flatnonzero(np.r_[True, r_obs[1:] != r_obs[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        obs.append(int(r_obs[s])); arcs.append(a_all[s:e]); frac.append(f_all[s:e]); pr.append(p_all[s:e])
    return Trellis(graph, P, T, obs, arcs, frac, pr)
