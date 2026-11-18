"""Candidate generation + labels + features for one trip.

Everything except `align_labels` works from .track/.nodes/.arcs ONLY, so the API can reuse it at
inference time. `.route` (reference) is used only to build training labels / server-side scoring.

Steps
 1. read + clean the track, project lon/lat to local metres (per trip; trips are in different cities)
 2. candidates: every arc within R_M metres of the GPS point (nearest K_MAX kept), spatial index
 3. labels : monotone alignment (Viterbi over route position) of GPS points to the route arcs
 4. features: geometry, heading, speed, time gaps, neighbour-point context, connectivity
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_M = 60.0          # candidate search radius (metres)
K_MAX = 16          # max candidates per GPS point
LONG_ARC_M = 150.0  # arcs longer than this are checked by brute force (KD-tree uses midpoints)
HEAD_MIN_SPEED = 1.5  # m/s; below this GPS heading is unreliable
GAP_S = 5.0         # time gap (s) treated as a GPS gap in label alignment

FEATURES = ["dist_m", "proj_frac", "along_m", "arc_len_m", "dist_rank", "dist_gap", "dist_ratio",
            "n_cands", "min_dist_m", "speed_mps", "heading_valid", "cos_heading", "abs_heading_deg",
            "dt_prev_s", "dt_next_s", "d_prev1", "d_next1", "d_prev3", "d_next3",
            "same_prev_best", "adj_prev_best", "same_next_best", "adj_next_best",
            "out_deg_to", "in_deg_from"]


# ------------------------------------------------------------------ IO and geometry
def read_trip(trip_dir: Path, include_route: bool = True):
    tid = trip_dir.name
    rd = lambda ext, **kw: pd.read_csv(trip_dir / f"{tid}.{ext}", sep=r"\s+", header=None, **kw)
    track = rd("track").to_numpy(float)
    nodes = rd("nodes").to_numpy(float)
    arcs = rd("arcs").to_numpy(int)
    rpath = trip_dir / f"{tid}.route"
    route = rd("route").to_numpy(int).ravel() if include_route and rpath.exists() else None
    return track, nodes, arcs, route


def clean_track(track: np.ndarray):
    """Drop non-finite rows and non-increasing timestamps. Returns cleaned array + n_dropped."""
    ok = np.isfinite(track).all(1)
    t = track[ok]
    keep = np.ones(len(t), bool)
    last = -np.inf
    for i, ts in enumerate(t[:, 2]):
        if ts <= last:
            keep[i] = False
        else:
            last = ts
    return t[keep], len(track) - int(keep.sum())


def project(lonlat, lon0, lat0):
    return np.c_[(lonlat[:, 0] - lon0) * 111320.0 * np.cos(np.radians(lat0)),
                 (lonlat[:, 1] - lat0) * 110574.0]


def seg_dist(P, A, B):
    AB = B - A
    t = np.clip(((P - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-9), 0.0, 1.0)
    return np.linalg.norm(P - (A + AB * t[..., None]), axis=-1), t


# ------------------------------------------------------------------ candidates
def generate_candidates(P, A, B, R=R_M, K=K_MAX):
    """Return (pt_idx, arc_idx, dist, t) sorted by (pt, dist), at most K per point."""
    L = np.linalg.norm(B - A, axis=1)
    mid = (A + B) / 2
    short = np.where(L <= LONG_ARC_M)[0]
    long_ = np.where(L > LONG_ARC_M)[0]
    pts, arcs = [], []
    if len(short):
        lists = cKDTree(mid[short]).query_ball_point(P, R + LONG_ARC_M / 2)
        cnt = np.fromiter((len(x) for x in lists), int, len(P))
        if cnt.sum():
            pts.append(np.repeat(np.arange(len(P)), cnt))
            arcs.append(short[np.concatenate([np.asarray(x, int) for x in lists if len(x)])])
    for lo in range(0, len(P), 2000):  # long arcs: brute force in chunks
        if len(long_):
            hi = min(lo + 2000, len(P))
            d, _ = seg_dist(P[lo:hi, None, :], A[long_][None], B[long_][None])
            pi, aj = np.where(d <= R)
            pts.append(pi + lo); arcs.append(long_[aj])
    pt = np.concatenate(pts) if pts else np.empty(0, int)
    ar = np.concatenate(arcs) if arcs else np.empty(0, int)
    d, t = seg_dist(P[pt], A[ar], B[ar]) if len(pt) else (np.empty(0), np.empty(0))
    keep = d <= R
    pt, ar, d, t = pt[keep], ar[keep], d[keep], t[keep]
    # points with no arc within R: fall back to their 3 nearest arcs so every point has candidates
    missing = np.setdiff1d(np.arange(len(P)), pt)
    if len(missing):
        kq = min(40, len(A))
        _, nn = cKDTree(mid).query(P[missing], k=kq)
        nn = np.atleast_2d(nn).reshape(len(missing), -1)
        mp = np.repeat(missing, nn.shape[1]); ma = nn.ravel()
        md, mt = seg_dist(P[mp], A[ma], B[ma])
        o = np.lexsort((md, mp)); mp, ma, md, mt = mp[o], ma[o], md[o], mt[o]
        first = np.searchsorted(mp, missing)  # each missing point has kq >= 3 rows here
        sel = (first[:, None] + np.arange(min(3, kq))[None]).ravel()
        pt = np.concatenate([pt, mp[sel]]); ar = np.concatenate([ar, ma[sel]])
        d = np.concatenate([d, md[sel]]); t = np.concatenate([t, mt[sel]])
    o = np.lexsort((d, pt))
    pt, ar, d, t = pt[o], ar[o], d[o], t[o]
    first = np.searchsorted(pt, np.arange(len(P)))
    rank = np.arange(len(pt)) - first[pt] + 1
    k = rank <= K
    return pt[k], ar[k], d[k], t[k], rank[k]


# ------------------------------------------------------------------ labels
def gps_motion(P, T):
    n = len(P)
    i = np.arange(n)
    im, ip = np.maximum(i - 1, 0), np.minimum(i + 1, n - 1)
    im2, ip2 = np.maximum(i - 2, 0), np.minimum(i + 2, n - 1)
    dt = T[ip] - T[im]
    speed = np.where(dt > 0, np.linalg.norm(P[ip] - P[im], axis=1) / np.where(dt > 0, dt, 1), 0.0)
    hv = P[ip2] - P[im2]
    hn = np.linalg.norm(hv, axis=1)
    valid = (speed >= HEAD_MIN_SPEED) & (hn > 1e-6)
    h = np.where(valid[:, None], hv / np.maximum(hn, 1e-9)[:, None], 0.0)
    return speed, h, valid


def align_labels(P, T, A, B, route, head_w=15.0, c1=2.0, c2=12.0, c3=30.0, gap_c=5.0):
    """Monotone alignment of GPS points to route positions (stay / +1 / +2 / +3, or any forward
    jump across a time gap). Cost = distance to the arc + heading mismatch. Returns aligned arc id
    per point plus the distance from each point to its aligned arc."""
    n, R = len(P), len(route)
    RA, RB = A[route], B[route]
    u = RB - RA
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    _, h, valid = gps_motion(P, T)
    e = np.empty((n, R), np.float32)
    step = max(1, int(4_000_000 // R))
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        d, _ = seg_dist(P[lo:hi, None, :], RA[None], RB[None])
        cosd = h[lo:hi] @ u.T
        e[lo:hi] = d + np.where(valid[lo:hi, None], head_w * (1.0 - cosd), 0.0)
    ar = np.arange(R)
    src = np.zeros((n, R), np.int32)
    prev = e[0].astype(np.float64)
    inf = np.inf
    for i in range(1, n):
        if T[i] - T[i - 1] > GAP_S:  # GPS gap: allow any forward jump
            cm = np.minimum.accumulate(prev)
            s = np.maximum.accumulate(np.where(prev <= cm, ar, 0))
            cost = cm + gap_c
        else:
            opts = [prev]
            for sh, c in ((1, c1), (2, c2), (3, c3)):
                opts.append(np.concatenate([np.full(sh, inf), prev[:-sh]]) + c if R > sh else np.full(R, inf))
            stack = np.stack(opts)
            k = stack.argmin(0)
            cost = stack[k, ar]
            s = ar - k
        src[i] = s
        prev = cost + e[i]
    j = np.empty(n, np.int64)
    j[-1] = int(prev.argmin())
    for i in range(n - 1, 0, -1):
        j[i - 1] = src[i, j[i]]
    d_al, _ = seg_dist(P, RA[j], RB[j])
    return route[j], d_al


# ------------------------------------------------------------------ main entry
def build_trip(track, nodes, arcs, route=None, trip_id=0, partition="train", R=R_M, K=K_MAX):
    track, n_dropped = clean_track(track)
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    P = project(track[:, :2], lon0, lat0)
    T = track[:, 2]
    N = project(nodes, lon0, lat0)
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    n = len(P)
    pt, ar, d, t, rank = generate_candidates(P, A, B, R, K)
    L = np.linalg.norm(B - A, axis=1)
    speed, h, valid = gps_motion(P, T)
    arc_u = (B - A) / np.maximum(L, 1e-9)[:, None]
    cosh = np.where(valid[pt], (h[pt] * arc_u[ar]).sum(1), 0.0)
    first = np.searchsorted(pt, np.arange(n))
    mind = d[first]
    ncand = np.bincount(pt, minlength=n)
    # neighbour-point context: distance of neighbouring GPS points to the same arc
    i = np.arange(n)
    ctx = {}
    for name, idx in (("d_prev1", np.maximum(i - 1, 0)), ("d_next1", np.minimum(i + 1, n - 1)),
                      ("d_prev3", np.maximum(i - 3, 0)), ("d_next3", np.minimum(i + 3, n - 1))):
        ctx[name] = seg_dist(P[idx[pt]], A[ar], B[ar])[0]
    best = ar[first]  # nearest candidate arc per point (label-free)
    af, at = arcs[:, 0], arcs[:, 1]
    def rel(bidx):
        b = best[bidx][pt]
        return (ar == b), ((ar == b) | (at[ar] == af[b]) | (af[ar] == at[b]))
    same_p, adj_p = rel(np.maximum(i - 1, 0))
    same_n, adj_n = rel(np.minimum(i + 1, n - 1))
    out_deg = np.bincount(af, minlength=len(nodes)); in_deg = np.bincount(at, minlength=len(nodes))
    dtp = np.r_[0.0, np.diff(T)]; dtn = np.r_[np.diff(T), 0.0]
    df = pd.DataFrame({
        "trip_id": np.int32(trip_id), "obs_idx": pt.astype(np.int32), "arc_id": ar.astype(np.int32),
        "arc_from": af[ar].astype(np.int32), "arc_to": at[ar].astype(np.int32)})
    stats = {"n_points": n, "n_dropped_points": n_dropped, "n_nodes": len(nodes), "n_arcs": len(arcs)}
    keep_obs = np.ones(n, bool)
    if route is not None:
        aligned, d_al = align_labels(P, T, A, B, route)
        # Observations farther than R from their aligned reference arc have an unreliable reference
        # (e.g. breaks in the supplied route): excluded from the table and counted in the stats.
        keep_obs = d_al <= R
        df["label"] = (ar == aligned[pt]).astype(np.int8)
        m = df["label"].to_numpy() == 1
        lab_rank = np.full(n, 99); lab_rank[pt[m]] = rank[m]
        stats.update({"route_len": len(route), "unref_dropped_frac": float(1 - keep_obs.mean()),
                      "label_found_frac": float((lab_rank[keep_obs] < 99).mean()),
                      "aligned_d_mean": float(d_al[keep_obs].mean()),
                      "aligned_d_p95": float(np.percentile(d_al[keep_obs], 95)),
                      "aligned_within30": float((d_al[keep_obs] <= 30).mean()),
                      **{f"miss_at_{k}": float((lab_rank[keep_obs] > k).mean()) for k in (1, 3, 5, 8, 12, 16)}})
    df["partition"] = partition
    feats = {
        "dist_m": d, "proj_frac": t, "along_m": t * L[ar], "arc_len_m": L[ar], "dist_rank": rank,
        "dist_gap": d - mind[pt], "dist_ratio": d / (mind[pt] + 1.0), "n_cands": ncand[pt],
        "min_dist_m": mind[pt], "speed_mps": speed[pt], "heading_valid": valid[pt],
        "cos_heading": cosh, "abs_heading_deg": np.degrees(np.arccos(np.clip(cosh, -1, 1))) * valid[pt],
        "dt_prev_s": dtp[pt], "dt_next_s": dtn[pt], **ctx,
        "same_prev_best": same_p, "adj_prev_best": adj_p, "same_next_best": same_n, "adj_next_best": adj_n,
        "out_deg_to": out_deg[at[ar]], "in_deg_from": in_deg[af[ar]]}
    for k in FEATURES:
        df[k] = np.asarray(feats[k], np.float32)
    df = df[keep_obs[pt]].reset_index(drop=True)
    return df, stats
