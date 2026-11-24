# lstm_single_bridge.py
"""LSTM-VAE bridges for scenario signatures — exact training contract.

single (lstmvae_single + dataset_single):
    per step: [d_norm, d_dot_norm, psi_norm, psi_dot_norm] + path_emb
    global normalization WITH clipping, 30->10 Hz decimation
pair (lstmvae_pair + dataset_pair, Option B1):
    per step: [v1(4), v2(4), d_rel, psi_rel, |d_rel|, dv] + SHARED path emb
    global normalization WITHOUT clipping

Importable without PyTorch (geometry/IO stay usable; available() == False).
"""
import json
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:  # torch optional at import time
    torch = None
    nn = None
    _TORCH_OK = False


# ---------------- IO ----------------

def load_paths_json(paths_json="cleaned/paths_library.json"):
    """Load the global path cache. [] if missing/invalid."""
    try:
        if os.path.exists(paths_json):
            with open(paths_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("paths", [])
    except Exception:
        pass
    return []


def load_single_model(path, device="cpu", model_cls=None):
    """Load a checkpoint -> (model, norms).

    model_cls: LSTMVAE_Single (default) or LSTMVAE_Pair for pair checkpoints.
    GLOBAL_* must come from the checkpoint (patched trainer saves them);
    the defaults below are fallbacks only.
    """
    assert _TORCH_OK, "PyTorch is required"
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception:  # older checkpoints with non-tensor payloads
        ckpt = torch.load(path, map_location=device, weights_only=False)
    num_paths = ckpt.get("num_paths")
    if num_paths is None:
        raise RuntimeError("Checkpoint missing 'num_paths'.")
    cls = model_cls or LSTMVAE_Single
    model = cls(num_paths=num_paths,
                latent_dim=ckpt.get("latent_dim", 16),
                hidden_dim=ckpt.get("hidden_dim", 128),
                emb_dim=ckpt.get("emb_dim", 8)).to(device)
    model.load_state_dict(ckpt["state_dict"])  # strict
    model.eval()
    cfg_data = (ckpt.get("config") or {}).get("data") or {}
    win = int(ckpt.get("window_size", cfg_data.get("window_size", 40)))
    norms = {
        "lane_half_width_px": ckpt.get("lane_half_width_px",
                                       cfg_data.get("lane_half_width_px", 40.0)),
        "GLOBAL_DDOT": ckpt.get("GLOBAL_DDOT", 0.2),      # patched-trainer value preferred
        "GLOBAL_PSIDOT": ckpt.get("GLOBAL_PSIDOT", 0.1),
        "window_size": win,
        "stride": int(ckpt.get("stride", cfg_data.get("stride", 5))),
        "target_len": win,  # legacy alias used by scenario_rule_gui
    }
    return model, norms


# ---------------- models (exact mirrors of the training classes) ----------------

if _TORCH_OK:

    class LSTMVAE_Single(nn.Module):
        """Mirror of training/lstmvae_single.py: 4 + emb_dim input, 4 output."""

        def __init__(self, num_paths, latent_dim=16, hidden_dim=128, emb_dim=8):
            super().__init__()
            self.num_paths, self.emb_dim = int(num_paths), int(emb_dim)
            self.path_emb = nn.Embedding(self.num_paths, self.emb_dim)
            self.input_dim = 4 + self.emb_dim
            self.encoder = nn.LSTM(self.input_dim, hidden_dim, num_layers=1, batch_first=True)
            self.to_mu = nn.Linear(hidden_dim, latent_dim)
            self.to_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.LSTM(latent_dim + self.emb_dim, hidden_dim,
                                   num_layers=1, batch_first=True)
            self.to_output = nn.Linear(hidden_dim, 4)

        def encode(self, x, path_id):
            B, T, _ = x.shape
            emb = self.path_emb(path_id).unsqueeze(1).expand(1, T, -1)
            _, (h_last, _) = self.encoder(torch.cat([x, emb], dim=2))
            h = h_last[-1]
            return self.to_mu(h), self.to_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z, path_id, seq_len):
            emb = self.path_emb(path_id).unsqueeze(1).expand(1, seq_len, -1)
            zin = torch.cat([z.unsqueeze(1).expand(1, seq_len, -1), emb], dim=2)
            out, _ = self.decoder(zin)
            return self.to_output(out)

        def forward(self, x, path_id):
            mu, logvar = self.encode(x, path_id)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, path_id, x.shape[1]), mu, logvar

    class LSTMVAE_Pair(nn.Module):
        """Mirror of training/lstmvae_pair.py (Option B1):
        12 + 2*emb_dim input (SHARED embedding for both vehicles), 12 output."""

        def __init__(self, num_paths, latent_dim=16, hidden_dim=128, emb_dim=8):
            super().__init__()
            self.num_paths, self.emb_dim = int(num_paths), int(emb_dim)
            self.path_emb = nn.Embedding(self.num_paths, self.emb_dim)
            self.input_dim = 12 + 2 * self.emb_dim
            self.encoder = nn.LSTM(self.input_dim, hidden_dim, num_layers=1, batch_first=True)
            self.to_mu = nn.Linear(hidden_dim, latent_dim)
            self.to_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.LSTM(latent_dim + 2 * self.emb_dim, hidden_dim,
                                   num_layers=1, batch_first=True)
            self.to_output = nn.Linear(hidden_dim, 12)

        def encode(self, x, path1_id, path2_id):
            B, T, _ = x.shape
            e1 = self.path_emb(path1_id).unsqueeze(1).expand(1, T, -1)
            e2 = self.path_emb(path2_id).unsqueeze(1).expand(1, T, -1)
            _, (h_last, _) = self.encoder(torch.cat([x, e1, e2], dim=2))
            h = h_last[-1]
            return self.to_mu(h), self.to_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z, path1_id, path2_id, seq_len):
            e1 = self.path_emb(path1_id).unsqueeze(1).expand(1, seq_len, -1)
            e2 = self.path_emb(path2_id).unsqueeze(1).expand(1, seq_len, -1)
            zin = torch.cat([z.unsqueeze(1).expand(1, seq_len, -1), e1, e2], dim=2)
            out, _ = self.decoder(zin)
            return self.to_output(out)

        def forward(self, x, path1_id, path2_id):
            mu, logvar = self.encode(x, path1_id, path2_id)
            z = self.reparameterize(mu, logvar)
            return self.decode(z, path1_id, path2_id, x.shape[1]), mu, logvar

else:
    class _NoTorch:
        def __init__(self, *a, **k):
            raise RuntimeError("PyTorch is required for model classes")
    LSTMVAE_Single = _NoTorch
    LSTMVAE_Pair = _NoTorch


# ---------------- geometry (30 Hz pixel space) ----------------

def _nearest(p, poly):
    """Closest point on polyline -> (q, seg_idx, t, dist)."""
    p = np.asarray(p, np.float32)
    best_q, best_i, best_t, best_d = poly[0], 0, 0.0, 1e18
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-8 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best_q, best_i, best_t, best_d = q, i, t, d
    return best_q, best_i, best_t, best_d


def _curv_s(poly):
    poly = np.asarray(poly, np.float32)
    if len(poly) < 2:
        return np.zeros(len(poly), np.float32)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(seg)]


def _project_frame(pts, P):
    """Project pts onto centerline P -> {q, s, tan, d} (d = signed lateral, left normal)."""
    pts = np.asarray(pts, np.float32)
    P = np.asarray(P, np.float32)
    if len(P) < 2 or len(pts) == 0:
        return None
    sP = _curv_s(P)
    qs, ss, tans, ds = [], [], [], []
    for p in pts:
        q, i, t, _ = _nearest(p, P)
        seg = P[i + 1] - P[i]
        L = np.linalg.norm(seg) + 1e-8
        tan = seg / L
        qs.append(q)
        ss.append(sP[i] + t * (sP[i + 1] - sP[i]))
        tans.append(tan)
        ds.append(float(np.dot(p - q, np.array([-tan[1], tan[0]], np.float32))))
    return {"q": np.vstack(qs), "s": np.asarray(ss, np.float32),
            "tan": np.vstack(tans), "d": np.asarray(ds, np.float32)}


def _single_residuals(points, centerline, decim=3):
    """[d, d_dot, psi, psi_dot] per 10 Hz step (training contract).

    decim: 3 == 30 Hz -> 10 Hz; set max(1, fps // 10) for other input rates.
    """
    pts = np.asarray(points, np.float32)
    if decim > 1:
        pts = pts[::decim]
    if len(pts) < 3:
        return None
    frame = _project_frame(pts, centerline)
    if frame is None:
        return None
    d = frame["d"]
    vel = np.gradient(pts, axis=0)
    theta_v = np.arctan2(vel[:, 1], vel[:, 0] + 1e-6)
    theta_c = np.arctan2(frame["tan"][:, 1], frame["tan"][:, 0] + 1e-6)
    psi = (theta_v - theta_c + np.pi) % (2 * np.pi) - np.pi
    return np.stack([d, np.gradient(d), psi, np.gradient(psi)], axis=1).astype(np.float32)


# ---------------- normalization (training contract) ----------------

def _normalize_single_raw(feats_raw, GLOBAL_DDOT, GLOBAL_PSIDOT, lane_half_width_px, clip):
    """(T,4) raw -> (T,4) normalized. clip=True: single-model contract;
    clip=False: pair-dataset contract (no clipping)."""
    d, d_dot, psi, psi_dot = feats_raw[:, 0], feats_raw[:, 1], feats_raw[:, 2], feats_raw[:, 3]
    eps = 1e-6
    d_n = d / (lane_half_width_px + eps)
    psi_n = psi / np.pi
    dd_n = d_dot / GLOBAL_DDOT
    pd_n = psi_dot / GLOBAL_PSIDOT
    if clip:
        d_n = np.clip(d_n, -1.25, 1.25)
        psi_n = np.clip(psi_n, -0.50, 0.50)
        dd_n = np.clip(dd_n, -1.00, 1.00)
        pd_n = np.clip(pd_n, -1.00, 1.00)
    return np.stack([d_n, dd_n, psi_n, pd_n], axis=1).astype(np.float32)


def _pair_b1(feats1_norm, feats2_norm):
    """B1 symmetric pair block (dataset_pair contract):
    (T,12) = [v1(4), v2(4), d_rel, psi_rel, |d_rel|, dv]."""
    d1, d1d, p1, _ = feats1_norm.T
    d2, d2d, p2, _ = feats2_norm.T
    d_rel = d2 - d1
    psi_rel = p2 - p1
    pair = np.stack([d_rel, psi_rel, np.abs(d_rel), d2d - d1d], axis=1).astype(np.float32)
    return np.concatenate([feats1_norm, feats2_norm, pair], axis=1).astype(np.float32)


def _windows(feats, window_size, stride):
    """Training-distribution windows: (T,C) -> list of (window_size,C)."""
    T = len(feats)
    if T < window_size:
        return [np.vstack([feats, np.repeat(feats[-1:], window_size - T, axis=0)])]
    return [feats[i:i + window_size] for i in range(0, T - window_size + 1, stride)]


# ---------------- path assignment / route centerline ----------------

def assign_path_id(trajectory_xy, paths, *, source_hint=None,
                   max_dist_px=120.0, min_forward_idx=2.0):
    """Match a 30 Hz pixel trajectory to the global path library. Index or -1."""
    if len(trajectory_xy) < 3 or not paths:
        return -1
    cand_idx = [i for i, p in enumerate(paths)
                if source_hint is None or p.get("source") == source_hint]
    if not cand_idx:
        cand_idx = list(range(len(paths)))
    pts = np.asarray(trajectory_xy, np.float32)
    p0, pN = pts[0], pts[-1]
    scores = []
    for i in cand_idx:
        P = np.asarray(paths[i]["polyline"], np.float32)
        if len(P) < 2:
            continue
        q0, i0, t0, d0 = _nearest(p0, P)
        qN, iN, tN, dN = _nearest(pN, P)
        prog = (iN + tN) - (i0 + t0)
        if d0 > max_dist_px or dN > max_dist_px or prog < min_forward_idx:
            continue
        score = d0 + dN + 0.1 * float(np.hypot(*(qN - q0)))
        scores.append((score, i))
    if not scores:
        return -1
    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def build_route_centerline(lane_ids, meta, max_gap_px=25.0):
    """Fallback: stitch per-lane pixel centerlines from meta along a route."""
    if not lane_ids or not meta:
        return None
    out = []
    for rid in lane_ids:
        e = meta.get(str(rid))
        if not isinstance(e, dict):
            continue
        cl = None
        for key in ("centerline", "center_points", "center", "polyline"):
            if isinstance(e.get(key), list) and len(e[key]) >= 2:
                cl = e[key]
                break
        if not cl:
            continue
        pts = []
        for p in cl:
            try:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
                elif isinstance(p, dict) and "x" in p and "y" in p:
                    pts.append((float(p["x"]), float(p["y"])))
            except (TypeError, ValueError):
                pass
        if len(pts) < 2:
            continue
        if out:
            gap = math.hypot(pts[0][0] - out[-1][0], pts[0][1] - out[-1][1])
            if gap > max_gap_px:
                continue
            if gap < 1e-3:
                pts = pts[1:]
        out.extend(pts)
    return out if len(out) >= 2 else None


# ---------------- bridges ----------------

class LSTMSingleBridge:
    """Single-vehicle bridge: windowed encode -> mean latent + robust scalars."""

    def __init__(self, model=None, device="cpu", norms=None, decim=3,
                 window_size=None, stride=None, target_len=None):
        self.model = model
        self.device = device
        self.norms = norms or {"lane_half_width_px": 40.0,
                               "GLOBAL_DDOT": 0.2, "GLOBAL_PSIDOT": 0.1}
        self.decim = int(decim)
        self.window_size = int(window_size or target_len or self.norms.get("window_size", 40))
        self.stride = int(stride or self.norms.get("stride", 5))

    def available(self):
        return _TORCH_OK and self.model is not None

    def features(self, poly_pts, centerline):
        """(T,4) normalized 10 Hz features (single contract, clipped) or None."""
        feats_raw = _single_residuals(poly_pts, centerline, decim=self.decim)
        if feats_raw is None:
            return None
        return _normalize_single_raw(
            feats_raw,
            self.norms.get("GLOBAL_DDOT", 0.2),
            self.norms.get("GLOBAL_PSIDOT", 0.1),
            self.norms.get("lane_half_width_px", 40.0),
            clip=True,
        )

    def _window_mse(self, seg, path_id=0):
        x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(self.device)
        p = torch.tensor([int(path_id)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(x, p)
            recon = self.model.decode(mu, p, seg.shape[0])
        return float(((recon - x) ** 2).mean())

    def signature(self, poly_pts, centerline, path_id=0):
        if not self.available() or poly_pts is None or centerline is None:
            return None
        try:
            feats = self.features(poly_pts, centerline)
            if feats is None or len(feats) < 5:
                return None
            num_paths = getattr(self.model, "num_paths", None)
            if num_paths is not None and not (0 <= int(path_id) < num_paths):
                return None  # nn.Embedding silently wraps negative ids

            wins = _windows(feats, self.window_size, self.stride)
            x = torch.stack([torch.tensor(w, dtype=torch.float32) for w in wins]
                            , dim=0).to(self.device)
            pid = torch.full((len(wins),), int(path_id), dtype=torch.long,
                             device=self.device)
            with torch.no_grad():
                mu, logvar = self.model.encode(x, pid)
                z_std = torch.exp(0.5 * logvar)
                recon = self.model.decode(mu, pid, self.window_size)

            z_mu = mu.detach().cpu().numpy()
            z_mean = z_mu.mean(axis=0)
            d, d_dot, psi, psi_dot = (feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3])
            return {
                "z_mu": z_mean.tolist(),
                "z_std": z_std.mean(axis=0).detach().cpu().numpy().tolist(),
                "z_norm": float(np.linalg.norm(z_mean)),
                "z_mu_spread": float(np.mean(np.linalg.norm(z_mu - z_mean, axis=1))),
                "n_windows": len(wins),
                "recon_err": float(((recon - x) ** 2).mean()),
                "mean_abs_d": float(np.mean(np.abs(d))),
                "std_d": float(np.std(d)),
                "mean_abs_psi": float(np.mean(np.abs(psi))),
                "std_psi": float(np.std(psi)),
                "max_abs_d_norm": float(np.max(np.abs(d))),
                "aggressiveness": float(np.mean(np.abs(d_dot)) + np.mean(np.abs(psi_dot))),
            }
        except Exception as e:
            print(f"[LSTMSingleBridge] signature failed: {e}")
            return None


class LSTMPairBridge:
    """Interaction-pair bridge (Option B1 contract):
    (T,12) = [v1(4), v2(4), d_rel, psi_rel, |d_rel|, dv].

    Per-vehicle features use the pair-dataset normalization (NO clipping).
    path2_id defaults to path1_id (shared path embedding).
    """

    def __init__(self, model=None, device="cpu", norms=None, decim=3,
                 window_size=None, stride=None, target_len=None):
        self.model = model
        self.device = device
        self.norms = norms or {"lane_half_width_px": 40.0,
                               "GLOBAL_DDOT": 0.2, "GLOBAL_PSIDOT": 0.1}
        self.decim = int(decim)
        self.window_size = int(window_size or target_len or self.norms.get("window_size", 40))
        self.stride = int(stride or self.norms.get("stride", 5))

    def available(self):
        return _TORCH_OK and self.model is not None

    def _vehicle_features(self, poly_pts, centerline):
        feats_raw = _single_residuals(poly_pts, centerline, decim=self.decim)
        if feats_raw is None:
            return None
        return _normalize_single_raw(
            feats_raw,
            self.norms.get("GLOBAL_DDOT", 0.2),
            self.norms.get("GLOBAL_PSIDOT", 0.1),
            self.norms.get("lane_half_width_px", 40.0),
            clip=False,  # pair contract: no clipping
        )

    def features(self, pts1, pts2, centerline1, centerline2):
        """(T,12) pair block for length-aligned vehicles, or None."""
        f1 = self._vehicle_features(pts1, centerline1)
        f2 = self._vehicle_features(pts2, centerline2)
        if f1 is None or f2 is None:
            return None
        n = min(len(f1), len(f2))
        if n < 3:
            return None
        return _pair_b1(f1[:n], f2[:n])

    def signature(self, pts1, pts2, centerline1, centerline2,
                  path1_id=0, path2_id=None):
        if not self.available():
            return None
        if path2_id is None:
            path2_id = path1_id
        try:
            feats = self.features(pts1, pts2, centerline1, centerline2)
            if feats is None or len(feats) < 5:
                return None
            num_paths = getattr(self.model, "num_paths", None)
            if num_paths is not None and not (
                0 <= int(path1_id) < num_paths
                and 0 <= int(path2_id) < num_paths
            ):
                return None  # nn.Embedding silently wraps negative ids

            wins = _windows(feats, self.window_size, self.stride)
            x = torch.stack([torch.tensor(w, dtype=torch.float32) for w in wins]
                            , dim=0).to(self.device)
            p1 = torch.full((len(wins),), int(path1_id), dtype=torch.long,
                            device=self.device)
            p2 = torch.full((len(wins),), int(path2_id), dtype=torch.long,
                            device=self.device)
            with torch.no_grad():
                mu, logvar = self.model.encode(x, p1, p2)
                z_std = torch.exp(0.5 * logvar)
                recon = self.model.decode(mu, p1, p2, self.window_size)

            z_mu = mu.detach().cpu().numpy()
            z_mean = z_mu.mean(axis=0)
            d1, d2 = feats[:, 0], feats[:, 4]
            d_rel, dv = feats[:, 8], feats[:, 11]
            return {
                "z_mu": z_mean.tolist(),
                "z_std": z_std.mean(axis=0).detach().cpu().numpy().tolist(),
                "z_norm": float(np.linalg.norm(z_mean)),
                "z_mu_spread": float(np.mean(np.linalg.norm(z_mu - z_mean, axis=1))),
                "n_windows": len(wins),
                "recon_err": float(((recon - x) ** 2).mean()),
                "mean_abs_d1": float(np.mean(np.abs(d1))),
                "mean_abs_d2": float(np.mean(np.abs(d2))),
                "mean_abs_d_rel": float(np.mean(np.abs(d_rel))),
                "std_d_rel": float(np.std(d_rel)),
                "mean_abs_dv": float(np.mean(np.abs(dv))),
            }
        except Exception as e:
            print(f"[LSTMPairBridge] signature failed: {e}")
            return None
