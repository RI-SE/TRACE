
# lstm_single_bridge.py (standalone, no external deps)
import os
import json
import numpy as np
import torch.nn as nn

try:
    import torch
except Exception:
    torch = None

class LSTMVAE_Single(nn.Module):
    """
    Residual + path embedding LSTM-VAE (standalone)
    -------------------------------------------------------
    Input per step = [d_norm, d_dot_norm, psi_norm, psi_dot_norm] + path_emb(emb_dim)
    Latent size = latent_dim
    Hidden size = hidden_dim
    """
    def __init__(self, num_paths, latent_dim=16, hidden_dim=128, emb_dim=8):
        super().__init__()
        self.num_paths = int(num_paths)
        self.emb_dim = int(emb_dim)

        # Path embedding
        self.path_emb = nn.Embedding(self.num_paths, self.emb_dim)

        # 4 residual features + emb_dim
        self.input_dim = 4 + self.emb_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=self.input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder (latent + emb)
        self.decoder_in_dim = latent_dim + self.emb_dim
        self.decoder = nn.LSTM(
            input_size=self.decoder_in_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True
        )
        self.to_output = nn.Linear(hidden_dim, 4)  # normalized residuals

    def encode(self, x, path_id):
        """x: (B,T,4), path_id: (B,)"""
        assert torch is not None, "PyTorch is required"
        B, T, _ = x.shape
        emb = self.path_emb(path_id)              # (B,emb)
        emb_exp = emb.unsqueeze(1).repeat(1, T, 1)
        x_in = torch.cat([x, emb_exp], dim=2)     # (B,T,4+emb)
        _, (h_last, _) = self.encoder(x_in)
        h_last = h_last[-1]
        mu = self.to_mu(h_last)
        logvar = self.to_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, path_id, seq_len):
        assert torch is not None, "PyTorch is required"
        B = z.size(0)
        emb = self.path_emb(path_id).unsqueeze(1)   # (B,1,emb)
        z_exp = z.unsqueeze(1).repeat(1, seq_len, 1)
        emb_exp = emb.repeat(1, seq_len, 1)
        dec_in = torch.cat([z_exp, emb_exp], dim=2)
        out, _ = self.decoder(dec_in)
        return self.to_output(out)

    def forward(self, x, path_id):
        mu, logvar = self.encode(x, path_id)
        z = self.reparameterize(mu, logvar)
        dec = self.decode(z, path_id, seq_len=x.shape[1])
        return dec, mu, logvar

# ---------------- IO helpers ----------------

def load_paths_json(paths_json="cleaned/paths_library.json"):
    """Load precomputed global path cache if available. Returns [] if not found."""
    try:
        if os.path.exists(paths_json):
            with open(paths_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("paths", data if isinstance(data, list) else [])
    except Exception:
        pass
    return []


def load_single_model(path, device="cpu"):
    """Load SINGLE-VAE checkpoint and instantiate model. Returns (model, norms).
       norms contains optional training-time constants when saved in ckpt.
    """
    assert torch is not None, "PyTorch is required"
    ckpt = torch.load(path, map_location=device)
    num_paths  = ckpt.get("num_paths")
    latent_dim = ckpt.get("latent_dim", 16)
    hidden_dim = ckpt.get("hidden_dim", 128)
    emb_dim    = ckpt.get("emb_dim", 8)
    if num_paths is None:
        raise RuntimeError("Checkpoint missing 'num_paths'.")
    model = LSTMVAE_Single(num_paths=num_paths, latent_dim=latent_dim,
                           hidden_dim=hidden_dim, emb_dim=emb_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])  # strict by default
    model.eval()
    norms = {
        "lane_half_width_px": ckpt.get("lane_half_width_px", 40.0),
        "GLOBAL_DDOT":        ckpt.get("GLOBAL_DDOT", 0.2),
        "GLOBAL_PSIDOT":      ckpt.get("GLOBAL_PSIDOT", 0.1),
        "target_len":         ckpt.get("target_len", 160),
    }
    return model, norms

# ---------------- geometry helpers ----------------

def _nearest_on_poly(p, P):
    p = np.asarray(p, np.float32)
    best = (P[0], 0, 0.0, 1e18)
    for i in range(len(P)-1):
        a, b = P[i], P[i+1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-8 else np.clip(np.dot(p-a, ab)/denom, 0.0, 1.0)
        q = a + t*ab
        d = float(np.linalg.norm(p-q))
        if d < best[3]:
            best = (q, i, t, d)
    return best


def assign_path_id(trajectory_xy, paths, *, source_hint=None, max_dist_px=120.0, min_forward_idx=2.0):
    if len(trajectory_xy) < 3 or not paths:
        return -1
    cand_idx = [i for i, p in enumerate(paths) if (source_hint is None or p.get("source") == source_hint)]
    if not cand_idx:
        cand_idx = list(range(len(paths)))
    pts = np.asarray(trajectory_xy, np.float32)
    p0, pN = pts[0], pts[-1]
    scores = []
    for i in cand_idx:
        P = np.asarray(paths[i]["polyline"], np.float32)
        (q0, i0, t0, d0) = _nearest_on_poly(p0, P)
        (qN, iN, tN, dN) = _nearest_on_poly(pN, P)
        prog = (iN + tN) - (i0 + t0)
        if d0 > max_dist_px or dN > max_dist_px or prog < min_forward_idx:
            continue
        score = d0 + dN + 0.1 * np.hypot(*(qN - q0))
        scores.append((score, i))
    if not scores:
        return -1
    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def _curvilinear_s(pts):
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2:
        return np.zeros(len(pts), np.float32)
    diff = np.diff(pts, axis=0)
    seg = np.linalg.norm(diff, axis=1)
    return np.r_[0.0, np.cumsum(seg)]


def _nearest(p, poly):
    p = np.asarray(p, np.float32)
    best = (poly[0], 0, 0.0, 1e18)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i+1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-8 else np.clip(np.dot(p - a, ab) / denom, 0, 1)
        q = a + t*ab
        d = float(np.linalg.norm(p - q))
        if d < best[3]:
            best = (q, i, t, d)
    return best

# ---------------- residuals ----------------

def _normalize_residuals_global(feats_raw, GLOBAL_DDOT, GLOBAL_PSIDOT, lane_half_width_px):
    """Dataset-global normalization + conservative clamping."""
    d      = feats_raw[:,0]
    d_dot  = feats_raw[:,1]
    psi    = feats_raw[:,2]
    psi_dot= feats_raw[:,3]
    eps = 1e-6
    d_norm      = d   / (lane_half_width_px + eps)
    psi_norm    = psi / np.pi
    d_dot_norm  = d_dot   / (GLOBAL_DDOT + eps)
    psi_dot_norm= psi_dot / (GLOBAL_PSIDOT + eps)
    # clamps for stability
    d_norm       = np.clip(d_norm,      -1.25,  1.25)
    psi_norm     = np.clip(psi_norm,    -0.50,  0.50)
    d_dot_norm   = np.clip(d_dot_norm,  -1.00,  1.00)
    psi_dot_norm = np.clip(psi_dot_norm,-1.00,  1.00)
    feats_norm = np.stack([d_norm, d_dot_norm, psi_norm, psi_dot_norm], axis=1).astype(np.float32)
    scalers = {
        "lane_half_width_px": float(lane_half_width_px),
        "GLOBAL_DDOT": float(GLOBAL_DDOT),
        "GLOBAL_PSIDOT": float(GLOBAL_PSIDOT),
    }
    return feats_norm, scalers


def _project_residuals(points, centerline):
    pts = np.asarray(points, np.float32)
    # Downsample 30Hz -> 10Hz by default
    pts = pts[::3]
    P = np.asarray(centerline, np.float32)
    if len(pts) < 3:
        return None
    sP = _curvilinear_s(P)
    qs, tans, nors = [], [], []
    for p in pts:
        q, idx, t, _ = _nearest(p, P)
        seg = P[idx+1] - P[idx]
        L = np.linalg.norm(seg) + 1e-8
        tan = seg / L
        nor = np.array([-tan[1], tan[0]], np.float32)
        qs.append(q); tans.append(tan); nors.append(nor)
    qs   = np.vstack(qs)
    tans = np.vstack(tans)
    nors = np.vstack(nors)
    # lateral deviation
    d = np.sum((pts - qs) * nors, axis=1)
    # heading diff
    vel = np.gradient(pts, axis=0)
    theta_v = np.arctan2(vel[:,1], vel[:,0] + 1e-6)
    theta_c = np.arctan2(tans[:,1], tans[:,0] + 1e-6)
    psi = (theta_v - theta_c + np.pi) % (2*np.pi) - np.pi
    d_dot = np.gradient(d)
    psi_dot = np.gradient(psi)
    return np.stack([d, d_dot, psi, psi_dot], axis=1).astype(np.float32)

# ---------------- bridge ----------------

class LSTMSingleBridge:
    """Encode observed polyline into SINGLE LSTM-VAE latent & signatures."""
    def __init__(self, model=None, device='cpu', norms=None, target_len=160):
        self.model = model
        self.device = device
        self.norms = norms or {
            'lane_half_width_px': 40.0,
            'GLOBAL_DDOT': 0.2,
            'GLOBAL_PSIDOT': 0.1,
        }
        self.target_len = int(target_len)

    def available(self):
        return (torch is not None) and (self.model is not None)

    def signature(self, poly_pts, centerline, path_id=0):
        if not self.available() or poly_pts is None or centerline is None:
            return None
        try:
            feats_raw = _project_residuals(poly_pts, centerline)
            if feats_raw is None or len(feats_raw) < 5:
                return None
            feats_norm, _ = _normalize_residuals_global(
                feats_raw,
                self.norms.get('GLOBAL_DDOT', 0.2),
                self.norms.get('GLOBAL_PSIDOT', 0.1),
                self.norms.get('lane_half_width_px', 40.0),
            )
            # pad/trim
            T = len(feats_norm)
            if T > self.target_len:
                feats_norm = feats_norm[: self.target_len]
            elif T < self.target_len:
                pad = np.repeat(feats_norm[-1:], self.target_len - T, axis=0)
                feats_norm = np.vstack([feats_norm, pad])

            x_t = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
            pid = torch.tensor([int(path_id)], dtype=torch.long).to(self.device)
            with torch.no_grad():
                mu, logvar = self.model.encode(x_t, pid)
                z_std = torch.exp(0.5 * logvar)
                z_mu = mu.squeeze(0).detach().cpu().numpy()
                z_sd = z_std.squeeze(0).detach().cpu().numpy()

            d      = feats_norm[:T, 0]
            d_dot  = feats_norm[:T, 1]
            psi    = feats_norm[:T, 2]
            psi_dot= feats_norm[:T, 3]
            return {
                'z_mu': z_mu.tolist(),
                'z_std': z_sd.tolist(),
                'z_norm': float(np.linalg.norm(z_mu)),
                'mean_abs_d': float(np.mean(np.abs(d))),
                'std_d': float(np.std(d)),
                'mean_abs_psi': float(np.mean(np.abs(psi))),
                'std_psi': float(np.std(psi)),
                'max_abs_d_norm': float(np.max(np.abs(d))),
                'aggressiveness': float(np.mean(np.abs(d_dot)) + np.mean(np.abs(psi_dot))),
            }
        except Exception:
            return None
