#!/usr/bin/env python3
"""
scroll_conjoint_hb.py

Hierarchical Bayes for large-scale scroll-based conjoint analysis.
Python equivalent of scroll_conjoint_hb.php using numpy / scipy.

Three separate models:
  selection — P(liked | visible)     positive-constrained betas, horseshoe prior
  rejection — P(disliked | visible)  positive-constrained betas, horseshoe prior
  stage2    — P(chosen | liked set)  unconstrained betas, horseshoe prior

Scroll-resolved attention:
  Items with position_row >= scroll_rows_visible are structurally unseen and
  excluded from the likelihood.  Visible-but-not-selected items are true
  neutral observations — no attention mixture needed.

Sparsity:
  Horseshoe prior on population means drives near-zero attributes toward zero
  without penalising the few attributes that genuinely drive choice.

Interactions:
  One cross-attribute product term per attribute pair, shared across respondents,
  with an independent horseshoe prior.

Usage:
  python scroll_conjoint_hb.py [--config scroll_conjoint_config.json]
                                [--catalogue scroll_catalogue.csv]
                                [--responses scroll_responses.csv]
                                [--out_dir .] [--seed 42]
                                [--iters 4000] [--burn_in 1500]

Requirements:
  pip install numpy scipy

Output files (identical format to PHP version):
  hb_selection_betas.csv    individual part-worths for liked-ness
  hb_rejection_betas.csv    individual part-worths for disliked-ness
  hb_stage2_betas.csv       individual part-worths for preference within liked set
  hb_population.csv         population summary + interaction effects (all 3 models)
"""

from __future__ import annotations
import numpy as np
from scipy.stats import truncnorm, invgamma, norm
from scipy.special import expit
import csv
import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

# ── data structures ────────────────────────────────────────────────────────────

@dataclass
class Config:
    attributes: list
    grid_rows: int
    grid_cols: int
    K: int
    M: int
    att_offsets: list
    interaction_pairs: list
    mcmc: dict


@dataclass
class RespArrays:
    """Pre-computed numpy arrays for one respondent — built once before MCMC."""
    rid: str
    X_main: np.ndarray       # (n_vis, K) main-effect design matrix, visible items
    X_inter: np.ndarray      # (n_vis, M) interaction design matrix, visible items
    y_liked: np.ndarray      # (n_vis,)  1 = liked or chosen
    y_disliked: np.ndarray   # (n_vis,)  1 = disliked
    XL_main: np.ndarray      # (n_liked, K) main effects, liked set only
    XL_inter: np.ndarray     # (n_liked, M) interactions, liked set only
    chosen_idx: int          # index of chosen item in liked set; -1 if no stage-2 data
    n_vis: int
    n_liked: int

# ── config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Config:
    with open(path) as f:
        c = json.load(f)

    atts = c["attributes"]
    K, offsets = 0, []
    for att in atts:
        offsets.append((K, K + att["levels"] - 2))
        K += att["levels"] - 1

    pairs = [
        (a, b, offsets[a][0], offsets[b][0])
        for a in range(len(atts) - 1)
        for b in range(a + 1, len(atts))
    ]

    mcmc = {"iterations": 4000, "burn_in": 1500,
            "adapt_every": 50, "target_accept": 0.234}
    mcmc.update(c.get("mcmc", {}))

    return Config(attributes=atts, grid_rows=c["grid_rows"], grid_cols=c["grid_cols"],
                  K=K, M=len(pairs), att_offsets=offsets,
                  interaction_pairs=pairs, mcmc=mcmc)

# ── feature engineering ─────────────────────────────────────────────────────────

def effects_code(level: int, n_levels: int) -> np.ndarray:
    if n_levels <= 1:
        return np.zeros(0)
    c = np.zeros(n_levels - 1)
    if level < n_levels - 1:
        c[level] = 1.0
    else:
        c[:] = -1.0
    return c

def build_main(levels: list, atts: list) -> np.ndarray:
    return np.concatenate([effects_code(lv, a["levels"]) for lv, a in zip(levels, atts)])

def build_inter(x_main: np.ndarray, pairs: list) -> np.ndarray:
    if not pairs:
        return np.zeros(0)
    return np.array([x_main[ia] * x_main[ib] for _, _, ia, ib in pairs])

# ── data loading ───────────────────────────────────────────────────────────────

def load_catalogue(path: str, config: Config) -> dict:
    items = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            iid    = row["item_id"]
            levels = [int(row[f"att{i}_level"]) for i in range(len(config.attributes))]
            xm     = build_main(levels, config.attributes)
            xi     = build_inter(xm, config.interaction_pairs)
            items[iid] = {"x_main": xm, "x_inter": xi}
    return items


def load_and_prepare(path: str, catalogue: dict, config: Config) -> list[RespArrays]:
    """Load responses CSV and convert each respondent's data to pre-computed numpy arrays."""
    raw: dict[str, list] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            raw.setdefault(row["respondent_id"], []).append(row)

    K, M = config.K, config.M
    prepared = []

    for rid, rows in raw.items():
        # The 'chosen' marker row has an empty position_row field
        chosen_id = next(
            (r["item_id"] for r in rows
             if r["response"] == "chosen" and r["position_row"] == ""),
            None,
        )
        scroll_rows = int(rows[0]["scroll_rows_visible"])

        vis_main, vis_inter   = [], []
        y_lk, y_dk            = [], []
        lk_main, lk_inter     = [], []
        lk_ids                = []

        for r in rows:
            # Skip bare 'chosen' marker and explicitly unseen rows
            if r["response"] in ("chosen", "unseen") and r["position_row"] == "":
                continue
            if r["response"] == "unseen":
                continue
            if r["position_row"] == "" or int(r["position_row"]) >= scroll_rows:
                continue
            iid = r["item_id"]
            if iid not in catalogue:
                continue

            resp     = r["response"]
            is_liked = resp in ("liked", "chosen")
            vis_main.append(catalogue[iid]["x_main"])
            vis_inter.append(catalogue[iid]["x_inter"])
            y_lk.append(float(is_liked))
            y_dk.append(float(resp == "disliked"))

            if is_liked:
                lk_main.append(catalogue[iid]["x_main"])
                lk_inter.append(catalogue[iid]["x_inter"])
                lk_ids.append(iid)

        if len(vis_main) < 5:
            continue

        chosen_idx = lk_ids.index(chosen_id) if chosen_id in lk_ids else -1

        prepared.append(RespArrays(
            rid       = rid,
            X_main    = np.array(vis_main)  if vis_main  else np.zeros((0, K)),
            X_inter   = np.array(vis_inter) if vis_inter else np.zeros((0, M)),
            y_liked   = np.array(y_lk,  dtype=np.float64),
            y_disliked= np.array(y_dk,  dtype=np.float64),
            XL_main   = np.array(lk_main)  if lk_main  else np.zeros((0, K)),
            XL_inter  = np.array(lk_inter) if lk_inter else np.zeros((0, M)),
            chosen_idx= chosen_idx,
            n_vis     = len(vis_main),
            n_liked   = len(lk_ids),
        ))
    return prepared

# ── likelihood (vectorised with numpy) ─────────────────────────────────────────

def binary_ll(beta: np.ndarray, gamma: np.ndarray, tau: float,
              X: np.ndarray, Z: np.ndarray, y: np.ndarray) -> float:
    """Binary logit log-likelihood for all visible items of one respondent."""
    u = X @ beta + Z @ gamma - tau
    p = np.clip(expit(u), 1e-10, 1.0 - 1e-10)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def stage2_ll(beta: np.ndarray, gamma: np.ndarray,
              XL: np.ndarray, ZL: np.ndarray, chosen_idx: int) -> float:
    """MNL log-likelihood for stage-2 choice from liked set."""
    if XL.shape[0] < 2 or chosen_idx < 0:
        return 0.0
    u = XL @ beta + ZL @ gamma
    u -= u.max()
    return float(u[chosen_idx] - np.log(np.sum(np.exp(u))))


def resp_ll(beta: np.ndarray, gamma: np.ndarray, tau: float,
            r: RespArrays, model: str) -> float:
    if model == "stage2":
        return stage2_ll(beta, gamma, r.XL_main, r.XL_inter, r.chosen_idx)
    y = r.y_liked if model == "liked" else r.y_disliked
    return binary_ll(beta, gamma, tau, r.X_main, r.X_inter, y)


def log_prior_beta(beta: np.ndarray, alpha: np.ndarray,
                   sigma_sq: np.ndarray, positive: bool) -> float:
    if positive and np.any(beta < 0.0):
        return -np.inf
    sd = np.sqrt(np.maximum(sigma_sq, 1e-6))
    return float(np.sum(norm.logpdf(beta, loc=alpha, scale=sd)))

# ── sampling helpers ───────────────────────────────────────────────────────────

def sample_inv_gamma_vec(shape: float, rate: np.ndarray) -> np.ndarray:
    """Vectorised InvGamma(shape, rate) — scipy uses scale = rate."""
    return invgamma.rvs(a=shape, scale=np.maximum(rate, 1e-10))


def sample_tnorm_pos(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Sample from N(mu, sigma²) truncated to [0, ∞) — vectorised over K."""
    a = np.where(sigma > 0, -mu / np.maximum(sigma, 1e-10), 0.0)
    return truncnorm.rvs(a, np.inf, loc=mu, scale=sigma)

# ── horseshoe prior updates (Makalic & Schmidt 2016) ───────────────────────────
# Hierarchy:
#   param_k | λ_k, τ ~ N(0, λ_k² τ²)
#   λ_k² | ν_k ~ InvGamma(½, 1/ν_k)
#   ν_k         ~ InvGamma(½, 1)
#   τ²  | ξ    ~ InvGamma((K+1)/2, 1/ξ)
#   ξ           ~ InvGamma(½, 1)

def hs_update_lambda_sq(param: np.ndarray, tau_sq: float, nu: np.ndarray) -> np.ndarray:
    rate = 1.0 / np.maximum(nu, 1e-10) + param ** 2 / (2.0 * max(tau_sq, 1e-10))
    return sample_inv_gamma_vec(1.0, rate)


def hs_update_nu(lambda_sq: np.ndarray) -> np.ndarray:
    return sample_inv_gamma_vec(1.0, 1.0 + 1.0 / np.maximum(lambda_sq, 1e-10))


def hs_update_tau_sq(param: np.ndarray, lambda_sq: np.ndarray, xi: float) -> float:
    K    = len(param)
    rate = 1.0 / max(xi, 1e-10) + float(
        np.sum(param ** 2 / (2.0 * np.maximum(lambda_sq, 1e-10)))
    )
    return float(invgamma.rvs(a=(K + 1) / 2.0, scale=max(rate, 1e-10)))


def hs_update_xi(tau_sq: float) -> float:
    return float(invgamma.rvs(a=1.0, scale=1.0 + 1.0 / max(tau_sq, 1e-10)))


def hs_update_alpha(betas: np.ndarray, sigma_sq: np.ndarray,
                    lambda_sq: np.ndarray, tau_sq: float,
                    positive: bool) -> np.ndarray:
    """Conjugate update for population mean alpha_k for all k simultaneously.
    betas: (n, K); returns updated alpha (K,).
    """
    n          = betas.shape[0]
    sig2       = np.maximum(sigma_sq, 1e-6)
    prior_prec = 1.0 / np.maximum(lambda_sq * tau_sq, 1e-10)
    post_prec  = n / sig2 + prior_prec
    post_var   = 1.0 / post_prec
    post_mean  = post_var * (betas.sum(axis=0) / sig2)

    if positive:
        return sample_tnorm_pos(post_mean, np.sqrt(post_var))
    return np.random.normal(post_mean, np.sqrt(post_var))

# ── main MCMC ─────────────────────────────────────────────────────────────────

def run_model(respondents: list[RespArrays], config: Config,
              model: str, rng: np.random.Generator) -> dict:
    """
    Estimate one HB model via MCMC.

    Parameters
    ----------
    respondents : pre-computed arrays for all respondents
    config      : design config
    model       : 'liked' | 'disliked' | 'stage2'
    rng         : seeded numpy Generator

    Returns
    -------
    dict with keys: beta_mean (n×K), alpha_mean (K,), gamma_mean (M,),
                    tau, tau_sq, n_samples, accept_rates, rids
    """
    K, M     = config.K, config.M
    n        = len(respondents)
    positive = (model != "stage2")
    it_cfg   = config.mcmc
    iters    = it_cfg["iterations"]
    burn_in  = it_cfg["burn_in"]
    adap_ev  = it_cfg["adapt_every"]
    target   = it_cfg["target_accept"]

    # ── initialisations ──────────────────────────────────────────────────────
    betas = (np.abs(rng.standard_normal((n, K))) * 0.05 if positive
             else rng.standard_normal((n, K)) * 0.05)

    # Main-effect population horseshoe
    alpha_pop  = np.full(K, 0.05)
    sigma_sq   = np.ones(K)
    lambda_sq  = np.ones(K)
    nu         = np.ones(K)
    tau_sq     = 1.0
    xi         = 1.0

    # Interaction horseshoe (gamma is shared across respondents)
    gamma       = np.zeros(M)
    g_lambda_sq = np.ones(M)
    g_nu        = np.ones(M)
    g_tau_sq    = 1.0
    g_xi        = 1.0
    step_gamma  = 0.1

    # Shared threshold
    tau      = 0.0
    step_tau = 0.1

    # Per-respondent step sizes and counters
    steps     = np.full(n, 0.3)
    adapt_acc = np.zeros(n, int)
    adapt_cnt = np.zeros(n, int)
    post_acc  = np.zeros(n, int)
    post_cnt  = np.zeros(n, int)

    g_acc = g_cnt = tau_acc = tau_cnt = 0

    # Posterior accumulators
    beta_acc  = np.zeros((n, K))
    alpha_acc = np.zeros(K)
    gamma_acc = np.zeros(M)
    n_samples = 0

    # ── MCMC ─────────────────────────────────────────────────────────────────
    for it in range(iters):

        # 1. Individual beta MH (per respondent)
        noise = rng.standard_normal((n, K))
        for i, r in enumerate(respondents):
            prop = betas[i] + steps[i] * noise[i]
            if positive:
                prop = np.abs(prop)  # reflect at zero boundary

            lp_old = log_prior_beta(betas[i], alpha_pop, sigma_sq, positive)
            lp_new = log_prior_beta(prop,     alpha_pop, sigma_sq, positive)
            if lp_new == -np.inf:
                continue

            ll_old = resp_ll(betas[i], gamma, tau, r, model)
            ll_new = resp_ll(prop,     gamma, tau, r, model)

            if np.log(rng.random()) < (ll_new + lp_new) - (ll_old + lp_old):
                betas[i] = prop
                adapt_acc[i] += 1
                if it >= burn_in:
                    post_acc[i] += 1
            adapt_cnt[i] += 1
            if it >= burn_in:
                post_cnt[i] += 1

            # Robbins-Monro step-size adaptation
            if it < burn_in and adapt_cnt[i] >= adap_ev:
                rate      = adapt_acc[i] / adapt_cnt[i]
                steps[i] *= float(np.exp(rate - target))
                steps[i]  = float(np.clip(steps[i], 1e-4, 5.0))
                adapt_acc[i] = adapt_cnt[i] = 0

        # 2. Shared threshold tau (binary models only)
        if model != "stage2":
            prop_tau  = tau + rng.normal(0, step_tau)
            ll_old_t  = sum(resp_ll(betas[i], gamma, tau,      r, model) for i, r in enumerate(respondents))
            ll_new_t  = sum(resp_ll(betas[i], gamma, prop_tau, r, model) for i, r in enumerate(respondents))
            if np.log(rng.random()) < ll_new_t - ll_old_t:
                tau = prop_tau
                tau_acc += 1
            tau_cnt += 1
            if it < burn_in and tau_cnt >= adap_ev:
                step_tau *= float(np.exp(tau_acc / tau_cnt - target))
                step_tau  = float(np.clip(step_tau, 1e-3, 2.0))
                tau_acc = tau_cnt = 0

        # 3. Shared interaction gamma (MH)
        if M > 0:
            prop_g    = gamma + rng.normal(0, step_gamma, M)
            prior_sd  = np.sqrt(np.maximum(g_lambda_sq * g_tau_sq, 1e-10))
            lp_g_old  = float(np.sum(norm.logpdf(gamma,  0.0, prior_sd)))
            lp_g_new  = float(np.sum(norm.logpdf(prop_g, 0.0, prior_sd)))
            ll_g_old  = sum(resp_ll(betas[i], gamma,  tau, r, model) for i, r in enumerate(respondents))
            ll_g_new  = sum(resp_ll(betas[i], prop_g, tau, r, model) for i, r in enumerate(respondents))
            if np.log(rng.random()) < (ll_g_new + lp_g_new) - (ll_g_old + lp_g_old):
                gamma = prop_g
                g_acc += 1
            g_cnt += 1
            if it < burn_in and g_cnt >= adap_ev:
                step_gamma *= float(np.exp(g_acc / g_cnt - target))
                step_gamma  = float(np.clip(step_gamma, 1e-4, 2.0))
                g_acc = g_cnt = 0

        # 4. Population horseshoe for main effects (vectorised over K)
        ss         = np.sum((betas - alpha_pop[None, :]) ** 2, axis=0)
        sigma_sq   = np.clip(sample_inv_gamma_vec(1.0 + n / 2.0, 1.0 + ss / 2.0), 1e-6, 10.0)
        lambda_sq  = hs_update_lambda_sq(alpha_pop, tau_sq, nu)
        nu         = hs_update_nu(lambda_sq)
        alpha_pop  = hs_update_alpha(betas, sigma_sq, lambda_sq, tau_sq, positive)
        xi         = hs_update_xi(tau_sq)
        tau_sq     = hs_update_tau_sq(alpha_pop, lambda_sq, xi)

        # 5. Horseshoe for interaction gamma
        if M > 0:
            g_lambda_sq = hs_update_lambda_sq(gamma, g_tau_sq, g_nu)
            g_nu        = hs_update_nu(g_lambda_sq)
            g_xi        = hs_update_xi(g_tau_sq)
            g_tau_sq    = hs_update_tau_sq(gamma, g_lambda_sq, g_xi)

        # 6. Accumulate
        if it >= burn_in:
            beta_acc  += betas
            alpha_acc += alpha_pop
            gamma_acc += gamma
            n_samples += 1

        if it % 500 == 0:
            print(f"[{model}] iter {it:4d}/{iters}  tau={tau:.3f}  "
                  f"tau_sq={tau_sq:.4f}  step={steps.mean():.3f}  "
                  f"samples={n_samples}", file=sys.stderr)

    ns = max(1, n_samples)
    rids = [r.rid for r in respondents]
    ar   = np.where(post_cnt > 0, post_acc / np.maximum(post_cnt, 1), 0.0)

    return {
        "beta_mean":    beta_acc / ns,          # (n, K)
        "alpha_mean":   alpha_acc / ns,          # (K,)
        "gamma_mean":   gamma_acc / ns,          # (M,)
        "tau":          tau,
        "tau_sq":       tau_sq,
        "n_samples":    n_samples,
        "accept_rates": dict(zip(rids, ar.tolist())),
        "rids":         rids,
    }

# ── output ─────────────────────────────────────────────────────────────────────

def col_names(config: Config) -> list[str]:
    cols = ["respondent_id"]
    for i, att in enumerate(config.attributes):
        for l in range(att["levels"] - 1):
            cols.append(f"att{i}_{att['name']}_l{l}")
    cols.append("accept_rate")
    return cols


def write_betas(path: str, result: dict, config: Config) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(col_names(config))
        w.writerow(["POPULATION"]
                   + [f"{v:.6f}" for v in result["alpha_mean"]]
                   + [""])
        for i, rid in enumerate(result["rids"]):
            acc = result["accept_rates"].get(rid, 0.0)
            w.writerow([rid]
                       + [f"{v:.6f}" for v in result["beta_mean"][i]]
                       + [f"{acc:.4f}"])
    print(f"Written: {path}")


def write_population(path: str, sel: dict, rej: dict, s2: dict,
                     config: Config) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "parameter", "sel_value", "rej_value", "s2_value"])

        k = 0
        for i, att in enumerate(config.attributes):
            for l in range(att["levels"] - 1):
                w.writerow([
                    "main", f"att{i}_{att['name']}_l{l}",
                    f"{sel['alpha_mean'][k]:.6f}",
                    f"{rej['alpha_mean'][k]:.6f}",
                    f"{s2['alpha_mean'][k]:.6f}",
                ])
                k += 1

        w.writerow(["threshold", "tau",
                    f"{sel['tau']:.4f}", f"{rej['tau']:.4f}", ""])

        for pi, (a, b, *_) in enumerate(config.interaction_pairs):
            label = (f"int_att{a}_{config.attributes[a]['name']}"
                     f"__att{b}_{config.attributes[b]['name']}")
            w.writerow([
                "interaction", label,
                f"{sel['gamma_mean'][pi]:.6f}",
                f"{rej['gamma_mean'][pi]:.6f}",
                f"{s2['gamma_mean'][pi]:.6f}",
            ])
    print(f"Written: {path}")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Scroll conjoint HB estimation")
    p.add_argument("--config",    default="scroll_conjoint_config.json")
    p.add_argument("--catalogue", default="scroll_catalogue.csv")
    p.add_argument("--responses", default="scroll_responses.csv")
    p.add_argument("--out_dir",   default=".")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--iters",     type=int, default=None)
    p.add_argument("--burn_in",   type=int, default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # scipy.stats draws from numpy global state

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    if args.iters:   config.mcmc["iterations"] = args.iters
    if args.burn_in: config.mcmc["burn_in"]    = args.burn_in
    print(f"K={config.K}  M={config.M} interaction pairs")

    print(f"Loading catalogue: {args.catalogue}")
    catalogue = load_catalogue(args.catalogue, config)
    print(f"{len(catalogue)} items")

    print(f"Loading responses: {args.responses}")
    respondents = load_and_prepare(args.responses, catalogue, config)
    n = len(respondents)
    print(f"{n} respondents with usable data")
    if n < 5:
        sys.exit("Too few respondents — need at least 5")

    n_vis = sum(r.n_vis   for r in respondents)
    n_lk  = sum(r.n_liked for r in respondents)
    n_dk  = sum(int(r.y_disliked.sum()) for r in respondents)
    n_ch  = sum(1 for r in respondents if r.chosen_idx >= 0)
    print(f"Avg visible: {n_vis/n:.1f}  liked: {n_lk/n:.1f}  "
          f"disliked: {n_dk/n:.1f}  stage-2 choices: {n_ch}/{n}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Selection model (P(liked | visible)) ===")
    sel = run_model(respondents, config, "liked", rng)

    print("\n=== Rejection model (P(disliked | visible)) ===")
    rej = run_model(respondents, config, "disliked", rng)

    print("\n=== Stage-2 model (P(chosen | liked set)) ===")
    s2  = run_model(respondents, config, "stage2", rng)

    write_betas(str(out / "hb_selection_betas.csv"), sel, config)
    write_betas(str(out / "hb_rejection_betas.csv"), rej, config)
    write_betas(str(out / "hb_stage2_betas.csv"),    s2,  config)
    write_population(str(out / "hb_population.csv"), sel, rej, s2, config)

    print(f"\nDone. Posterior samples: {sel['n_samples']} per model.")
    print(f"tau_sel={sel['tau']:.3f}  tau_rej={rej['tau']:.3f}")
    print(f"Global shrinkage: sel={sel['tau_sq']:.4f}  "
          f"rej={rej['tau_sq']:.4f}  s2={s2['tau_sq']:.4f}")


if __name__ == "__main__":
    main()
