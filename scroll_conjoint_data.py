#!/usr/bin/env python3
"""
scroll_conjoint_data.py

Generates synthetic test data for scroll_conjoint_hb.py.
Equivalent to scroll_conjoint_data.php — same scenario (hotel booking),
same CSV output formats, different RNG so numbers differ.

Outputs:
  scroll_conjoint_config.json  (created if absent)
  scroll_catalogue.csv
  scroll_responses.csv

Attribute coding (hotel scenario)
  att0  star_rating  5 levels: 0=1-star … 4=5-star
  att1  price_band   4 levels: 0=budget … 3=luxury
  att2  location     3 levels: 0=outskirts 1=suburban 2=central
  att3  amenities    4 levels: 0=basic … 3=premium
  att4  review_score 3 levels: 0=good 1=very_good 2=excellent

True latent segments:
  Segment 0 "Quality"  (40%): strong selection on star_rating, review_score
  Segment 1 "Value"    (40%): strong rejection of high price_band
  Segment 2 "Location" (20%): strong selection on location, amenities

Response values:
  liked    — item was positively selected in stage-1
  disliked — item was negatively selected in stage-1
  neutral  — item was visible but neither liked nor disliked
  chosen   — item selected as preferred in stage-2 (additional row, empty position fields)
  unseen   — item was below the scroll point (excluded from likelihood in HB)
"""

import numpy as np
from scipy.special import expit
import csv
import json
from pathlib import Path

RNG = np.random.default_rng(42)

# ── config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "attributes": [
        {"name": "star_rating",  "levels": 5},
        {"name": "price_band",   "levels": 4},
        {"name": "location",     "levels": 3},
        {"name": "amenities",    "levels": 4},
        {"name": "review_score", "levels": 3},
    ],
    "grid_rows": 10,
    "grid_cols":  5,
    "mcmc": {"iterations": 4000, "burn_in": 1500,
             "adapt_every": 50,  "target_accept": 0.234},
}

ATTRIBUTES = CONFIG["attributes"]
GRID_ROWS  = CONFIG["grid_rows"]
GRID_COLS  = CONFIG["grid_cols"]
N_ATTS     = len(ATTRIBUTES)
K = sum(a["levels"] - 1 for a in ATTRIBUTES)  # 4+3+2+3+2 = 14

# ── feature helpers ─────────────────────────────────────────────────────────────

def effects_code(level: int, n_levels: int) -> np.ndarray:
    if n_levels <= 1:
        return np.array([])
    c = np.zeros(n_levels - 1)
    if level < n_levels - 1:
        c[level] = 1.0
    else:
        c[:] = -1.0
    return c

def build_x(levels: list) -> np.ndarray:
    return np.concatenate([effects_code(lv, a["levels"]) for lv, a in zip(levels, ATTRIBUTES)])

# ── true segment parameters ────────────────────────────────────────────────────
# K=14 layout: star_rating[0:4], price_band[4:7], location[7:9],
#              amenities[9:12], review_score[12:14]

SEG_SEL = np.array([
    # Quality: strong star_rating and review_score selection
    [1.2, 0.8, 0.4, 0.0,  0.0, 0.0, 0.0,  0.1, 0.0,  0.1, 0.0, 0.0,  0.8, 0.4],
    # Value: weak selection (mostly review_score)
    [0.3, 0.2, 0.1, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.5, 0.3],
    # Location: strong location and amenities
    [0.2, 0.1, 0.0, 0.0,  0.0, 0.0, 0.0,  0.9, 0.5,  0.7, 0.4, 0.1,  0.2, 0.1],
])

SEG_REJ = np.array([
    # Quality: reject bad stars and reviews
    [0.8, 0.5, 0.2, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0,  0.6, 0.3],
    # Value: reject high price (price_band level-2 = premium vs luxury ref)
    [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.9,  0.0, 0.0,  0.0, 0.0, 0.0,  0.2, 0.1],
    # Location: reject bad location
    [0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.9, 0.4,  0.0, 0.0, 0.0,  0.0, 0.0],
])

SEG_PROBS  = [0.40, 0.40, 0.20]
TAU_SEL    = 0.5   # logit threshold for liked
TAU_REJ    = 0.3   # logit threshold for disliked

# ── catalogue: 200 items ───────────────────────────────────────────────────────

N_ITEMS        = 200
N_RESPONDENTS  = 80
ITEMS_PER_PAGE = GRID_ROWS * GRID_COLS  # 50

item_ids = [f"I{i:03d}" for i in range(1, N_ITEMS + 1)]
item_levels = {
    iid: [RNG.integers(0, a["levels"]) for a in ATTRIBUTES]
    for iid in item_ids
}
item_x = {iid: build_x(levels) for iid, levels in item_levels.items()}

config_path = Path("scroll_conjoint_config.json")
if not config_path.exists():
    config_path.write_text(json.dumps(CONFIG, indent=2))
    print(f"Created {config_path}")

with open("scroll_catalogue.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_id"] + [f"att{i}_level" for i in range(N_ATTS)])
    for iid in item_ids:
        w.writerow([iid] + item_levels[iid])
print(f"Wrote scroll_catalogue.csv ({N_ITEMS} items, K={K})")

# ── responses ──────────────────────────────────────────────────────────────────

stats = {"liked": 0, "disliked": 0, "neutral": 0, "unseen": 0, "chosen": 0, "total": 0}

with open("scroll_responses.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["respondent_id", "item_id", "position_row", "position_col",
                "scroll_rows_visible", "response"])

    for r in range(1, N_RESPONDENTS + 1):
        rid = f"R{r:03d}"

        # Assign segment
        seg = RNG.choice(3, p=SEG_PROBS)

        # Individual betas: segment mean + heterogeneity noise, clipped to ≥ 0
        b_sel = np.maximum(0.0, SEG_SEL[seg] + RNG.normal(0, 0.15, K))
        b_rej = np.maximum(0.0, SEG_REJ[seg] + RNG.normal(0, 0.15, K))

        # Pick 50 items and assign random grid positions
        shown_ids = RNG.choice(item_ids, size=ITEMS_PER_PAGE, replace=False).tolist()
        positions = [(row, col)
                     for row in range(GRID_ROWS)
                     for col in range(GRID_COLS)]
        RNG.shuffle(positions)  # randomise grid order
        pos_map = {iid: positions[i] for i, iid in enumerate(shown_ids)}

        # Scroll depth: 70% chance of seeing ≥7 rows
        scroll_rows = int(RNG.integers(7, GRID_ROWS + 1) if RNG.random() < 0.70
                          else RNG.integers(4, 7))

        liked_items = {}  # iid -> utility for stage-2 softmax

        for iid in shown_ids:
            row, col = pos_map[iid]
            visible  = row < scroll_rows
            stats["total"] += 1

            if not visible:
                stats["unseen"] += 1
                w.writerow([rid, iid, row, col, scroll_rows, "unseen"])
                continue

            x = item_x[iid]
            u_sel = float(b_sel @ x) - TAU_SEL + RNG.normal(0, 0.3)
            u_rej = float(b_rej @ x) - TAU_REJ + RNG.normal(0, 0.3)

            p_liked    = float(expit(u_sel))
            p_disliked = float(expit(u_rej)) * (1.0 - p_liked)

            rn = RNG.random()
            if rn < p_liked:
                response = "liked"
                liked_items[iid] = float(b_sel @ x)
                stats["liked"] += 1
            elif rn < p_liked + p_disliked:
                response = "disliked"
                stats["disliked"] += 1
            else:
                response = "neutral"
                stats["neutral"] += 1

            w.writerow([rid, iid, row, col, scroll_rows, response])

        # Stage-2: softmax choice from liked set (if ≥2 liked items)
        if len(liked_items) >= 2:
            iids_l = list(liked_items.keys())
            utils  = np.array([liked_items[i] for i in iids_l])
            utils -= utils.max()
            probs   = np.exp(utils)
            probs  /= probs.sum()
            chosen  = RNG.choice(iids_l, p=probs)
            # Append a 'chosen' marker row (empty position fields)
            w.writerow([rid, chosen, "", "", scroll_rows, "chosen"])
            stats["chosen"] += 1

seen = stats["total"] - stats["unseen"]
print(f"Wrote scroll_responses.csv")
print(f"Stats: total={stats['total']}  "
      f"liked={stats['liked']} ({100*stats['liked']//max(1,seen)}%)  "
      f"disliked={stats['disliked']} ({100*stats['disliked']//max(1,seen)}%)  "
      f"neutral={stats['neutral']}  unseen={stats['unseen']}  chosen={stats['chosen']}")
print(f"Respondents: {N_RESPONDENTS}  Segments: Quality=40% Value=40% Location=20%")
print(f"K={K} main-effect params  True betas are sparse (horseshoe should recover this)")
