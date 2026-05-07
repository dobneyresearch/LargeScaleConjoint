<?php
/**
 * scroll_conjoint_data.php
 *
 * Generates synthetic test data for scroll_conjoint_hb.php.
 * Simulates a hotel-booking-style scroll page with:
 *   - 200-item catalogue
 *   - 80 respondents, each seeing 50 items in a 10×5 grid
 *   - Scroll depth 60–100% (rows visible)
 *   - Sparse true utilities: respondents belong to one of 3 segments
 *   - Like rate ~20–30%, dislike rate ~10–20%, rest neutral
 *   - Stage-2 choice from liked set when ≥2 items liked
 *
 * Outputs:
 *   scroll_conjoint_config.json   (writes if not present)
 *   scroll_catalogue.csv
 *   scroll_responses.csv
 *
 * Data formats
 * ============
 * scroll_catalogue.csv
 *   item_id  : string  e.g. "I001"
 *   att0_level..attN_level : int (0-based level index for each attribute)
 *
 * scroll_responses.csv
 *   respondent_id      : string  e.g. "R001"
 *   item_id            : string  e.g. "I047"
 *   position_row       : int     0-based row in the display grid
 *   position_col       : int     0-based column in the display grid
 *   scroll_rows_visible: int     number of rows the respondent scrolled to see
 *                                Items with position_row >= scroll_rows_visible are UNSEEN
 *                                and must be excluded from the likelihood
 *   response           : string  'liked' | 'disliked' | 'neutral' | 'chosen'
 *                                'chosen' = selected in stage-2 preference pick
 *                                (only one item per respondent may be 'chosen')
 *
 * Attribute coding (hotel scenario)
 * ===================================
 * att0  star_rating  5 levels: 0=1-star … 4=5-star   (higher = better for quality seekers)
 * att1  price_band   4 levels: 0=budget … 3=luxury    (lower = better for value seekers)
 * att2  location     3 levels: 0=outskirts 1=suburban 2=central  (higher = better)
 * att3  amenities    4 levels: 0=basic … 3=premium    (higher = better)
 * att4  review_score 3 levels: 0=good 1=very_good 2=excellent  (higher = better)
 *
 * True latent segments (selection betas are positive-constrained):
 *   Segment 0 "Quality"  (40%): strong weight on star_rating, review_score
 *   Segment 1 "Value"    (40%): strong rejection of high price, moderate on review_score
 *   Segment 2 "Location" (20%): strong selection on location, amenities
 */

mt_srand(42);

// ── helpers ────────────────────────────────────────────────────────────────

function rnd(): float { return mt_rand() / mt_getrandmax(); }

function rand_normal_gen(float $mu = 0.0, float $sigma = 1.0): float {
    static $spare = null;
    if ($spare !== null) { $v = $spare; $spare = null; return $mu + $sigma * $v; }
    do { $u = rnd()*2-1; $v = rnd()*2-1; $s = $u*$u + $v*$v; } while ($s >= 1 || $s == 0);
    $m = sqrt(-2*log($s)/$s);
    $spare = $v * $m;
    return $mu + $sigma * ($u * $m);
}

function logistic_gen(float $x): float {
    return 1.0 / (1.0 + exp(-max(-35.0, min(35.0, $x))));
}

function effects_code_gen(int $level, int $nLevels): array {
    if ($nLevels === 1) return [];
    $c = array_fill(0, $nLevels - 1, 0.0);
    if ($level < $nLevels - 1) { $c[$level] = 1.0; }
    else                        { $c = array_fill(0, $nLevels - 1, -1.0); }
    return $c;
}

function build_x(array $levels, array $attributes): array {
    $x = [];
    foreach ($attributes as $i => $att) {
        foreach (effects_code_gen($levels[$i], $att['levels']) as $v) $x[] = $v;
    }
    return $x;
}

function dot_gen(array $a, array $b): float {
    $s = 0.0;
    for ($i = 0; $i < min(count($a), count($b)); $i++) $s += $a[$i] * $b[$i];
    return $s;
}

// ── config ─────────────────────────────────────────────────────────────────

$config_path = 'scroll_conjoint_config.json';
if (!file_exists($config_path)) {
    file_put_contents($config_path, json_encode([
        'attributes' => [
            ['name'=>'star_rating',  'levels'=>5],
            ['name'=>'price_band',   'levels'=>4],
            ['name'=>'location',     'levels'=>3],
            ['name'=>'amenities',    'levels'=>4],
            ['name'=>'review_score', 'levels'=>3],
        ],
        'grid_rows' => 10,
        'grid_cols' =>  5,
        'mcmc' => ['iterations'=>4000, 'burn_in'=>1500, 'adapt_every'=>50, 'target_accept'=>0.234],
    ], JSON_PRETTY_PRINT));
    echo "Created $config_path\n";
}

$config     = json_decode(file_get_contents($config_path), true);
$attributes = $config['attributes'];
$grid_rows  = $config['grid_rows'];   // 10
$grid_cols  = $config['grid_cols'];   // 5
$n_atts     = count($attributes);

// K = number of main-effect parameters (sum of levels-1)
$K = 0;
foreach ($attributes as $att) $K += $att['levels'] - 1;
// K = 4+3+2+3+2 = 14

// ── catalogue: 200 items ───────────────────────────────────────────────────

$n_items       = 200;
$n_respondents = 80;
$items_per_screen = $grid_rows * $grid_cols;  // 50

$catalogue = [];
$cat_fh = fopen('scroll_catalogue.csv', 'w');
$cat_header = ['item_id'];
foreach ($attributes as $i => $att) $cat_header[] = "att{$i}_level";
fputcsv($cat_fh, $cat_header);

for ($i = 1; $i <= $n_items; $i++) {
    $iid    = sprintf('I%03d', $i);
    $levels = [];
    foreach ($attributes as $att) $levels[] = mt_rand(0, $att['levels'] - 1);
    $catalogue[$iid] = [
        'levels' => $levels,
        'x'      => build_x($levels, $attributes),
    ];
    $row = [$iid, ...$levels];
    fputcsv($cat_fh, $row);
}
fclose($cat_fh);
echo "Wrote scroll_catalogue.csv ($n_items items, K=$K)\n";

// ── true segment betas ─────────────────────────────────────────────────────
// Parameter layout (K=14):
//   att0 star_rating : params 0-3  (4 params)
//   att1 price_band  : params 4-6  (3 params)
//   att2 location    : params 7-8  (2 params)
//   att3 amenities   : params 9-11 (3 params)
//   att4 review_score: params 12-13(2 params)
//
// Selection betas: positive values = drives liking
// Rejection betas: positive values = drives disliking
//
// Thresholds: utility > tau_sel → liked; utility > tau_rej → disliked
// (independent logit models, not ordered)

$seg_sel = [
    // Segment 0: Quality — strong star_rating and review_score
    [1.2, 0.8, 0.4, 0.0,   // star_rating (levels 0-3, last implied)
     0.0, 0.0, 0.0,         // price_band (don't care much)
     0.1, 0.0,              // location (mild)
     0.1, 0.0, 0.0,         // amenities (mild)
     0.8, 0.4],             // review_score

    // Segment 1: Value — moderate review, mostly doesn't drive selection
    [0.3, 0.2, 0.1, 0.0,   // star_rating (mild)
     0.0, 0.0, 0.0,         // price_band (price drives rejection not selection)
     0.0, 0.0,              // location
     0.0, 0.0, 0.0,         // amenities
     0.5, 0.3],             // review_score

    // Segment 2: Location — strong location and amenities drive selection
    [0.2, 0.1, 0.0, 0.0,   // star_rating (mild)
     0.0, 0.0, 0.0,         // price_band
     0.9, 0.5,              // location (central = big selection driver)
     0.7, 0.4, 0.1,         // amenities
     0.2, 0.1],             // review_score
];

$seg_rej = [
    // Segment 0: Quality — reject low review scores (but betas are for repulsion direction)
    // In rejection model: positive beta_k means "this feature makes rejection more likely"
    // High star_rating→ low repulsion; low star_rating → high repulsion
    // Effects coding: level 0 gives code [1,0,0,0] for 5-level att
    // So param 0 = β for level-0-vs-last; a positive β here means level 0 increases rejection
    // That's correct: low star rating increases rejection
    [0.8, 0.5, 0.2, 0.0,   // star_rating repulsion (bad stars → rejected)
     0.0, 0.0, 0.0,         // price_band (quality seg doesn't reject on price)
     0.0, 0.0,              // location
     0.0, 0.0, 0.0,         // amenities
     0.6, 0.3],             // review_score repulsion

    // Segment 1: Value — reject high price
    [0.0, 0.0, 0.0, 0.0,   // star_rating (don't reject on stars)
     0.0, 0.0, 0.9,         // price_band: param 2 = level-2-vs-last(3)=luxury rejection
     0.0, 0.0,              // location
     0.0, 0.0, 0.0,         // amenities
     0.2, 0.1],             // review_score (mild)

    // Segment 2: Location — reject bad location
    [0.0, 0.0, 0.0, 0.0,   // star_rating
     0.0, 0.0, 0.0,         // price_band
     0.9, 0.4,              // location: params 7,8 = outskirts/suburban repulsion
     0.0, 0.0, 0.0,         // amenities
     0.0, 0.0],             // review_score
];

$seg_probs = [0.40, 0.40, 0.20];  // segment membership probabilities
$tau_sel   = 0.5;   // threshold: utility > 0.5 → liked
$tau_rej   = 0.3;   // threshold: repulsion utility > 0.3 → disliked

// ── generate responses ─────────────────────────────────────────────────────

$resp_fh = fopen('scroll_responses.csv', 'w');
fputcsv($resp_fh, [
    'respondent_id', 'item_id',
    'position_row', 'position_col',
    'scroll_rows_visible', 'response',
]);

$item_ids = array_keys($catalogue);
$stats    = ['liked'=>0,'disliked'=>0,'neutral'=>0,'chosen'=>0,'unseen'=>0,'total'=>0];

for ($r = 1; $r <= $n_respondents; $r++) {
    $rid = sprintf('R%03d', $r);

    // Assign segment
    $p   = rnd();
    $seg = 0;
    $cum = 0.0;
    foreach ($seg_probs as $s => $sp) {
        $cum += $sp;
        if ($p < $cum) { $seg = $s; break; }
    }

    // Individual betas: segment mean + noise (heterogeneity)
    $true_sel = [];
    $true_rej = [];
    for ($k = 0; $k < $K; $k++) {
        $true_sel[$k] = max(0.0, $seg_sel[$seg][$k] + rand_normal_gen(0, 0.15));
        $true_rej[$k] = max(0.0, $seg_rej[$seg][$k] + rand_normal_gen(0, 0.15));
    }

    // Pick 50 items randomly (without replacement)
    $shown = array_splice($item_ids, 0, 0);  // clone
    shuffle($item_ids);
    $screen_items = array_slice($item_ids, 0, $items_per_screen);

    // Assign grid positions (row-major)
    $positions = [];
    $idx = 0;
    for ($row = 0; $row < $grid_rows; $row++) {
        for ($col = 0; $col < $grid_cols; $col++) {
            if ($idx >= $items_per_screen) break 2;
            $positions[$idx] = [$row, $col];
            $idx++;
        }
    }
    // Shuffle positions so items appear in random grid locations
    shuffle($positions);

    // Scroll depth: respondent sees rows 0..(scroll_rows_visible-1)
    // Simulate: 70% chance of seeing ≥7 rows, 30% partial view
    $scroll_rows = (rnd() < 0.70)
        ? mt_rand(7, $grid_rows)
        : mt_rand(4, 6);

    // Evaluate visible items
    $liked_items = [];

    foreach ($screen_items as $pos_idx => $iid) {
        [$row, $col] = $positions[$pos_idx];
        $visible = ($row < $scroll_rows);

        if (!$visible) {
            $stats['unseen']++;
            $stats['total']++;
            // Unseen items are NOT written to the responses file —
            // the HB script infers unseen from scroll_rows_visible vs position_row
            // (we still write them so the researcher can verify scroll cutoff)
            fputcsv($resp_fh, [$rid, $iid, $row, $col, $scroll_rows, 'unseen']);
            continue;
        }

        $x = $catalogue[$iid]['x'];

        $u_sel = dot_gen($true_sel, $x) - $tau_sel + rand_normal_gen(0, 0.3);
        $u_rej = dot_gen($true_rej, $x) - $tau_rej + rand_normal_gen(0, 0.3);

        $p_liked    = logistic_gen($u_sel);
        $p_disliked = logistic_gen($u_rej) * (1.0 - $p_liked);  // dislike conditional on not liked

        $rn = rnd();
        if ($rn < $p_liked) {
            $response = 'liked';
            $liked_items[$iid] = dot_gen($true_sel, $x);  // store utility for stage-2
            $stats['liked']++;
        } elseif ($rn < $p_liked + $p_disliked) {
            $response = 'disliked';
            $stats['disliked']++;
        } else {
            $response = 'neutral';
            $stats['neutral']++;
        }

        fputcsv($resp_fh, [$rid, $iid, $row, $col, $scroll_rows, $response]);
        $stats['total']++;
    }

    // Stage-2: choose preferred from liked set (if ≥2 liked items)
    if (count($liked_items) >= 2) {
        // Soft-max choice weighted by utility
        $max_u   = max($liked_items);
        $weights = array_map(fn($u) => exp($u - $max_u), $liked_items);
        $sum_w   = array_sum($weights);
        $rn      = rnd() * $sum_w;
        $cum     = 0.0;
        $chosen  = null;
        foreach ($weights as $iid => $w) {
            $cum += $w;
            if ($rn <= $cum) { $chosen = $iid; break; }
        }
        if ($chosen !== null) {
            // Overwrite that item's response row to 'chosen'
            // Since we've already written it as 'liked', we track separately
            // The HB script treats 'chosen' as a superset of 'liked' for Stage-1
            // We use a second pass — instead, append a 'chosen' marker row
            // In practice: the response file uses 'chosen' instead of 'liked' for the chosen item
            // For the generator, we'd need to buffer — simpler: write chosen as a separate note column
            // Simplest: in the responses file, the chosen item already appears as 'liked';
            // we append one extra row with response='chosen' and the same item to flag it.
            // HB script: if an item_id appears as 'chosen', it is treated as liked for Stage-1
            //            and contributes to the Stage-2 MNL.
            fputcsv($resp_fh, [$rid, $chosen, '', '', $scroll_rows, 'chosen']);
            $stats['chosen']++;
        }
    }
}

fclose($resp_fh);

echo "Wrote scroll_responses.csv\n";
echo sprintf(
    "Stats: total=%d  liked=%d (%.0f%%)  disliked=%d (%.0f%%)  neutral=%d  unseen=%d  chosen=%d\n",
    $stats['total'],
    $stats['liked'],    100*$stats['liked']   /max(1,$stats['total']-$stats['unseen']),
    $stats['disliked'], 100*$stats['disliked']/max(1,$stats['total']-$stats['unseen']),
    $stats['neutral'],
    $stats['unseen'],
    $stats['chosen']
);
echo sprintf("Respondents: %d  Segments: Quality=40%% Value=40%% Location=20%%\n", $n_respondents);
echo "K=$K main-effect params  True betas are sparse (horseshoe should recover this)\n";
