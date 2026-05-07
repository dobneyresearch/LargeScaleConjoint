<?php
/**
 * scroll_conjoint_hb.php
 *
 * Hierarchical Bayes estimation for large-scale scroll-based conjoint analysis.
 *
 * Three separate models, each producing individual-level part-worth estimates:
 *   Selection model  — P(liked | visible)    positive-constrained betas, horseshoe prior
 *   Rejection model  — P(disliked | visible)  positive-constrained betas, horseshoe prior
 *   Stage-2 model    — P(chosen | liked set)  unconstrained betas, horseshoe prior
 *
 * Population interactions (shared across respondents) estimated jointly in each model.
 *
 * Scroll-resolved attention:
 *   Items with position_row >= scroll_rows_visible are structurally unseen and
 *   excluded from the likelihood. Items that are visible but not selected are
 *   true neutral observations (no mixture model needed).
 *
 * Usage:
 *   php scroll_conjoint_hb.php \
 *     --config scroll_conjoint_config.json \
 *     --catalogue scroll_catalogue.csv \
 *     --responses scroll_responses.csv \
 *     [--out_dir .] [--seed 42]
 *
 * Input formats — see scroll_conjoint_data.php for full documentation.
 *
 * Output:
 *   hb_selection_betas.csv   respondent_id, att0_name_l0..., accept_rate
 *   hb_rejection_betas.csv   same layout
 *   hb_stage2_betas.csv      same layout
 *   hb_population.csv        parameter, sel_alpha, rej_alpha, s2_alpha,
 *                            sel_interaction, rej_interaction, s2_interaction
 */

// ── constants ───────────────────────────────────────────────────────────────

define('R_LIKED',    'liked');
define('R_DISLIKED', 'disliked');
define('R_NEUTRAL',  'neutral');
define('R_CHOSEN',   'chosen');
define('R_UNSEEN',   'unseen');

// ── config ──────────────────────────────────────────────────────────────────

function sc_load_config(string $path): array {
    $c = json_decode(file_get_contents($path), true);
    if (!$c) die("Cannot read config: $path\n");

    // K = total main-effect parameters
    $K = 0;
    $att_offsets = [];  // [start_idx, end_idx] per attribute in the K-vector
    foreach ($c['attributes'] as $att) {
        $att_offsets[] = [$K, $K + $att['levels'] - 2];
        $K += $att['levels'] - 1;
    }
    $c['K']           = $K;
    $c['att_offsets'] = $att_offsets;

    // Interaction pairs: one cross-product term per attribute pair
    // Uses the first effects-coded parameter of each attribute as representative
    $pairs = [];
    $nA    = count($c['attributes']);
    for ($a = 0; $a < $nA - 1; $a++) {
        for ($b = $a + 1; $b < $nA; $b++) {
            $pairs[] = [$a, $b, $att_offsets[$a][0], $att_offsets[$b][0]];
        }
    }
    $c['interaction_pairs'] = $pairs;
    $c['M']  = count($pairs);

    $c['mcmc'] += ['iterations'=>3000,'burn_in'=>1000,'adapt_every'=>50,'target_accept'=>0.234];
    return $c;
}

// ── catalogue loading ───────────────────────────────────────────────────────

function sc_load_catalogue(string $path, array $config): array {
    $items = [];
    $fh    = fopen($path, 'r');
    $hdr   = fgetcsv($fh);
    while (($row = fgetcsv($fh)) !== false) {
        $rec    = array_combine($hdr, $row);
        $iid    = $rec['item_id'];
        $levels = [];
        foreach ($config['attributes'] as $i => $att) {
            $levels[] = (int)($rec["att{$i}_level"] ?? 0);
        }
        $xm = sc_build_main($levels, $config['attributes']);
        $xi = sc_build_interact($xm, $config['interaction_pairs']);
        $items[$iid] = ['x_main' => $xm, 'x_inter' => $xi, 'levels' => $levels];
    }
    fclose($fh);
    return $items;
}

// ── responses loading ───────────────────────────────────────────────────────

function sc_load_responses(string $path, array $catalogue, array $config): array {
    // Returns respondent_id → [
    //   'visible'   => [item records for all seen items],
    //   'liked_set' => [item records for liked+chosen],
    //   'chosen'    => item_id|null
    // ]
    //
    // A row with response='chosen' flags the stage-2 pick; that item_id was
    // also written as 'liked' in the data. The chosen row may have empty
    // position fields (it is only used to identify which item was chosen).

    $raw = [];
    $fh  = fopen($path, 'r');
    $hdr = fgetcsv($fh);
    while (($row = fgetcsv($fh)) !== false) {
        $rec = array_combine($hdr, $row);
        $raw[$rec['respondent_id']][] = $rec;
    }
    fclose($fh);

    $respondents = [];
    foreach ($raw as $rid => $rows) {
        // Find chosen item (may appear as a separate 'chosen' row with empty position)
        $chosen = null;
        foreach ($rows as $rec) {
            if ($rec['response'] === R_CHOSEN) { $chosen = $rec['item_id']; break; }
        }

        // Scroll threshold (same for all rows of this respondent)
        $scroll_rows = (int)($rows[0]['scroll_rows_visible'] ?? PHP_INT_MAX);

        $visible   = [];
        $liked_set = [];

        foreach ($rows as $rec) {
            $resp = $rec['response'];
            $iid  = $rec['item_id'];

            // Skip the bare 'chosen' marker row (no position data)
            if ($resp === R_CHOSEN && $rec['position_row'] === '') continue;
            // Skip structurally unseen
            if ($resp === R_UNSEEN) continue;
            // Skip if row position is beyond scroll (belt-and-braces check)
            $pos_row = (int)$rec['position_row'];
            if ($pos_row >= $scroll_rows) continue;
            // Skip unknown items
            if (!isset($catalogue[$iid])) continue;

            $is_liked = in_array($resp, [R_LIKED, R_CHOSEN]);
            $item = [
                'item_id'   => $iid,
                'row'       => $pos_row,
                'col'       => (int)$rec['position_col'],
                'x_main'    => $catalogue[$iid]['x_main'],
                'x_inter'   => $catalogue[$iid]['x_inter'],
                'liked'     => (int)$is_liked,
                'disliked'  => (int)($resp === R_DISLIKED),
            ];

            $visible[] = $item;
            if ($is_liked) $liked_set[] = $item;
        }

        if (count($visible) < 5) continue;  // too little data for this respondent

        $respondents[$rid] = [
            'visible'    => $visible,
            'liked_set'  => $liked_set,
            'chosen'     => $chosen,
        ];
    }
    return $respondents;
}

// ── feature engineering ─────────────────────────────────────────────────────

function sc_effects_code(int $level, int $nLevels): array {
    if ($nLevels <= 1) return [];
    $c = array_fill(0, $nLevels - 1, 0.0);
    if ($level < $nLevels - 1) $c[$level] = 1.0;
    else                        $c = array_fill(0, $nLevels - 1, -1.0);
    return $c;
}

function sc_build_main(array $levels, array $attributes): array {
    $x = [];
    foreach ($attributes as $i => $att) {
        foreach (sc_effects_code($levels[$i], $att['levels']) as $v) $x[] = (float)$v;
    }
    return $x;
}

function sc_build_interact(array $x_main, array $pairs): array {
    $v = [];
    foreach ($pairs as $pair) {
        [, , $ia, $ib] = $pair;
        $v[] = $x_main[$ia] * $x_main[$ib];
    }
    return $v;
}

// ── math helpers ─────────────────────────────────────────────────────────────

function sc_dot(array $a, array $b): float {
    $s = 0.0;
    $n = min(count($a), count($b));
    for ($i = 0; $i < $n; $i++) $s += $a[$i] * $b[$i];
    return $s;
}

function sc_logistic(float $x): float {
    if ($x >  35.0) return 1.0;
    if ($x < -35.0) return 0.0;
    return 1.0 / (1.0 + exp(-$x));
}

function sc_rand_normal(float $mu = 0.0, float $sigma = 1.0): float {
    static $spare = null;
    if ($spare !== null) { $v = $spare; $spare = null; return $mu + $sigma * $v; }
    do { $u = lcg_value()*2-1; $v = lcg_value()*2-1; $s = $u*$u+$v*$v; } while ($s>=1||$s==0);
    $m = sqrt(-2*log($s)/$s);
    $spare = $v * $m;
    return $mu + $sigma * ($u * $m);
}

function sc_rand_gamma(float $shape, float $rate = 1.0): float {
    if ($shape < 1.0) return sc_rand_gamma(1.0 + $shape, $rate) * pow(lcg_value(), 1.0/$shape);
    $d = $shape - 1.0/3.0; $c = 1.0/sqrt(9.0*$d);
    while (true) {
        do { $x = sc_rand_normal(); $v = 1.0 + $c*$x; } while ($v <= 0.0);
        $v = $v*$v*$v; $u = lcg_value();
        if ($u < 1.0 - 0.0331*$x*$x*$x*$x) return $d*$v/$rate;
        if (log($u) < 0.5*$x*$x + $d*(1-$v+log($v))) return $d*$v/$rate;
    }
}

function sc_rand_inv_gamma(float $shape, float $rate): float {
    return 1.0 / sc_rand_gamma($shape, $rate);
}

function sc_log_normal_pdf(float $x, float $mu, float $sigma): float {
    $z = ($x - $mu) / $sigma;
    return -0.5*$z*$z - log($sigma) - 0.9189385332;  // -0.5*log(2π)
}

// Truncated normal: sample from N(mu, sigma²) on [0, ∞)
// Uses exponential-envelope rejection (Robert 1995) when mu < 0
function sc_rand_tnorm_pos(float $mu, float $sigma): float {
    if ($mu >= 0) {
        while (true) { $x = sc_rand_normal($mu, $sigma); if ($x >= 0.0) return $x; }
    }
    $a = -$mu / $sigma;
    $lambda = ($a + sqrt($a*$a + 4.0)) / 2.0;
    while (true) {
        $z = $mu + $sigma * ($a + (-log(lcg_value())) / $lambda);
        if ($z < 0.0) continue;
        $rho = exp(-0.5 * (($z/$sigma - $mu/$sigma) - $lambda) ** 2);
        if (lcg_value() < $rho) return $z;
    }
}

// ── horseshoe prior (Makalic & Schmidt 2016) ─────────────────────────────────
// Hierarchy:
//   alpha_k | lambda_k, tau ~ N(0, lambda_k² tau²)    [population mean for param k]
//   lambda_k² | nu_k ~ InvGamma(1/2, 1/nu_k)
//   nu_k ~ InvGamma(1/2, 1)
//   tau² | xi ~ InvGamma((K+1)/2, 1/xi)
//   xi ~ InvGamma(1/2, 1)
//
// For the selection/rejection models, alpha_k is positive-constrained (truncated normal).

function hs_update_lambda_sq(float $alpha_k, float $tau_sq, float $nu_k): float {
    $rate = 1.0/$nu_k + $alpha_k*$alpha_k / (2.0*max(1e-10, $tau_sq));
    return sc_rand_inv_gamma(1.0, $rate);
}

function hs_update_nu(float $lambda_sq): float {
    return sc_rand_inv_gamma(1.0, 1.0 + 1.0/max(1e-10, $lambda_sq));
}

function hs_update_tau_sq(array $alpha, array $lambda_sq, float $xi): float {
    $K    = count($alpha);
    $rate = 1.0/$xi;
    for ($k = 0; $k < $K; $k++) {
        $rate += $alpha[$k]*$alpha[$k] / (2.0*max(1e-10, $lambda_sq[$k]));
    }
    return sc_rand_inv_gamma(($K+1)/2.0, $rate);
}

function hs_update_xi(float $tau_sq): float {
    return sc_rand_inv_gamma(1.0, 1.0 + 1.0/max(1e-10, $tau_sq));
}

// Conjugate update for alpha_k (population mean) given individual betas
// sigma_sq_k: between-respondent variance for param k
function hs_update_alpha(array $betas_k, float $sigma_sq_k, float $lambda_sq_k,
                          float $tau_sq, bool $positive_only): float {
    $n         = count($betas_k);
    $post_prec = $n / max(1e-6, $sigma_sq_k)
               + 1.0 / max(1e-10, $lambda_sq_k * $tau_sq);
    $post_var  = 1.0 / $post_prec;
    $post_mean = $post_var * array_sum($betas_k) / max(1e-6, $sigma_sq_k);

    if ($positive_only) return sc_rand_tnorm_pos($post_mean, sqrt($post_var));
    return sc_rand_normal($post_mean, sqrt($post_var));
}

// ── likelihood functions ──────────────────────────────────────────────────────

// Binary logit log-likelihood for all visible items of one respondent
// beta_main: individual main-effect part-worths [K]
// gamma:     population interaction effects [M]
// tau:       intercept threshold
// target:    'liked' (selection) or 'disliked' (rejection)
function sc_binary_ll(array $beta_main, array $gamma, float $tau,
                       array $visible, string $target): float {
    $ll = 0.0;
    $key = ($target === 'liked') ? 'liked' : 'disliked';
    foreach ($visible as $item) {
        $u = sc_dot($beta_main, $item['x_main'])
           + sc_dot($gamma,     $item['x_inter'])
           - $tau;
        $p = sc_logistic($u);
        $p = max(1e-10, min(1.0 - 1e-10, $p));
        $ll += $item[$key] ? log($p) : log(1.0 - $p);
    }
    return $ll;
}

// MNL log-likelihood for stage-2 choice from liked set
function sc_stage2_ll(array $beta_main, array $gamma, array $liked_set,
                       string $chosen_id): float {
    if (count($liked_set) < 2) return 0.0;
    $max_u = -INF;
    $utils = [];
    foreach ($liked_set as $item) {
        $u = sc_dot($beta_main, $item['x_main']) + sc_dot($gamma, $item['x_inter']);
        $utils[$item['item_id']] = $u;
        if ($u > $max_u) $max_u = $u;
    }
    $sum_exp = 0.0;
    foreach ($utils as $u) $sum_exp += exp($u - $max_u);
    return ($utils[$chosen_id] - $max_u) - log($sum_exp);
}

// Log prior for one respondent's beta vector (positive-constrained or free)
function sc_log_prior(array $beta, array $alpha_pop, array $sigma_sq,
                       bool $positive_only): float {
    $lp = 0.0;
    $K  = count($beta);
    for ($k = 0; $k < $K; $k++) {
        if ($positive_only && $beta[$k] < 0.0) return -INF;
        $lp += sc_log_normal_pdf($beta[$k], $alpha_pop[$k], sqrt(max(1e-6, $sigma_sq[$k])));
    }
    return $lp;
}

// ── individual beta update (MH) ───────────────────────────────────────────────

// Returns [new_beta, accepted]
// For positive_only: proposal is reflected at zero (approximate detailed balance;
// accurate when posterior mass is well away from zero, which the horseshoe ensures
// for active parameters).
function sc_mh_beta(array $beta, float $step, array $alpha_pop, array $sigma_sq,
                    array $gamma, float $tau, array $data, string $model,
                    bool $positive_only): array {
    $K = count($beta);

    // Compute old log-posterior
    if ($model === 'stage2') {
        $ll_old = ($data['chosen'] !== null && count($data['liked_set']) >= 2)
            ? sc_stage2_ll($beta, $gamma, $data['liked_set'], $data['chosen'])
            : 0.0;
    } else {
        $ll_old = sc_binary_ll($beta, $gamma, $tau, $data['visible'], $model);
    }
    $lp_old = sc_log_prior($beta, $alpha_pop, $sigma_sq, $positive_only);

    // Propose
    $prop = $beta;
    for ($k = 0; $k < $K; $k++) {
        $prop[$k] += sc_rand_normal(0, $step);
        if ($positive_only && $prop[$k] < 0.0) $prop[$k] = -$prop[$k]; // reflect
    }

    $lp_new = sc_log_prior($prop, $alpha_pop, $sigma_sq, $positive_only);
    if ($lp_new === -INF) return [$beta, false];

    if ($model === 'stage2') {
        $ll_new = ($data['chosen'] !== null && count($data['liked_set']) >= 2)
            ? sc_stage2_ll($prop, $gamma, $data['liked_set'], $data['chosen'])
            : 0.0;
    } else {
        $ll_new = sc_binary_ll($prop, $gamma, $tau, $data['visible'], $model);
    }

    $log_a = ($ll_new + $lp_new) - ($ll_old + $lp_old);
    if (log(lcg_value()) < $log_a) return [$prop, true];
    return [$beta, false];
}

// ── population interaction update (MH, shared across respondents) ─────────────

function sc_mh_gamma(array $gamma, float $step_gamma, array $gamma_alpha,
                      array $gamma_lambda_sq, float $gamma_tau_sq,
                      array $beta_all, float $tau, array $respondents,
                      string $model): array {
    $M = count($gamma);

    // Sum ll over all respondents (shared parameter)
    $ll_old = 0.0;
    foreach ($respondents as $rid => $resp) {
        if ($model === 'stage2') {
            if ($resp['chosen'] !== null && count($resp['liked_set']) >= 2)
                $ll_old += sc_stage2_ll($beta_all[$rid], $gamma, $resp['liked_set'], $resp['chosen']);
        } else {
            $ll_old += sc_binary_ll($beta_all[$rid], $gamma, $tau, $resp['visible'], $model);
        }
    }
    // Horseshoe prior on gamma
    $lp_old = 0.0;
    for ($m = 0; $m < $M; $m++) {
        $lp_old += sc_log_normal_pdf($gamma[$m], $gamma_alpha[$m],
                                      sqrt(max(1e-10, $gamma_lambda_sq[$m] * $gamma_tau_sq)));
    }

    // Propose
    $prop = $gamma;
    for ($m = 0; $m < $M; $m++) $prop[$m] += sc_rand_normal(0, $step_gamma);

    $ll_new = 0.0;
    foreach ($respondents as $rid => $resp) {
        if ($model === 'stage2') {
            if ($resp['chosen'] !== null && count($resp['liked_set']) >= 2)
                $ll_new += sc_stage2_ll($beta_all[$rid], $prop, $resp['liked_set'], $resp['chosen']);
        } else {
            $ll_new += sc_binary_ll($beta_all[$rid], $prop, $tau, $resp['visible'], $model);
        }
    }
    $lp_new = 0.0;
    for ($m = 0; $m < $M; $m++) {
        $lp_new += sc_log_normal_pdf($prop[$m], $gamma_alpha[$m],
                                      sqrt(max(1e-10, $gamma_lambda_sq[$m] * $gamma_tau_sq)));
    }

    if (log(lcg_value()) < ($ll_new + $lp_new) - ($ll_old + $lp_old)) return $prop;
    return $gamma;
}

// ── main MCMC ─────────────────────────────────────────────────────────────────

function sc_run_model(array $respondents, array $config, string $model): array {
    // model: 'liked' | 'disliked' | 'stage2'

    $K    = $config['K'];
    $M    = $config['M'];
    $n    = count($respondents);
    $rids = array_keys($respondents);

    $mcmc           = $config['mcmc'];
    $iters          = $mcmc['iterations'];
    $burn_in        = $mcmc['burn_in'];
    $adapt_every    = $mcmc['adapt_every'];
    $target_accept  = $mcmc['target_accept'];
    $positive_only  = ($model !== 'stage2');

    // ── initialise individual betas ──────────────────────────────────────────
    $beta = [];
    foreach ($rids as $rid) {
        $b = [];
        for ($k = 0; $k < $K; $k++) {
            $b[] = $positive_only ? abs(sc_rand_normal(0, 0.05)) : sc_rand_normal(0, 0.05);
        }
        $beta[$rid] = $b;
    }

    // ── population main-effect horseshoe hyperparams ─────────────────────────
    $alpha_pop  = array_fill(0, $K, 0.05);
    $sigma_sq   = array_fill(0, $K, 1.0);   // between-respondent variance
    $lambda_sq  = array_fill(0, $K, 1.0);
    $nu         = array_fill(0, $K, 1.0);
    $tau_sq     = 1.0;
    $xi         = 1.0;

    // ── population interaction horseshoe hyperparams ─────────────────────────
    $gamma         = array_fill(0, $M, 0.0);
    $g_alpha       = array_fill(0, $M, 0.0);  // population mean for each gamma
    $g_lambda_sq   = array_fill(0, $M, 1.0);
    $g_nu          = array_fill(0, $M, 1.0);
    $g_tau_sq      = 1.0;
    $g_xi          = 1.0;
    $step_gamma    = 0.1;
    $g_adapt_acc   = 0;
    $g_adapt_count = 0;

    // ── threshold ────────────────────────────────────────────────────────────
    $tau      = 0.0;
    $step_tau = 0.1;
    $tau_acc  = 0; $tau_count = 0;

    // ── per-respondent step sizes and acceptance tracking ────────────────────
    $step       = array_fill_keys($rids, 0.3);
    $adapt_acc  = array_fill_keys($rids, 0);
    $adapt_cnt  = array_fill_keys($rids, 0);
    $total_acc  = array_fill_keys($rids, 0);
    $total_cnt  = array_fill_keys($rids, 0);

    // ── posterior accumulators ───────────────────────────────────────────────
    $beta_sum  = [];
    $alpha_sum = array_fill(0, $K, 0.0);
    $gamma_sum = array_fill(0, $M, 0.0);
    foreach ($rids as $rid) $beta_sum[$rid] = array_fill(0, $K, 0.0);
    $n_samples = 0;

    // ── MCMC loop ─────────────────────────────────────────────────────────────
    for ($iter = 0; $iter < $iters; $iter++) {

        // 1. Update individual betas (MH per respondent)
        foreach ($rids as $rid) {
            [$new_beta, $accepted] = sc_mh_beta(
                $beta[$rid], $step[$rid], $alpha_pop, $sigma_sq,
                $gamma, $tau, $respondents[$rid], $model, $positive_only
            );
            $beta[$rid] = $new_beta;

            $adapt_acc[$rid]  += (int)$accepted;
            $adapt_cnt[$rid]  += 1;
            $total_cnt[$rid]  += 1;
            if ($iter >= $burn_in && $accepted) $total_acc[$rid]++;

            // Adaptive step (Robbins-Monro)
            if ($iter < $burn_in && $adapt_cnt[$rid] >= $adapt_every) {
                $acc_rate = $adapt_acc[$rid] / $adapt_cnt[$rid];
                $step[$rid] *= exp($acc_rate - $target_accept);
                $step[$rid]  = max(1e-4, min(5.0, $step[$rid]));
                $adapt_acc[$rid] = $adapt_cnt[$rid] = 0;
            }
        }

        // 2. Update shared threshold tau (MH, model='liked'|'disliked' only)
        if ($model !== 'stage2') {
            $prop_tau = $tau + sc_rand_normal(0, $step_tau);
            $ll_old = $ll_new = 0.0;
            foreach ($rids as $rid) {
                $ll_old += sc_binary_ll($beta[$rid], $gamma, $tau,      $respondents[$rid]['visible'], $model);
                $ll_new += sc_binary_ll($beta[$rid], $gamma, $prop_tau, $respondents[$rid]['visible'], $model);
            }
            if (log(lcg_value()) < $ll_new - $ll_old) {
                $tau = $prop_tau;
                $tau_acc++;
            }
            $tau_count++;
            // Adapt threshold step
            if ($iter < $burn_in && $iter % $adapt_every === $adapt_every - 1) {
                $step_tau *= exp($tau_acc / max(1,$tau_count) - $target_accept);
                $step_tau  = max(0.001, min(2.0, $step_tau));
                $tau_acc = $tau_count = 0;
            }
        }

        // 3. Update population interaction gamma (shared MH)
        if ($M > 0) {
            $old_gamma = $gamma;
            $gamma = sc_mh_gamma($gamma, $step_gamma, $g_alpha, $g_lambda_sq, $g_tau_sq,
                                  $beta, $tau, $respondents, $model);
            $g_accepted = ($gamma !== $old_gamma) ? 1 : 0;
            $g_adapt_acc++;
            $g_adapt_count++;
            if ($iter < $burn_in && $g_adapt_count >= $adapt_every) {
                $step_gamma *= exp($g_adapt_acc / $g_adapt_count - $target_accept);
                $step_gamma  = max(1e-4, min(2.0, $step_gamma));
                $g_adapt_acc = $g_adapt_count = 0;
            }
        }

        // 4. Update population horseshoe hyperparameters (main effects)
        for ($k = 0; $k < $K; $k++) {
            $betas_k = array_map(fn($rid) => $beta[$rid][$k], $rids);

            // Between-respondent variance (inverse-gamma conjugate)
            $ss = 0.0;
            foreach ($betas_k as $bk) $ss += ($bk - $alpha_pop[$k]) ** 2;
            $sigma_sq[$k] = sc_rand_inv_gamma(1.0 + $n/2.0, 1.0 + $ss/2.0);
            $sigma_sq[$k] = max(1e-6, min(10.0, $sigma_sq[$k]));

            // Horseshoe local and global shrinkage
            $lambda_sq[$k] = hs_update_lambda_sq($alpha_pop[$k], $tau_sq, $nu[$k]);
            $nu[$k]        = hs_update_nu($lambda_sq[$k]);
            $alpha_pop[$k] = hs_update_alpha($betas_k, $sigma_sq[$k],
                                              $lambda_sq[$k], $tau_sq, $positive_only);
        }
        $xi     = hs_update_xi($tau_sq);
        $tau_sq = hs_update_tau_sq($alpha_pop, $lambda_sq, $xi);

        // 5. Update interaction gamma horseshoe hyperparameters
        if ($M > 0) {
            for ($m = 0; $m < $M; $m++) {
                $g_lambda_sq[$m] = hs_update_lambda_sq($g_alpha[$m], $g_tau_sq, $g_nu[$m]);
                $g_nu[$m]        = hs_update_nu($g_lambda_sq[$m]);
                // Population mean for gamma (unconstrained, horseshoe)
                $post_prec = 1.0 / max(1e-10, $g_lambda_sq[$m] * $g_tau_sq);
                // gamma is a single shared value; "data" = gamma[m] itself (degenerate)
                // Use the gamma value as a point observation of the prior
                $post_var  = 1.0 / ($post_prec + 1.0);
                $post_mean = $post_var * $gamma[$m] * $post_prec;
                $g_alpha[$m] = sc_rand_normal($post_mean, sqrt($post_var));
            }
            $g_xi     = hs_update_xi($g_tau_sq);
            $g_tau_sq = hs_update_tau_sq($g_alpha, $g_lambda_sq, $g_xi);
        }

        // 6. Collect posterior samples
        if ($iter >= $burn_in) {
            foreach ($rids as $rid) {
                for ($k = 0; $k < $K; $k++) $beta_sum[$rid][$k] += $beta[$rid][$k];
            }
            for ($k = 0; $k < $K; $k++) $alpha_sum[$k] += $alpha_pop[$k];
            for ($m = 0; $m < $M; $m++) $gamma_sum[$m] += $gamma[$m];
            $n_samples++;
        }

        if ($iter % 500 === 0) {
            $mean_step = array_sum($step) / max(1, $n);
            fwrite(STDERR, sprintf(
                "[%s] iter %4d/%d  tau=%.3f  tau_sq=%.3f  mean_step=%.3f  samples=%d\n",
                $model, $iter, $iters, $tau, $tau_sq, $mean_step, $n_samples
            ));
        }
    }

    // ── posterior means ────────────────────────────────────────────────────
    $ns = max(1, $n_samples);
    $beta_mean  = [];
    foreach ($rids as $rid) {
        $beta_mean[$rid] = array_map(fn($s) => $s / $ns, $beta_sum[$rid]);
    }

    return [
        'beta_mean'    => $beta_mean,
        'alpha_mean'   => array_map(fn($s) => $s / $ns, $alpha_sum),
        'gamma_mean'   => array_map(fn($s) => $s / $ns, $gamma_sum),
        'tau'          => $tau,
        'tau_sq'       => $tau_sq,
        'n_samples'    => $n_samples,
        'accept_rate'  => array_map(fn($rid) => $total_cnt[$rid] > 0
                              ? $total_acc[$rid] / $total_cnt[$rid] : 0.0, array_combine($rids, $rids)),
    ];
}

// ── output ─────────────────────────────────────────────────────────────────────

function sc_beta_col_names(array $config): array {
    $cols = ['respondent_id'];
    foreach ($config['attributes'] as $i => $att) {
        for ($l = 0; $l < $att['levels'] - 1; $l++) {
            $cols[] = "att{$i}_{$att['name']}_l{$l}";
        }
    }
    $cols[] = 'accept_rate';
    return $cols;
}

function sc_write_betas(string $path, array $result, array $config): void {
    $K    = $config['K'];
    $fh   = fopen($path, 'w');
    fputcsv($fh, sc_beta_col_names($config));

    // Population row
    $row = ['POPULATION'];
    foreach ($result['alpha_mean'] as $v) $row[] = round($v, 6);
    $row[] = '';
    fputcsv($fh, $row);

    foreach ($result['beta_mean'] as $rid => $betas) {
        $row = [$rid];
        foreach ($betas as $v) $row[] = round($v, 6);
        $row[] = round($result['accept_rate'][$rid] ?? 0.0, 4);
        fputcsv($fh, $row);
    }
    fclose($fh);
    echo "Written: $path\n";
}

function sc_write_population(string $path, array $sel, array $rej, array $s2,
                              array $config): void {
    $K    = $config['K'];
    $M    = $config['M'];
    $pairs = $config['interaction_pairs'];
    $atts  = $config['attributes'];
    $fh    = fopen($path, 'w');
    fputcsv($fh, ['type','parameter','sel_value','rej_value','s2_value']);

    // Main effects
    $k = 0;
    foreach ($atts as $a => $att) {
        for ($l = 0; $l < $att['levels'] - 1; $l++) {
            fputcsv($fh, [
                'main',
                "att{$a}_{$att['name']}_l{$l}",
                round($sel['alpha_mean'][$k], 6),
                round($rej['alpha_mean'][$k], 6),
                round($s2['alpha_mean'][$k],  6),
            ]);
            $k++;
        }
    }

    // Thresholds
    fputcsv($fh, ['threshold', 'tau', round($sel['tau'],4), round($rej['tau'],4), '']);

    // Interaction terms
    foreach ($pairs as $pi => $pair) {
        [$a, $b] = $pair;
        $label = "int_att{$a}_{$atts[$a]['name']}__att{$b}_{$atts[$b]['name']}";
        fputcsv($fh, [
            'interaction',
            $label,
            round($sel['gamma_mean'][$pi], 6),
            round($rej['gamma_mean'][$pi], 6),
            round($s2['gamma_mean'][$pi],  6),
        ]);
    }

    fclose($fh);
    echo "Written: $path\n";
}

// ── CLI entry point ────────────────────────────────────────────────────────────

$opts           = getopt('', ['config:','catalogue:','responses:','out_dir:','seed:']);
$config_file    = $opts['config']    ?? 'scroll_conjoint_config.json';
$catalogue_file = $opts['catalogue'] ?? 'scroll_catalogue.csv';
$responses_file = $opts['responses'] ?? 'scroll_responses.csv';
$out_dir        = rtrim($opts['out_dir'] ?? '.', '/');
$seed           = (int)($opts['seed'] ?? 42);

srand($seed);

echo "Loading config: $config_file\n";
$config = sc_load_config($config_file);
echo "K={$config['K']}  M={$config['M']} interaction pairs\n";

echo "Loading catalogue: $catalogue_file\n";
$catalogue = sc_load_catalogue($catalogue_file, $config);
echo count($catalogue) . " items\n";

echo "Loading responses: $responses_file\n";
$respondents = sc_load_responses($responses_file, $catalogue, $config);
$n = count($respondents);
echo "$n respondents with usable data\n";

if ($n < 5) die("Too few respondents — need at least 5\n");

// Descriptive summary
$n_vis = $n_lk = $n_dk = $n_ch = 0;
foreach ($respondents as $r) {
    $n_vis += count($r['visible']);
    $n_lk  += count($r['liked_set']);
    $n_dk  += array_sum(array_column($r['visible'], 'disliked'));
    if ($r['chosen'] !== null) $n_ch++;
}
printf("Avg visible/resp: %.1f  liked: %.1f  disliked: %.1f  stage-2 choices: %d/%d\n",
    $n_vis/$n, $n_lk/$n, $n_dk/$n, $n_ch, $n);

echo "\n=== Selection model (P(liked | visible)) ===\n";
$sel = sc_run_model($respondents, $config, 'liked');

echo "\n=== Rejection model (P(disliked | visible)) ===\n";
$rej = sc_run_model($respondents, $config, 'disliked');

echo "\n=== Stage-2 model (P(chosen | liked set)) ===\n";
$s2  = sc_run_model($respondents, $config, 'stage2');

sc_write_betas("$out_dir/hb_selection_betas.csv", $sel, $config);
sc_write_betas("$out_dir/hb_rejection_betas.csv", $rej, $config);
sc_write_betas("$out_dir/hb_stage2_betas.csv",    $s2,  $config);
sc_write_population("$out_dir/hb_population.csv", $sel, $rej, $s2, $config);

echo "\nDone. Posterior samples: {$sel['n_samples']} per model.\n";
printf("tau_sel=%.3f  tau_rej=%.3f\n", $sel['tau'], $rej['tau']);
printf("Global shrinkage: sel tau_sq=%.4f  rej tau_sq=%.4f  s2 tau_sq=%.4f\n",
    $sel['tau_sq'], $rej['tau_sq'], $s2['tau_sq']);
