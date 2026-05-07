"""
DCC Benchmark -- Step 4b: Human Annotation Scoring and ICC Computation
=======================================================================
Aggregates completed human ratings, computes the Intraclass Correlation
Coefficient (ICC) with a 95 % confidence interval, filters causal pairs by
mean rating, and optionally applies the decisions back to the full DCC
annotation JSON.

Implements the human validation described in Section 3.2 of the paper:

    "Three annotators view merged clips and rate causality
     (1=definitely not, 5=definitely yes), achieving an intraclass
     correlation coefficient (ICC) of 0.85 (95 %, CI: 0.83–0.91).
     Results: 98.5 % score >= 3."

ICC model
---------
    ICC(2,1) -- two-way random effects, single measures, absolute agreement
    (Shrout & Fleiss, 1979, Model 2).
    This is appropriate when the three raters are treated as a random sample
    from a larger annotator pool and each pair is rated by a different set
    of three annotators.

Filtering rule
--------------
    A causal pair is retained if its mean rating across the three annotators
    is >= 3.0 ("at least borderline causal").  Pairs rated by fewer than
    three annotators are flagged but not automatically rejected.

Input: completed annotation CSV
-------------------------------
    pair_id,rater_1,rater_2,rater_3
    actnet_v123_1_2,4,5,4
    yc2_recipe01_vid01_2_3,2,2,3
    ...

    Produced by annotators completing annotations_template.csv from Step 4a.

Usage
-----
    # Basic: score + report only
    python dcc_human_study_score.py \\
        --tasks        human_study/tasks.json \\
        --annotations  human_study/annotations_completed.csv \\
        --output-dir   human_study/results/

    # Full pipeline: also filter the DCC annotation JSONs
    python dcc_human_study_score.py \\
        --tasks        human_study/tasks.json \\
        --annotations  human_study/annotations_completed.csv \\
        --activitynet  dcc_actnet_vlm.json \\
        --youcook2     dcc_yc2_vlm.json \\
        --output-dir   human_study/results/
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# scipy is used only for the F-distribution quantiles needed for the ICC CI.
# If unavailable the CI is reported as None with a clear message.
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# ICC(2,1) computation  (Shrout & Fleiss, 1979, two-way random, absolute)
# ---------------------------------------------------------------------------

def _two_way_anova(ratings: List[List[float]]) -> Tuple[float, float, float]:
    """
    Compute mean squares for a two-way ANOVA without interaction.

    Parameters
    ----------
    ratings : list of lists, shape (n_subjects, n_raters)
              Each inner list contains one rating per rater for that subject.

    Returns
    -------
    (MS_r, MS_c, MS_e)
        MS_r : mean square between subjects (rows)
        MS_c : mean square between raters  (columns)
        MS_e : mean square residual (error)
    """
    n = len(ratings)          # number of subjects (pairs)
    k = len(ratings[0])       # number of raters

    grand_mean = sum(sum(row) for row in ratings) / (n * k)
    row_means  = [sum(row) / k for row in ratings]
    col_means  = [
        sum(ratings[i][j] for i in range(n)) / n
        for j in range(k)
    ]

    SS_r = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    SS_c = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    SS_total = sum(
        (ratings[i][j] - grand_mean) ** 2
        for i in range(n)
        for j in range(k)
    )
    SS_e = SS_total - SS_r - SS_c

    MS_r = SS_r / (n - 1)
    MS_c = SS_c / (k - 1)
    MS_e = SS_e / ((n - 1) * (k - 1))

    return MS_r, MS_c, MS_e


def compute_icc21(
    ratings: List[List[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute ICC(2,1): two-way random effects, single measures, absolute agreement.

    Formula (Shrout & Fleiss, 1979):
        ICC(2,1) = (MS_r - MS_e) /
                   (MS_r + (k-1)*MS_e + k*(MS_c - MS_e)/n)

    95 % CI from the F-distribution (requires scipy; reported as None otherwise).

    Parameters
    ----------
    ratings : list of lists, shape (n_subjects, n_raters)
    alpha   : significance level for the confidence interval (default 0.05)

    Returns
    -------
    dict with keys: icc, ci_lower, ci_upper, n_subjects, n_raters,
                    MS_r, MS_c, MS_e, F_obs
    """
    n = len(ratings)
    k = len(ratings[0])

    MS_r, MS_c, MS_e = _two_way_anova(ratings)

    # ICC(2,1) formula
    denom = MS_r + (k - 1) * MS_e + k * (MS_c - MS_e) / n
    icc   = (MS_r - MS_e) / denom if denom != 0 else 0.0
    icc   = max(0.0, min(1.0, icc))

    # F-statistic for CI calculation (Shrout & Fleiss formula)
    F_obs = MS_r / MS_e if MS_e > 0 else float("inf")
    df1   = n - 1
    df2   = (n - 1) * (k - 1)

    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    if _HAS_SCIPY and math.isfinite(F_obs):
        # Critical F values
        F_lo = _scipy_stats.f.ppf(alpha / 2,       df1, df2)
        F_hi = _scipy_stats.f.ppf(1.0 - alpha / 2, df1, df2)

        # Shieh (2012) / Shrout & Fleiss bounds adapted for ICC(2,1)
        def _bound(F_crit: float) -> float:
            num = F_obs / F_crit - 1.0
            den = F_obs / F_crit + k - 1.0
            return max(0.0, min(1.0, num / den)) if den != 0 else 0.0

        ci_lower = _bound(F_hi)   # F_hi produces the lower ICC bound
        ci_upper = _bound(F_lo)   # F_lo produces the upper ICC bound
    else:
        if not _HAS_SCIPY:
            print(
                "  [ICC] scipy not available -- CI not computed. "
                "Install scipy to get the 95 % CI."
            )

    return {
        "icc":        round(icc, 4),
        "ci_lower":   round(ci_lower, 4) if ci_lower is not None else None,
        "ci_upper":   round(ci_upper, 4) if ci_upper is not None else None,
        "alpha":      alpha,
        "n_subjects": n,
        "n_raters":   k,
        "MS_r":       round(MS_r, 6),
        "MS_c":       round(MS_c, 6),
        "MS_e":       round(MS_e, 6),
        "F_obs":      round(F_obs, 4) if math.isfinite(F_obs) else None,
        "df1":        df1,
        "df2":        df2,
        "model":      "ICC(2,1) -- two-way random, single measures, absolute agreement",
        "reference":  "Shrout & Fleiss (1979)",
    }


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def _load_ratings(annotations_csv: str) -> Dict[str, List[Optional[int]]]:
    """
    Load completed annotations from the CSV returned by annotators.

    Expected columns: pair_id, rater_1, rater_2, rater_3
    Ratings must be integers in [1, 5]; empty cells are stored as None.

    Returns
    -------
    dict mapping pair_id -> [rating_1, rating_2, rating_3]
    """
    ratings: Dict[str, List[Optional[int]]] = {}

    with open(annotations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"].strip()
            rs: List[Optional[int]] = []
            for col in ("rater_1", "rater_2", "rater_3"):
                val = row.get(col, "").strip()
                if val == "":
                    rs.append(None)
                else:
                    try:
                        r = int(val)
                        if not (1 <= r <= 5):
                            raise ValueError(f"Rating {r} outside [1,5]")
                        rs.append(r)
                    except ValueError as exc:
                        print(f"  [load_ratings] {pair_id} {col}: {exc} -- treating as missing")
                        rs.append(None)
            ratings[pair_id] = rs

    return ratings


def _load_tasks(tasks_path: str) -> Dict[str, Dict[str, Any]]:
    """Load tasks.json and return a dict mapping pair_id -> task metadata."""
    with open(tasks_path) as f:
        tasks = json.load(f)
    return {t["pair_id"]: t for t in tasks}


# ---------------------------------------------------------------------------
# Pair-level scoring
# ---------------------------------------------------------------------------

def _score_pairs(
    tasks: Dict[str, Dict[str, Any]],
    ratings: Dict[str, List[Optional[int]]],
    retention_threshold: float = 3.0,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    Compute per-pair mean ratings and retention decisions.

    Parameters
    ----------
    tasks                : task metadata dict from tasks.json
    ratings              : completed ratings dict from the annotation CSV
    retention_threshold  : mean rating >= this value -> pair retained

    Returns
    -------
    (pair_results, icc_matrix)
        pair_results : list of per-pair result dicts
        icc_matrix   : list of [r1, r2, r3] rows for pairs with all 3 ratings
                       (used for ICC computation)
    """
    pair_results: List[Dict[str, Any]] = []
    icc_matrix:   List[List[float]]    = []

    for pair_id, task in tasks.items():
        rs = ratings.get(pair_id, [None, None, None])

        present  = [r for r in rs if r is not None]
        n_rated  = len(present)
        mean_r   = sum(present) / n_rated if present else None
        retained = mean_r is not None and mean_r >= retention_threshold

        result: Dict[str, Any] = {
            "pair_id":          pair_id,
            "dataset":          task.get("dataset"),
            "video_id":         task.get("video_id"),
            "cause_event_id":   task.get("cause_event_id"),
            "effect_event_id":  task.get("effect_event_id"),
            "cause_description": task.get("cause_description"),
            "effect_description": task.get("effect_description"),
            "n_events_in_video": task.get("n_events_in_video"),
            "high_event_stratum": task.get("high_event_stratum"),
            "ratings":          rs,
            "n_ratings_received": n_rated,
            "mean_rating":      round(mean_r, 3) if mean_r is not None else None,
            "retained":         retained,
            "incomplete":       n_rated < 3,
        }
        pair_results.append(result)

        # Only include fully-rated pairs in the ICC matrix
        if n_rated == 3:
            icc_matrix.append([float(r) for r in present])

    return pair_results, icc_matrix


# ---------------------------------------------------------------------------
# Apply filtering to DCC annotation JSON
# ---------------------------------------------------------------------------

def _extract_edges(events: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    edges = []
    for event in events:
        src = event["event_id"]
        for dst in event["chain"].get("effect_event", []):
            edges.append((src, int(dst)))
    return edges


def _rebuild_chains(
    events: List[Dict[str, Any]],
    valid_edges: Set[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Prune chains to validated edges and recompute event_type flags."""
    ids = {e["event_id"] for e in events}
    new_effect: Dict[int, List[int]] = {e["event_id"]: [] for e in events}
    new_cause:  Dict[int, List[int]] = {e["event_id"]: [] for e in events}

    for src, dst in valid_edges:
        if src in ids and dst in ids:
            new_effect[src].append(dst)
            new_cause[dst].append(src)

    updated = []
    for event in events:
        eid       = event["event_id"]
        effect_ev = sorted(new_effect[eid])
        cause_ev  = sorted(new_cause[eid])
        is_cause  = len(effect_ev) > 0
        is_effect = len(cause_ev) > 0

        if is_cause and is_effect:
            event_type = "causal"
        elif is_cause:
            event_type = "cause"
        elif is_effect:
            event_type = "effect"
        else:
            event_type = "independent"

        updated.append({
            **event,
            "chain": {
                "is_cause":    is_cause,
                "is_effect":   is_effect,
                "cause_event": cause_ev,
                "effect_event": effect_ev,
            },
            "event_type": event_type,
        })
    return updated


def _apply_human_filter(
    annotations: Dict[str, Any],
    pair_results: List[Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Remove pairs flagged by human validators from the annotation JSON.

    Only pairs whose dataset field matches *dataset_name* are considered.
    All other pairs (not in the human sample) are kept unchanged.

    Returns the filtered annotation dict.
    """
    # Build lookup: (video_id, cause_id, effect_id) -> retained
    pair_lookup: Dict[Tuple[str, int, int], bool] = {}
    for r in pair_results:
        if r["dataset"] != dataset_name:
            continue
        key = (r["video_id"], r["cause_event_id"], r["effect_event_id"])
        pair_lookup[key] = r["retained"]

    filtered: Dict[str, Any] = {}
    for video_id, video_data in annotations.items():
        if video_data is None:
            filtered[video_id] = None
            continue

        events = video_data.get("events", [])
        edges  = _extract_edges(events)

        valid_edges: Set[Tuple[int, int]] = set()
        for src_id, dst_id in edges:
            key = (video_id, src_id, dst_id)
            if key in pair_lookup:
                # In human sample: apply decision
                if pair_lookup[key]:
                    valid_edges.add((src_id, dst_id))
            else:
                # Not in human sample: keep (already passed Tiers 1 and 2)
                valid_edges.add((src_id, dst_id))

        updated_events = _rebuild_chains(events, valid_edges)
        filtered[video_id] = {**video_data, "events": updated_events}

    return filtered


# ---------------------------------------------------------------------------
# Main scoring entry point
# ---------------------------------------------------------------------------

def score_annotations(
    tasks_path: str,
    annotations_csv: str,
    output_dir: str,
    actnet_path: Optional[str] = None,
    yc2_path: Optional[str] = None,
    retention_threshold: float = 3.0,
    alpha: float = 0.05,
) -> None:
    """
    Aggregate human ratings, compute ICC(2,1), filter pairs, and write outputs.

    Parameters
    ----------
    tasks_path           : tasks.json from dcc_human_study_tasks.py
    annotations_csv      : completed annotation CSV (pair_id, rater_1..3)
    output_dir           : directory for all output files
    actnet_path          : optional Step-3 ActivityNet JSON to apply filtering to
    yc2_path             : optional Step-3 YouCook2 JSON to apply filtering to
    retention_threshold  : minimum mean rating for pair retention (default: 3.0)
    alpha                : significance level for ICC CI (default: 0.05)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[score_annotations]")
    print(f"  tasks            : {tasks_path}")
    print(f"  annotations CSV  : {annotations_csv}")
    print(f"  threshold        : mean >= {retention_threshold}")
    print(f"  ICC CI alpha     : {alpha}")

    tasks   = _load_tasks(tasks_path)
    ratings = _load_ratings(annotations_csv)

    n_missing = sum(1 for pid in tasks if pid not in ratings)
    if n_missing:
        print(f"  Warning: {n_missing} tasks have no ratings in the CSV")

    # -----------------------------------------------------------------------
    # Per-pair scoring
    # -----------------------------------------------------------------------
    pair_results, icc_matrix = _score_pairs(tasks, ratings, retention_threshold)

    n_total     = len(pair_results)
    n_complete  = sum(1 for r in pair_results if not r["incomplete"])
    n_retained  = sum(1 for r in pair_results if r["retained"])
    n_rejected  = sum(1 for r in pair_results if not r["retained"] and not r["incomplete"])
    n_incomplete = sum(1 for r in pair_results if r["incomplete"])
    retention   = n_retained / n_total if n_total else 0.0

    # Dataset-level breakdown
    def _ds_stats(ds: str) -> Dict[str, Any]:
        ds_pairs = [r for r in pair_results if r["dataset"] == ds]
        ds_ret   = sum(1 for r in ds_pairs if r["retained"])
        ds_mean  = (
            sum(r["mean_rating"] for r in ds_pairs if r["mean_rating"] is not None)
            / max(1, sum(1 for r in ds_pairs if r["mean_rating"] is not None))
        )
        return {
            "total_pairs":    len(ds_pairs),
            "retained_pairs": ds_ret,
            "rejected_pairs": len(ds_pairs) - ds_ret,
            "retention_rate": round(ds_ret / len(ds_pairs), 4) if ds_pairs else 0.0,
            "mean_rating":    round(ds_mean, 3),
        }

    # -----------------------------------------------------------------------
    # ICC(2,1) computation
    # -----------------------------------------------------------------------
    icc_result: Dict[str, Any] = {}
    if len(icc_matrix) >= 10:
        print(f"\n  Computing ICC(2,1) on {len(icc_matrix)} fully-rated pairs ...")
        icc_result = compute_icc21(icc_matrix, alpha=alpha)
        print(
            f"  ICC(2,1) = {icc_result['icc']:.3f}  "
            f"(95 % CI: [{icc_result['ci_lower']}, {icc_result['ci_upper']}])"
        )
    else:
        print(
            f"  Warning: only {len(icc_matrix)} fully-rated pairs -- "
            f"ICC computation skipped (need >= 10)."
        )

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    report: Dict[str, Any] = {
        "aggregate": {
            "total_pairs":     n_total,
            "complete_ratings": n_complete,
            "incomplete_ratings": n_incomplete,
            "retained_pairs":  n_retained,
            "rejected_pairs":  n_rejected,
            "retention_rate":  round(retention, 4),
            "retention_threshold": retention_threshold,
            "pct_score_ge3":   round(
                sum(1 for r in pair_results
                    if r["mean_rating"] is not None and r["mean_rating"] >= 3.0)
                / max(1, n_complete), 4
            ),
        },
        "icc": icc_result,
        "by_dataset": {
            "activitynet": _ds_stats("activitynet"),
            "youcook2":    _ds_stats("youcook2"),
        },
        "by_stratum": {
            "high_event": {
                "total":    sum(1 for r in pair_results if r["high_event_stratum"]),
                "retained": sum(1 for r in pair_results if r["high_event_stratum"] and r["retained"]),
            },
            "low_event": {
                "total":    sum(1 for r in pair_results if not r["high_event_stratum"]),
                "retained": sum(1 for r in pair_results if not r["high_event_stratum"] and r["retained"]),
            },
        },
    }

    print(f"\n  Retained {n_retained}/{n_total} pairs ({100 * retention:.1f} %)")
    pct_ge3 = report["aggregate"]["pct_score_ge3"]
    print(f"  Pairs with mean >= 3.0: {100 * pct_ge3:.1f} %  (paper: 98.5 %)")

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    report_path = out / "human_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Validation report -> {report_path}")

    pairs_path = out / "pair_results.json"
    with open(pairs_path, "w") as f:
        json.dump(pair_results, f, indent=2)
    print(f"  Per-pair results  -> {pairs_path}")

    # Write filtered CSVs (retained / rejected) for auditing
    retained_csv  = out / "pairs_retained.csv"
    rejected_csv  = out / "pairs_rejected.csv"

    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        keys = ["pair_id", "dataset", "video_id", "cause_event_id", "effect_event_id",
                "mean_rating", "ratings", "retained"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "ratings": str(row["ratings"])})

    _write_csv(retained_csv, [r for r in pair_results if r["retained"]])
    _write_csv(rejected_csv, [r for r in pair_results if not r["retained"]])
    print(f"  Retained pairs    -> {retained_csv}")
    print(f"  Rejected pairs    -> {rejected_csv}")

    # Optionally apply filtering back to the annotation JSONs
    if actnet_path is not None:
        with open(actnet_path) as f:
            actnet_data = json.load(f)
        filtered_actnet = _apply_human_filter(actnet_data, pair_results, "activitynet")
        out_actnet = out / "dcc_actnet_final.json"
        with open(out_actnet, "w") as f:
            json.dump(filtered_actnet, f, indent=2)
        print(f"  Final ActivityNet  -> {out_actnet}")

    if yc2_path is not None:
        with open(yc2_path) as f:
            yc2_data = json.load(f)
        filtered_yc2 = _apply_human_filter(yc2_data, pair_results, "youcook2")
        out_yc2 = out / "dcc_youcook2_final.json"
        with open(out_yc2, "w") as f:
            json.dump(filtered_yc2, f, indent=2)
        print(f"  Final YouCook2     -> {out_yc2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DCC benchmark -- human annotation scoring and ICC computation (Step 4b of 4)"
        )
    )
    parser.add_argument(
        "--tasks", required=True, metavar="FILE",
        help="tasks.json from dcc_human_study_tasks.py",
    )
    parser.add_argument(
        "--annotations", required=True, metavar="FILE",
        help="Completed annotation CSV (columns: pair_id, rater_1, rater_2, rater_3)",
    )
    parser.add_argument(
        "--output-dir", default="human_study/results", metavar="DIR",
        help="Directory for all output files (default: human_study/results/)",
    )
    parser.add_argument(
        "--activitynet", default=None, metavar="FILE",
        help="Step-3 ActivityNet annotation JSON to apply final filtering to (optional)",
    )
    parser.add_argument(
        "--youcook2", default=None, metavar="FILE",
        help="Step-3 YouCook2 annotation JSON to apply final filtering to (optional)",
    )
    parser.add_argument(
        "--threshold", type=float, default=3.0, metavar="FLOAT",
        help="Mean rating threshold for pair retention (default: 3.0)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, metavar="FLOAT",
        help="Significance level for ICC confidence interval (default: 0.05)",
    )
    args = parser.parse_args()

    score_annotations(
        tasks_path          = args.tasks,
        annotations_csv     = args.annotations,
        output_dir          = args.output_dir,
        actnet_path         = args.activitynet,
        yc2_path            = args.youcook2,
        retention_threshold = args.threshold,
        alpha               = args.alpha,
    )


if __name__ == "__main__":
    main()
