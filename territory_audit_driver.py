import argparse
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "old_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from votekit.cvr_loaders import load_scottish

from edouard.MeekGraph import MeekGraph
from old_src.edouard.src.margin_audits import end_to_end_synthetic_audit_3seat


def parse_fraction(text):
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        return Fraction(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid fraction value: {text}") from exc


def format_fraction(value):
    if value is None:
        return ""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def clean_path(raw_value):
    if raw_value is None:
        return ""
    cleaned = raw_value.strip()
    while cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 1:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def candidate_fractions(start_fraction, step_denominator, min_denominator, max_steps):
    if start_fraction is None:
        return []
    if start_fraction.numerator != 1:
        return [start_fraction]
    fractions = []
    denominator = start_fraction.denominator
    for _ in range(max_steps):
        if denominator < min_denominator:
            break
        fractions.append(Fraction(1, denominator))
        denominator -= step_denominator
    return fractions


def run_trials(profile, audit_graph, sample_fraction, trials, max_failures, alpha, noise_level, epsilon, m):
    successes = 0
    failures = 0
    for _ in range(trials):
        ok = end_to_end_synthetic_audit_3seat(
            profile,
            audit_graph,
            int(profile.total_ballot_wt // 100),
            sample_size_fraction=float(sample_fraction),
            alpha=alpha,
            noise_level=noise_level,
            hypergeo_var_bounds=[],
            _m=m,
            epsilon=epsilon,
            verbose=False,
        )
        if ok:
            successes += 1
        else:
            failures += 1
        if failures > max_failures:
            break
    return failures <= max_failures, successes, failures


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic audits for territory paths listed in CSV.")
    parser.add_argument(
        "--driver-csv",
        default="data/territory_audit_driver.csv",
        help="CSV containing audit paths and metadata.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/audit_results/territory_audit_results.csv",
        help="Output CSV with audit results.",
    )
    parser.add_argument("--m", type=int, default=2, help="Number of winners to elect.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Risk limit for audit.")
    parser.add_argument("--noise-level", type=float, default=0.02, help="Noise level for audit.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon for audit.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per sample size fraction.")
    parser.add_argument("--max-failures", type=int, default=1, help="Max failures allowed per fraction.")
    parser.add_argument(
        "--step-denominator",
        type=int,
        default=100,
        help="Denominator decrement when trying larger samples (1/x to 1/(x-100)).",
    )
    parser.add_argument(
        "--min-denominator",
        type=int,
        default=100,
        help="Minimum denominator to try when stepping sample fractions.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of denominators to try.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from an existing output CSV.",
    )
    parser.add_argument(
        "--rerun-errors",
        action="store_true",
        help="Rerun rows that previously ended in error when resuming.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    driver_path = Path(args.driver_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    completed = {}

    with driver_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    fieldnames = [
        "territory",
        "year",
        "path",
        "recommended_fraction",
        "smallest_passing_fraction",
        "successes",
        "failures",
        "total_nodes",
        "nodes_per_layer",
        "status",
        "error",
    ]

    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            existing_reader = csv.DictReader(handle)
            for existing in existing_reader:
                key = (existing.get("territory", ""), existing.get("year", ""), existing.get("path", ""))
                completed[key] = existing

    open_mode = "a" if resume else "w"
    with output_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not resume or output_path.stat().st_size == 0:
            writer.writeheader()

        for row in rows:
            territory = row.get("") or row.get("territory") or row.get("Territory") or ""
            year = row.get("Year", "")
            margin_text = row.get("Margin", "")
            rec_fraction_text = row.get("Recommended Fraction", "")
            path_text = row.get("Path", "")

            result = {
                "territory": territory,
                "year": year,
                "path": "",
                "recommended_fraction": rec_fraction_text,
                "smallest_passing_fraction": "",
                "successes": "",
                "failures": "",
                "total_nodes": "",
                "nodes_per_layer": "",
                "status": "",
                "error": "",
            }

            try:
                path = clean_path(path_text)
                result["path"] = path
                key = (territory, year, path)
                existing = completed.get(key)
                if resume and existing:
                    status = existing.get("status", "")
                    if status in {"pass", "no_pass"} or (status == "error" and not args.rerun_errors):
                        continue

                margin_value = int(margin_text) if margin_text else 0
                margins = [margin_value] * 3

                profile = load_scottish(path)[0]
                audit_graph = MeekGraph(
                    profile=profile,
                    m=args.m,
                    auditable_margins_per_deg=margins,
                    use_numerical_labels=False,
                )
                if not audit_graph.check_coherence():
                    raise RuntimeError("Incoherent audit graph (degree-m winner sets do not match).")
                audit_graph.print_analysis()
                stats = audit_graph.get_stats()

                result["total_nodes"] = stats.get("total_nodes", "")
                result["nodes_per_layer"] = json.dumps(stats.get("nodes_per_layer", {}), sort_keys=True)

                start_fraction = parse_fraction(rec_fraction_text)
                candidates = candidate_fractions(
                    start_fraction,
                    step_denominator=args.step_denominator,
                    min_denominator=args.min_denominator,
                    max_steps=args.max_steps,
                )

                for fraction in candidates:
                    passed, successes, failures = run_trials(
                        profile,
                        audit_graph,
                        sample_fraction=fraction,
                        trials=args.trials,
                        max_failures=args.max_failures,
                        alpha=args.alpha,
                        noise_level=args.noise_level,
                        epsilon=args.epsilon,
                        m=args.m,
                    )
                    if passed:
                        result["smallest_passing_fraction"] = format_fraction(fraction)
                        result["successes"] = successes
                        result["failures"] = failures
                        result["status"] = "pass"
                        break

                if not result["status"]:
                    result["status"] = "no_pass"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)

            writer.writerow(result)
            handle.flush()


if __name__ == "__main__":
    main()
