import argparse
import csv
import gc
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "old_src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from votekit.cleaning import remove_and_condense_rank_profile
from votekit.cvr_loaders import load_scottish
from old_src.edouard.ranking_election.meek import MeekSTV

from edouard.MeekGraph import MeekGraph
from old_src.edouard.src.margin_audits import end_to_end_synthetic_audit_3seat


def normalize_name(name):
    return " ".join(name.replace(",", " ").split()).lower()


def swap_first_last(name):
    parts = name.replace(",", " ").split()
    if len(parts) < 2:
        return name
    first = " ".join(parts[:-1])
    last = parts[-1]
    return f"{last} {first}"


def round_down_1000(value):
    return int(value // 1000) * 1000


def find_largest_coherent_margin(profile, m, original_margin):
    high = max(0, int(original_margin))
    low = 0
    best_margin = None
    while high - low > 1000:
        mid = round_down_1000((low + high) / 2)
        if mid <= low:
            mid = low + 1000
        if mid >= high:
            break
        graph = MeekGraph(
            profile=profile,
            m=m,
            auditable_margins_per_deg=[mid] * m,
            use_numerical_labels=False,
        )
        if graph.check_coherence():
            best_margin = mid
            low = mid
        else:
            high = mid
    if best_margin is None:
        return round_down_1000(low)
    return best_margin


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic audits for state profiles.")
    parser.add_argument(
        "--base-dir",
        default="[/path/to/votekit_csv_dir/]",
        help="Base directory containing the votekit CSV data.",
    )
    parser.add_argument(
        "--years",
        default="2016,2019,2022,2025",
        help="Comma-separated years to include.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/audit_results/state_audit_results.csv",
        help="Output CSV with audit results.",
    )
    parser.add_argument("--m", type=int, default=3, help="Number of winners to elect.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Risk limit for audit.")
    parser.add_argument("--noise-level", type=float, default=0.02, help="Noise level for audit.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon for audit.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per sample size fraction.")
    parser.add_argument("--min-successes", type=int, default=9, help="Minimum successes to accept a fraction.")
    parser.add_argument("--start-denominator", type=int, default=1000, help="Initial denominator to try.")
    parser.add_argument("--step-denominator", type=int, default=100, help="Denominator step size.")
    parser.add_argument("--min-denominator", type=int, default=100, help="Minimum denominator to try.")
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum number of fraction attempts.")
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


def tightest_margin(
    df: pd.DataFrame,
    quota_label: str = "Quota",
    skip_first_n_cols: int = 2,
    atol: float = 1e-9,
):
    cand = df.drop(index=quota_label, errors="ignore")
    cols = list(cand.columns)[skip_first_n_cols:]
    rows = []
    for col in cols:
        s = cand[col]
        nz = s[s.ne(0)]
        if len(nz) < 3:
            continue
        if quota_label in df.index:
            q = df.at[quota_label, col]
            if np.isfinite(q) and np.all(np.isclose(nz.to_numpy(), q, atol=atol, rtol=0.0)):
                continue
        third = nz.iloc[2]
        last = nz.iloc[-1]
        margin = float(third - last)
        rows.append(
            {
                "column": col,
                "margin": margin,
                "abs_margin": abs(margin),
                "third_row": nz.index[2],
                "third_value": float(third),
                "last_row": nz.index[-1],
                "last_value": float(last),
            }
        )
    results = pd.DataFrame(rows)
    if results.empty:
        return None, results
    results = results.sort_values(["abs_margin", "column"], ascending=[True, True]).reset_index(drop=True)
    return results.iloc[0], results


def find_state_paths(base_dir, years):
    base = Path(base_dir)
    paths = []
    year_order = {year: idx for idx, year in enumerate(sorted(years, reverse=True))}
    state_order = {"tas": 0, "sa": 1, "wa": 2, "qld": 3, "vic": 4, "nsw": 5}
    for year in years:
        year_dir = base / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("*_votekit.csv")):
            name = path.name.lower()
            if name.startswith("act") or name.startswith("nt"):
                continue
            paths.append(path)

    def sort_key(path):
        name = path.name.lower()
        state = name.split("_", 1)[0]
        year = int(path.parent.name)
        return (
            state_order.get(state, 999),
            year_order.get(year, 999),
            name,
        )

    return sorted(paths, key=sort_key)


def run_trials(profile, audit_graph, sample_fraction, trials, min_successes, alpha, noise_level, epsilon, m):
    successes = 0
    for i in range(trials):
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
        if trials - (i + 1) + successes < min_successes:
            break
    return successes >= min_successes, successes


def search_best_fraction(
    profile,
    audit_graph,
    start_denominator,
    step_denominator,
    min_denominator,
    max_steps,
    n_voters_adjusted,
    trials,
    min_successes,
    alpha,
    noise_level,
    epsilon,
    m,
):
    denom = start_denominator
    best_pass = None
    tried = set()

    def try_denom(denom_value):
        if denom_value < min_denominator or denom_value in tried:
            return None
        tried.add(denom_value)
        passed, successes = run_trials(
            profile,
            audit_graph,
            sample_fraction=Fraction(1, denom_value),
            trials=trials,
            min_successes=min_successes,
            alpha=alpha,
            noise_level=noise_level,
            epsilon=epsilon,
            m=m,
        )
        return passed, successes

    steps_used = 0
    initial = try_denom(denom)
    if initial is None:
        return None
    passed, successes = initial
    steps_used += 1
    if not passed:
        denom = max(min_denominator, denom - step_denominator)
        while steps_used < max_steps:
            result = try_denom(denom)
            if result is None:
                break
            passed, successes = result
            steps_used += 1
            if passed:
                best_pass = (Fraction(1, denom), successes)
                break
            denom = max(min_denominator, denom - step_denominator)
        return best_pass

    best_pass = (Fraction(1, denom), successes)

    last_success_denom = denom
    fail_denom = None

    while steps_used < max_steps:
        denom *= 2
        result = try_denom(denom)
        if result is None:
            break
        passed, successes = result
        steps_used += 1
        if passed:
            last_success_denom = denom
            best_pass = (Fraction(1, denom), successes)
            continue
        fail_denom = denom
        break

    if fail_denom is None:
        return best_pass

    low = last_success_denom
    high = fail_denom
    while steps_used < max_steps and high - low > 1:
        half_width = 0.5 * (1 / low - 1 / high) * n_voters_adjusted
        if half_width < 30:
            break
        mid = (low + high) // 2
        result = try_denom(mid)
        if result is None:
            break
        passed, successes = result
        steps_used += 1
        if passed:
            low = mid
            best_pass = (Fraction(1, mid), successes)
        else:
            high = mid

    return best_pass


def main():
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    years = [int(item) for item in args.years.split(",") if item.strip()]
    paths = find_state_paths(args.base_dir, years)

    fieldnames = [
        "path",
        "candidates_original",
        "candidates_viable",
        "n_voters_adjusted",
        "auditable_margin",
        "auditable_margin_over_n",
        "sample_size_fraction",
        "sample_size_n",
        "successes",
        "total_nodes",
        "nodes_per_layer",
        "status",
        "error",
    ]

    resume = not args.no_resume
    completed = {}
    if resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as handle:
            existing_reader = csv.DictReader(handle)
            for existing in existing_reader:
                key = existing.get("path", "")
                completed[key] = existing

    open_mode = "a" if resume else "w"
    with output_path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not resume or output_path.stat().st_size == 0:
            writer.writeheader()

        for path in paths:
            result = {
                "path": str(path),
                "candidates_original": "",
                "candidates_viable": "",
                "n_voters_adjusted": "",
                "auditable_margin": "",
                "auditable_margin_over_n": "",
                "sample_size_fraction": "",
                "sample_size_n": "",
                "successes": "",
                "total_nodes": "",
                "nodes_per_layer": "",
                "status": "",
                "error": "",
            }

            try:
                existing = completed.get(str(path))
                if resume and existing:
                    status = existing.get("status", "")
                    if status in {"pass", "no_pass"} or (status == "error" and not args.rerun_errors):
                        continue
                    resume_from_margin = (
                        status == "computed_auditable_margin"
                        and not (existing.get("sample_size_fraction") or "").strip()
                    )
                else:
                    resume_from_margin = False

                print(f"Loading profile from {path}")
                profile = load_scottish(str(path))[0]
                c_original = len(profile.candidates)
                result["candidates_original"] = c_original

                if path.name.lower().startswith("nsw_2016"):
                    manual_remove = [
                    "Ken CANNING",
                    "Susan PRICE",
                    "Sharlene LEROY-DYER",
                    "Howard BYRNES",
                    "Brian Malcolm TUCKER",
                    "Maree NICHOLS",
                    "Allan THOMAS",
                    "Bruce RELPH",
                    "Mitch CARR",
                    "Sam KEARNS",
                    "Darren McINTOSH",
                    "Ian Robert BRYCE",
                    "Dee ELLIS",
                    "Christopher BUCKMAN",
                    "Methuen MORGAN",
                    "James COGAN",
                    "John DAVIS",
                    "Anthony Geno BELCASTRO",
                    "Robert BUTLER",
                    "Ann LAWLER",
                    "Rob BRYDEN",
                    "Daniel KIRKNESS",
                    "Eric GREENING",
                    "Andy THOMPSON",
                    "Paul QUINN",
                    "Gregory FREARSON",
                    "Ross FITZGERALD",
                    "Sue RAYE",
                ]+['Jason Mark OLBOURNE', 'Ray GOODLASS', 'Phil JOBE', 'Nella HALL', 'Sam DASTYARI', 'Ray THORPE', 'Tom HARRIS', 'Brian BURSTON', 'Gillian EVANS', 'Steven LOPEZ', 'Lee RHIANNON', 'Arthur SINODINOS', 'William BOURKE', 'Ken STEVENS', 'James JANSSON', 'Dave VINCENT', 'Teresa VAN LIESHOUT', 'Peter BREEN', 'Adam WASHBOURNE', 'Berge Anthony DER SARKISSIAN', 'Paul McCORMACK', 'Karl HOUSEMAN', 'Deborah Ann Jane LIONS', 'Shayne HIGSON', 'Christine DONAYRE', 'Marise PAYNE', 'Sarah FERNANDES', 'Peter RAHME', 'Gordon ELKINGTON', 'Peter JOHNSON', 'Kate PARKER', 'Tania Stephanie PIPER', 'Alexandra COSTELLO', 'Kerry KOLIADIS', 'Dean MACKIN', 'Sally VINCENT', 'Robert James MARKS', 'Susan PERROW', 'Charles KNOX', 'Janise FARRELL', 'Beth SMITH', 'Andrew PHILLIPS', 'Jai COOPER', 'Nathan SPATARO', 'Christine Pamela BERNIER', 'Fiona NASH', 'Danny LIM', 'Jane OAKLEY', "Deborah O'NEILL", 'Archie LEA', 'Andrew John KATELARIS', 'Christina HO', 'Rhonda AVASALU', 'Leanne PAFF', 'Colin BROADBRIDGE', 'Nigel James SMITH', 'Greg GRAHAM', 'John WILLIAMS', 'Jenny McALLISTER', 'Concetta FIERRAVANTI-WELLS', 'Warren GRZIC', 'Jim MOLAN', 'Anthony DONA', 'Lena EL-DAGHL', 'Chris OSBORNE', 'Mark BRADBURY', 'James WRIGHT', 'Sang OK', 'Nicholas GLEDHILL', 'Paul Yi-Wen HAN', 'Nick CHAPMAN', 'Doug CAMERON', 'Arthur John EMMETT', 'Fiona LEVINY', 'Eve SLAVICH', 'Bryan LAMBERT', 'Cara Melissa DONNELLY', 'David ASH', 'Ash ROSE', 'Tara MORIARTY', 'Jane WARD', 'Michael OSBORNE', 'Stacey DOWSON', 'Liam MUNDAY', 'Simon McCAFFREY', 'Peter GOOLEY', 'Maree Ann CRUZE', 'Jananie JANARTHANA', 'Vivien THOMSON', 'John COOPER', 'Joanna RZETELSKI', 'Santa SPRUCE-PEET-BOYD', 'Kathryn MAIDEN', 'Peter WALLACE', 'Shuo ZHOU', 'Peter MULLER', "Mary O'SULLIVAN", 'Wes FANG', 'Sam KENNARD', 'Richelle Girado TSAY', 'Marika KONTELLIS', 'Sarah RICHARDS', 'Jagath BANDARA', 'Hollie HUGHES', 'Ron POULSEN', 'Victoria McGAHEY', 'Dawn WILLIS', 'Gareth BRYANT', 'Miriam RIZVI', 'Stephen MULLER', 'Leonard BROWN', 'Ingrid RALPH', 'Colin BENNETT']
                    profile = remove_and_condense_rank_profile(manual_remove, profile)
                    if len(profile.candidates) > np.iinfo(np.int8).max:
                        raise RuntimeError(
                            f"Candidate count {len(profile.candidates)} still exceeds int8 limit after manual removal."
                        )

                print(f"Identifying non-viable candidates for {path.name}")
                meek_elec = MeekSTV(profile, m=args.m, tiebreak="random")
                score_df = meek_elec.get_score_df()
                non_viable_cands = []
                non_viable_cands_5pct = []
                for idx, row in score_df.iterrows():
                    last_nonzero_index = row[row != 0.0].last_valid_index()
                    initial_quota = score_df.loc[score_df.index[-1]].iloc[0]
                    last_score = row[last_nonzero_index]
                    if last_score < 0.10 * initial_quota:
                        non_viable_cands.append(idx)
                    if last_score < 0.05 * initial_quota:
                        non_viable_cands_5pct.append(idx)

                viable_profile = remove_and_condense_rank_profile(non_viable_cands, profile)
                viable_score_df = None
                use_existing_margin = False
                if resume_from_margin:
                    existing_margin = (existing.get("auditable_margin") or "").strip()
                    if existing_margin:
                        try:
                            use_existing_margin = float(existing_margin) > 0
                        except ValueError:
                            use_existing_margin = False
                if not use_existing_margin:
                    viable_elec = MeekSTV(viable_profile, m=args.m, tiebreak="random")
                    viable_score_df = viable_elec.get_score_df()

                del profile, meek_elec, score_df
                gc.collect()

                result["candidates_viable"] = max(0, c_original - len(non_viable_cands_5pct))
                n_voters_adjusted = float(viable_profile.total_ballot_wt) * 1.01
                result["n_voters_adjusted"] = n_voters_adjusted

                if use_existing_margin:
                    auditable_margin = float(existing.get("auditable_margin", 0))
                    result["auditable_margin"] = auditable_margin
                    result["auditable_margin_over_n"] = auditable_margin / n_voters_adjusted
                else:
                    print(f"Computing auditable margin for {path.name}")
                    best, _ = tightest_margin(viable_score_df)
                    if best is None:
                        raise RuntimeError("Unable to compute auditable margin from viable score dataframe.")
                    auditable_margin = float(best["margin"]) // 1000 * 1000
                    result["auditable_margin"] = auditable_margin
                    result["auditable_margin_over_n"] = auditable_margin / n_voters_adjusted
                    result["status"] = "computed_auditable_margin"
                    writer.writerow(result)
                    handle.flush()

                print(f"Building audit graph for {path.name}")
                audit_graph = MeekGraph(
                    profile=viable_profile,
                    m=args.m,
                    auditable_margins_per_deg=[auditable_margin] * 3,
                    use_numerical_labels=False,
                )

                if not audit_graph.check_coherence():
                    print(f"Incoherent graph for {path.name}; searching coherent margin")
                    adjusted_margin = find_largest_coherent_margin(viable_profile, args.m, auditable_margin)
                    auditable_margin = adjusted_margin
                    result["auditable_margin"] = auditable_margin
                    result["auditable_margin_over_n"] = auditable_margin / n_voters_adjusted
                    audit_graph = MeekGraph(
                        profile=viable_profile,
                        m=args.m,
                        auditable_margins_per_deg=[auditable_margin] * 3,
                        use_numerical_labels=False,
                    )
                    if not audit_graph.check_coherence():
                        raise RuntimeError("Unable to find coherent margin within search bounds.")

                stats = audit_graph.get_stats()
                result["total_nodes"] = stats.get("total_nodes", "")
                result["nodes_per_layer"] = json.dumps(stats.get("nodes_per_layer", {}), sort_keys=True)
                result["status"] = "computed_graph"
                writer.writerow(result)
                handle.flush()

                print(f"Searching for sample size fraction for {path.name}")
                best_pass = search_best_fraction(
                    viable_profile,
                    audit_graph,
                    start_denominator=args.start_denominator,
                    step_denominator=args.step_denominator,
                    min_denominator=args.min_denominator,
                    max_steps=args.max_steps,
                    n_voters_adjusted=n_voters_adjusted,
                    trials=args.trials,
                    min_successes=args.min_successes,
                    alpha=args.alpha,
                    noise_level=args.noise_level,
                    epsilon=args.epsilon,
                    m=args.m,
                )
                if best_pass is None:
                    result["status"] = "no_pass"
                else:
                    fraction, successes = best_pass
                    result["sample_size_fraction"] = f"{fraction.numerator}/{fraction.denominator}"
                    result["sample_size_n"] = float(fraction) * n_voters_adjusted
                    result["successes"] = successes
                    result["status"] = "pass"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
            finally:
                if "viable_profile" in locals():
                    del viable_profile
                if "viable_elec" in locals():
                    del viable_elec
                if "viable_score_df" in locals():
                    del viable_score_df
                if "audit_graph" in locals():
                    del audit_graph
                gc.collect()

            writer.writerow(result)
            handle.flush()


if __name__ == "__main__":
    main()
