# Bayesian Ballot-Comparison Audit for STV Elections

This module implements a naive first pass of Rivest's Bayesian ballot-comparison audit framework, adapted for Single Transferable Vote (STV) elections.

## Quick Start

```python
from src.bayesian_comparison import (
    initialize_audit_state,
    generate_noised_sample,
    simulate_audit_from_sample,
    sequential_stopping_check
)
from src.utils import convert_pf_to_numpy_arrays

# 1. Load an election profile
from votekit.cvr_loaders import load_scottish
profile = load_scottish('data/scot-elex/5_cands/aberdeen_2017_ward11.csv')

# 2. Convert to ballot matrix
ballot_matrix, weights, _ = convert_pf_to_numpy_arrays(profile)

# 3. Generate noised sample using existing noise infrastructure
BAL, CVR = generate_noised_sample(
    profile, 
    sample_size=100,
    noise_level=0.02,
    num_ghosts=10,
    seed=42
)

# 4. Run audit from sample with partial auditing for posterior uncertainty
audit_state = simulate_audit_from_sample(
    CVR=CVR,
    BAL=BAL,
    candidates=list(profile.candidates),
    prior="laplace",
    num_winners=3,
    unseen_prior_mass=1.0,  # Allocate prior probability to unseen ballot types
    audit_fraction=0.5,      # Audit 50% of sample ballots (leaves rest for posterior)
    seed=123
)

# 5. Check stopping condition  
can_stop, results = sequential_stopping_check(
    audit_state,
    risk_limit=0.05,
    num_simulations=1000
)

print(f"Upset probability: {results['upset_probability']:.5f}")
print(f"Can stop: {can_stop}")
```

## Architecture

### Core Components

1. **Ballot Type System** (`BallotType`, `ballot_row_to_type`, `enumerate_ballot_types`)
   - Converts STV ballots to immutable type representations
   - Handles invalid ballots (-2), exhausted ballots (-127), and padding (-126)
   - Provides deterministic indexing for ballot types

2. **Audit State** (`AuditState` dataclass)
   - `t`: number of **observed** ballot types (from CVR)
   - `n`: total ballots in election
   - `R`: reported type counts (length t)
   - `C`: comparison matrix (t × **t+1**), where C[j,k] = # of ballots reported as j, actually k
     - **Columns 0 to t-1**: known ballot types
     - **Column t**: all unseen/novel ballot types (security feature)
   - `alpha`: prior hyperparameters (t × t+1)
   - `unseen_prior_mass`: probability mass allocated to unseen types (default: 1.0)
   - Tracks reported outcome, candidates, and STV parameters

3. **Unseen Type Protection** 🔒
   - **Security Feature**: Prevents exploitation via ballot types not in CVR
   - Model allocates prior probability to ballot patterns not yet observed
   - Conservative treatment: unseen ballots treated as exhausted in STV
   - Configurable via `unseen_prior_mass` parameter:
     - Higher values (e.g., 5.0): more cautious, slower stopping
     - Lower values (e.g., 0.5): less cautious, faster stopping
     - Default (1.0): symmetric Dirichlet assumption
   - See [UNSEEN_TYPES_SECURITY_IMPROVEMENT.md](UNSEEN_TYPES_SECURITY_IMPROVEMENT.md) for details

4. **Initialization** (`initialize_audit_state`)
   - Creates initial audit state from a VoteKit `PreferenceProfile`
   - Supports three priors:
     - `"haldane"`: all zeros (improper prior, faster convergence)
     - `"laplace"`: all ones (uniform prior)
     - `"custom"`: user-provided alpha matrix
   - Runs FastSTV to determine reported outcome
   - Automatically includes unseen type column in C and alpha matrices

5. **Sequential Updates** (`record_comparison`, `record_comparisons_batch`)
   - Records ballot comparisons: C[reported_type, actual_type] += 1
   - Maintains audit state across sequential sampling

6. **Posterior Sampling** (`sample_row_posterior`, `sample_full_election_tally`)
   - Uses gamma/Dirichlet conjugacy for efficient sampling
   - No MCMC needed - direct posterior simulation
   - For row j: sample from Gamma(alpha[j,k] + C[j,k]) for each k (including unseen column)
   - Handles unaudited ballots by sampling from prior

7. **STV Integration** (`run_stv_on_tally`, `create_fast_stv`, `filter_invalid_ballots`)
   - Converts ballot type tallies to VoteKit profiles
   - Monkey-patches FastCore to work with ballot matrices
   - Filters invalid ballots before STV (fpv < 0, internal negatives)
   - Normalizes -126 (padding) → -127 (exhausted) for FastCore compatibility
   - **Unseen types (column t) treated as exhausted ballots** in STV

8. **Audit Logic** (`run_posterior_simulations`, `sequential_stopping_check`)
   - Simulates many election outcomes from posterior
   - Computes upset probability: P(outcome ≠ reported outcome | comparisons)
   - Stopping rule: upset_prob ≤ risk_limit
   - Returns unique outcomes and their frequencies

9. **Noise Integration** (`generate_noised_sample`, `simulate_audit_from_sample`)
   - Wraps existing `bal_cvr_sample_constructor` from `src.edouard.noise`
   - 4-type noise model: delete, insert, swap, replace
   - Ghost ballots for undervote/overvote simulation
   - Builds audit state directly from CVR/BAL matrices
   - **Partial auditing** via `audit_fraction` parameter (default 0.5):
     - Only audits a fraction of the provided ballots
     - Leaves unaudited ballots for posterior uncertainty
     - Essential for proper Bayesian inference (without it, posterior is deterministic)
     - Lower values (e.g., 0.3) = more posterior uncertainty, slower stopping
     - Higher values (e.g., 0.8) = less posterior uncertainty, faster stopping

## Key Design Decisions

### Partial Auditing & Posterior Uncertainty
- **Critical design element**: Not all ballots in the sample are audited
- `audit_fraction` parameter controls what percentage of ballots are compared
- Unaudited ballots have `unaudited_count > 0`, allowing posterior randomness
- Without partial auditing: posterior samples become deterministic (all identical)
- With partial auditing: posterior properly quantifies uncertainty about unaudited ballots
- Default `audit_fraction=0.5` provides good balance between efficiency and uncertainty

### Ballot Encoding
- Candidate indices: 0, 1, 2, ...
- Exhausted: -127
- Padding: -126
- Invalid marker: -2
- FastCore only understands -127, so we normalize -126 → -127 before STV

### Invalid Ballot Handling
- Ballots with fpv < 0 are excluded from STV  
- Ballots with internal negatives (except -127, -126) are excluded
- Invalid ballots are tracked in the type system but filtered before outcome determination

### Type System & Unseen Types 🔒
- Ballots are mapped to types deterministically
- Invalid ballots get their own types but are excluded from winner determination
- **Open-world assumption**: Matrix includes extra column (index t) for unseen ballot types
- **When BAL has type not in CVR**: recorded to unseen column, not as new type
- **Security rationale**: Prevents adversarial exploitation via novel ballot injection
  - Without this feature: attacker could inject types not in CVR and evade detection
  - With this feature: model anticipates unknown types, requires more evidence before stopping
- See [UNSEEN_TYPES_SECURITY_IMPROVEMENT.md](UNSEEN_TYPES_SECURITY_IMPROVEMENT.md) for attack scenarios

### Prior Selection
- Haldane (alpha=0): faster convergence, more aggressive
- Laplace (alpha=1): conservative, uniform prior
- Custom: expert knowledge can be encoded

### Audit State from Samples
- `simulate_audit_from_sample` treats CVR as the reported distribution
- R is built from CVR sample counts (including invalid ballots)
- By default, only 50% of CVR vs BAL comparisons are recorded (via `audit_fraction`)
- This partial auditing ensures proper posterior uncertainty
- Appropriate for sample-based audits where full CVR may be unavailable

## Demonstration Notebook

See `bayesian_audit_demo.ipynb` for:
- Toy example with 4 candidates, 2 seats
- Ballot type enumeration demonstration  
- Posterior sampling verification
- Sequential update mechanics
- End-to-end Scottish election audit with synthetic noise
- Adversarial stress test showing audit behavior under strategic manipulation

### Adversarial Stress Test Results
- **Attack scenario**: 59 ballots (1.97%) strategically moved from "E>A" to "D>A"
- **Outcome change**: CVR reports (A,C) as winners, true outcome is (A,B)
- **Audit behavior** with `audit_fraction=0.5`:
  - n=50: upset_prob=0.915 (high uncertainty, correctly suspicious)
  - n=100: upset_prob=0.081 (borderline)
  - n=500: upset_prob=0.603 (correctly detects potential manipulation)
  - n=2997 (full): upset_prob=0.248 (continues to flag concern)
- **Key insight**: Audit produces realistic intermediate probabilities, not binary 0/1 jumps
- Demonstrates proper Bayesian uncertainty quantification

## Limitations & Future Work

1. **Naive Implementation**: This is a first pass for research purposes, not production-ready
2. **No Stratification**: Could be more efficient with stratified sampling by ballot type
3. **No Adaptive Sampling**: Sample size is fixed upfront, not optimized sequentially
4. **Fixed Audit Fraction**: `audit_fraction` is constant; could be adaptive
5. **Simple Priors**: Could incorporate structural knowledge about ballot correlations
6. **Partial Auditing Overhead**: Auditing only 50% of sampled ballots may seem wasteful
   - Trade-off: posterior uncertainty vs. audit efficiency
   - Future work: explore optimal audit fractions for different scenarios
7. **Conservative Unseen Types**: Treating unseen types as exhausted is pessimistic
   - Could model unseen types more sophisticatedly with domain knowledge

## Testing

Run module self-tests:
```bash
python -m src.bayesian_comparison
```

Tests include:
- Tiny toy contest (3 candidates, 2 seats)
- Posterior dimension verification  
- Zero-error full-sample audit
- Sequential update mechanics
- Integration with `bal_cvr_sample_constructor`

## Dependencies

- numpy: array operations, sampling
- votekit: STV election profiles, ballots
- src.utils: ballot matrix conversion
- src.edouard.FastSTV: STV tabulator
- src.edouard.noise: ballot noise generation

## References

- Rivest, R.L. "Bayesian Tabulation Audits Explained and Extended" (2018)
- Meek, B.L. "A New Approach to the Single Transferable Vote" (1969)
- VoteKit documentation: https://github.com/mggg/VoteKit
