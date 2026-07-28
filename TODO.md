# COBRA audits:

We want to use the [COBRA](http://arxiv.org/abs/2304.01010) framework to do a full, rigorous, graph-based audit of the Portland D1 election -- initially just for WIGM STV. The components of the codebase we will need for this are:

- [x] An audit graph constructor that will be able to build an optimal graph for this election.
- [x] An edge-local assorter+compiler object that will take for input information about a vertex in the graph, as well as an escape edge from that vertex, to initialize a COBRA martingale that it will be able to update as the audit progresses.
- [x] An implicit random sampler that will take in a `RankProfile` or `NumpyRankProfile` and sample rows, with or without replacement, from the ballot matrix, with probabilities determined by their weights in the `wt_vec`.
- [x] A global driver that will take the constructed graph and decide what local compilers it needs to initialize, then update these compilers with the sampled ballots as the audits progress.

## Edge-local COBRA compilers.

In this section I use the words "compiler" and "test process" interchangeably -- the latter term is better to describe a mathematical concept, whereas the former is better suited to describe a computer science object.

We need to extend the codebase with a new subdir, `src/test_processes`, and put a first script in there called `cobra.py`. The main object in this script will be an edge-local test process corresponding to an "escape edge" of the underlying graph. This is an edge that *was not* included by the graph constructor; its base is a vertex in the graph, but its tip would be a vertex not in the graph.

This compiler will have to:
- [x] Handle initialization, which involves:
    - [x] Deciding what the critical margin is;
    - [x] Determining the candidate indices relevant to that margin;
    - [x] Pre-compute some values for the COBRA test process.
- [x] Have an optimal lambda search-function, and an alternative naive search-function.
- [x] Have an assorter method, which takes for input two rows of the ballot matrix (one row corresponding to a CVR, the other to a paper ballot), and projects them onto the relevant candidates, then assigns a conservative overstatement estimate to this pair.
- [x] Have an internal COBRA-based martingale M, with an updater method, which calls the assorter method, then converts it to a T-value, which it multiplies by M to get the new value of M.
- [x] Have some debug/formatting methods, that the user can use to understand the internals of the compiler.

As we work on this, bear in mind that our global audit driver will have to keep many of these objects in memory at a time -- probably thousands for large audits -- so keeping them lean is a goal.

### Initialization

This escape edge could be an elimination edge or an election edge, and the reason it was not included always depends on a candidate-to-candidate or candidate-to-quota margin, although which margin specifically is something the local compiler will have to decide.

```
class CobraCompiler():
    def __init__(
        self,
        base_vertex: ElectionState      # from src/election_graphs/datatypes.py
        candidate_index: int            # the index of the candidate who would be eliminated or seated in the escape edge
        candidate_strings: list[string] # a list with candidate strings ordered canonically
        edge_type: EdgeAction           # the action corresponding to the escape edge, enum defined in src/election_graphs/datatypes.py
        LAM: int | float                # the "least auditable margin" for the underlying graph
        quota: int                      # the value of quota for the election -- we force this to be fixed for now.
        N: int                          # the number of votes in the election
        noise_level_guess: float        # the fraction of ballot discrepancies we expect overall
        simultaneous: bool = False      # whether or not this election allowed for simultaneous seating of candidates with quota
    ):
    [...]
```

The first thing the compiler will have to do during initialization is validate that the parameters it was handed make sense -- in particular, the candidate corresponding to `candidate_index` should still be in `base_vertex.hopefuls` (note: to convert a candidate index `idx` to a string, use `candidate_strings[idx]`); and the LAM, N, and quota should all be positive, with `LAM < 2*quota/3` (although that last validation should just raise a flag, rather than error).

The second thing the compiler will have to do during initialization is decide which margin is the most relevant to the escape edge we are looking at. This depends on the `edge_type`, and the property `tallies = base_vertex.tallies` already computed by the underlying graph. This is a dict with string keys and float values, corresponding to the tallies of each candidate computed by the CVR. This decision process looks like:

1. If `edge_type` is ELIMINATE in the enum: either there is another tally that is more than LAM in excess of quota, or there is a candidate whose tally is more than LAM below our candidate_index.
    1. First, look for any `tally` in the values of `tallies` such that `tally >= quota + LAM`. In such a case, the corresponding candidate index `w` should be force-elected, and the critical margin corresponding to our local edge is the candidate-to-quota margin of `w`.
    2. If there are no such tallies forcing an election, instead look for the candidate index `L` with the smallest tally in `tallies`; this tally `T_L` should satisfy `T_L + LAM < T_c`, where `T_c` is the tally of `candidate_index`, in which case the critical margin is the candidate-to-candidate margin `M_{cL}=T_c-T_L`.
    3. If the condition `T_L + LAM < T_c` is not met, print out a flag that something unusual is happening (specifying the candidate strings and tallies involved), but proceed with initialization, unless `L` is the same candidate as `candidate_index`, in which case we should throw an error (again, printing out a helpful autopsy).
2. If `edge_type` is ELECT in the enum: either the tally `T_c` of `candidate_index` is more than LAM below quota, or there is another candidate `w` whose tally `T_w` is more than LAM above `T_c` (note: the latter case only matters if `simultaneous = False`).
    1. If there is a `T_w` such that `T_w > T_c + LAM` *and* `simultaneous = False`, the relevant margin is the candidate-to-candidate margin `M_{wc} = T_w - T_c`.
    2. Otherwise, it should be the case that `T_c + LAM < quota`; if so the relevant margin is the candidate-to-quota margin `quota - T_c`.
    3. If none of the above two options are met, throw an error and print an autopsy.

Once we have identified the margin, we should (1) figure out who are the relevant candidates to the margin (this will determine the behavior of our assorter method), then (2) there are some quantities we need to pre-compute during initialization.

For step (1), we want to internally store three lists (or sets, or numpy arrays, or direct indices, up to you) of candidate indices: the indices of winners who are already seated in `base_vertex`, the indices of candidates that matter for our critical margin, and the indices of eliminated candidates.
- To find the winners, just use the `seated_at` property of the `base_vertex`. Store this as `self.seated_winners`.
- To find the candidates that matter to the critical margin -- there is always just one or two such candidates, and they should be stored directly as properties of the compiler -- not as a list:
    - If we are doing a candidate-to-candidate margin, these should be stored as `self.canonical_victor` and `self.canonical_loser`. Which index is which should be picked so that the tally of `canonical_victor` is always greater than the tally of `canonical_loser` in the `base_vertex`.
    - If we are doing a candidate-to-quota margin, there are two cases: either we are proving that a candidate definitely has quota (case 1.1 above), in which case we should call that candidate `self.canonical_winner`. Oterwise, we are proving that a candidate doesn't have a quota (case 2.2 above), in which case we should call that candidate `self.canonical_non_winner`.
- Finally, all the candidates that are not seated and not in `base_vertex.hopefuls` should be stored in a list `self.eliminated_candidates`.

Next, we need to compute some quantities during intialization -- these are mainly `self.v= M/N`, where `M` is the critical margin according to the tallies in `base_vertex` (note: `M` should always be positive! so in case 2.2 above this margin should really be `quota - Tc`). Once we have `self.v`, we should also compute and store `self.a = 1/(2-self.v)`.

The last part of initialization is we should set `self.M = 1` to initialize our martingale capital, then call a method `self.find_lambda(...)` to decide the optimal COBRA betting parameter, which is what I discuss next.

### Lambda calculation.

The COBRA paper gives two ways to decide how to set lambda; one that is naive and deterministic, and the other which involves solving a rational expression numerically. We should use the numerical approach by default.

For the naive version, we'll use `p_0 = (1- noise_level)`, although even that is not accurate, since `noise_level` is the probability of a *given ballot* showing a discrepancy, not a *given position in the ballot* -- but this is conservative. In this case we should return `lambda = (2- 4*self.a*p_0)/(1-2*self.a)`.

Otherwise, we should set `p_2 = noise_level / len(candidates)`, `p_1 = noise_level - p_2`, and `p_0 = 1 - p_1 - p_2`, then use the value of lambda that is a root of the following expression:
$$\frac{(a-1/2)p_0}{1+\lambda(a-1/2)} + \frac{(a-1)p_1}{2-\lambda(1-a)} - \frac{p_2}{2-\lambda}$$

I don't really care what numerical optimization approach we use to find this value of lambda, as long as it returns a value of lambda that is within [0,2). If there is no such optimum in that interval, we should raise a flag and default to the naive version.

Whatever value of lambda we end up with should be stored in `self.lambda_value`.

### Assorter method.

The assorter method needs to take in two numpy arrays, corresponding to rows (call them `row_c` for CVR and `row_b` for Ballot) of the ballot matrix as recorded in the CVR vs in the paper ballots. These rows will often be identical; in such a case, the assorter can instantly return `self.a`.

Even when the rows are not identical, there is a chance the discrepancy does not matter at the level of the candidates relevant to our critical margin. To determine this, we need to turn both numpy arrays into an array `winner_prefix` (initialized with `np.zeros_like(self.seated_winners)`) and an numpy.int8 or int16 `fpv`. To compute those, mask out the eliminated candidate indices from the row, and look at the first non-masked position; if that position is in `self.seated_winners`, add it to `winner_prefix`, otherwise, that position contains the `fpv` for that row.

After projecting both rows, we end up with two arrays `winner_prefix_c` and `winner_prefix_b`, and two fpv candidate indices `fpv_c` and `fpv_b` (those fpvs might be the sentinel value of -127, which is fine). If these are all pairwise equal, the discrepancy does not matter to us, and we can return `self.a`.

Otherwise, we first use `fpv_c` and `fpv_c` to determine the "raw" overstatement or understatement `w` of the sample. This `w` can take on values in {-2, -1, 0, 1, 2} for candidate-to-candidate margins, and {-1, 0, 1} for candidate-to-quota.

For candidate-to-candidate: `w = (fpv_c == self.canonical_victor) - (fpv_b == self.canonical_victor) + (fpv_b == self.canonical_loser) - (fpv_c == self.canonical_loser)`.
For candidate-above-quota: `w = (fpv_c == self.canonical_winner) - (fpv_b == self.canonical_winner)`.
For candidate-below-quota: `w = (fpv_b == self.canonical_non_winner) - (fpv_c == self.canonical_non_winner)`.

Finally, if the `winner_prefix` arrays are not both still `numpy.zeros` (indicating that one or both ballots had a non-empty winner prefix), we should floor `w` with 1 -- i.e. do `w = max(1, w)`. This is the only generalization we do to extend COBRA to the non-linear STV setting -- we are just conservative with respect to discrepancies that affect transfer values in any way.

Finally, map the given `w` with an output:
- `w=2` -> return `0`
- `w=1` -> return `a/2`
- `w=0` -> return `a` (but this should not happen if we detected a discrepancy earlier)
- `w=-1` -> return `3*a/2`
- `w=-2` -> return `2*a`.

### Martingale Updater Method.

With the previous two parts done, the updater is very easy to handle. The updater is handed two numpy ballot rows, `row_b` and `row_c`; it passes them to the assorter method to get a value `x` in {0, a/2, a, 3a/2, 2a}; and then it does `self.M *= (1+self.lambda_value*(x-1/2))`. The updater should return the updated value of `M`.

### Debug and Formatting Methods.

We probably want a method `self.print_info` which prints out and explains the internal variables of the compiler -- that is, show:
- What the critical margin is (candidate-to-candidate, candidate-above-quota, or candidate-below-quota), 
- who the candidates involved are, who the hopefuls are (print both indices and strings for all involved candidates), who the seated candidates are, and 
- what the internal vales of `v`, `lambda_value`, `a` were initialized as.

We probbly also want an optional initialization parameter `verbose = False` to initialization and the updater which will print the value of `self.M` as well as any detected discrepancy every time the updater is called.

## Random Ballot Matrix Row-Samplers with Implicit Noising.

We need to make an implicit row-sampler that can construct artificially noised pairs of ballot-CVR rows and feed them to our global audit coordinator, which will in turn feed them to each of its compilers.

Some desiderata we have for this random sampler:
- It should be memory efficient -- avoid materializing the whole sample at once, and rather one materialize cvr-ballot rows when they are called.
- It should be computationally efficient -- only perform operations on a given row when it has been selected for sampling, otherwise just use a direct view of the ballot matrix.
- It should support both sampling with replacement and without; for our current COBRA audits, we will only ever use the with replacement option, but we will eventually want without replacement for other compilers.
- It should be seeded and fully reproducible; the sampler should be initialized with an internally stored seed, which can be used to fully reproduce the way it noises the sample.

Here are the parts we will need to code for this sampler:
- [x] Initialization: upon initialization, the sampler will store some global properties (including its seed), then create and store an array of row-indices of the ballot matrix it plans to sample, along with a boolean mask for that array indicating which positions it intends to noise.
- [x] Hash method: I believe the easiest way to balance reproducibility with efficiency is to use the position of each index selected for noising to hash the globally owned key, then store the hashed keys separately. The hash function we use is up to your choice; I just need it to be fast -- in particular security is not a concern here.
- [x] Noising method: Each row index selected for noising will be noised in one of 4 ways described below; the way this noising occurs should be deterministic with respect to the hash function.
- [x] `self.sample(i)` method: this is the method the audit coordinator will call, which should return the row pair `row_c, row_b`.

### Initialization.

I expect something like this:

```
class ImplicitSampler():
    def __init__(
        self,
        noise_level: float,                          # the fraction of the sample that should bear any discrepancy
        n_cands: int                                 # the number of candidates in the election.
        wt_vec: NDarray,                             # the wt_vec corresponding to the ballot_matrix we are sampling from.
        sample_size: int | None = None,              # the desired maximum sample_size for the audit -- can be none, which indicates we want a fractional sample size instead.
        fractional_sample_size: float | None = None, # can replace sample_size
        with_replacement: bool = False,              # whether to use sampling with replcement or not
        seed: int = None                             # if None, generate and store an internal seed
        cache_rows: bool = False                     # if True, the Implicit Sampler will cache rows of the ballot_matrix that it has previously had to noise
    ):
    [...]
```

If `seed` is not specified, use the current date in Melbourne Australia to generate an appropriate seed -- should be the same on any given day.

During validation, we need exactly one of `sample_size` or `fractional_sample_size` to be specified and non-negative, and if we are sampling without replacement we need `sample_size < wt_vec.sum()` or `fractional_sample_size < 1` (and you should raise a flag if these conditions are not met for sampling with replacement -- that would be weird). Having zero noise should be allowed and completely fine.

Also during validation, we should check that the `wt_vec` we were given has integer values, despite its entries being typed as floats -- i.e. their decimals should all be 0.

If the sample size is specified via the fractional method, we should decide during initialization what the integer sample size will be, by doing `int(wt_vec.sum()* fractional_sample_size)`.

I am deliberately not providing the underlying ballot_matrix during initialization; I want to avoid copying it as a property of the sampler, so instead my plan is to pass it as an argument to `self.sample(i)` every time it is called. Ideally we don't store the `wt_vec` locally after initialization either.

Also during initialization, we should construct two arrays; one first array `sampled_indices` that indicates which row indices we plan to sample, satisfying `len(sampled_indices)==self.sample_size`, and if we are sampling without replacement, `np.bincount(sampled_indices)[i]< wt_vec[i]` -- i.e. in the without replacement regime, we are still allowed to see the same row index multiple times, up to `wt_vec[i]`. This should be outsoured to a helper method; you can have a look at the function `numpy_random_transfer` in `votekit.elections.election_types.ranking.stv.utils.py` for inspiration for the without replacement version of that method. For sampling with replacement, you should ensure that if two sampled indices are literally the same (as in, even before accumulating the sampled indices to their per-row owners using the wt_vec, they were meant to represent the same ballot out of the possible `wt_vec.sum()` choices), and one of these is selected for noising, then the other one should be noised too, and it should be noised in the same way.

The second array we need to make is a boolean array `noised_mask` with length equal to `sample_indices` indicating which rows have been selected for noising. These noised indices should be picked randomly to so that the number of 1s (or trues) in this array divided by its length is close to `noise_level` (maybe it's easier to store this as an array of indices poiting to entries in `sample_indices`? up to you).

Finally we should construct and store one last array that tells us, for each row selected for noising, what the seed is for that row's noise -- this should call a vectorized hashing method, whose design is up to you.

### Noising.

When a row is selected for noising, we should have `self.sample` make a copy of that row, and noise it in one of four ways:
1. Replace a random candidate index in the row with a random one not in the row.
2. Insert a random candidate not in the row into a random position in the row.
3. Randomly select two non-empty positions in the ranking and swap them.
4. Randomly delete a candidate index in the ranking (and then add back in a padding -127 at the end to keep the row the same length).

Note that option 1 and 2 will not work when `len(row.nonzero()) == n_cands` -- in this case, we should instead apply one of options 3 or 4 to that row, but we need to make sure that every row that gets selected for noising truly does get noise.

Option 4 is allowed to delete the only non-negative entry in a row; in this case, the row should become all -127, indicating an empty ranking -- this is fine.

The noising method should be deterministic with respect to the hash key that comes with it.

Finally, if globally we have `cache_rows = True`, we should store an internal record of only those rows that we have noised, remembering what the noised output looked like; if the `sample(i)` calls for the same row again, we can just use the cache rather than re-noising.

## Global Audit Driver.

The driver takes in a constructed graph, determines all the compilers it needs to initialize, initializes them, then initializes an implicit sampler, and uses it to update each compiler (broadcasting the same pair of rows for all compilers still running). When a compiler's capital `compiler.M` grows above a set value of `1/alpha`, the driver optionally removes the compiler from the update pool. The driver stops when all compilers have certified, or when the sample size has exceeded our pre-set maximum.

For now, our drive will only allow COBRA compilers, but eventually the plan is to have the drive be able to initilize other compilers as well; in fact, eventually, the compiler might initialize different compilers for different edges. The driver should expect `WIGMGraphConstructor` only -- this drive does not need to know how to handle a `MeekAuditGraph` yet.

Here are the parts we need:
- [x] Initialization:
    - [x] Set up an implicit sampler.
    - [x] Using the provided audit graph, determine the escape edges, then initialize a compiler for each.
- [x] Main method `.run()`:
    - [x] Reset all of the compilers' capitals `self.M` to 1.
    - [x] Initialize the compiler pool with all the compilers inside of it.
    - [x] Iterating over the row pairs provided by our sampler, update all of the compilers in the pool. Then, if an global option is set, remove all compilers whose capital has exceeded a threshold from the pool, and check if the pool is now empty.
    - [x] Unless an option `print_diagnostics_every` is set to `0`, print information about the remaining compilers ever `print_diagnostic_every` samples, using their built-in `print_info` method.

### Initialization.

For initialization, I should pass:
- an `audit_graph: WIGMGraphConstructor`, although we should need to copy or store this graph after initialization.
- a `sample_size` or `fractional_sample_size`, a `noise_level`, and an optional `seed` which it will use to initialize the implicit sampler. Also note that for COBRA compilers, this sampler should be initialized with replacement. The arguments `n_cands` and `wt_vec` it should be able to extract from the audit graph.
- a `compiler_type: str = "cobra", which it will use to decide what compilers to setup. Later, this argument might become more complicated, but for now it can stay simple.
- default `print_diagnostics_every` to 30 if not specified.
- default `self.alpha` to 0.05 if not specified; the cutoff capital value is 1/alpha.
- initialize `keep_certified_compilers` to False if not spefified; when this parameter is False, we remove compilers from the update pool once their capital exceeds the threshold, to save on computations.

To determine what compilers to setup: Iterating over the layers of the graph, iterating over the non-terminal vertices in that layer, identify the edges leaving from that vertex (using logic similar to the `.lookup_edges_from` method of the graph). Then, for every candidate in that vertice's `hopefuls` frozenset (access via `vertex.key.hopefuls`):
- if no edges leaving the vertex have an edge action of eliminate, we should add a compiler corresponding to that escape edge;
- if no edges leaving the vertex have an edge action of elect or force_elect, add a compiler corresponding to that escape edge.

### Running the driver.

The `.run()` method should mainly take for input a ballot matrix, which it will use to get random samples from the implicit sampler. It can optionally take an argument `num_steps` which will just run that many steps from the current point. And maybe another optional argument `reset_compilers = True` to restart all the compilers' capitals at 1, and add them all back to the pool, restarting `self.i` at `0` in the next step.

It iterates over the indices `i` (maybe store `i` as a global property) within range of the sample size, passing those indices to the sampler, and broadcasting the rows it gets to update all compilers in the pool. After each update, check if we are done -- when `keep_certified_compilers` is True, this means checking that all compiler's capital is more than the threshold, otherwise it just means checking that the pool isn't empty yet.

You might want to delegate the updating logic to a helper method `update_pool`, which can then be called manually by the user if they want to run a single step of the process.

When `i` is a multiple of `print_diagnostics_every`, except `i==0`, print some global information about the pool -- how many compilers remain, and for the lowest `5` capital values remaining in the pool, some minimal information about the corresponding compilers -- what was the escape edge, who is the compiler's critical margin comparing them to, and label the compiler with an escape edge ID that the user can lookup; this ID should have an alphabetical component corresponding to the layer of its base vertex, and a numerical component corresponding to its position within that layer, similar to the labeling conventions for vertices of the audit graphs. This diagnostic method should probably also be a separate helper method, so the user can call it on an interrupted audit.

Finally, if and when the audit terminates -- meaning that the pool is empty, or `self.i` has exceeded the maximum sample size -- you should have the driver print out some information about how the audit went; if successful, what our final sample size ended up being, and what the last three compilers were to pass threshold, and if the audit failed, print full diagnostics for the three compilers with the lowest capitals.

# Fitting these Audits to the Graph Seeding Process.

Occasionally, we use a so-called *seeding* process to construct our graphs, wherein we batch-eliminate a prescribed group of weak candidates all at once, and seed the graph after this batch elimination with a set of vertices, each of which must contain a prescribed set of strong candidates, as well as any permutation of "in-between" candidates -- these are candidates that are neither on the weak or strong list. Occasionally, we also allow some very strong candidates to be seated before the batch elimination; in such a case, the seeded graph will have a few initial vertices after the root corresponding to some very strong seatings, then the batch elimination (which looks like a bunch of empty layers in the audit graph), and finally a post-seed audit graph constructed in the same way as usual.

The strong set is usually prescribed by the user -- although it must satisfy the property that all strong candidates, if they were the only candidates left in the election would have a tally that is definitely below quota. The weak set, on the other hadn, is defined implicitly by the strong set. A candidate is weak with respect to a strong set if its "frozen mentions" are definitely less than the lowest strong candidate tally in the last vertex before the batch elimination -- the below code block illustrates how to determine this set ("definitely less" than means "more than a LAM below" in this context).

```
from votekit.pref_profile.numpy_profile import NumpyRankProfile, remove_and_condense_numpy_profile, numpy_profile_fpv
from votekit.utils import mentions_from_numpy_arrays

def find_weak_from_strong(numpy_profile, strong_cands, m, LAM, verify_strong = False):
    if verify_strong:
        non_strong_cands = [c for c in numpy_profile.candidates if c not in strong_cands]
        strong_cands_pf = remove_and_condense_numpy_profile(numpy_profile, non_strong_cands)
        strong_cands_fpv = numpy_profile_fpv(strong_cands_pf)
        quota = math.floor(numpy_profile.total_ballot_wt/(m+1)) + 1
        for c, score in strong_cands_fpv.items():
            if score + LAM >= quota:
                    raise ValueError(
                f"Strong candidate {c} has FPV {score} which is more than quota {quota} minus LAM {LAM}")
    initial_fpv = numpy_profile_fpv(numpy_profile)
    smallest_strong_fpv = min(score for c, score in initial_fpv.items() if c in strong_cands)
    frozen_mentions = mentions_from_numpy_arrays(numpy_profile, freeze_behind=strong_cands)
    weak_cands = [c for c, score in frozen_mentions.items() if score + LAM <= smallest_strong_fpv]
    return weak_cands
```

When we construct an audit graph using the seeding process (we do this for computational efficiency), we have an additional set of assertions to check:
1. If there are any very strong candidates, they definitely have a quota of votes in their respective vertices.
2. After electing the very strong candidates, all strong candidates would definitely not have a quota if they (collectively) were the only candidates left in the election.
3. For each weak candidate, their frozen mentions would definitely be lower than the lowest strong candidate tally in the last vertex before the batch elimination.

The first assertion corresponds to a typical candidate-above-quota margin, which our compilers know how to handle; only the way the constructor currently constructs the base vertices of the graph is atypical, so we'll want to update our constructors to make sure they are storing enough information in the vertices to formulate these assertions. We'll also want to update the WIGM constructor to properly allow simultaneous elections, which is something I've been putting off until now.

The second assertion correspods to a typical candidate-below-quota margin, although the base vertex will not be obvious to find. The seeding process should always include a vertex with only the strong candidates somewhere in the seeds (by design), but we should just verify that there aren't any election edges from that vertex within the graph -- otherwise we're in trouble.

The third assertion corresponds to a margin type we haven't seen before -- a candidate-to-mentions margin. We will have to extend our current COBRA compiler to be able to initialize and assort for such a margin, and then we'll have to tell the driver which of these compilers in needs to initialize as a function of the strong and weak candidates the graph was seeded with.

So our TODOs look like:
- [x] Update/modernize the wigm graph constructor:
    - [x] Allow an optional argument `simultaneous = False` to be passed to constructor; when True, and when multiple candidates are within LAM of quota, they might be seated simlutaneously (this might get very messy!)
    - [x] For the `seeded_build` method, allow a set of `very_strong` candidates to be specified; if so, these candidates will be seated before batch elimination, when possible.
    - [x] Create an optimal weak-vs-strong search function that will look for set of strong candidates that induces the largest set of weak candidates, and allow the seeded constructor to use this search function if strong / weak candidates are not specified.
    - [x] When the graph was constructed using the seeding process, globally store the set of weak and strong candidates 
- [x] Update our edge-local COBRA compiler to deal with candidate-to-mention assertions.
    - [x] Initialization will change: we will still be given a candidate and a base vertex, but we will also need to be told who the strong candidates are, and optionally the compiler can be handed the pre-computed frozen mentions.
    - [x] We will need a custom assorter function to recognize when a ballot should or shouldn't count for the frozen mentions; although the mapping from net discrepancy effects in {-2,-1,0,1,2} to the martingale updater will still be the same.
- [x] Update the driver to add the necessary compilers for batch eliminations
    - [x] When the graph passed to the driver was computed via `seeded_build`, store the very strong, strong, and weak candidates locally
    - [x] Compute and store the frozen mentions once at the start of audit
    - [x] For each weak candidate, add a candidate-to-mentions audit comparing the lowest tally of a strong candidate in the last vertex before the batch elimination to hte mentions of the weak candidate.

A note towards generalization; the reason we only allow very strong candidates to be seated before batch eliminations is just because of the empirical profiles we apply these auditing methods too -- these tend to have very strong winners early in the election, followed by many weak candidates who would get eliminated roughly at the same time, which would cause the number of vertices in the graph to blow up combinatorially if we didn't batch-eliminate them. But, it is perfectly plausible to imagine profiles that would need two batches of eliminations -- although we shouldn't plan our current graph infrastructure around that. 

## Modernized WIGM Graph Constructor.

So far my process for constructing seeded graphs has been pretty ad-hoc, and we should normalize and streamline some of it.

### Simultaneous Election Setting.

So far, the constructor only allows for candidates to be elected one at a time; but in some municipalities, when multiple candidates reach quota in the same round, they can be elected simultaneously. This makes a difference for how the votes transfer between winners; for non-simultaneous elections, votes can transfer from the first winner to the next (because there is a round of transfers in between), but for simultaneous elections, all winners compute their transfer values simultaneously, and when a vote from one winner lists another winner as their next preference, that vote skips over that next winner if they are elected simultaneously. Care should be given that we re-weigh the wt_vec correctly for simultaneous elections -- the `already_elected` and `transfer_values` parameters of the `seeded_build` method should show how to do this (these methods should also be removed once we have correctly implemented simultaneous elections).

We should also give care to how we store the edges corresponding to simultaneous elections; so far, edges have only ever needed to point from one layer of vertices to the next, but simultaneous election edges will point to layers further down. This is still compatible with the current architecture of our edge layers, since we specify a `dst:VertexRef` for each edge, which can specify the destination layer for the edge -- but care should be taken that this is handled properly.

We also will need to make sure that the solution we go with is still compatible with the child vertices' use of the `seated_at` property to search for the most recent `wt_vec` to construct their tallies; to handle this cleanly, it will probably be cleanest to have simultaneous elections actually create duplicate edges for each winner in the simultaneous election step, each coming and pointing to the same place -- although you might be able to get away with storing a `wt_vec` in only the last of those duplicates you create, if the children's `seated_at` property also follow that same order (since the children should only need to lookup the `wt_vec` of its latest `seated_at`). All these duplicate edges can just re-use the same `EdgeAction.ELECT` -- I don't see a need to create a new edge action category here.

In terms of election logic, we allow a simultaneous election whenever multiple candidates are within LAM of quota; we should add one simultaneous election edge for each permutation of such candidates (e.g. if 3 candidates are within LAM of quota, we have 7 options; 3 edges for single elections of one candidate, 3 options for simultaneous elections of 2 candidates -- adding 6 edges -- and 1 option for the simultaneous election of all three). We should make sure that we never simultaneously elect more candidates than there are seats remaining in the election; if that were possible, we should check if the lowest candidate is more than a LAM below the highest, in which case we shouldn't allow for their simultaneous election; otherwise just allow simultaneous elections of magnitude up to the number of seats left.

Finally, if one of the candidates is more than a LAM above quota, we should still allow them to be combined with other candidates who are up to a LAM below quota, *but* we should not allow for simultaneous elections of permutations of candidates that do not include the forced winner.

So, here are some example tallies to investigate the edge cases -- assume quota is 1000 and LAM is 150, m is 3.
- tallies = [1100, 900, 1000, 500, 300] -> could elect {0} or {1} or {2} or {0, 1} or {0, 2} or {1, 2} or {1, 2, 3}.
- tallies = [900, 900, 900, 900, 300] -> could elect {0} or {1} or {2} or {3} or {0, 1} or {0, 2} or {0, 3} or {1, 2} or {1, 3} or {2, 3} or {0, 1, 2} or {1, 2, 3} or {0, 2, 3} or {0, 1, 3} *but not* {0, 1, 2, 3}.
- tallies = [1200, 900, 800, 300, 300] -> could elect {0} or {0, 1} *but not* {1}.
- tallies = [0, 0, 1100, 900, 300] *and* two seats are already filled -> could elect {2} *but not* {3} because 3's tally is more than a LAM below 1100.

### Strong and Weak Candidate Search.

Currently, the seeded builder requires the user to specify the full strong and weak candidate set, as well as very strong candidates and their transfer values if necessary. The constructor should allow any or all of these arguments not to be fully specified -- really we should only need the (unordered) very strong candidate set.

When the very strong candidate set is specified, start constructing the graph allowing the election of these candidates (if their elections would be forced), using the simultaneous or non-simultaneous protocol the constructor was initialized with, and compute the transfer values + wt_vec locally rather than have it be inputted by the user. Then it should stop at the last vertex resulting from such a forced election, and determine the weak and strong candidates if they were not specified.

If the strong candidates are specified but the weak are not, the constructor should use the code blurb above (which can be stored in a new scipt in `src/election_graphs/utils.py`) to determine the induced set of weak candidates.

If the strong candidates are not specified, the constructor should search for the set of strong candidates that induces the largest set of weak candidates, while still satisfying the property that the strong candidates would not have a quota if alone in the election. To do this, I recommend including candidates with the next highest current fpv one at a time (starting with enough candidates to fill the remaining number of seats, and proressively expanding the strong set). Compute the fpv tallies of the strong set if they were alone in the election, and confirm that it is more than a LAM below quota. If so, compute the induced weak set. Then, keep expanding the strong set until the weak set stops growing (the smallest strong set is not always the one that induces the largest weak set, because including more strong candidates also freezes more votes for fpv computation -- but in practice the optimal strong set does tend to be small).

Finally, we should globally store the very strong, strong, and weak candidates, as well as the frozen mentions corresponding to the strong set we constructed the graph with, so that we can pass them to any local compilers that will require them.

### Updating the COBRA compilers to deal with candidate-to-mention margins.

When we audit a graph that was seed-constructed, we have to verify some additional assertions justifying the seeding. In particular, we will have to verify that the frozen mentions of each weak candidate is definitely below the lowest strong candidate tally in the last vertex before the batch elimination.

When the graph owns a copy of the frozen mentions, the driver will pass that copy to the compiler for initialization, as well as the list of strong candidates that were used to construct it. The compiler will use those to initialize its internal variables (in particular `v` is still `critical_margin/N`, but `critical_margin` would now be computed as `lowest_strong_tally - frozen_mentions[L]`), and to decide how its assorter method should interpret rows of ballots. The `base_vertex` for this compiler can be the last vertex before the batch elimination, although the hopefuls of that base vertex will not be the ones use to project ballot rows in the assorter.

This assorter should essentially mask all non-strong, non-L, non-very-strong candidates in the ballot (so only `L` and the strong candidates are not masked), then find the first non-masked positions `fpv_c` and `fpv_b` of both `row_c` and `row_b` and use it to compute the overstatement `w` of the ballot (if there are very strong candidates, we should also compute the prefix of very strong candidates for both). Let's call `L` the weak candidate and `s` the strong candidate with the lowest tally in the last vertex before batch-elimination (note that the fpv might be neither `L` or `s`, but might instead be any other strong candidate) The way we compute the overstatement `w` should be:

`w = (fpv_c == s) - (fpv_b == s) + (fpv_b == L) - (fpv_c == L)`.

Again, assuming there is a discrepancy somewhere in the projected rows (so at least one of `fpv_b == fpv_c` or `prefix_b==preix_c` is False), we should floor the discrepancy effect with `1` if the prefixes are not both `np.zeros`, to be conservative with respect to non-linear effects. So:

- both fpvs are same but prefixes are not -> w is 1
- fpvs differ but prefixes are both zeros -> w could be negative (determined exclusively by fpvs)
- fpvs differ and prefixes are not both zeros -> w is at least 1 (could still be 2)

The mapping from the overstatement `w` to the martingale capital updater should still be the same as before.

### Updated driver for seeded graphs.

Finally, we need to tell the driver how to properly test the extra assertions needed to justify a seeded graph. To do this, it is probably best to add a method `add_compiler()` to the driver, which will just add a passed compiler (with an already-identified `base_vertex` and `critical_margin` etc) to the internal pool of compilers the driver broadcasts updates to -- this will also allow the user to manually add compilers, if this is ever needed.

When it is passed an audit graph that was constructed via seeding, the driver should detect that and slightly modify the assertions it constructs to deal with them. Have the driver print a flag to point out that it has detected a seeded graph, and print out the candidate string names of all weak, strong, and very strong candidates that were used for this seeding.

First, we should make sure that very strong candidates are dealt with properly. All very strong candidates should be elected before the batch elimination of weak candidates, and they should be elected via a forced election; although they might be elected simultaneously or not in any order. If the graph follows this pattern of forced elections in the initial rounds before the batch elimination, our current driver should already know how to add the necessary candidate-above-quota compilers to the pool, so we shouldn't need to add any behavior there 

That being said : we should make sure that 
1. The compiler knows how to deal with simultaneous elections properly (in particular, simultaneous forced election; in such a case, it is only necessary to check that the topmost tally among forced winners is definitely above quota). The current logic should already be sufficient to justify the non-inclusion of escape edges corresponding to simultaneous elections; if such an edge is an escape edge, it would be because one of the candidates involved is definitely below quota, and this is already an assertion we test in the current framework.
2. It would be good to add some deduplication method to the driver, to remove compilers from the pool when they have the same base vertex and same critical margin. I *believe* such duplicate drivers should only be added when there is a forced election edge justifying the non-elimination of many other candidates simultaneously -- so maybe we should just have the driver treat vertices with leaving forced election edges differently, and only create one compiler for such edges (although we will still have to justify, for forced elections, why another winner wasn't simultaneously elected -- so we should still create candidate-below-quota compilers for such vertices when `simultaneous` is True). But regardless, it is probably good to create a method that deduplicates the compiler pool in general (have this method print out how many duplicate compilers it deleted, when it is called).

Second, the driver should identify the vertex in the graph that only has the strong candidates as hopefuls -- if none exist, throw an error -- and make sure there are no election edges leaving that vertex included in the graph; if there are, throw an error. Nothing additional should be needed from that point, since if the above is satisfied we should already have candidate-below-quota compilers created for each strong candidate in that vertex, under the current logic.

Third, we need to create a candidate-to-mention compiler for each weak candidate, comparing their frozen mentions in the last vertex before batch elimination to the lowest strong candidate tally. This is the main source of properly new compilers that our driver will have to create for seeded graphs.

# More Resilient Graph Seeding via "Black-Boxed Elections."

As a reminder, the audit graphs are supposed to embody the election paths we plausibly run across if the true ballots for the election were within a fixed neighborhood of the recorded CVRs.
The reason we needed batch eliminations was because the early layers of these graphs tend to experience combinatorial explosion.
This is because, in real-world data, the early rounds of the election tend to correspond to the elimination of "weak candidates" whose tallies are all orders of magnitude lower than those of the "strong candidates."
Because these candidates all have tallies that are so low, it is usually "plausible" to eliminate any given one of them at any time; as a result, we see these early layers grow combinatorially, creating what I call an "elimination cloud."
Batch eliminations are a way to deal with this when the timing of elections is very certain with respect to this elimination cloud: winners either definitely make quota before the cloud (so-called "very strong candidate"), or they definitely don't have a quota until after all of the weak candidates are eliminated.

But not all elections fall cleanly into that pattern.
Sometimes, one of the reported winners of the election is very close to quota -- either just below or just above -- during the elimination cloud stage of the election. 
In such a case it will not be possible to do a batch-elimination of weak candidates, because there is no small "strong set" of candidates where the winner will definitely have quota (or any such set will induce a very small set of weak candidates).
The tabulation for Portland D3 (below) is a good example of this (and so is Victoria 2022).
In this tabulation, Steve Novick starts out the election around 400 votes below quota, although the Least Auditable Margin (LAM) is a lot larger than 400 -- the top three candidates start out over 10,000 votes higher than the next strongest candidate, and stay so until the end of the election, when they are all seated.

|                                 |   Round 0 |   Round 1 |   Round 2 |   Round 3 |   Round 4 |   Round 5 |   Round 6 |   Round 7 |   Round 8 |   Round 9 |   Round 10 |   Round 11 |   Round 12 |   Round 13 |   Round 14 |   Round 15 |   Round 16 |   Round 17 |   Round 18 | Round 19   | Round 20   | Round 21   | Round 22   | Round 23   | Round 24   | Round 25   | Round 26   | Round 27   | Round 28   |
|:--------------------------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|:-----------|
| Steve Novick                    |     20470 |     20481 |     20502 |     20513 |     20523 |     20531 |     20552 |     20566 |     20590 |     20620 |      20668 |      20698 |      20735 |      20789 |      20844 |      20873 |      20936 |      21015 |      21141 | Elected    | Elected    | Elected    | Elected    | Elected    | Elected    | Elected    | Elected    | Elected    | Elected    |
| Angelita Morillo                |     16405 |     16417 |     16426 |     16431 |     16472 |     16489 |     16519 |     16554 |     16568 |     16589 |      16607 |      16616 |      16696 |      16799 |      16869 |      17098 |      17168 |      17262 |      17725 | 17731.27   | 17868.35   | 18393.79   | 18636.89   | 18854.93   | 19253.4    | 19387.48   | 20596.71   | 22326.91   | Elected    |
| Tiffany Koyama Lane             |     16338 |     16344 |     16349 |     16356 |     16376 |     16387 |     16405 |     16424 |     16429 |     16449 |      16484 |      16494 |      16545 |      16624 |      16690 |      16818 |      16879 |      17036 |      17406 | 17410.97   | 17627.07   | 17888.28   | 18118.37   | 18312.4    | 18682.1    | 18915.2    | 20114.09   | 21677.47   | Elected    |
| Kezia Wanner                    |      5317 |      5318 |      5320 |      5323 |      5331 |      5341 |      5344 |      5353 |      5369 |      5374 |       5397 |       5425 |       5448 |       5466 |       5485 |       5505 |       5558 |       5696 |       5714 | 5719.05    | 5911.13    | 6035.24    | 6334.38    | 6812.48    | 7027.87    | 8813.23    | 9640.52    | 10530.73   | 10685.86   |
| Rex Burkholder                  |      3953 |      3960 |      3966 |      3976 |      3978 |      3990 |      3997 |      4008 |      4016 |      4031 |       4048 |       4064 |       4093 |       4135 |       4160 |       4191 |       4279 |       4374 |       4443 | 4447.64    | 4638.77    | 4785.24    | 4953.33    | 5130.38    | 5632.31    | 5901.45    | 6823.94    | 0.0        | 0.0        |
| Jesse Cornett                   |      3861 |      3864 |      3867 |      3876 |      3881 |      3891 |      3906 |      3916 |      3919 |      3927 |       3940 |       3955 |       4014 |       4052 |       4077 |       4099 |       4146 |       4195 |       4326 | 4330.65    | 4452.7     | 4542.98    | 4838.13    | 4980.15    | 5222.08    | 5445.19    | 0.0        | 0.0        | 0.0        |
| Harrison Kass                   |      2787 |      2792 |      2798 |      2801 |      2801 |      2815 |      2819 |      2823 |      2831 |      2836 |       2853 |       2897 |       2912 |       2929 |       2941 |       2949 |       3007 |       3083 |       3105 | 3105.91    | 3183.95    | 3215.98    | 3472.05    | 3810.08    | 3967.18    | 0.0        | 0.0        | 0.0        | 0.0        |
| Philippe Knab                   |      1553 |      1555 |      1558 |      1560 |      1563 |      1567 |      1571 |      1578 |      1580 |      1587 |       1608 |       1616 |       1657 |       1660 |       1670 |       1689 |       1732 |       1765 |       1803 | 1817.74    | 1939.85    | 2306.13    | 2393.19    | 2489.21    | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Sandeep Bali                    |      1410 |      1411 |      1414 |      1418 |      1419 |      1426 |      1429 |      1432 |      1441 |      1449 |       1474 |       1512 |       1522 |       1537 |       1583 |       1622 |       1716 |       1871 |       1906 | 1906.41    | 2017.42    | 2114.45    | 2229.48    | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Daniel DeMelo                   |      1577 |      1579 |      1580 |      1581 |      1583 |      1590 |      1600 |      1609 |      1620 |      1624 |       1642 |       1661 |       1683 |       1743 |       1765 |       1787 |       1822 |       1883 |       1950 | 1950.92    | 2061.99    | 2112.01    | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Cristal Azul Otero              |      1408 |      1408 |      1408 |      1410 |      1422 |      1428 |      1436 |      1449 |      1453 |      1465 |       1477 |       1487 |       1505 |       1522 |       1555 |       1624 |       1657 |       1736 |       1809 | 1812.03    | 1909.1     | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Jonathan (Jon) Walker           |      1325 |      1327 |      1337 |      1342 |      1344 |      1349 |      1360 |      1364 |      1369 |      1374 |       1394 |       1425 |       1447 |       1462 |       1487 |       1497 |       1541 |       1609 |       1665 | 1665.85    | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Chris Flanary                   |      1243 |      1249 |      1253 |      1260 |      1266 |      1275 |      1289 |      1302 |      1317 |      1327 |       1337 |       1345 |       1396 |       1450 |       1471 |       1525 |       1559 |       1608 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Melodie Beirwagen               |      1136 |      1137 |      1139 |      1145 |      1156 |      1161 |      1165 |      1176 |      1183 |      1196 |       1209 |       1252 |       1266 |       1275 |       1304 |       1326 |       1396 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Matthew (Matt) Anderson         |       755 |       756 |       760 |       774 |       780 |       789 |       790 |       796 |       805 |       817 |        827 |        854 |        878 |        893 |        924 |        964 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Ahlam K Osman                   |       710 |       717 |       718 |       719 |       725 |       726 |       734 |       746 |       749 |       761 |        767 |        772 |        784 |        794 |        827 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Heart Free Pham                 |       551 |       553 |       557 |       566 |       571 |       578 |       583 |       589 |       598 |       606 |        612 |        632 |        649 |        661 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Luke Zak                        |       549 |       552 |       555 |       560 |       566 |       571 |       586 |       595 |       601 |       603 |        612 |        616 |        633 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Brian Conley                    |       508 |       509 |       518 |       527 |       531 |       541 |       549 |       554 |       566 |       575 |        581 |        598 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Terry Parker                    |       375 |       377 |       387 |       400 |       409 |       412 |       413 |       418 |       435 |       444 |        458 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Dan Gilk                        |       332 |       333 |       341 |       347 |       350 |       351 |       352 |       355 |       358 |       372 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Christopher Brummer             |       264 |       265 |       270 |       270 |       271 |       273 |       275 |       279 |       284 |         0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| John Sweeney                    |       240 |       243 |       247 |       255 |       257 |       261 |       264 |       271 |         0 |         0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
...
| Kenneth (Kent) R Landgraver III |       176 |       178 |       180 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| David O'Connor                  |       175 |       177 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Clifford Higgins                |       105 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |         0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 |          0 | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        | 0.0        |
| Quota                           |     21090 |     21090 |     21090 |     21090 |     21090 |     21090 |     21090 |     21090 |     21090 |     21090 |      21090 |      21090 |      21090 |      21090 |      21090 |      21090 |      21090 |      21090 |      21090 | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    | 21090.0    |

But a batch elimination would not be appropriate for this graph, because there would be no clear "strong set" of candidates such that none of them would have a quota if left alone. 
The problem is that Steve Novick is not "very strong" (he doesn't definitely get elected early in the election), but he is more than just "strong" (in that it would be difficult to verify that he doesn't have a quota at any given point in the tabulation).
We have to devise a new seeding process to deal with elections such as this one. 

The idea I'm going with is to use a "black-boxed election."
In essence, we will seed the graph with vertices where Steve Novick is already elected, although there will be no specific edge that the seeds will point to as the place of his seating.
Instead, the seeds will use worst-case bounds on Novick's transfer value to compute their two sets of tallies, and will allow outgoing edges that would be consistent with either set of tallies.
Later, when compilers will audit the escape edges for these vertices, they will the transfer value corresponding to the tightest critical margin as their ground truth.

To make this concrete: we partition the candidates into 5 sets, VS, S, w, L, and B. When we define these sets, "definitely" means "by more than a LAM worth of votes" -- e.g. "w *definitely* has quota" means "w's tally is more than LAM votes in excess of quota." The 5 sets of candidates are:
- Very Strong (VS) Candidates: these are candidates that definitely have quota in the first round of the election, or in a subsequent round after a sequence of very strong candidates getting seated. In Portland D3, there are no such candidates; In Victoria 22, there are 4 -- namely "HENDERSON Sarah", "WHITE Linda", "STEWART Jana", and "McKENZIE Bridget". Note also that the latter two candidates only definitely make quota after the first two are seated and have transferred their votes; this is fine, as long as no candidates need to be eliminated before a very strong candidate gets seated. After all these candidates are seated, our audit graph ends up in a single vertex with a deterministic set of transfer values for each of the seated candidates (and hence a deterministic weight for all ballots in the profile). We call that vertex the *base vertex* for the black-boxed election (n.b. if the order in which the very strong candidates are seated is uncertain, there might be multiple base verticess -- in which case we would conduct one black-boxed election for each base vertex -- but in practice this rarely happens).
- The candidate w: this is the candidate whose election we are black boxing. For now, there will only ever be one such candidate. Usually, w will be within a LAM of quota from the start of the election. In Portland D3, w is Steve Novick; in Victoria 22, w is Lydia Thorpe.
- Strong (S) candidates: this is any set of candidates such that, if those candidates were alone in the election with w, w would definitely have a quota, and everyone else definitely wouldn't. Usually we want to pick S so as to induce the largest weak set possible. In the above tabulation, I would wager that the remaining candidates in Round 21 or 22 of the election would make a good strong set.
- Weak (L) candidates: when the previous three sets of candidates are fixed, there is a deterministic way to compute the "maximum possible tally" of any candidate in their complement. When a candidate's maximum possible tally is definitely less than the lowest strong candidate tally in the base vertex, we call them weak.
- "In-betweener" (B) candidates: these are candidates that fit into none of the above categories -- in particular, they were not included in our chosen strong set, but their maximum possible tally is within LAM of the tally of one of the strong candidates (or w) in the base vertex.

To determine a maximum possible tally, given fixed sets VS, S, and {w}, we first need to determine the maximum possible surplus of w. To do this, for every candidate c in the complement of those three sets, do the following:
- start with the votes weighted as they were in the base vertex after the seating of the very strong candidates.
- remove all the candidates not in S or {w,c} from the election, as if those were the only remaining hopeful candidates left.
- count the weight of ballots whose first-place goes for c and whose second place goes for w in this projected state.
This number is the maximum surplus induced by c for candidate w.

Then, compute this value for all candidates c in $W\cup N$, and find the maximum such value, calling it the maximum surplus $s$ of w. Next, w's maximum transfer value $\overline{\tau}$ is $s/(q+s)$. 

Finally, the maximum possible tally of a candidate c is the tally they would have if it was just them and the strong candidates, and w's transfer value was $\overline{\tau}$ (this is the same thing as what we were previously calling "frozen mentions" -- the frozen candidates are the strong candidates -- and we should use frozen mentions rather than explicitly re-projecting the whole ballot matrix to determine these maximum possible tallies).

# Noise-Filtered Linearization for Better Compilers.

Currently, we only have one edge-local compiler to audit the margins coming from escape edges -- cf. the section on "Edge-local COBRA compilers."
These first prototypes were very conservative with respect to non-linear noise: whenever a sampled ballot-cvr pair of rows had a non-linear discrepancy, we treated their impact on the margin as a systematically lowering integral impact -- even when the sample should actually have been considered to help the margin.
We're going to overhaul these compilers to use better rounding procedures, which should give us better sampling numbers in time.

The idea is to have a separate set of "noise-filter" assertions verify, for each edge, that the total number of discrepant ballot-cvr row pairs corresponding to that edge are less than a fixed radius `r` (initially, it will be easiest to set `r=2*LAM`), but we should probably be smarter about this eventually.
Then, whenever we sample a discrepancy, we run an edge-local optimization to find the worst-case impact of that discrepancy conditioned on living within the L^1 ball (or 'diamond') of radius `r` centered on the base point for that edge (I'll explain the coordinate system we use in more detail later).
Next, we map this discrepancy into a compiler-level updater (similar to how the previous compilers mapped `w` to a discrete set `{2a, 3a/2, a, a/2, 0}`, except now we will have more than 5 possible outputs now), and we use this assorted output to update an internal martingale capital as before.

I already added an extensive filtered optimizer module in `stv_partial_optimizer.py` -- so the main work we'll have to do is to wire it so that it can interface with the vertices of our graph (we'll probably want vertex-local audit drivers to interpret the impact of a ballot-cvr pair in terms of the coordinates used by these optimizers), and then build a compiler that can use its outputs, similar to the current COBRA compilers. 
Next, we'll have to update our global drivers to generate noise-filter compilers for each edge (these can look virtually identical to our current ultra-conservative compilers). 
I also want to eventually add better profiling for these compilers, and also add an ALPHA version of the noise-filted compilers.

Here's the list of tasks we should try to achieve in this section:
- [x] Vertex-local interpreters: We need make a new class that is able to project standard ballot-matrix formatted rows into the coordinate system used by our noise-filtered optimizers. *N.b.: these are only needed for vertices with non-zero degree: if the vertex has no seated winners, it does not need an interpreter.*
    - [x] We will have to modify the graph constructor to also store the `pos_vec` as information contained in each vertex. For all but the largest elections, these are int8 vectors with length equal to the ballot matrix; if this is still too large of a memory overhead (it might be), it will be sufficient to store a boolean mask with each winner edge that shows which rows transferred through the winner. 
    - [x] Then, when we sample a discrepancy relevant to a given vertex, for each edge affected by that discrepancy, we'll have to project those helper vecs into a different edge-specific set of coordinates used by the noise-filtered optimizers. We should similarly project the sampled ballot-cvr rows into those coordinates, since the noise filter will need to minimize the partial of the margin function in the direction of that discrepancy.
- [x] Modernized COBRA compilers: the noise filtered optimizer will output a constant bound when the vertex-local interpreter hands it a discrepancy converted in its coordinate system. This bound will not fit neatly within the previous {-2,-1,0,1,2} outputs we had used to update COBRA, but the mapping from it to an updater value will work the same way. We should make a v2 version of COBRA that can take these richer outputs and assort them to update a Martingale capital. We will similarly need to make v2 versions for the candidate-below-quota, candidate-above-quota, and candidate-to-mention compilers we currently have.
- [x] Adapted noise-filtered version of COBRA: these less-conservative compilers are only justified if we separately have a compiler verify that the noise level for a given margin is below a fixed level. This will be a relatively straightforward and linear compiler, but it needs to be done.
- [x] Modernized global driver: we will also need to make a v2 global driver -- it needs to know how to create vertex-local interpreters, and then hand sampled rows to them instead of the edge-level compilers directly. It could also be good to have this driver dynamically decide which compiler to use for each given edge, as a function of the margin and how easy it is to audit. Large margins can use ultra-conservative rounding, and we can spare the computational cost of running an optimizer for them. A good starting rule of thumb is to use something ultra-conservative when the critical margin is more than twice of LAM, the default radius for our noise filter diamonds.
- [x] Better profiling: I would like to optionally select some compilers to track stats during a run of the global driver, to allow for plotting of their capital over time.
- [ ] ALPHA compilers: this is an alternative but similar test process to COBRA, that allows for sampling without replacement.

Throughout this section, I'm going to be referencing two elections to benchmark the performance of these new noise-linearized compilers. 
The first is a synthetic 3-seat test-election I built to be auditable while still capturing a lot of the edge cases we're trying to deal with:
```
def my_map(bleh, lst):
    return (frozenset([item]) for item in lst)

test_profile = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(my_map(frozenset, ["W1"])), weight=25000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W1", "S2", "W3"])), weight=15000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W1", "S2", "S3"])), weight=15000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W1", "L1"])), weight=5000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W1", "S3"])), weight=30000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W1", "W3"])), weight=60000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W2", "W3"])), weight=40000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W2", "S3"])), weight=40000),
            RankBallot(ranking=tuple(my_map(frozenset, ["W3"])), weight=50000),
            RankBallot(ranking=tuple(my_map(frozenset, ["S3"])), weight=70000),
            RankBallot(ranking=tuple(my_map(frozenset, ["L1", "W3"])), weight=5000),
            RankBallot(ranking=tuple(my_map(frozenset, ["L2", "W3"])), weight=4999),
            RankBallot(ranking=tuple(my_map(frozenset, ["S1", "W2"])), weight=20000),
            RankBallot(ranking=tuple(my_map(frozenset, ["S2", "W3"])), weight=10000),
            RankBallot(ranking=tuple(my_map(frozenset, ["S2", "W2"])), weight=10000),
        ]
    ),
    max_ranking_length=3,
)
```
This election is simple enough that I should be able to compute vertex-local coordinate projections to double-check your work. 
It can have a LAM of up to 9000 while still having a very manageable audit graph.

The second benchmark election will be Portland D1, which was also a 3-seat election:
```
from votekit.cvr_loaders import load_scottish, load_numpy

portland_D1 = load_scottish("/home/nardo/research/data/election_data/votekit/portland/portland_d1_2024.csv")[0]
```
This election is small enough to compute a complete audit graph for, so we will be able to test it before making generalized candidate-to-mentions compilers. 
This election can have a LAM of up to 670; the ultra-conservative sampling number to beat is around 600 votes.

## Vertex-Local Interpreters.

The noise filtered optimizers use a separate edge-local coordinate system that projects each ballot type into exactl one of $3\cdot 2^{d}$ unit basis vectors, where $d$ is the *degree* (i.e. number of already-seated winners) of the base vertex.
These dimensions consist of a "winner prefix" $S$ and a "candidate suffix" $x$. 
$S$ is a set of winners, of which there are $2**d$ possible. 
$x$ is an edge-local candidate choice; there are only $3$ pertinent possibilities for a candidate-to-candidate margin between candidates $c$ and $l$ -- namely $x=c$, $x=l$, or $x=o$, indicating any *o*ther candidate.
When we generalize candidate-to-quota margins, one of $c$ or $l$ will be the candidate $w$ (depending on whether w is below or above quota), and the other will be constant zero.

To identify a coordinate in this system, we just need to specify $S$ and $x$. 
In the degree-d setting, we will use the notational shorthand where S is specified as a concatenated winner indices; so `S=023` indicates that a row transferred through winners 0, 2, and 3 (i.e. this ballot sat with those winners when they got seated), but not through winner 1.
Coordinates are therefore specified by a pair such as `(023, c)`.
The empty prefix is identified by placing nothing before the comma in this tuple.
To emphasize that we are using this canonical coordinate system (as indeed there exist other re-parameterizations of the same problem), we subscript this tuple onto a $t$; so $t_{03, o}$ indicates those ballots which transferred through winner index 0 and 3 before getting tallied for some candidate that is not c or l in the current margin.

The noise-filtered optimizers arrange these coordinates in a `2**d` by `3` array; each row `j` corresponds to a subset $S$ via the binary decomposition of `j`, and the columns represent $x = c$, then $x= l$, and finally $x = o$. 
So for example, in the degree $d=4$ setting, coordinate `(3,0)` corresponds to the symbolic variable $t_{01, c}$; coordinate `(10,1)` corresponds to $t_{13,l}$; corrdinate `(7,2)` corresponds to $t_{012,o}$; and so forth.

A the level of a single vertex in the audit graph, the `ballot_matrix` and `wt_vec` owned globally by the auditgraph induce a unique winner prefix $S$ for each row of the ballot matrix.
To construct `S`, we just need to look back at all the edges in the `seated_at` property of the vertex, and use the `fpv_vec` for the base vertex of that election edge to determine which rows of the ballot matrix transferred through the newly seated winner.
It will be a good idea to cache these `fpv_vec`s in election edges (initially, let's do this even when `memory_lite = False`), because they should be relatively cheap on memory.
If these fpv vecs are not cached, we will have to recompute them in place; in such a case, we can at least streamline the process by re-using the bool_ballot_matrix from one `seated_at` ancestor to compute the `bool_ballot_matrix` of the next -- just iteratively batch-eliminate candidates, then seat a winner, then batch-eliminate all the other candidates until we get to the next winner, etc.
It should also not be necessary to re-compute the `wt_vec` for this process.

To convert the ballot matrix information into the `t_*` basis of coordinates, we will use a new `VertexInterpreter` class. The job of this class will be to:
1. Using the `fpv_vec`s of the vertex's ancestors, reconstruct an array indicating the index `j` that each row of the ballot matrix should occupy in the row-wise coordinate system of `t_*`. This should just use vectorized binary numpy logic: just convert each seating ancestor's fpv vec into one that has a single bit in every position corresponding to a newly seated winner, and take a cumulative OR. Reconstruct this winner prefix array every time the Interpreter needs to update its compilers with a discrepancy, since they will be expensive to store.
2. Iterating through the edges the Interpreter needs to update:
    2.1. Convert the discrepant ballot-cvr pair into `t_*` coordinate (i.e. do `theta=t(bal_row) - t(cvr_row)`). Check if this directional vector `theta` has already been encountered and is cached in the edge-local compiler. If so, don't bother running the optimization again, and just use the cached update value; otherwise, 
    2.2. Use the cached winner prefix + a vertex-local (possibly reconstructed) `fpv_vec` to create a base point in `t_*` coordinates representing the CVR profile in that space.
    2.3. Pass those `t_*` coordinate vectors to the optimizer, then return their outputs to the edge-local compilers.
     
Let's get the following tasks done, and then test them before moving on:
- [x] Create the vertex-local Interpreters
- [x] Update the graph constructors to store `fpv_vec`s in election edges by default (maybe add a hidden parameter `store_fpv_vec = True` to `init`).
- [x] For two escape edges of the test profile, test the Interpreter in two ways: have it (1) compute the full base point in `t_*` coordinates, (2) use the optimization module to verify that the numerical value of the margin it gets is the same as the recorded value in the graph, and (3) find the full impact of every possible ballot-cvr discrepancy `theta` localized at this basepoint, with a given radius of either `r=4000` or `r=18000`.

The first escape edge we'll test is degree-1: it should be interesting to look at the critical margin `M_{cl}` between candidates `c = S2` and `l=L1` in the only vertex of (zero-indexed) layer-1 of the graph that still contains `L1`. 
To construct this graph:
```
constructor = WIGMGraphConstructor(test_profile, m=3, LAM = 9000, memory_lite = True,simultaneous=True)
constructor.build()
```
According to my handcount, the `t_*` coordinate basepoint for this margin should be:
```
[[20_000, 5_000, 224_999],
[30_000, 5_000, 115_000]]
```

The second escape edge we'll test is degree-2: for this one, we can compare S3 and W3 in the vertex in layer 6 resulting from the sequence where W2 is elected first from layer 4, then S2 is eliminated (in the graph when I construct it, this is the 0th vertex in this layer 6).
For this margin, the coordinates I get are:
```
[[69_999, 70_000, 10_000],
[75_000, 45_000, 30_000],
[40_000, 40_000, 20_000],
[0, 0, 0]]
```

## Modernized COBRA Compilers.

It's worth saying that I'm now envisioning a slightly different balance of responsibilities between the interpreter and the compilers -- the interpreter should just compute and cache the winner prefix and candidate suffix, but then hand those directly to the compilers, who will themselve be in charge of converting those into a `t_*` coordinate theta-vector, determining if that vector has already been encountered before, and handing it off to a noise-filtered linearizer if not. 

There's a couple things we'll need to do to make the `CobraV2` compilers.

- [x] Accept and assort non-discrete values for `w` from the noise-filtered linearizers, and use them to update the internal Martingale capital.
- [x] Make sure these compilers are properly optimizing simultaneous-winner situations by killing the appropriate rows of the optimizer inputs.
- [x] Adapt a version of these noise-filetered compilers for candidate-above-quota and candidate-below-quota critical margins.
- [x] Make a compiler that works for noise-filter assertions.
- [x] (Later) Adapt the candidate-to-mentions compilers we need for seeded graph construction.
- [x] Better Profiling.

### Assorting non-discrete w values.

*Big disclaimer ahead of this section*: the interpretation of `w` has flipped compared to the previous section on Cobra Compilers. 
In the current setting, $w<0$ corresponds to an effect that tightens the margin, so that $w=-2$ is the worst case scenario, whereas previously $w=2$ was the worst case scenario.
This is fine, as long as we stay consistent with how a given ballot-cvr discrepancy updates the martingale updater rule.

Previously, our COBRA compilers were only designed to call their updater method on one of 5 ballot-cvr discrepancy patterns, corresponding to values of w in {-2, -1, 0, 1, 2}.

These w patterns can now take on any value in [-2, 2] -- the possible outputs of our noise-filtered optimizers -- but the update rule will still be similar. 
The compiler initially computes and stores a value of `v = M/N` where M is the recorded margin of the critical edge and N is the total number of ballots in the profile.
We no longer need to precompute an store a value of `a`; and we can use the same rules as before to compute `lambda` -- using a constant fallback near `lambda = 2` if those rules give us a negative output or one that is near zero.
The compiler also now initializes with an int parameter `r` which control the noise allowance we give to its optimizer.
By default, we'll take `r` to be twice of the `LAM` we used for the underlying audit graph (or specifically the vertex the edge is leaving from, if we ever implement non-uniform LAM graph constructors).

To map the output `w` of the noise-filtered optimizer to the martingale update coefficient `X`, we do:
$
    X = (1 +  w/2)/(2-v).
$
This immediately gets us our martingale update coefficient `T`: $T = 1+ \lambda (X-1/2)$ (note: it would be fine to compute `T` directly from `w`, we don't have to waste time computing and storing `X` in between).

As mentioned before, I also think the appropriate way to break down tasks between the Interpreter and Compiler is to have the interpreter compute exactly the information that is shared by all vertex-local compilers -- namely the local `fpv_vec` and the winner prefix index for each row of the ballot matrix.
Then, it is up to the compiler to turn these helper vectors into theta coordinates, verify its cache to see if it has already run an optimizer on that theta, and run the optimization (materializing the necessary base point in place) otherwise.

### Dealing with simultaneous winners.

None of our current test-cases have simultaneous winners, but our real-world use-cases frequently do. 
When this happens, the way we call the noise-filtered linearizer is slightly different.

The big idea is that when i and j are simultaneous, then any `t_{S,x}` coordinate where the winner prefix $S$ contains both `i` and `j` is zero by definition, and the optimizer should not even bother doing gradient descent in that direction.
The optimization module should already have something wired in that allows a caller to specify `dead_row` indices, which are precisely row indices that should be zero'd out this way.
Our Interpreter should tell the compilers when two winners in the base vertex and simultaenous.
Then, the V2 compilers should know how to recognize when members of their winner set are simultaenous, and zero out the appropriate rows when this is the case.

This essentially amounts to specifying all row indices `k` such that `k ^ ((1 << i)|(1<<j)) = ((1 << i)|(1<<j))` as dead rows (you might be able to come up with some cleverer binary logic to detect and kill these rows).

### Candidate-to-Quota margins.

Our previous philosophy was to treat candidate-to-quota margins identically to candidate-to-candidate margins, only treating quota as a 'candidate' with a fixed tally who can not receive or lose votes.
I'm now slightly less sure about the mathematical validity of this approach, but we'll deal with it later if it's a problem.

For now, we'll do the same thing: when the critical margin is candidate-below-quota, we'll treat it as a margin $M_{cl}$ where $c$ is quota and $l$ is the non-winner $w$.
When the critical margin is candidate-above-quota, we'll treat it as a margin $M_{cl}$ where $c$ is $w$ and $l$ is quota.
The golden rule is that candidates should be chosen such that critical margins are always positive in the CVR, and negative discrepancies are evidence against our margin (again, n.b. that this is this is flipped from how we were previously updating COBRA).

In this setting, since quota can't 'receive' votes, we should zero out the column corresponding to the role $x$ it is playing in theta space.
The suffix of a vote is either $w$, or it is anyone else, but we should only optimize worst-case non-linear impacts insofar as they affect $w$, because they can not affect quota.

As before, the optimizers have a parameter `dead_column` that allows the compiler to specify when the gradient descent should ignore a complete column in theta space.

One the optimizers spit out an overstatement bound `w`, we will map it to a martingale updater factor in exactly the same way as for candidate-to-candidate margins (this is the part I am mathematically unsure about).

### Noise-filter compilers.

Since our optimizers work under the assumption of a noise filter, we need a separate compiler to verify this noise filter for each escape edge.

This is rather simple: it looks exactly like one of our old ultra-conservative COBRA compilers.

The diluted margin `v=r/2N` is the noise allowance divided by the total number of votes. 
*Please note*: this allowance is very deliberately multiplied by a factor of 1/2. 
This is because the optimizers use `r` as a bound on the total variation from their base point in *theta space*; but a single ballot-cvr discrepancy usually has a total length of 2 in theta space, so actually our noise filter requires less than r/2 ballots show discrepancies.

To update the compiler, whenever we sample a discrepant ballot-cvr pair (in the sense that there is any discrepancy at all as far as the winner prefix or candidate suffix for the local critical margin coordinates), treat it as a discrepancy `w=1` in the old setting, so update the martingale capital by using $X=a/2$.
Otherwise, update the martingale capital with $X=a$ when there was no edge-local noise in the sample (either because there was no noise in the sample at all, or the noise was not relevant to the critical margin).

### Better Profiling.

The compilers should have an optional init parameter `profile = False`. If set to true, the compiler should remember the martingale capital it had after every single update since since initialization.

The `discrepancies` property of the v2 driver should also be updated to track flattened theta vectors + the numerical optimizer output corresponding to every discrepancy.

And we should write a general-purpose plotting function to plot the history of a compiler whose profiling was set to true. 
This should look like a scatter plot of the martingale capital over time, with a bar plot on the same plot but a different y-axis scale showing the numerical output of the optimizer when noise was sampled.
This should show us slighlty negative bars when the martingale capital drops, and slightly positive bars when it increases.
Ideally, also label each bar of the bar plot with the flattened coordinates of the theta vector it came from.

### Candidate-to-Mentions Compilers.

There are two types of seeded audit graph constructions: batch eliminations and black-boxed seatings.
We'll deal only with the former for now.

The old audit driver already had a framework to initialize the constructors needed for a batch elimination.
They identified a critical margin for each weak candidate, comparing them to the weakest strong candidate in the base vertex.
They also had a separate set of compilers to verify that the strong candidates would not have quota if alone in a vertex.

These are still exactly the assertions we'll need in the V2 framework, except we now use noise-filtered compilers for each of these critical margins, plus an additional ultra-conservative compiler to verify this noise level. 
So each candidate-to-mention assertion induces two compilers -- one ultra-conservative `CobraNoiseFilterCompiler`, and one new version of our V2 noise-filtered compilers.
These can be owned by an interpreter that lives on the base vertex of the batch elimination.

For a given ballot, its suffix will be computed in exactly the same way as before, except we now also need to worry about the winner prefix of a ballot.
This means that we use the wt_vec of the base vertex only to compute the numerical value of the margin for a compiler, but then we still need to turn the profile into a theta space base point when we update a compiler.

So a candidate-to-mentions assertion between strong candidate `s` and weak candidate `l` has the same theta-space shape as a standard candidate-to-candidate margin `M_{cl}`, where a ballot's suffix is `c` if its fpv in the base vertex is `s`, and the suffix is `l` if the ballot contains no strong candidates before candidate `l`. 
The prefix of a ballot of these candidate-to-mentions is just its winner prefix among the seated very strong candidates in the base vertex.
A ballot-cvr pair of rows still induces a well-defined theta vector with flattened coordinates `(u,v)` indicating the positions of its unique +1 and -1 entries, so the optimizers should just go through as is.

## Modernized Global Driver.

Next up, we should wire these new noise-filtered compilers into a `V2` version of our Global Audit Driver.

The global driver should now take a separate parameter `r` corresponding to the noise-filter radius it'll give to all of the V2 Cobra Compilers.

The overall logic will be very similar to the `V1` compiler -- for each vertex, find all the escape edges leaving from that vertex, for each such edge determine the critical margin, and set up one of the `V2` Cobra compilers for this margin.
Additionally now set up a noise filter compiler for that same margin, and also a vertex interpreter for the base vertex. 
Only set up one such interpreter per vertex; this interpreter should internally keep track of the compilers it will be in charge of updating.

Let's not do anything subtle yet about having the driver dynamically use noise-filters for some edge and ultra-conservative rounding for others; let's just use noise-filters for everything, for now.

The global driver should also have a new optional argument `profile = False`. 
If True, it will run the audit normally once, and then identify the three compilers that were the slowest to certify.
It will then re-run those three compilers with profiling set to True, with the same seeded RNG, and then plot their profiled history side-by-side.

Everything else should mimick the old `V1` global driver as much as possible.

## More aggressive noising for the implicit sampler.

We should add an optional parameter `aggressive = True` by default to the implicit sampler.
When true it uses an exponential distribution to target rows that are ealier in a row chosen for implicit sampling.
This means that if a position gets deleted, we tend to target earlier positions; if a ranking gets inserted, it tends to get inserted early; if two rankings are swapped, the positions of those rankings tend to be early; if a ranking gets replaced, that ranking tends to be early in the row. 
For ballot-completion noise, there is nothing to do with such an index, so there is no needed change in the aggressive setting

We should pre-generate a list of target indices using the same exponential distribution (try to make it so that 90% of the support of the indices is in ${0, ..., ncands}$), seeded deterministically by the sampler's global seed, and then use those indices iteratively as we select more rows for sampling.

## Support for Black-boxed elections.

In principle, generalizing to black-boxed elections is not very difficult.
As far as the compilers are concerned, they still just need a conservative estimate of the critical margin corresponding to an edge, and then the rest of the noise-linearized machinery goes through as before.
The big difference is that uncertain votes for a candidate `l` get receive the winner prefix they had in the base vertex, and a suffix for candidate `l`.

As a reminder: in a black boxed election, we have a base vertex that is the one resulting from the seating of all very strong candidates, and then an uncertain candidate (call them `w`) whose timing is insecure.
The graph will partition the rest of the candidates into strong candidates, weak candidates, and in-between candidates.
When we are doing candidate-to-mentions assertions comparing the mentions of the weak candidates to the fpv votes of the strong candidates in the base vertex, a vote is considered *uncertain* for a candidate `l` if all of the following occur:
- the uncertain candidate `w` is listed in the ballot, but not as its first place vote (aka suffix) in the base vertex.
- the candidate `l` occurs *after* `w` in the ranking on the ballot, but *before* any other strong candidates.

In this case, we are unsure if the ballot should be considered to have transferred through `w` before getting to `l`, so we will treat it conservatively, and treat it as having the worst weight possible for the critical margin.

## ALPHA Compilers.
