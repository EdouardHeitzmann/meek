Contains replication code for my graph-based Meek auditing paper.

## Graph-Based Meek Auditing

1. I recommend you get started with the `portland_d1_demo.ipynb` notebook. It walks through all the key components of the graph-based auditing process, and also shows how to construct the graph for Portland D1.
2. The results for the Scottish election audits can be replicated in `audit_all_scottish.ipynb`.
3. The other Portland audits can be replicated by swapping out the loaded profile in the demo notebook.
4. The Australian audits can be replicated by the `territory_audit_driver.py` and `state_audit_driver.py` scripts; you will have to manually re-populate the votekit csv for these elections, and specify a path to them in the script.
5. The full Portland D4 graph can be built via `build_d4_graph.py`; fair warning that this script will take several days to run, and costs a lot of memory. The `stream_stats.py` script can be used to print out minimal information about this graph after it is constructed.

## Bayesian Ballot-Comparison Auditing for STV

A naive first-pass implementation of Rivest's Bayesian ballot-comparison audit framework adapted for Single Transferable Vote (STV) elections.

- **Documentation**: See [BAYESIAN_AUDIT_README.md](BAYESIAN_AUDIT_README.md) for detailed API documentation
- **Demo Notebook**: `bayesian_audit_demo.ipynb` contains examples, stress tests, and usage patterns
- **Module**: `src/bayesian_comparison.py` contains the core implementation

Key features:
- Direct posterior sampling via gamma/Dirichlet conjugacy (no MCMC)
- Partial auditing for proper uncertainty quantification
- Unseen type protection against adversarial attacks
- Integration with VoteKit and existing noise infrastructure
- Sequential stopping rules based on upset probability