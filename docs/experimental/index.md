# Ragas Experimental

## Hello World 👋

1. Setup a sample experiment. 

```
ragas hello-world
```

2. Run your first experiment with Ragas CLI.

```
ragas evals hello_world/evals.py --dataset test_data --metrics accuracy
```

```
Running evaluation: hello_world/evals.py
Dataset: test_data
Getting dataset: test_data
✓ Loaded dataset with 10 rows
Running experiment: 100%|████████████████████████████████████████████████| 20/20 [00:00<00:00, 4872.00it/s]
✓ Completed experiments successfully
╭────────────────────────── Ragas Evaluation Results ──────────────────────────╮
│ Experiment: lucid_codd                                                       │
│ Dataset: test_data (10 rows)                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
  Numerical Metrics   
┏━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric   ┃ Current ┃
┡━━━━━━━━━━╇━━━━━━━━━┩
│ accuracy │   0.100 │
└──────────┴─────────┘
✓ Experiment results displayed
✓ Evaluation completed successfully
```

3. Inspect the results 

```
tree hello_world/experiments
```

```
hello_world/experiments
└── lucid_codd.csv

0 directories, 1 files
```

4. View the results in a spreadsheet application.

```
open hello_world/experiments/lucid_codd.csv
```

