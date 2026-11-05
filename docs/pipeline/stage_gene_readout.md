# Stage `gene_readout`: gene-level readout of concept membership

Directory `07_gene_readout/`. Inputs: the activation store, `sae` and `concepts`. Core environment, CPU (scikit-learn).

## What it computes

Whether concept membership of *unseen* genes can be decoded from a representation at all. Each measured gene is represented by the mean of its token activations over the section: its mean hidden state (`hidden`), its mean activation of the selected SAE (`sae`) and of the untrained SAE (`untrained`). Genes are split into `n_splits` whole-gene folds (10 % test genes each, the rest train+validation; the split rule of `concept_eval` with seeds `seed`, `seed + 1`, ...). For every concept with 5–100 panel genes (and at least one test gene and two training genes in the fold) an L2-regularized logistic regression (`C` = 1, balanced class weights) is fitted on the train+validation genes and scored by the AUROC on the test genes. Inputs are standardized on the training genes; because standardization inflates the rare activations of sparse features, a center-only variant is reported as a sensitivity analysis. The best single dimension (chosen by training AUROC; both signs for non-SAE representations) is scored the same way.

Controls, scored on the same genes, splits and concepts:

- `random`: Gaussian vectors of the hidden dimension (one per hidden size among the layers);
- `coexpression`: each gene's Pearson correlation with every panel gene over the cells (log-normalized expression; at most `max_cells` cells);
- `coexpression_pc50`: the first `n_pcs` principal components of that matrix;
- `celltype`: mean log-normalized expression per cell type (types with at least `min_type_cells` cells), z-scored per gene across types; needs `dataset.labels`;
- `trivial`: mean expression and detection rate.

## Parameters (`stages.gene_readout`)

| key | default | meaning |
|---|---|---|
| `n_splits` | 5 | whole-gene folds |
| `C` | 1.0 | inverse regularization strength |
| `scaling` | `[standardize, center]` | input scaling variants reported |
| `controls` | `[random, coexpression, coexpression_pc50, celltype, trivial]` | |
| `concept_size` | `[5, 100]` | concepts scored |
| `split` | `{valid: 0.2, test: 0.1}` | the whole-gene split fractions |
| `max_cells` | 100000 | cells used for the expression controls (random subset) |
| `min_type_cells` | 50 | |
| `n_pcs` | 50 | |
| `seed` | `run.seed` | splits, cell subset and random vectors |
| `max_iter` | 3000 | lbfgs iterations |
| `dedup_cells` | true | first copy of duplicated cells only |
| `max_shards`, `device`, `token_vocab`, `checkpoints`, `store_root` | | as in `concept_eval` |

## Outputs

```
07_gene_readout/
  gene_readout.json             per representation: mean +- SD over splits of the probe AUROC (per scaling)
                                and of the single-dimension AUROC, share of concepts >= 0.7
  gene_readout.csv              per representation and split
  per_concept.csv.gz            split, representation, family, concept, n_genes, single_dim, single_sign,
                                single_train_auroc, single_test_auroc, probe_test_auroc_<scaling>
  splits.json                   the gene universe and the test genes of every split
  expression_reps.npz           the expression controls over the panel
  <layer>/gene_means.npz        keys, count, hidden, sae, untrained (genes x dimensions)
```

Representations are named `<kind>|<layer>` (`hidden|layer_1.norm2`) and controls `<control>|-`.
