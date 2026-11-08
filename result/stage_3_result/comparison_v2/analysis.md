# Stage 3 experimental results

| Dataset | Configuration | Selected | Validation accuracy | Test accuracy | Macro F1 |
|---|---|---|---:|---:|---:|
| MNIST | baseline | False | 98.80% | 99.02% | 99.01% |
| MNIST | deeper | True | 99.17% | 99.40% | 99.40% |
| ORL | baseline | False | 97.50% | 100.00% | 100.00% |
| ORL | deeper | True | 97.50% | 97.50% | 96.67% |
| CIFAR | baseline | False | 85.72% | 85.70% | 85.68% |
| CIFAR | deeper | True | 90.56% | 89.71% | 89.71% |

All selection.json files are written before the corresponding dataset test scores.
Models are initialized independently. The split, seed and epoch budget are fixed within each dataset.
ORL labels are shifted from 1-40 to 0-39; use the stored label_offset to interpret predictions.
ORL has 40 test images: one error changes accuracy by 2.5 percentage points.
One seed only: these results do not establish statistical superiority.
Training metrics include dropout and, for CIFAR, random crops/flips. Validation disables both.
