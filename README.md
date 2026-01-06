# WD40DomainPred
A prediction model repository of WD40 protein domains. The repository is an official implementation of ....

## Introduction

**input**: PDB structure files and domain label files

**output**: domain intervals(if domain doesn't exist, the domain is empty)

In training stage, we prepared 1076 proteins, which distributed as follows.
| train_stage     | dataset_train  | dataset_val  | dataset_test  |
|-----------------|----------------|--------------|---------------|
| `positive`      | 260            | 33           | 33            |
| `negative`      | 59             | 7            | 8             |
| `negative2`     | 540            | 68           | 68            |
| **totally**     | **859**        | **108**      | **109**       |

We evaluated the model on both an in-house and an external test dataset consisting of 860 manually annotated sequences extracted from WD40. Considering domain interval error as 5bp, two datasets results are as follows.

| in-house test   | Precision | Recall | F1-score | Support |
|-----------------|:---------:|:------:|:--------:|:-------:|
| 0(negative)     |   0.9487  | 0.9737 |  0.9610  |    76   |
| 1(positive)     |   0.9355  | 0.8788 |  0.9062  |    33   |
|                 |           |        |          |         |
| **Macro avg**   |   0.9421  | 0.9262 |  0.9336  |   109   |
| **Weighted avg**|   0.9445  | 0.9450 |  0.9445  |   109   |
| **Accuracy**    |           |        |  0.9450  |   109   |

| external test   | Precision | Recall | F1-score | Support |
|-----------------|:---------:|:------:|:--------:|:-------:|
| 0(negative)     | 0.9152    | 0.9877 | 0.9500   |   568   |
| 1(positive)     | 0.9717    | 0.8219 | 0.8905   |   292   |
| **Macro avg**   | 0.9434    | 0.9048 | 0.9203   |   860   |
| **Weighted avg**| 0.9344    | 0.9314 | 0.9298   |   860   |
| **Accuracy**    |           |        | 0.9314   |   860   |

Then we computed classification statistics in residue level of all sequences on both in-house and external test dataset. We also compared the results within and without applying the domain-assignment rules.

<table border="1" cellspacing="0" cellpadding="6" style="text-align:center; vertical-align:middle;">
  <thead>
    <tr>
      <!-- 第一级表头：Class 列上下居中，左右合并 -->
      <th rowspan="2" style="vertical-align:middle;">In-house test</th>
      <th colspan="3">Original Prediction</th>
      <th colspan="3">After Applying Rules</th>
    </tr>
    <tr>
      <!-- 第二级表头 -->
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0 (negative)</td>
      <td>0.9854</td>
      <td>0.9813</td>
      <td>0.9834</td>
      <td>0.9870</td>
      <td>0.9807</td>
      <td>0.9839</td>
    </tr>
    <tr>
      <td>1 (positive)</td>
      <td>0.9444</td>
      <td>0.9561</td>
      <td>0.9502</td>
      <td>0.9430</td>
      <td>0.9611</td>
      <td>0.9520</td>
    </tr>
    <tr>
      <td><strong>Macro Avg</strong></td>
      <td><strong>0.9649</strong></td>
      <td><strong>0.9687</strong></td>
      <td><strong>0.9668</strong></td>
      <td><strong>0.9650</strong></td>
      <td><strong>0.9709</strong></td>
      <td><strong>0.9679</strong></td>
    </tr>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td>—</td>
      <td>—</td>
      <td><strong>0.9751</strong></td>
      <td>—</td>
      <td>—</td>
      <td><strong>0.9759</strong></td>
    </tr>
  </tbody>
</table>

<table border="1" cellspacing="0" cellpadding="6" style="text-align:center; vertical-align:middle;">
  <thead>
    <tr>
      <th rowspan="2" style="vertical-align:middle;">External test</th>
      <th colspan="3">Original Prediction</th>
      <th colspan="3">After Applying Rules</th>
    </tr>
    <tr>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0 (negative)</td>
      <td>0.9777</td>
      <td>0.9918</td>
      <td>0.9847</td>
      <td>0.9780</td>
      <td>0.9935</td>
      <td>0.9857</td>
    </tr>
    <tr>
      <td>1 (positive)</td>
      <td>0.9708</td>
      <td>0.9238</td>
      <td>0.9467</td>
      <td>0.9769</td>
      <td>0.9247</td>
      <td>0.9501</td>
    </tr>
    <tr>
      <td><strong>Macro avg</strong></td>
      <td><strong>0.9742</strong></td>
      <td><strong>0.9578</strong></td>
      <td><strong>0.9657</strong></td>
      <td><strong>0.9775</strong></td>
      <td><strong>0.9591</strong></td>
      <td><strong>0.9679</strong></td>
    </tr>
    <tr>
      <td><strong>Accuracy</strong></td>
      <td>—</td>
      <td>—</td>
      <td><strong>0.9762</strong></td>
      <td>—</td>
      <td>—</td>
      <td><strong>0.9778</strong></td>
    </tr>
  </tbody>
</table>


## Installation

After cloning this repository, please follow SaProt([SaProt](https://github.com/westlake-repl/SaProt)) installation guide to install SaProt and Foldseek.


## Pipeline

![pipeline](../assets/WDPpipeline.png)

1. Obtain structural embeddings with SaProt ([SaProt](https://github.com/westlake-repl/SaProt)):

    - Feed the PDB file to Foldseek to generate a "structural sequence" that expands the amino-acid vocabulary from 21 to 441 tokens (see `src/s1_generate_foldseek_seqs.py`).
    - Generate the embedding extraction by [westlake-repl/SaProt_650M_PDB](https://huggingface.co/westlake-repl/SaProt_650M_PDB) on the structural sequence (see `src/s2_generate_saprot_embddings.py`).

2. Train with the extracted embeddings (see `src/s3_train_WDP.py`). We tried `mlp/lstm/cnn+lstm/cnn+transformers` prediction head, and `LSTM` performs as the best model.

3. Apply Post-processing rules:
    - **rule 1 - length filtering**: Sequences shorter than 200 residues are deemed unable to form a domain and are assigned as 0.
    - **rule 2 - noise filtering**: predicted 1-runs ≤ 3 residues long are flipped to 0; predicted 0-runs ≤ 3 residues long are flipped to 1.
    - **rule 3 - interval merging**: merge adjacent domain intervals if the gap between them is < $ \alpha \cdot \min(\text {current interval length}, \text {next interval length}) $ ; otherwise keep them separate.
    - **rule 4 - Final domain selection**: from the merged candidate intervals, discard any shorter than 200 residues and select the longest remaining interval as the final predicted domain.

4. Evaluation criteria:
    - For positive samples (domain exists), a prediction is considered correct if both start and end positions are within ±5 bp of the true domain boundaries.
    - For negative samples (domain doesn't exsit), a prediction is correct only if no domain is predicted (empty prediction).


## Citation

```
@article{su2023saprot,
  title={SaProt: Protein Language Modeling with Structure-aware Vocabulary},
  author={Su, Jin and Han, Chenchen and Zhou, Yuyang and Shan, Junjie and Zhou, Xibin and Yuan, Fajie},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Reference

[1]: Su, J., Han, C., Zhou, Y., Shan, J., Zhou, X., & Yuan, F. (2023). SaProt: Protein Language Modeling with Structure-aware Vocabulary. *bioRxiv*.



