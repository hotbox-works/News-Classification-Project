# CS390AIF-Project1

News classification across datasets using TF-IDF features with logistic regression and an MLP.

## Files

- `project1_full.ipynb` has all the code for Parts 1 to 5 and the Selected Component.
- `dataset/` has the five CSV files the notebook reads.
- `requirements.txt` lists the packages needed to run it locally.

## Running in Google Colab (recommended)

1. Open Colab and go to **File > Upload notebook**, then pick `project1_full.ipynb`.
2. Since the repo is private, the notebook needs a GitHub token to clone it. Click the **key icon** in the left sidebar, add a secret named `GITHUB_TOKEN` with your token as the value, and turn on **Notebook access**.
3. Go to **Runtime > Change runtime type** and pick **T4 GPU**.
4. Go to **Runtime > Run all**.

The first cell clones the repo and moves into it, so the notebook finds the CSVs in `dataset/` on its own. Colab already has every package the notebook uses. A full run takes about 15 to 20 minutes on a T4.

## Running locally

From the repo root:

```
pip install -r requirements.txt
jupyter notebook project1_full.ipynb
```

Skip the first cell, since it only works in Colab. Then run the rest of the cells in order. The notebook runs on CPU, but it's a lot slower without a GPU.

## Reproducibility

All splits, samples, and models use random seed 0. Every reported result comes with a table showing its text representation, classifier, hyperparameters, number of epochs, seed, and training-set size.
