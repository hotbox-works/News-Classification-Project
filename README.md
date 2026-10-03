# CS390AIF-Project1

All of our executable code is in one Jupyter notebook file called `project1_aif.ipynb`. Running it from top to bottom reproduces every result, table, and figure from our report across all of the sections.

## File Structure
 
The notebook expects the data comes from a `dataset/` folder next to it so make sure the file structure looks like this:
 
```
project1/
|-- project1_aif.ipynb
|-- requirements.txt
|-- README.md
|-- dataset/
    |-- dataset_a_train.csv
    |-- dataset_a_test.csv
    |-- dataset_b_test.csv
    |-- dataset_c_train.csv
    |-- dataset_c_test.csv
```

The file names must match exactly as is written in this file tree and the notebook file reads these files with relative paths so start Jupyter from the 'project1/' folder. There should not be any output files and all results are output directly into the notebook file.

## Run on Purdue RCAC Scholar with GPU

We recommend running this project using the Purdue RCAC Scholar GPU because the MLP and LR are trained in PyTorch on TF-IDF vectors. The notebook runs on CPU but it is much faster to run it on the Scholar GPUs and also our results were obtained by running this project on GPUs so it might help for reproducibility up to extreme decimal places.

### 1. Upload the project to Scholar
 
1. Go to the Scholar Open OnDemand gateway, https://gateway.scholar.rcac.purdue.edu/, and log in with your Purdue career account.
2. Open **Files → Home Directory**, create a `project1` folder, and upload the notebook file `project1_aif.ipynb`, `requirements.txt`, this README file and the `dataset/` folder, keeping the file structure in the order mentioned above.

It is also possible to copy the files over with `scp` or open a terminal on Scholar and unzip a zipped copy of our folder there.

### 2. Create the conda environment

Open a terminal on Scholar, either **Clusters → Scholar Shell Access** in the gateway or `ssh <career_account>@scholar.rcac.purdue.edu`, then run this to set it up:
 
```bash
# load conda (try running "module avail conda" or "module avail anaconda" if it says this name isn't found)
module load conda
 
# create and activate a fresh environment
conda create -n aif-project1 python=3.11 -y
conda activate aif-project1
 
# install the packages we used
cd ~/project1
pip install -r requirements.txt
 
# register this new environment as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name aif-project1 --display-name "Python (aif-project1)"
```
 
If `conda activate` doesn't work, you can run `source activate aif-project1` instead.

### 3. Start a GPU Jupyter session from Purdue RCAC Scholar
 
1. In the gateway, open **Interactive Apps → Jupyter Notebook**.
2. Choose the GPU option for the queue and request 1 GPU. Ask for at least 2 hours of wall time and click **Launch**. (You can also just start a quick Jupyter Session)
3. When the session is ready, click **Connect to Jupyter** and open `project1/project1_aif.ipynb`.
4. Select **Kernel → Change Kernel → Python (aif-project1)**.

## Running locally without connecting to a Scholar session
 
```bash
conda create -n aif-project1 python=3.11 -y
conda activate aif-project1
pip install -r requirements.txt
pip install jupyter
jupyter notebook   # make sure you start this from the project1/ folder
```
 
Our notebook file sets up and uses CUDA automatically when it's available otherwise uses the CPU so everything will still run locally, just slower without the GPU.

## Running the notebook
 
1. Select **Kernel → Restart & Run All**. Run the whole notebook from the top in a fresh kernel because all the later parts depend on objects created in the earlier cells such as for Parts 4 and 5, they use the frozen Part 3 models and the Data A vectorizer that are kept in memory.
2. The third code cell prints `DEVICE: cuda` if the GPU is being used and `DEVICE: cpu` otherwise.
3. Results are in this order:

Part 1: Explore Data A: Class counts, missing, empty and duplicate checks, length box plots, example articles, then common title openings

Part 2: In-domain baseline: Validation training curves, refit on all of Data A train, Data A test metrics and confusion matrices, and then error analysis

Part 3: Regularization and Data B: Validation sweep on 4 LR and 4 MLP settings, selected models, Data A and Data B test metrics, per-class tables, confusion matrices, post hoc results for all settings, and then a Data A vs Data B comparison

Part 4: Transfer to Data C: Shared-labels 2–4 and label-1 metrics, label-5 rate, confusion matrices, World error samples, vocabulary-gap table, and then in-domain Data C reference models

Part 5: Limited Data C examples: Results for every budget and sampling strategy, label-1 F1 and Data C macro-F1 plots, and retained Data A and Data B performance plots

Selected component: Unigram & bigram TF-IDF: validation sweep, Data A and Data B test results, per-class comparison, confusion matrices, repeated Part 5 experiments, and then comparison plots against unigram

## Reproducibility
 
- **Seed:** Every random step uses seed 0. This covers the train and validation splits (`random_state=0`), NumPy and PyTorch (`np.random.seed(0)`, `torch.manual_seed(0)`, `torch.cuda.manual_seed_all(0)`), batch shuffling, the error-analysis samples, and in Part 5 the Data C sampling.
- **Package versions:** `requirements.txt` has the packages we used. Follow the setup rules we discussed earlier to setup the environment.
- **GPU vs CPU:** Some GPU operations in PyTorch are not bit-for-bit deterministic, so a rerun on a different GPU or on a CPU, can cause differences in the third or fourth decimal place, however our conclusions in the report don't depend on differences that small so it shouldn't cause any issues.
- **No test-set tuning:** Every TF-IDF vectorizer is fit only on the training data of the model that uses it. Model selection uses only validation splits made from training data, and Data B is never used for training or fitting.
- **Model configurations:** The notebook displays a configuration table for every final model, with representation, optimizer, learning rate, weight decay, dropout, batch size, epochs, seed, training-set size and vocabulary size. These same configurations are listed in our report.

### Results Comparison to Confirm Reproducibility

| Result | LR | MLP |
|---|---|---|
| Part 2 baseline, Data A test macro-F1 | 0.5171 | 0.5375 |
| Part 3 final models, Data A test macro-F1 | 0.5647 | 0.5749 |
| Part 3 final models, Data B test macro-F1 | 0.7888 | 0.7941 |
| Part 4, Data C shared-label macro-F1 (labels 2–4) | 0.6286 | 0.6378 |
| Part 4, Data C label-1 F1 (transferred model) | 0.4983 | 0.4763 |
| Part 4, Data C label-1 F1 (Data C reference model) | 0.6080 | 0.6257 |
| Part 5, balanced strategy with 1,500 added examples, Data C macro-F1 (labels 1–4) | 0.6201 | 0.6230 |
| Part 5, balanced strategy with 1,500 added examples, Data C label-1 F1 | 0.5836 | 0.5812 |
| Selected component (uni+bigram), Part 3 Data A test macro-F1 | 0.5667 | 0.5715 |
| Selected component (uni+bigram), Part 3 Data B test macro-F1 | 0.7255 | 0.7988 |
| Selected component (uni+bigram), Part 5 balanced strategy with 1,500 added examples, Data C macro-F1 | 0.6082 | 0.6058 |
