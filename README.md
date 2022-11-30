# scME
scME: A Dual-Modality Factor Model for Single-Cell Multi-Omics Embedding


## Installation
1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. Install dependencies:
    `pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib scanpy anndata`
    or use conda environment
    `conda env create -f environment.yaml`

## Prepare matrix and label files
1. scME accepts as input the log-transformed gene matrix in the MatrixMarket format usually end in ".mtx", where rows are cells and columns are genes. 
2. The label file can be either the CSV format or the TSV format, one label per line.
3. [Data](https://github.com/ZengFLab/scClassifier2/tree/main/data) gives some examples of matrix and label files.



## Usage
```
usage: python scme.py [-h] [--rna RNA] [--protein PROTEIN] [--output-dir OUTPUT_DIR]
               [--max-epochs MAX_EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
               [--latentdim LATENTDIM]
               [--aux-loss-multiplier AUX_LOSS_MULTIPLIER]
               [--rna-latentdim RNA_LATENTDIM]
               [--protein-latentdim PROTEIN_LATENTDIM]


```
## Option
```
optional arguments:
  -h, --help            show this help message and exit
  --rna RNA             rna csv data path
  --protein PROTEIN     protein csv data path
  --output-dir OUTPUT_DIR
                        output directory
  --max-epochs MAX_EPOCHS
                        max epochs
  --batch-size BATCH_SIZE
                        batch size
  --lr LR               learning rate
  --latentdim LATENTDIM
                        latent dimension
  --aux-loss-multiplier AUX_LOSS_MULTIPLIER
                        auxiliary loss multiplier
  --rna-latentdim RNA_LATENTDIM
                        rna latent dimension
  --protein-latentdim PROTEIN_LATENTDIM
                        protein latent dimension
```

