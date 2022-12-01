# scME
scME: A Dual-Modality Factor Model for Single-Cell Multi-Omics Embedding


## Installation
1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. Install dependencies:
    `pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib scanpy anndata`
    or use conda environment
    `conda env create -f environment.yaml`

## Prepare matrix and label files
scME accepts as input the  RNA gene counts matrix data and raw protein ADTs counts matrix data in the CSV format usually end in ".csv", where rows are cells and columns are genes, and the columns 0 in csv file should be the cells ids. 



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

