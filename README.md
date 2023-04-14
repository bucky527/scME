# scME

scME: A Dual-Modality Factor Model for Single-Cell Multi-Omics Embedding

## Installation

1. Install [pytorch](https://pytorch.org/get-started/locally/) according to your computational platform
2. You can use git to clone our repository.

```
  git clone https://github.com/bucky527/scME.git
  cd SCME/
```

3. Install dependencies:

   you can install dependencies use pip

   ```
   pip3 install numpy scipy pandas scikit-learn pyro-ppl matplotlib scanpy anndata scvi-tools
   ```

   or use conda environment

   ```
   conda env create -f environment.yaml
   ```
4. Install scME package

```
  python setup.py install
```

## Prepare data

scME accepts as input the  RNA gene counts matrix data and raw protein ADTs counts matrix data in the CSV format usually end in ".csv", where rows are cells and columns are genes, and the columns 0 in csv file should be the cells ids.

## Usage

```
usage: scme.py [-h] --rna RNA --protein PROTEIN --output-dir OUTPUT_DIR
               [--max-epochs MAX_EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
               [--lr_classify LR_CLASSIFY] [--latentdim LATENTDIM]
               [--aux-loss-multiplier AUX_LOSS_MULTIPLIER]
               [--rna-latentdim RNA_LATENTDIM]
               [--protein-latentdim PROTEIN_LATENTDIM]
               [--lr-step LR_STEP [LR_STEP ...]] [--cuda CUDA]
               [--use-mnb USE_MNB]

```

## Option arguments description

```
optional arguments:
  -h, --help            show this help message and exit
  --rna RNA             rna count data .csv data path
  --protein PROTEIN     protein count data .csv data path
  --output-dir OUTPUT_DIR
                        output directory to save cells embeddings
  --max-epochs MAX_EPOCHS
                        train max epochs
  --batch-size BATCH_SIZE
                        train dataset batch size
  --lr LR               learning rate
  --lr_classify LR_CLASSIFY
                        learning rate for classify loss
  --latentdim LATENTDIM
                        dimension for embedding
  --aux-loss-multiplier AUX_LOSS_MULTIPLIER
                        auxiliary loss multiplier
  --rna-latentdim RNA_LATENTDIM
                        rna latent dimension
  --protein-latentdim PROTEIN_LATENTDIM
                        protein latent dimension
  --lr-step LR_STEP [LR_STEP ...]
                        learning rate decay step
  --cuda CUDA           use cuda
  --use-mnb USE_MNB     use mixture negative binomial distribution or not for
                        proteindata
```

### Get cell embedding for CITE-seq

You can use scme.py to easily obtain cell embeddings for CITE-seq data

```
python scme.py --rna [your rna gene counts csv file path] --protein [your protein ADTs counts csv file path] --output-dir [result save path] --batch-size 256
```

### Building model and training

```
#Prepare your data and build scME models

scme=build_scme(rnadata=rna
                ,proteindata=protein
                ,protein_dist="NB",#'NB' or 'MNB'
                rna_latent_dim=24,
                protein_latent_dim=20,
                latent_dim=32)

#Train scme
scme=train_model(model,
                max_epochs=200,
                batchsize=256,
                lr=1e-4,
                lr_cla=1e-4,
                milestones=[80],
                save_model=False,
                save_dir=None)

#Inference cell embedding

zm=scme.inference(rna_data,protein_data) 

```

### scME model example

see a running example in notebook [tutorial.ipynb](https://github.com/bucky527/scME/blob/master/tutorial.ipynb).
