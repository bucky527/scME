
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
import torch
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
import seaborn as sns
import umap
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay
from sklearn import metrics
import scanpy as sc
import anndata as ad

##DataSet 

#Dataset for the Model:
'''
must have attr:
self.rna    ->dataframe
self.efeatures      ->dataframe
self.label      ->DataFrame
self.layer      ->DataFrame
self.label_digit        ->np.ndarray
self.layer_digit        ->np.ndarray

self.cells      ->np.ndarray
self.rnanums
self.efeaturesnums
self.labelclassnum
self.layerclassnum

'''

class MyData(Dataset):
    def __init__(self,data_path='D:/zhoub/neuron/GANTransform/data/citeseq/seurat',rnafilename='',efeaturesfilename='',metafilename='',targetname1='celltype.l1',targetname2='celltype.l2',reduct=None):
        self.data_path=data_path
        self.rnafilename=rnafilename
        self.metafilename=metafilename
        self.efeaturesfilename=efeaturesfilename
        self.targetname1=targetname1
        self.targetname2=targetname2

        print(f"load {self.data_path}")
        self.meta=pd.read_csv(os.path.join(self.data_path,metafilename),index_col=0)
        self.cells=np.array(self.meta.index).squeeze()
        print('cellname shape:',self.cells.shape)
        self.rna_df=pd.read_csv(os.path.join(self.data_path,rnafilename),index_col=0)
        self.efeatures_df=pd.read_csv(os.path.join(self.data_path,efeaturesfilename),index_col=0)

        self.label1=self.meta[targetname1]
        self.label1=self.label1.loc[self.cells]
        self.label1=np.array(self.label1)

        self.label2=self.meta[targetname2]
        self.label2=self.label2.loc[self.cells]
        self.label2=np.array(self.label2)

        self.rna_df=self.rna_df.loc[self.cells,:]
        self.efeatures_df=self.efeatures_df.loc[self.cells,:]

        self.rna,self.efeatures=np.array(self.rna_df),np.array(self.efeatures_df)
        self.rnanums=self.rna.shape[1]
        self.ephysfeaturesnums=self.efeatures.shape[1]
        print(self.rna.shape,self.efeatures.shape)
        #Dataframe

        self.label1encoder=LabelEncoder()
        self.label1_digit=self.label1encoder.fit_transform(np.array(self.label1).squeeze())
        self.label1_digit=self.label1_digit.reshape((-1,1))
        self.labelclassnum1=len(self.label1encoder.classes_)
        self.label1_onehot=np.eye(self.labelclassnum1)[self.label1_digit.squeeze()]
        print('label.shape:',self.label1_digit.shape)

        self.label2encoder=LabelEncoder()
        self.label2_digit=self.label2encoder.fit_transform(np.array(self.label2).squeeze())
        self.label2_digit=self.label2_digit.reshape((-1,1))
        self.labelclassnum2=len(self.label2encoder.classes_)
        self.label2_onehot=np.eye(self.labelclassnum2)[self.label2_digit.squeeze()]

        self.reduct=reduct
        if reduct=='tsne':
            print("run pac and tsne...")
            pca=PCA(n_components=50)
            self.rna_pca=pca.fit_transform(self.rna)
            self.rcode=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(self.rna_pca)
            self.rcode=pd.DataFrame({'x':self.rcode[:,0],'y':self.rcode[:,1]})
            self.ecode=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(self.efeatures)
            self.ecode=pd.DataFrame({'x':self.ecode[:,0],'y':self.ecode[:,1]})
        elif reduct =='umap':
            print("run pca and umap...")
            pca=PCA(n_components=50)
            self.rna_pca=pca.fit_transform(self.rna)
            self.rcode=umap.UMAP(random_state=42).fit_transform(self.rna_pca)
            self.rcode=pd.DataFrame({'x':self.rcode[:,0],'y':self.rcode[:,1]})
            self.ecode=umap.UMAP(random_state=42).fit_transform(self.efeatures)
            self.ecode=pd.DataFrame({'x':self.ecode[:,0],'y':self.ecode[:,1]})
    
    def __getitem__(self, item):
        rnada=self.rna[item,:]
        rnada=torch.from_numpy(rnada).float()
        ephysda=self.efeatures[item,:]
        ephysda=torch.from_numpy(ephysda).float()
        labelda=self.label_digit[item,:]
        labelda=torch.from_numpy(labelda).long()
        return rnada,ephysda,labelda

    def __len__(self):

        return len(self.cells)


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else adata.X)
    )
    return adata

def rna_protein_preprocess(rnadata:pd.DataFrame,proteindata:pd.DataFrame,res=1)->ad.AnnData:
    
    if isinstance(rnadata,ad.AnnData):
        rna=rnadata
    else:
        rna=ad.AnnData(X=rnadata.values,obs=pd.DataFrame(index=rnadata.index), var=pd.DataFrame(index=rnadata.columns))
    if isinstance(proteindata,ad.AnnData):
        protein=proteindata
    else:
        protein=ad.AnnData(X=proteindata.values,obs=pd.DataFrame(index=proteindata.index), var=pd.DataFrame(index=proteindata.columns))
    print("preprocess rna and protein data...")
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes( 
        rna,
        n_top_genes=2000,
        flavor="seurat_v3",
        subset = True
    )
    rna.raw = rna
    rna = rna[:, rna.var.highly_variable]
    sc.pp.pca(rna,svd_solver='arpack')
    sc.pp.neighbors(rna, n_neighbors=30,n_pcs=40)   
    sc.tl.leiden(rna, key_added="rna_leiden",resolution=res)
    #protein
    print("preprocess protein data...")
    protein.var["control"] = protein.var_names.str.contains("control")
    # sc.pp.calculate_qc_metrics(
    # protein,
    # percent_top=(5, 10, 15),
    # var_type="antibodies",
    # qc_vars=("control",),
    # inplace=True,
    # )
    protein.layers["counts"] = protein.X.copy()
    protein=clr_normalize_each_cell(protein)
    sc.pp.pca(protein, svd_solver="arpack")
    sc.pp.neighbors(protein, n_neighbors=30) 
    sc.tl.leiden(protein, key_added="protein_leiden",resolution=res)
    return rna,protein

#dataset return rna,protein,rna_leiden,protein_leiden
class AnnDataset(Dataset):
    def __init__(self,rnadata,proteindata,to_onehot=False) -> None:
        super().__init__()
        self.rnadata=rnadata
        self.to_onehot=to_onehot
        if not isinstance(rnadata.X,np.ndarray):
            self.rna=np.array(rnadata.X.todense())
        else:
            self.rna=np.array(rnadata.X)

        self.rna_leiden=np.array(rnadata.obs['rna_leiden'])
        self.rna_leiden=self.rna_leiden.reshape((-1,1))
        self.rnaleiden_digit=LabelEncoder().fit_transform(np.array(rnadata.obs['rna_leiden']))
        self.rnaleiden_onehot=np.eye(len(np.unique(self.rnaleiden_digit)))[self.rnaleiden_digit.squeeze()]

        self.proteindata=proteindata
        if not isinstance(proteindata.X,np.ndarray):
            self.protein=np.array(proteindata.X.todense())
        else:
            self.protein=np.array(proteindata.X)
        self.protein_leiden=np.array(proteindata.obs['protein_leiden'])
        self.protein_leiden=self.protein_leiden.reshape((-1,1))
        self.protein_leiden_digit=LabelEncoder().fit_transform(np.array(proteindata.obs['protein_leiden']))
        self.protein_leiden_onehot=np.eye(len(np.unique(self.protein_leiden_digit)))[self.protein_leiden_digit.squeeze()]

        self.rna_class_num=len(np.unique(self.rnaleiden_digit))
        self.protein_class_num=len(np.unique(self.protein_leiden_digit))
    def __getitem__(self, item):
        if self.to_onehot:
            rnada=self.rna[item,:]
            rnada=torch.from_numpy(rnada).float()
            rnaclass=self.rnaleiden_onehot[item,:]
            rnaclass=torch.from_numpy(rnaclass).long()
            proteinda=self.protein[item,:]
            proteinda=torch.from_numpy(proteinda).float()
            proteinclass=self.protein_leiden_onehot[item,:]
            proteinclass=torch.from_numpy(proteinclass).long()
            return rnada,proteinda,rnaclass,proteinclass
        else:
            rnada=self.rna[item,:]
            rnada=torch.from_numpy(rnada).float()
            rnaclass=self.rnaleiden_digit[item]
            rnaclass=torch.tensor(rnaclass).long()
            proteinda=self.protein[item,:]
            proteinda=torch.from_numpy(proteinda).float()
            proteinclass=self.protein_leiden_digit[item]
            proteinclass=torch.tensor(proteinclass).long()
            return rnada,proteinda,rnaclass,proteinclass

    def __len__(self):
        return len(self.rnadata)


#return rnada,proteinda,rnaclass,proteinclass,batch
class AnnDataset_Batch(Dataset):
    def __init__(self,rnadata,proteindata,to_onehot=False,batch_key=None) -> None:
        super().__init__()
        self.rnadata=rnadata
        self.to_onehot=to_onehot
        if not isinstance(rnadata.X,np.ndarray):
            self.rna=np.array(rnadata.X.todense())
        else:
            self.rna=np.array(rnadata.X)

        self.rna_leiden=np.array(rnadata.obs['rna_leiden'])
        self.rna_leiden=self.rna_leiden.reshape((-1,1))
        self.rnaleiden_digit=LabelEncoder().fit_transform(np.array(rnadata.obs['rna_leiden']))
        self.rnaleiden_onehot=np.eye(len(np.unique(self.rnaleiden_digit)))[self.rnaleiden_digit.squeeze()]

        self.proteindata=proteindata
        if not isinstance(proteindata.X,np.ndarray):
            self.rna=np.array(proteindata.X.todense())
        else:
            self.protein=np.array(proteindata.X)
        self.protein_leiden=np.array(proteindata.obs['protein_leiden'])
        self.protein_leiden=self.protein_leiden.reshape((-1,1))
        self.protein_leiden_digit=LabelEncoder().fit_transform(np.array(proteindata.obs['protein_leiden']))
        self.protein_leiden_onehot=np.eye(len(np.unique(self.protein_leiden_digit)))[self.protein_leiden_digit.squeeze()]

        self.batch_key=batch_key
        self.batch=np.array(rnadata.obs[self.batch_key])
        self.batch_digit=LabelEncoder().fit_transform(self.batch)
        self.batch_onehot=np.eye(len(np.unique(self.batch_digit)))[self.batch_digit.squeeze()]
        
    def __getitem__(self, item):
        if self.to_onehot:
            rnada=self.rna[item,:]
            rnada=torch.from_numpy(rnada).float()
            rnaclass=self.rnaleiden_onehot[item,:]
            rnaclass=torch.from_numpy(rnaclass).long()
            proteinda=self.protein[item,:]
            proteinda=torch.from_numpy(proteinda).float()
            proteinclass=self.protein_leiden_onehot[item,:]
            proteinclass=torch.from_numpy(proteinclass).long()
            batch=self.batch_onehot[item,:]
            batch=torch.from_numpy(batch).long()
            return rnada,proteinda,rnaclass,proteinclass,batch
        else:
            rnada=self.rna[item,:]
            rnada=torch.from_numpy(rnada).float()
            rnaclass=self.rnaleiden_digit[item]
            rnaclass=torch.tensor(rnaclass).long()
            proteinda=self.protein[item,:]
            proteinda=torch.from_numpy(proteinda).float()
            proteinclass=self.protein_leiden_digit[item]
            proteinclass=torch.tensor(proteinclass).long()
            batch=self.batch_onehot[item,:]
            batch=torch.from_numpy(batch).long()
            return rnada,proteinda,rnaclass,proteinclass,batch

    def __len__(self):
        return len(self.rnadata)