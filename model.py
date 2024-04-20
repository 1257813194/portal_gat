import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA, IncrementalPCA
import sys
from tqdm import tqdm
from torch_geometric.loader import NeighborSampler
import anndata
from .networks import *
from .gat import *

class Model(object):
    def __init__(self, training_steps=1500,batch_size=128,seed=1234, hidden_dims=[512,20],lambdacos=20.0,lambdaAE=10,lambdaLA=10,lambdaGAN=1,lambdanb=1,margin=5,npcs=30,warmup=5,cutoff=6,lr_G=0.001,lr_D=0.001,attention=True,model_path="portal_gat", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.training_steps = training_steps
        self.hidden_dims=hidden_dims
        self.lambdacos = lambdacos  #20
        self.lambdaAE = lambdaAE 
        self.lambdaLA = lambdaLA
        self.lambdaGAN = lambdaGAN
        self.lambdanb = lambdanb
        self.margin = margin
        self.model_path = model_path+'/'+str(time.time())
        self.data_path = data_path
        self.result_path = result_path
        self.warmup=warmup
        self.lr_G=lr_G
        self.lr_D=lr_D
        self.npcs = npcs
        self.cutoff = cutoff
        self.attention=attention
        self.batch_size=batch_size

    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   hvg_num=None, # number of highly variable genes for each anndata
                   svg=False,
                   save_embedding=False # save low-dimensional embeddings or not
                   ):
        '''
        Performing preprocess for a pair of datasets.
        To integrate multiple datasets, use function preprocess_multiple_anndata in utils.py
        '''
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        if hvg_num!=None:
            print("Finding highly variable genes...")
            sc.pp.highly_variable_genes(adata_A, flavor='seurat_v3', n_top_genes=hvg_num)
            sc.pp.highly_variable_genes(adata_B, flavor='seurat_v3', n_top_genes=hvg_num)
            hvg_A = adata_A.var[adata_A.var.highly_variable == True].sort_values(by="highly_variable_rank").index
            hvg_B = adata_B.var[adata_B.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        elif svg:
            print("Finding spatially variable genes...")
            hvg_A = adata_A.var[adata_A.var.spatially_variable == True].sort_values(by="moranI").index
            hvg_B = adata_B.var[adata_B.var.spatially_variable == True].sort_values(by="moranI").index

        hvg_total = list(set(hvg_A) & set(hvg_B))
        self.hvg_num=len(hvg_total)
        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))
        else:
            print('{} highly variable genes'.format(self.hvg_num))

        print("Normalizing and scaling...")
        sc.pp.normalize_total(adata_A, target_sum=1e4)
        sc.pp.log1p(adata_A)
        adata_A = adata_A[:, hvg_total]
        sc.pp.scale(adata_A, max_value=10)

        sc.pp.normalize_total(adata_B, target_sum=1e4)
        sc.pp.log1p(adata_B)
        adata_B = adata_B[:, hvg_total]
        sc.pp.scale(adata_B, max_value=10)

        adata_total = adata_A.concatenate(adata_B, index_unique=None)
        print("Dimensionality reduction via PCA...")
        pca = PCA(n_components=self.npcs, svd_solver="arpack", random_state=0)
        adata_total.obsm["X_pca"] = pca.fit_transform(adata_total.X)

        adata_A_new= anndata.AnnData(adata_total.obsm["X_pca"][:adata_A.shape[0], :self.npcs].copy())
        adata_B_new= anndata.AnnData(adata_total.obsm["X_pca"][adata_A.shape[0]:, :self.npcs].copy())
        adata_A_new.obsm['spatial']=adata_A.obsm['spatial']
        adata_B_new.obsm['spatial']=adata_B.obsm['spatial']
        Cal_Spatial_Net(adata_A_new,k_cutoff=self.cutoff,model="KNN")
        Cal_Spatial_Net(adata_B_new,k_cutoff=self.cutoff,model="KNN")
        self.adata_A=adata_A_new
        self.adata_B=adata_B_new
        #Cal_Spatial_Net(adata_A,k_cutoff=self.cutoff,model="KNN")
        #Cal_Spatial_Net(adata_B,k_cutoff=self.cutoff,model="KNN")
        #self.adata_A=adata_A
        #self.adata_B=adata_B

    def train(self,transformed_data=None,id_A=[0,1],id_B=[1,0],graph_dict=None):
        if transformed_data==None:
            self.data_A = Transfer_pytorch_Data(self.adata_A).cuda()
            self.data_B = Transfer_pytorch_Data(self.adata_B).cuda()
        else:
            self.data_A = transformed_data[0].cuda()
            self.data_B = transformed_data[1].cuda()
        hidden_dims_A=[self.npcs] + self.hidden_dims
        hidden_dims_B=[self.npcs] + self.hidden_dims
        if self.attention:
            self.GRAPH=GAT(len(id_A),hidden_dims = hidden_dims_A).to(self.device)    
        else:
            self.GRAPH=GCN(len(id_A),hidden_dims = hidden_dims_A).to(self.device)           
        self.G_A= Generator(hidden_dims = hidden_dims_A).to(self.device)
        self.G_B= Generator(hidden_dims = hidden_dims_B).to(self.device)
        self.params_G = list(self.GRAPH.parameters())+list(self.G_A.parameters()) + list(self.G_B.parameters())
        self.optimizer_G = optim.Adam(self.params_G, lr=self.lr_G, weight_decay=0.0001)
        self.D_A = discriminator(hidden_dims_A).to(self.device)
        self.D_B = discriminator(hidden_dims_B).to(self.device)
        self.params_D = list(self.D_A.parameters()) + list(self.D_B.parameters())
        self.optimizer_D = optim.Adam(self.params_D, lr=self.lr_D, weight_decay=0.)
        if graph_dict!= None:
            self.GRAPH.load_state_dict(graph_dict)
        self.GRAPH.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()
        self.D=[]
        self.G=[]
        self.GAN=[]
        self.AE=[]
        self.cos=[]
        self.LA=[]
        self.nb=[]
        self.id_A=id_A
        self.id_B=id_B
        self.x_A = self.data_A.x.cuda()
        self.x_B = self.data_B.x.cuda()
        self.edge_A=self.data_A.edge_index.cuda()
        self.edge_B=self.data_B.edge_index.cuda() 


        with tqdm(total=self.training_steps, file=sys.stdout) as pbar:
            for step in range(self.training_steps):
                #nid_A,adjs_A=sample_graph(self.data_A,batch_size=self.batch_size,num_neighbors=self.cutoff)
                #nid_B,adjs_B=sample_graph(self.data_B,batch_size=self.batch_size,num_neighbors=self.cutoff)
                #s_x_A = self.data_A.x[nid_A,:].cuda()
                #s_x_B = self.data_B.x[nid_B,:].cuda()
                #s_edge_A=adjs_A.edge_index.cuda()
                #s_edge_B=adjs_B.edge_index.cuda() 
                z_A= self.GRAPH(self.x_A ,self.edge_A,self.id_A)
                z_B= self.GRAPH(self.x_B ,self.edge_B,self.id_B)
                x_Arecon=self.G_A(z_A)
                x_Brecon=self.G_B(z_B)
                x_AtoB = self.G_B(z_A)
                x_BtoA = self.G_A(z_B)
                z_AtoB = self.GRAPH(x_AtoB,self.edge_A,[0]*len(self.id_A))
                z_BtoA = self.GRAPH(x_BtoA,self.edge_B,[0]*len(self.id_A))
                # discriminator loss:
                self.optimizer_D.zero_grad()
                if step <= self.warmup:
                    # Warm-up
                    loss_D_A = (torch.log(1 + torch.exp(-self.D_A(self.x_A)))).mean() + (torch.log(1 + torch.exp(self.D_A(x_BtoA)))).mean()
                    loss_D_B = (torch.log(1 + torch.exp(-self.D_B(self.x_B)))).mean() + (torch.log(1 + torch.exp(self.D_B(x_AtoB)))).mean()
                else:
                    loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(self.x_A), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                    loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(self.x_B), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                loss_D = loss_D_A + loss_D_B
                loss_D.backward(retain_graph=True)
                #torch.nn.utils.clip_grad_norm_(self.params_D, 5)
                self.optimizer_D.step()

                # autoencoder loss:
                loss_AE_A = torch.mean((x_Arecon - self.x_A)**2)
                loss_AE_B = torch.mean((x_Brecon - self.x_B)**2)
                loss_AE = loss_AE_A + loss_AE_B

                # cosine correspondence:
                loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(self.x_A, p=2), 1)).mean()
                loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(self.x_B, p=2), 1)).mean()
                loss_cos = loss_cos_A + loss_cos_B

                # latent align loss:
                loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
                loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
                loss_LA = loss_LA_AtoB + loss_LA_BtoA

                # neighbour loss:
                loss_nb_A=neighbor_mean_mse_loss(z_A,self.edge_A)
                loss_nb_B=neighbor_mean_mse_loss(z_B,self.edge_B)
                loss_nb=loss_nb_A+loss_nb_B

                # generator loss
                self.optimizer_G.zero_grad()
                if step <= self.warmup:
                    # Warm-up
                    loss_G_GAN = (torch.log(1 + torch.exp(-self.D_A(x_BtoA)))).mean() + (torch.log(1 + torch.exp(-self.D_B(x_AtoB)))).mean()
                else:
                    loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                
                loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdanb * loss_nb
                loss_G.backward()
                #torch.nn.utils.clip_grad_norm_(self.params_G, 5)
                self.optimizer_G.step()
                self.G.append(loss_G.detach().cpu().numpy())
                self.D.append(loss_D.detach().cpu().numpy())
                self.GAN.append(self.lambdaGAN *loss_G_GAN.detach().cpu().numpy())
                self.AE.append(self.lambdaAE *loss_AE.detach().cpu().numpy())
                self.cos.append(self.lambdacos *loss_cos.detach().cpu().numpy())
                self.LA.append(self.lambdaLA *loss_LA.detach().cpu().numpy())
                self.nb.append(self.lambdanb *loss_nb.detach().cpu().numpy())
                pbar.set_description('processed: %d' % (1 + step))
                pbar.set_postfix({'loss_D':loss_D.detach().cpu().numpy(),
                                 'loss_GAN':self.lambdaGAN *loss_G_GAN.detach().cpu().numpy(),
                                 'loss_AE':self.lambdaAE *loss_AE.detach().cpu().numpy(),
                                 'loss_cos':self.lambdacos *loss_cos.detach().cpu().numpy(),
                                 'loss_LA':self.lambdaLA *loss_LA.detach().cpu().numpy(),
                                 'loss_nb':self.lambdanb *loss_nb.detach().cpu().numpy()
                                 })
                pbar.update(1)
        self.G_A.eval()
        self.G_B.eval()
        #self.E_A.eval()
        #self.E_B.eval()
        self.GRAPH.eval()
        self.D_A.eval()
        self.D_B.eval()
        z_A = self.GRAPH(self.x_A,self.edge_A,self.id_A)
        z_B = self.GRAPH(self.x_B,self.edge_B,self.id_B)
        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)
        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.A_space= np.concatenate((x_Arecon.detach().cpu().numpy(), x_BtoA.detach().cpu().numpy()), axis=0)
        self.B_space= np.concatenate((x_AtoB.detach().cpu().numpy(), x_Brecon.detach().cpu().numpy()), axis=0)
    def KL_process(self,training_steps=200,n_clust=7,lambdaKL=1):
        cluster=mclust(self.latent, num_cluster=n_clust)
        Mergefeature=pd.DataFrame(self.latent)
        Mergefeature['mclust']=cluster
        cluster_centers = np.asarray(Mergefeature.groupby("mclust").mean())
        self.mu = torch.nn.Parameter(torch.Tensor(n_clust,self.hidden_dims[-1]))    
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.mu.data = self.mu.data.to(self.device)
        self.GRAGH_A.train()
        self.GRAGH_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()
        self.KL=[]
        with tqdm(total=training_steps, file=sys.stdout) as pbar:
            for step in range(training_steps):    
                z_A= self.GRAGH_A(self.x_A,self.edge_A)
                z_B= self.GRAGH_B(self.x_B,self.edge_B)          
                edge_A_I=torch.tensor([list(range(self.x_A.shape[0])),list(range(self.x_A.shape[0]))], dtype=torch.long).cuda()
                edge_B_I=torch.tensor([list(range(self.x_B.shape[0])),list(range(self.x_B.shape[0]))], dtype=torch.long).cuda()
                x_Arecon=self.G_A(z_A)
                x_Brecon=self.G_B(z_B)
                x_AtoB = self.G_B(z_A)
                x_BtoA = self.G_A(z_B)
                z_AtoB = self.GRAGH_B(x_AtoB,edge_A_I)
                z_BtoA = self.GRAGH_A(x_BtoA,edge_B_I)

                # discriminator loss:
                self.optimizer_D.zero_grad()
                loss_D_A = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(self.x_A), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean()
                loss_D_B = (torch.log(1 + torch.exp(-torch.clamp(self.D_B(self.x_B), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                loss_D = loss_D_A + loss_D_B
                loss_D.backward(retain_graph=True)
                self.optimizer_D.step()

                # autoencoder loss:
                loss_AE_A = torch.mean((x_Arecon - self.x_A)**2)
                loss_AE_B = torch.mean((x_Brecon - self.x_B)**2)
                loss_AE = loss_AE_A + loss_AE_B

                # cosine correspondence:
                loss_cos_A = (1 - torch.sum(F.normalize(x_AtoB, p=2) * F.normalize(self.x_A, p=2), 1)).mean()
                loss_cos_B = (1 - torch.sum(F.normalize(x_BtoA, p=2) * F.normalize(self.x_B, p=2), 1)).mean()
                loss_cos = loss_cos_A + loss_cos_B

                # latent align loss:
                loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
                loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
                loss_LA = loss_LA_AtoB + loss_LA_BtoA
                
                # neighbour loss:
                loss_nb_A=neighbor_mean_mse_loss(z_A,self.edge_A)
                loss_nb_B=neighbor_mean_mse_loss(z_B,self.edge_B)
                loss_nb=loss_nb_A+loss_nb_B


                # KL loss
                z=torch.cat((z_A, z_B), dim=0)
                if step%3 == 0:
                    Q = get_q(z,self.mu)
                    q = Q.detach().data.cpu().numpy().argmax(1)              
                    
                q = get_q(z,self.mu)
                p = get_p(Q.detach())
                loss_KL=KL_div(p, q)

                # generator loss
                self.optimizer_G.zero_grad()
                loss_G_GAN = (torch.log(1 + torch.exp(-torch.clamp(self.D_A(x_BtoA), -self.margin, self.margin)))).mean() + (torch.log(1 + torch.exp(-torch.clamp(self.D_B(x_AtoB), -self.margin, self.margin)))).mean()
                
                loss_G = self.lambdaGAN * loss_G_GAN + self.lambdacos * loss_cos + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + lambdaKL * loss_KL + self.lambdanb * loss_nb
                loss_G.backward()
                #torch.nn.utils.clip_grad_norm_(self.params_G, 5)
                self.optimizer_G.step()
                self.G.append(loss_G.detach().cpu().numpy())
                self.D.append(loss_D.detach().cpu().numpy())
                self.GAN.append(self.lambdaGAN *loss_G_GAN.detach().cpu().numpy())
                self.AE.append(self.lambdaAE *loss_AE.detach().cpu().numpy())
                self.cos.append(self.lambdacos *loss_cos.detach().cpu().numpy())
                self.LA.append(self.lambdaLA *loss_LA.detach().cpu().numpy())
                self.nb.append(self.lambdanb *loss_nb.detach().cpu().numpy())
                self.KL.append(lambdaKL *loss_KL.detach().cpu().numpy())
                pbar.set_description('processed: %d' % (1 + step))
                pbar.set_postfix({'loss_D':loss_D.detach().cpu().numpy(),
                                 'loss_GAN':self.lambdaGAN *loss_G_GAN.detach().cpu().numpy(),
                                 'loss_AE':self.lambdaAE *loss_AE.detach().cpu().numpy(),
                                 'loss_cos':self.lambdacos *loss_cos.detach().cpu().numpy(),
                                 'loss_LA':self.lambdaLA *loss_LA.detach().cpu().numpy(),
                                 'loss_nb':self.lambdanb *loss_nb.detach().cpu().numpy(),
                                 'loss_KL':lambdaKL *loss_KL.detach().cpu().numpy()
                                 })
                pbar.update(1)
        self.G_A.eval()
        self.G_B.eval()
        #self.E_A.eval()
        #self.E_B.eval()
        self.GRAGH_A.eval()
        self.GRAGH_B.eval()
        self.D_A.eval()
        self.D_B.eval()
        z_A = self.GRAGH_A(self.x_A,self.edge_A)
        z_B = self.GRAGH_B(self.x_B,self.edge_B)
        self.KL_latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)


    def save(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        state = {'D_A': self.D_A.state_dict(), 'D_B': self.D_B.state_dict(),
                 'GRAGH_A': self.GRAGH_A.state_dict(), 'GRAGH_B': self.GRAGH_B.state_dict(),
                 'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict(),}
        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))