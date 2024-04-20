import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean
import torch.nn.functional as F

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def sample_graph(data, batch_size, num_neighbors):
    """
    对单个图进行采样，并返回采样结果。
    """
    from torch_geometric.loader import NeighborSampler
    sampler = NeighborSampler(data.edge_index, node_idx=None, sizes=[num_neighbors], batch_size=batch_size, shuffle=True)
    for subset, n_id, adjs in sampler:
        # 注意：这里仅为演示，实际使用时可能需要根据adjs等进一步处理数据
        # 返回第一个batch的采样结果作为示例
        return n_id, adjs
    return None

def keep_nodes(edge_index, nodes_to_keep):
    """
    仅保留指定节点ID的边，并返回新的边索引。
    
    参数:
        edge_index (Tensor): 图的边索引，形状为 [2, num_edges]。
        nodes_to_keep (list or Tensor): 要保留的节点ID列表。
        
    返回:
        Tensor: 仅包含指定节点及其边的新边索引。
    """
    # 确保 nodes_to_keep 是 tensor
    if not isinstance(nodes_to_keep, torch.Tensor):
        nodes_to_keep = torch.tensor(nodes_to_keep, dtype=torch.long)
    
    # 获取所有要保留的边的掩码
    mask = torch.isin(edge_index[0], nodes_to_keep) & torch.isin(edge_index[1], nodes_to_keep)
    
    # 应用掩码以保留涉及指定节点的边
    new_edge_index = edge_index[:, mask]
    
    # 创建一个映射字典，将原始节点ID映射到新的索引
    node_id_to_new_id = {node_id.item(): new_id for new_id, node_id in enumerate(nodes_to_keep)}
    
    # 使用映射字典更新边索引
    new_edge_index[0] = torch.tensor([node_id_to_new_id[node_id.item()] for node_id in new_edge_index[0]])
    new_edge_index[1] = torch.tensor([node_id_to_new_id[node_id.item()] for node_id in new_edge_index[1]])
    
    return new_edge_index

def final_sample(data1,data2,batch_size=128,num_neighbors=6):
    sampled_n_id_1,_ = sample_graph(data1, batch_size=batch_size, num_neighbors=num_neighbors)
    sampled_n_id_2,_  = sample_graph(data2, batch_size=batch_size, num_neighbors=num_neighbors)
    min_num_nodes = min(len(sampled_n_id_1), len(sampled_n_id_2))
    sampled_n_id_1=sampled_n_id_1[:min_num_nodes]
    sampled_n_id_2=sampled_n_id_2[:min_num_nodes]
    trimmed_data_1=Data(x=data1.x[sampled_n_id_1,:],edge_index=keep_nodes(data1.edge_index,sampled_n_id_1))
    trimmed_data_2=Data(x=data2.x[sampled_n_id_2,:],edge_index=keep_nodes(data2.edge_index,sampled_n_id_2))
    return(trimmed_data_1, trimmed_data_2)

def get_q(z,mu,beta=0.5):
    q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / beta) + 1e-8)
    q = q**(beta+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

def get_p(q):
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return(p)

def KL_div(p, q):
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss

def kl_loss(z,mu,beta=0.5):
    q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / beta) + 1e-8)
    q = q**(beta+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
    return loss

def mclust(data, num_cluster, modelNames = 'EEE', random_seed = 2020):
    """
    Mclust algorithm from R, similar to https://mclust-org.github.io/mclust/
    """ 
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()  
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(data, num_cluster, modelNames)
    return np.array(res[-2])

def neighbor_mean_mse_loss(embeddings, edge_index):
    row, col = edge_index
    neighbor_embeddings = embeddings[col]  # 获取邻居节点的嵌入表示
    center_embeddings = embeddings[torch.unique(row)]  # 获取中心节点的嵌入表示

    # 计算每个节点的邻居节点嵌入表示的均值
    neighbor_means = scatter_mean(neighbor_embeddings, row, dim=0)

    # 计算每个节点与其邻居节点均值之间的均方误差
    mse_loss = F.mse_loss(center_embeddings, neighbor_means)

    return mse_loss