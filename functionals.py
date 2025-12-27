import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np

def normalization(x: torch.Tensor, axis: int = 0, ntype: str = None) -> torch.Tensor:
    if ntype == None:
        print("ntype is missed -- original tensor is returned")
        return x
    elif ntype == 'standardization':
        return (x - x.mean(axis=axis)) / x.std(axis=axis)
    elif ntype == 'min-max':
        return (x - x.min()) / (x.max() - x.min())


def symmetric_normalization(matrix, im=None):
    tilde_matrix = matrix
    if im != None:
        tilde_matrix += im
    deg = torch.sum(tilde_matrix, dim=1)
    norm_deg = torch.pow(deg, -0.5)
    tilde_deg = torch.diag(norm_deg)  # degree matrix (D) tilde
    norm_matrix = torch.mm(torch.mm(tilde_deg, tilde_matrix), tilde_deg)

    return norm_matrix

def similarity_matrix(x:torch.Tensor, scale) -> torch.Tensor:
    def l2_distance_matrix(X: torch.Tensor):
        X_sq_norms = torch.sum(X ** 2, dim=1)  # squared Euclidean (l2) norm
        X1_sq_norms = X_sq_norms.view(-1, 1)
        X2_sq_norms = X_sq_norms.view(1, -1)
        vector_wise_dists = X1_sq_norms - 2 * torch.matmul(X, X.T) + X2_sq_norms
        l2_distance_matrix = torch.sqrt(torch.clamp(vector_wise_dists, min=0))

        return l2_distance_matrix

    def gaussian_kernel_function(distance_matrix, scale=0.05):
        norm_score = torch.exp(-distance_matrix / (2 * (scale ** 2)))
        return norm_score

    l2_dm = l2_distance_matrix(x) # squared_l2_distance_matrix() is too sensitive.
    norm_score = gaussian_kernel_function(l2_dm, scale=scale)

    return norm_score

def concatenate_fusion(*args):
    fused_feature = args[0]
    for arg in args[1:]:
        fused_feature = torch.cat((fused_feature, arg), axis = 1)
    return fused_feature


def ssm_fusion(ssm_1, ssm_2, nssm_1, nssm_2, k, t):
    def kneighbors(matrix, length, k):
        neighbors = np.zeros((length, k))

        neigh = NearestNeighbors(n_neighbors=k, p=1)
        for i, vector in enumerate(matrix):
            v = torch.unsqueeze(vector, axis=1)
            neigh.fit(v)
            neighbor = neigh.kneighbors([[vector[i]]], return_distance=False)
            neighbors[i] = neighbor.squeeze()

        neighbors = torch.from_numpy(neighbors).to(torch.long)
        return neighbors

    length = ssm_1.shape[0]

    skm_1 = torch.zeros((length, length), dtype=torch.float32)  # km means kernel matrix
    skm_2 = torch.zeros((length, length), dtype=torch.float32)

    f1_neighbors = kneighbors(ssm_1, length, k)
    f2_neighbors = kneighbors(ssm_2, length, k)

    # 1st feature based sparse kernel matrix construction
    for i in range(length):
        f1_ith_neighs = f1_neighbors[i]
        f1_elements = ssm_1[i][f1_ith_neighs]
        skm_1[i, f1_ith_neighs] = f1_elements

        f2_ith_neighs = f2_neighbors[i]
        f2_elements = ssm_2[i][f2_ith_neighs]
        skm_2[i, f2_ith_neighs] = f2_elements

    skm_1 = symmetric_normalization(skm_1 + skm_1.T - torch.diag(skm_1.diagonal()))
    skm_2 = symmetric_normalization(skm_2 + skm_2.T - torch.diag(skm_2.diagonal()))

    # make normalized weight matrices by iterating t times

    for _t in range(t):
        temp = nssm_1.clone()
        nssm_1 = torch.matmul(torch.matmul(skm_1, nssm_2.clone()), skm_1)
        nssm_2 = torch.matmul(torch.matmul(skm_2, temp), skm_2)

        if (t > 1 and _t < t - 1):
            nssm_1 = symmetric_normalization(nssm_1)
            nssm_2 = symmetric_normalization(nssm_2)

    fused_ssm = (nssm_1 + nssm_2) / 2

    return fused_ssm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
