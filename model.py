import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from functionals import symmetric_normalization


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels,
                 bias=False
                 ):
        super(GraphConvolution, self).__init__()

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.torch.empty(out_channels))
        #         else:
        #             self.register_parameter('bias',None)
        self.reset_parameters()

    def forward(self, x, adj):
        out = torch.mm(x, self.weight)
        out = torch.mm(adj, out)

        if self.bias is not None:
            out += self.bias
        return out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class GraphConvolutionalEncoder(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, activation, base_model=GraphConvolution):
        super(GraphConvolutionalEncoder, self).__init__()
        self.base_model = base_model

        self.gcn1 = base_model(in_channels, hid_channels, bias = True)
        self.gcn2 = base_model(hid_channels, out_channels, bias = True)

        self.activation = activation

    def forward(self, x: torch.Tensor, edges: torch.Tensor):
        if x.data.dim() == 3:
            x = x.squeeze()
        x1 = self.activation(self.gcn1(x, edges))
        x2 = self.activation(self.gcn2(x1, edges))
        return x2

class GRACE(nn.Module):
    def __init__(self, encoder: GraphConvolutionalEncoder, num_in: int, num_hidden: int, num_proj_hidden: int, num_out: int,
                 tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder = encoder
        self.tau: float = tau

        self.adj = None
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

        self.gcn_classifier = GraphConvolution(num_hidden, num_out, bias=False)
        self.fc_classifier = nn.Linear(num_hidden, num_out)
        self.activation = nn.CELU()
        self.num_hidden = num_hidden
        self.norm1d = nn.BatchNorm1d(num_in)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        #         x = self.norm1d(x)
        self.adj = edges
        self.adj.requires_grad_(True)
        x = x.unsqueeze(dim=0)
        return self.encoder(x, self.adj)

    def decoder(self, z):
        return self.activation(torch.mm(z, z.t()))

    def classification(self, z: torch.Tensor) -> torch.Tensor:
        #         return self.gcn_classifier(self.activation(z), adj)
        return self.fc_classifier(self.activation(z))

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, h1: torch.Tensor, h2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


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


def learner(id, model, optimizer, feature, adj, label, train_identifier, test_identifier, args, isdeap=False,
            verbose=True, earlystop=False):
    def accuracy(out, label, isbinary=False):
        out = torch.sigmoid(out) if isbinary == True else nn.functional.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        iscorrect = torch.eq(pred, label).float()
        return iscorrect.mean() * 100

    def disc_rank(feature, label, identifier, n_classes):

        X = feature[identifier]
        y = label[identifier]
        n_dims = X.shape[1]

        Sw = torch.zeros(n_dims)
        Sb = torch.zeros(n_dims)
        for i in range(n_dims):
            w = 0
            wa = 0

            global_mean_x = torch.mean(X[:, i])
            for j in range(n_classes):
                xc = X[torch.where(y == j)[0], i]
                mean_xc = torch.mean(xc)
                a = xc - mean_xc
                w += torch.dot(a, a)

                wa += torch.pow(mean_xc - global_mean_x, 2)
            Sb[i] = wa
            Sw[i] = w

        # Sw low, Sb high --> important feauture dimension
        disc_power = Sb / Sw
        # disc_power high --> important feature dimension

        max_power = disc_power.max()
        average_power = disc_power.mean()
        rank = (max_power - disc_power) / (max_power - average_power)
        # rank high -> unimportant feature dimension --> can be masked by high probablity

        return rank

    def edge_rank(edge_weights):
        weight_max = edge_weights.max()
        edge_weights_mean = edge_weights.mean()
        weights = (weight_max - edge_weights) / (weight_max - edge_weights_mean)
        return weights

    def drop_features(probability_weights, features, thresholds, threshold: float = 1.):
        probability_weights = probability_weights.where(probability_weights < threshold,
                                                        thresholds)
        drop_mask = torch.bernoulli(probability_weights).to(torch.bool)

        features_view = features.clone()
        features_view[:, drop_mask] = 0.

        return features_view

    def drop_edges(probability_weights, edge_weights, thresholds, zeros, threshold: float = 1.):
        probability_weights = probability_weights.where(probability_weights < threshold,
                                                        thresholds)
        drop_mask = torch.bernoulli(1. - probability_weights).to(torch.bool)

        edge_weights_view = edge_weights.where(drop_mask == True, zeros)

        return edge_weights_view

    best_acc = 0
    out_trigger = 0
    best_epoch = 0
    best_z = None

    device = feature.device
    # im = torch.eye(args.n_samples).to(device=device) # 이 부분에서 문제 발생 1512 -> 24x63 만족 해야함
    im = torch.eye(feature.shape[0]).to(device=device)

    thresholds = torch.ones_like(adj).to(device=device)
    feature_thresholds = torch.ones_like(feature[1]).to(device=device)
    zeros = torch.zeros_like(adj).to(device=device)

    rankf = disc_rank(feature, label, train_identifier, args.out_channels).to(device=device)
    rankf1 = rankf * args.pf1
    rankf2 = rankf * args.pf2
    ranke = edge_rank(adj)
    ranke1 = ranke * args.pe1
    ranke2 = ranke * args.pe2

    if earlystop:
        save_path = args.model_save_path + 'subject_independent_' + id + '.pt'
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=save_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1 = drop_features(rankf1, feature, feature_thresholds * args.pt, threshold=args.pt)
        x2 = drop_features(rankf2, feature, feature_thresholds * args.pt, threshold=args.pt)
        e1 = drop_edges(ranke1, adj, thresholds * args.pt, zeros, threshold=args.pt)
        e2 = drop_edges(ranke2, adj, thresholds * args.pt, zeros, threshold=args.pt)

        e1 = symmetric_normalization(e1, im)
        e2 = symmetric_normalization(e2, im)

        z1 = model(x1, e1)
        z1 = model.projection(z1)
        z2 = model(x2, e2)
        z2 = model.projection(z2)

        r_y = label[train_identifier]

        r1 = model.classification(z1)
        r1_pred = r1[train_identifier]
        labeled_loss1 = F.cross_entropy(r1_pred, r_y)
        r1_acc = accuracy(r1_pred, r_y, isdeap)

        r2 = model.classification(z2)
        r2_pred = r2[train_identifier]
        labeled_loss2 = F.cross_entropy(r2_pred, r_y)
        r2_acc = accuracy(r2_pred, r_y, isdeap)

        labeled_loss = (labeled_loss1 + labeled_loss2) / 2.

        contrastive_loss = model.loss(z1, z2)

        loss = labeled_loss + contrastive_loss * args.cl_coefficient

        loss.backward()
        optimizer.step()

        acc = (r1_acc + r2_acc) / 2.

        tr_y = label[test_identifier]
        tr1_pred = r1[test_identifier]
        tr1_loss = F.cross_entropy(tr1_pred, tr_y)
        tr1_acc = accuracy(tr1_pred, tr_y, isdeap)

        tr2_pred = r2[test_identifier]
        tr2_acc = accuracy(tr2_pred, tr_y, isdeap)
        tr2_loss = F.cross_entropy(tr2_pred, tr_y)

        tr_acc = tr1_acc if tr1_acc > tr2_acc else tr2_acc

        tr_loss = (tr1_loss + tr2_loss) / 2.

        if tr_acc > best_acc:
            best_acc = tr_acc
            best_epoch = epoch

            if tr1_acc > tr2_acc:
                best_pred = tr1_pred
                best_z = z1
            else:
                best_pred = tr2_pred
                best_z = z2

        if best_acc == 100.0:
           out_trigger = 1
           break

        if earlystop:
            early_stopping(tr_loss, model)
            if early_stopping.early_stop:
                out_trigger = 2
                break

        if verbose == True:
            if epoch % 100 == 0:
                print(
                    "Epoch {} - Train Acc : {}    Train Loss : {},    Test Acc : {},    Test Loss :{},    Total Acc : {}".format(
                        epoch, round(acc.item(), 2), round(loss.item(), 2), round(tr_acc.item(), 2),
                        round(tr_loss.item(), 2), round(total_acc.item(), 2)))


    cfm = confusion_matrix(tr_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(),
                           normalize='true')
    # f1_micro = f1_score(tr_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(), average='micro')
    # f1_macro = f1_score(tr_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(), average='macro')
    print(classification_report(tr_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(), target_names=['기쁨', '중립', '불안', '당황', '상처', '슬픔', '분노']))
    report = classification_report(tr_y.cpu().numpy(), best_pred.max(1, keepdim=True)[1].detach().cpu().numpy(), target_names=['기쁨', '중립', '불안', '당황', '상처', '슬픔', '분노'],output_dict=True)
    return best_acc, best_z, cfm, report, out_trigger, best_epoch
