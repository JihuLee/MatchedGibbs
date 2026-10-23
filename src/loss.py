import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import ot
from scipy.stats import wasserstein_distance



UNPREV = 0

def _loss(
        x, y, s, 
        model, 
        constraint_func, 
        args
):
    model.train()
    
    x = x.to(args.device)
    y = y.flatten().to(args.device)
    s = s.flatten().to(args.device)

    preds = model(x).flatten()

    
    if args.task_loss_func == 'bce':
        task_loss = _task_loss_bce(preds, y)
    
    
    if constraint_func == 'dp':
        constraint_loss = _fair_loss_dp(preds, s)
    elif constraint_func == 'mmd':
        constraint_loss = _fair_loss_mmd(preds, s)
    elif constraint_func == 'mdp':
        x_s0 = x[s == 0]; x_s1 = x[s == 1]
        y_s0 = y[s == 0]; y_s1 = y[s == 1]
        matching = get_batch_matching(x_s0, x_s1, y_s0, y_s1, args.margin)
        constraint_loss = _fair_loss_mdp(preds, matching, s, use_logits=True, use_clamp=False)
    elif constraint_func == 'wdp':
        constraint_loss = _fair_loss_wdp(preds, s)
    else:
        raise NotImplementedError()
    
    return task_loss, constraint_loss


def _task_loss_bce(preds, y):
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    bceloss = criterion(preds, y)

    return bceloss

def _fair_loss_dp(preds, s):
    
    s = s.flatten().type(torch.int)

    y_pred = torch.sigmoid(preds).flatten()
    y_pred0, y_pred1 = y_pred[s == UNPREV], y_pred[s == 1]
    
    score0 = y_pred0.mean() if len(y_pred0) > 0 else 0.0
    score1 = y_pred1.mean() if len(y_pred1) > 0 else 0.0

    dp = score0 - score1
    dp = abs(dp)
    
    return dp

def _fair_loss_dp_true(y_pred, s):
    s = s.flatten().type(torch.int)

    y_pred = y_pred.flatten().cpu().detach().numpy()
    y_pred0, y_pred1 = y_pred[s == UNPREV], y_pred[s == 1]
    
    score0 = (y_pred0 > 0.5).mean() if len(y_pred0) > 0 else 0.0
    score1 = (y_pred1 > 0.5).mean() if len(y_pred1) > 0 else 0.0

    dp = score0 - score1
    dp = abs(dp)

    return dp

def wasserstein_2_distance(preds0, preds1, n_quantiles=1000):
    
    preds0_sorted = torch.sort(preds0).values
    preds1_sorted = torch.sort(preds1).values
    qs = torch.linspace(0, 1, n_quantiles, device=preds0.device)
    
    idx0 = (len(preds0_sorted) - 1) * qs
    idx0_floor = torch.floor(idx0).long()
    idx0_ceil = torch.ceil(idx0).long()
    alpha = idx0 - idx0_floor.float()
    
    q_u = preds0_sorted[idx0_floor] * (1 - alpha) + preds0_sorted[torch.min(idx0_ceil, torch.tensor(len(preds0_sorted)-1, device=preds0.device))] * alpha
    
    idx1 = (len(preds1_sorted) - 1) * qs
    idx1_floor = torch.floor(idx1).long()
    idx1_ceil = torch.ceil(idx1).long()
    alpha = idx1 - idx1_floor.float()
    
    q_v = preds1_sorted[idx1_floor] * (1 - alpha) + preds1_sorted[torch.min(idx1_ceil, torch.tensor(len(preds1_sorted)-1, device=preds1.device))] * alpha
    
    return torch.sqrt(torch.mean((q_u - q_v) ** 2))

def _fair_loss_wdp(preds, s):
    preds0 = preds[s == UNPREV]
    preds1 = preds[s == 1]
    
    
    if len(preds0) == 0 or len(preds1) == 0:
        return torch.tensor(0.0, device=preds.device)
        
    return wasserstein_2_distance(preds0, preds1)

def _fair_loss_sdp(preds, s):
    preds0 = preds[s == UNPREV]
    preds1 = preds[s == 1]
    taus = np.linspace(0, 1, 10)
    dps = []
    for tau in taus:
        tau_dp = (preds0 > tau).float().mean() - (preds1 > tau).float().mean()
        dps.append(abs(tau_dp))
    sdp = np.mean(dps)
    ksdp = np.max(dps)

    return sdp, ksdp


def _Gaussian_kernel_matrix(Xi, Xj, sigma=1.0):
    if Xi.dim() == 1:
        Xi = Xi.unsqueeze(1)
    if Xj.dim() == 1:
        Xj = Xj.unsqueeze(1)
        
    matrix = - torch.cdist(Xi, Xj, p=2)**2
    matrix /= (2.0 * sigma**2)
    matrix = torch.exp(matrix)
    return matrix


def _compute_MMD(source_reps, target_reps, sigmas=[1.0]):
    mmd = 0.0
    for sigma in sigmas:
        KXX = _Gaussian_kernel_matrix(source_reps, source_reps, sigma=sigma)
        mmd += KXX.mean()
        del KXX
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        KXY = _Gaussian_kernel_matrix(source_reps, target_reps, sigma=sigma)
        mmd -= 2 * KXY.mean()
        del KXY
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        KYY = _Gaussian_kernel_matrix(target_reps, target_reps, sigma=sigma)
        mmd += KYY.mean()
        del KYY
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return mmd.mean()


def _fair_loss_mmd(preds, s):
    s = s.flatten().long()
    preds = preds.flatten()
    
    preds_s0 = preds[s == UNPREV]
    preds_s1 = preds[s == 1]
    
    mmd = _compute_MMD(preds_s0, preds_s1)
    return mmd


@torch.no_grad()
def get_batch_matching(x_s0, x_s1, y_s0=None, y_s1=None, margin=0.):
    x_s0_np = x_s0.cpu().detach().numpy()
    x_s1_np = x_s1.cpu().detach().numpy()
    
    if y_s0 is not None:
        y_s0_np = y_s0.cpu().detach().numpy().reshape(-1, 1)
    if y_s1 is not None:
        y_s1_np = y_s1.cpu().detach().numpy().reshape(-1, 1)
    
    s0_weight = np.ones(x_s0_np.shape[0]) / x_s0_np.shape[0]
    s1_weight = np.ones(x_s1_np.shape[0]) / x_s1_np.shape[0]
    
    M = ot.dist(x_s0_np, x_s1_np, metric='euclidean')
    if y_s0 is not None:
        M += margin * ot.dist(y_s0_np, y_s1_np, metric='minkowski', p=1)
    
    try:
        G = ot.emd(s0_weight, s1_weight, M)
        matching = torch.argmax(torch.tensor(G, device=x_s0.device), dim=1)
        return matching
    except Exception as e:
        print(f"Error in OT computation: {e}")
        return torch.randint(0, x_s1.size(0), (x_s0.size(0),), device=x_s0.device)
    

def _fair_loss_mdp(logits, matching, s, use_logits=True, use_clamp=True):
    if use_logits: 
        if use_clamp:
            probs = torch.clamp(torch.sigmoid(logits).squeeze(), 0.2, 0.8)
            res = probs
        else:
            res = torch.sigmoid(logits.squeeze())
    else: 
        if use_clamp:
            probs = torch.clamp(logits.squeeze(), 0.2, 0.8)
            res = probs
        else:
            res = logits.squeeze()
    
    s0_mask = (s == 0)
    s1_mask = (s == 1)
    
    if not s0_mask.any() or not s1_mask.any():
        return torch.tensor(0.0, device=logits.device)
    s0_res = res[s0_mask]
    matched_s1_res = res[s1_mask][matching]
    fair_loss = (s0_res - matched_s1_res).pow(2).mean()
    
    return fair_loss
    


class ConstraintLoss(nn.Module):
    def __init__(self, device, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        
        self.device = device
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons