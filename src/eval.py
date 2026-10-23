



import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import calibration as cal
from sklearn.metrics import brier_score_loss
from sklearn.neighbors import NearestNeighbors
from src.loss import _fair_loss_dp_true, _fair_loss_wdp, _fair_loss_dp, _fair_loss_sdp

def _eval(model, x, y, s, args, matching=None):
    model.eval()
    with torch.no_grad():
        y = y.flatten()
        logits = model(x)
        probs = nn.Sigmoid()(logits).flatten().cpu()

    preds = torch.round(probs)
    acc = (preds == y.cpu()).float().mean().item()
    if args.constraint_loss_func == 'dp':
        fair = _fair_loss_dp_true(probs, s.cpu()).item()
    elif args.constraint_loss_func == 'wdp':
        fair = _fair_loss_wdp(logits, s.cpu()).item()
    else:
        raise NotImplementedError()
        
    return acc, fair


def _eval_mfg(model, x, y, s, args, matching=None, num_samples=5):
    
    model.eval()
    
    with torch.no_grad():
        
        samples = []
        for _ in range(num_samples):
            logits, _ = model(x.to(args.device))
            logits = torch.sigmoid(logits).flatten()
            samples.append(logits.unsqueeze(0))
        logits = torch.cat(samples).mean(0).cpu()
        
        probs = logits.flatten().cpu()

    preds = torch.round(probs)
    acc = (preds == y.flatten().cpu()).float().mean().item()
    fair = _fair_loss_wdp(logits, s.cpu()).item()
    
    return acc, fair


def evaluate(probs, features, targets, sens, args, matchings=None, calc_prop=False):
    
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    if isinstance(sens, np.ndarray):
        sens = torch.from_numpy(sens)
        
    if targets.dtype != torch.float32 and targets.dtype != torch.float64:
        targets = targets.float()
        
    if isinstance(probs, list):
        probs = torch.mean(torch.stack(probs), dim=0)
    elif isinstance(probs, torch.Tensor) and probs.dim() > 1:
        probs = torch.mean(probs, dim=0)
        
    probs_clipped = torch.clamp(probs, 1e-7, 1 - 1e-7)
    logits = torch.logit(probs_clipped)
    preds = torch.round(probs)
    
    if args.constraint_eval == 'dp':
        dp_true_array, dp_array, wdp_array, sdp_array, ksdp_array = [], [], [], [], []
        if probs.dim() > 1:
            N = probs.shape[0]
            for i in range(N):
                prob = torch.clamp(probs[i], 1e-7, 1-1e-7)
                logit = torch.logit(prob)
                
                
                sdp, ksdp = _fair_loss_sdp(logit, sens)
                dp_true_array.append(_fair_loss_dp_true(prob, sens).item())
                dp_array.append(_fair_loss_dp(logit, sens).item())
                
                wdp_array.append(_fair_loss_wdp(probs[i], sens).item())
                sdp_array.append(sdp.item())
                ksdp_array.append(ksdp.item())

                fairness = (
                    np.mean(dp_true_array), 
                    np.mean(dp_array), 
                    np.mean(wdp_array), 
                    np.mean(sdp_array), 
                    np.mean(ksdp_array), 
                )
        else:
            prob = torch.clamp(probs, 1e-7, 1-1e-7)
            logit = torch.logit(prob)
            
            
            sdp, ksdp = _fair_loss_sdp(logit, sens)
            dp_true_array.append(_fair_loss_dp_true(prob, sens).item())
            dp_array.append(_fair_loss_dp(logit, sens).item())
            
            wdp_array.append(_fair_loss_wdp(probs, sens).item())
            sdp_array.append(sdp.item())
            ksdp_array.append(ksdp.item())

            fairness = (
                np.mean(dp_true_array), 
                np.mean(dp_array), 
                np.mean(wdp_array), 
                np.mean(sdp_array), 
                np.mean(ksdp_array), 
            )
            
    if args.task_loss_func == 'bce':
        acc = (preds == targets).float().mean().item()
        utility = (acc, )
        
    if args.task_loss_func == 'bce':
        
        criterion = torch.nn.BCELoss(reduction='mean')
        nll = criterion(probs, targets).item()
        
        probs_np = probs.detach().cpu().numpy()
        targets_int = targets.cpu().numpy().astype(int)
        
        ece = cal.lower_bound_scaling_ce(probs_np, targets_int, p=1, debias=False, num_bins=15, binning_scheme=cal.get_equal_bins, mode='marginal')
        brier = brier_score_loss(targets_int, probs_np)
        nbrs = NearestNeighbors(n_neighbors=5).fit(features)
        distances, indices = nbrs.kneighbors(features)
        con = (1 - torch.mean(torch.abs(preds - preds[indices].sum(axis = 1)/5))).item()

        uncertainty = (nll, ece, brier, con)
    else:
        raise NotImplementedError()
        
    return utility, uncertainty, fairness, probs


def get_nll_crps_regression(probs, targets, sigma_min=1e-6, return_tensors=False):
    if isinstance(probs, list):
        probs_tensor = torch.stack(probs)
        mean_pred = probs_tensor.mean(dim=0)
        std_pred = probs_tensor.std(dim=0, unbiased=False)
        samples = probs_tensor.permute(1, 0)
    else:
        mean_pred = probs
        std_pred = torch.zeros_like(mean_pred)  
        samples = None
    
    
    sigma = std_pred.clamp_min(sigma_min)
    var = sigma ** 2

    log2pi = math.log(2.0 * math.pi)
    nll_tensor = 0.5 * (log2pi + torch.log(var) + (targets - mean_pred) ** 2 / var)
    nll = nll_tensor.mean()

    
    if samples is not None:
        
        
        term1 = (samples - targets.unsqueeze(1)).abs().mean(dim=1)
        
        B, M = samples.shape
        diffs = (samples.unsqueeze(1) - samples.unsqueeze(2)).abs()
        term2 = diffs.sum(dim=(1, 2)) / (M * (M - 1)) if M > 1 else 1.0
        crps_tensor = term1 - 0.5 * term2
        crps = crps_tensor.mean()
    else:
        
        crps = (mean_pred - targets).abs().mean()
    
    if return_tensors:
        return nll, crps
    else:
        return nll.item(), crps.item()



def get_sufficiency_gap(probs, y_true, s, n_bins=15, agg='mean'):
    '''
    agg: 'mean' -> sufficiency gap, 
         'max' -> worst-case sufficiency gap
    '''
    edges = np.linspace(0., 1., n_bins + 1)
    bin_edges = np.digitize(probs, edges[1:-1])
    
    gap_sum, weight_sum, max_gap = 0.0, 0, 0.0

    for b in range(n_bins):
        idx_bin = bin_edges == b
        n_bin = idx_bin.sum()
        if n_bin == 0:
            continue

        idx_b0 = idx_bin & (s == 0)
        idx_b1 = idx_bin & (s == 1)
        
        r_b0 = y_true[idx_b0].mean() if idx_b0.any() else y_true[idx_bin].mean()
        r_b1 = y_true[idx_b1].mean() if idx_b1.any() else y_true[idx_bin].mean()
        
        gap = abs(r_b0 - r_b1)
        gap_sum += gap * n_bin
        weight_sum += n_bin
        max_gap = max(max_gap, gap)
    
    sufficiency_gap = gap_sum / weight_sum if weight_sum > 0 else 0.0
    if agg == 'mean':
        return sufficiency_gap
    elif agg == 'max':
        return max_gap
    else:
        raise ValueError(f"Unknown agg: {agg}")


def get_group_calibration_gap(probs, y_true, s):
    groups = np.unique(s)
    score_per_group = []

    for group in groups:
        mask = s == group
        probs_group = probs[mask]
        y_true_group = y_true[mask]

        if len(probs_group) == 0:
            score_per_group.append(0.0)
        else:
            score = cal.get_ece(probs_group, y_true_group)
            score_per_group.append(score)

    group_cal_gap = np.max(score_per_group) - np.min(score_per_group)
    return group_cal_gap

'''
Temperature scaling
'''
def temperature_scale(preds, T):
    temperature = nn.Parameter(torch.ones(1, device=preds.device) * T)
    temperature = temperature.unsqueeze(1).expand(preds.size(0), preds.size(1))

    return preds / temperature

def set_temperature(preds: torch.Tensor,
                    labels: torch.Tensor,
                    init_T: float = 1.5,
                    max_iter: int = 50,
                    lr: float = 0.01) -> float:
    """
    검증용 logits과 레이블을 입력받아, NLL을 최소화하는 최적의 온도 T를 찾습니다.
    
    Args:
        preds: torch.Tensor of shape [N, C], 모델이 출력한 logits
        labels: torch.LongTensor of shape [N], 정답 레이블
        init_T: 초기 온도 값
        max_iter: LBFGS 최대 반복 횟수
        lr: LBFGS 러닝레이트

    Returns:
        최적화된 온도 스칼라 (float)
    """
    
    T = nn.Parameter(torch.ones(1, device=preds.device) * init_T)
    
    
    nll_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([T], lr=lr, max_iter=max_iter)

    
    def _eval():
        optimizer.zero_grad()
        scaled_logits = temperature_scale(preds, T.item())
        loss = nll_criterion(scaled_logits, labels.float())
        loss.backward()
        return loss

    
    optimizer.step(_eval)

    
    return T.item()

    
    
    
    
    
    
    
    

    
    
    

    
    
    
    