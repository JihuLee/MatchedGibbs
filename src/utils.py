import torch
import numpy as np

SUBSAMPLE_SIZE = 2000
SCALED = False

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    else:
        return obj

def make_path(args, mode, pretrain, lmda=None):
    path = args.save_dir + mode
    if mode == 'pretrain':
        path += f"/"
    elif mode == 'gibbs':
        path += f"/pretrain={pretrain}/lr_gibbs={args.lr_gibbs},lmda={args.lmda_init},step_size={args.step_size},num_burn={args.num_burn}/"
    elif mode == 'matched_gibbs':
        path += f"/pretrain={pretrain}/lr_gibbs={args.lr_gibbs},lmda={args.lmda_init},step_size={args.step_size},permute_size={args.permute_size},num_burn={args.num_burn}/"
    elif mode == 'variational':
        path += f"/lmda={lmda}/"

    return path

def select_indices_with_thinning(arr, k, m):
    n = len(arr)
    thinning_indices = [i for i in range(n) if i % k == k - 1]

    if len(thinning_indices) < m:
        additional_needed = m - len(thinning_indices)
        additional_indices = list(range(n - 1, -1, -1))
        additional_indices = [i for i in additional_indices if i not in thinning_indices][:additional_needed]
        thinning_indices.extend(reversed(additional_indices))

    return np.array(thinning_indices)