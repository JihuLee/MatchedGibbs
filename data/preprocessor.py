import torch
from data.registry import get_preprocessor
import numpy as np


class SharedPreprocessor:
    @staticmethod
    def preprocess(args, seed: int) -> dict:
        preprocess_data = get_preprocessor(args.dataset)
        if args.dataset in ['crime']:
            seed = np.random.choice(np.arange(1, 11))
            binarize = args.task_loss_func == 'bce'
            x_train, x_test, y_train, y_test, s_train, s_test = preprocess_data(seed, binarize=binarize)
        else:
            x_train, x_test, y_train, y_test, s_train, s_test = preprocess_data(seed)
        device = args.device
        return {
            'train': {
                'x': torch.from_numpy(x_train).to(torch.float32).to(device),
                'y': torch.from_numpy(y_train).to(torch.float32).to(device),
                's': torch.from_numpy(s_train).to(torch.float32).to(device),
            },
            'test': {
                'x': torch.from_numpy(x_test).to(torch.float32).to(device),
                'y': torch.from_numpy(y_test).to(torch.float32).to(device),
                's': torch.from_numpy(s_test).to(torch.float32).to(device),
            }
        }