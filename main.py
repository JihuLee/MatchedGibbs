
import os
import torch
import argparse
import random
import numpy                as np
import pandas               as pd
import json
from argparse               import BooleanOptionalAction as BOA
from src.runner_bayesian    import BayesianRunner, GibbsRunner, MatchedGibbsRunner
from src.runner_variational import VariationalRunner
from src.utils              import make_path
from data.preprocessor      import SharedPreprocessor

parser = argparse.ArgumentParser(description='Fair Bayesian Inference')

parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--unprev', type=int, default=0)

parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--rep_dim', type=int, default=200)

parser.add_argument('--start_seed', type=int, default=2025)
parser.add_argument('--num_repeats', type=int, default=5)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--thres', type=float)
parser.add_argument('--clip_grad', default=False, action=BOA)

parser.add_argument('--task_loss_func', type=str, choices=['bce'], default='bce')
parser.add_argument('--constraint_loss_func', type=str, choices=['dp', 'mmd', 'mdp'], default='dp')
parser.add_argument('--constraint_eval', type=str, choices=['dp'], default='dp')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lmda_init', type=float)

parser.add_argument('--pretrain_seed', type=int, default=2025)
parser.add_argument('--lmda_grid', type=float, nargs='+', default=[0.0, 1.0, 11], help='lambda grid in form of (lower, upper, num_grid)')
parser.add_argument('--epochs_pre', type=int, default=5)

parser.add_argument('--use_pretrained', default=True, action=BOA)
parser.add_argument('--num_samples', type=int, default=5)
parser.add_argument('--num_burn', type=int, default=3000)
parser.add_argument('--thin_interval', type=int, default=10)
parser.add_argument('--L', type=int, default=1)
parser.add_argument('--grad_clip', type=float, default=1.)
parser.add_argument('--step_size', type=float, default=1e-3)
parser.add_argument('--mc_runs', type=int, default=5)

parser.add_argument('--variational_epochs', type=int, default=100)
parser.add_argument('--best_by_val', default=False, action=BOA)

parser.add_argument('--lr_gibbs', type=float, default=1.)
parser.add_argument('--sample_lmda', default=False, action=BOA)
parser.add_argument('--step_size_lmda', type=float, default=1e-3)
parser.add_argument('--constraint', default=False, type=BOA)
parser.add_argument('--tau', type=float, default=1.0, help='temperature on the prior for the matching')
parser.add_argument('--init_matching', type=str, choices=['ot'], default='ot')
parser.add_argument('--permute_size', type=int, default=10)

parser.add_argument('--ot_eps', type=float, default=None)
parser.add_argument('--ot_max_iter', type=int, default=1000)
parser.add_argument('--ot_stop_thr', type=float, default=1e-6)
parser.add_argument('--margin', type=float, default=1e10)
parser.add_argument('--large_value', type=float, default=1e10)

def parse_mode(mode):
    # 'gibbs_matching' is the old name of 'matched_gibbs'; kept so existing scripts still run
    if mode == 'gibbs_matching':
        return 'matched_gibbs'
    return mode

parser.add_argument(
    '--mode', type=parse_mode, 
    choices=['pretrain','matched_gibbs','gibbs', 'variational'], 
    nargs='+', 
    required=True
)
parser.add_argument('--save_dir', type=str, help='e.g. save/adult/')


args = parser.parse_args()
args.device = 'cuda:' + str(args.device)
cnt_eval = len(args.mode) - args.mode.count('pretrain')
print("Start running with the following arguments:")
for key, value in vars(args).items():
    if key in ['dataset', 'mode', 'lmda_init', 'epochs_pre', 'step_size', 'device']:
        print(f'[{key}]: {value}')
    else:
        continue

args = parser.parse_args()
args.device = 'cuda:' + str(args.device)
cnt_eval = len(args.mode) - args.mode.count('pretrain')

if (args.constraint_eval == 'dp')&(args.task_loss_func == 'bce'): 
    MEASURES = {
        'utility': ['ACC'], 
        'uncertainty': ['NLL', 'ECE', 'brier', 'con'], 
        'fairness': ['DP', 'meanDP', 'WDP', 'SDP', 'KSDP']
    }
else:
    raise NotImplementedError()
NUM_RES = {'utility': len(MEASURES['utility']), 'uncertainty': len(MEASURES['uncertainty']), 'fairness': len(MEASURES['fairness'])}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.runner_maps = {
            'gibbs':            (GibbsRunner,           ('sample', 'eval')), 
            'matched_gibbs':    (MatchedGibbsRunner,    ('sample', 'eval')), 
            'variational':      (VariationalRunner,     ('train', 'eval')),
        }
        
        self.args.lmda_grid[-1] = int(args.lmda_grid[-1])
        measures_index = MEASURES['utility'] + MEASURES['uncertainty'] + MEASURES['fairness']
        self.measure_table = {
            mode: {
                'train':    pd.DataFrame(None, index=measures_index, columns=['mean', 'se'], dtype=float), 
                'test':     pd.DataFrame(None, index=measures_index, columns=['mean', 'se'], dtype=float)
            } for mode in args.mode}
        self.pretrain_measure_table = {
            round(lmda, 2): {
                'train':    pd.DataFrame(None, index=measures_index, columns=['value'], dtype=float), 
                'test':     pd.DataFrame(None, index=measures_index, columns=['value'], dtype=float)
            } for lmda in np.linspace(*self.args.lmda_grid)}
        self.result = {
            mode: {
                'train':    {k: {} for k in ['utility', 'uncertainty', 'fairness']}, 
                'test':     {k: {} for k in ['utility', 'uncertainty', 'fairness']}
            } for mode in args.mode}
        self.pretrain_result = {
            round(lmda, 2): {
                'train':    {k: {} for k in ['utility', 'uncertainty', 'fairness']}, 
                'test':     {k: {} for k in ['utility', 'uncertainty', 'fairness']}
            } for lmda in np.linspace(*self.args.lmda_grid)}
        
        for mode in args.mode:
            for m in ['train', 'test']:
                for k in ['utility', 'uncertainty', 'fairness']:
                    for i in range(NUM_RES[k]):
                        self.result[mode][m][k][i] = []

    def run(self):
        if 'pretrain' in self.args.mode:
            seed = self.args.pretrain_seed
            set_global_seed(seed)
            dataset = SharedPreprocessor.preprocess(self.args, seed)
            runner = BayesianRunner(self.args, seed, dataset, MEASURES)
            runner.pretrain()

            res = runner.eval_pretrain()
            for lmda in self.pretrain_result.keys():
                for m in ['train', 'test']:
                    for k in ['utility', 'uncertainty', 'fairness']:
                        for i in range(NUM_RES[k]):
                            self.pretrain_result[lmda][m][k][i] = res[lmda][m][k][i]
            pretrain_save_dir = self.args.save_dir + f"pretrain/"
            if not os.path.exists(pretrain_save_dir):
                os.makedirs(pretrain_save_dir)
            with open(pretrain_save_dir + f"pretrain_result_seed={self.args.pretrain_seed}.json", "w") as f:
                json.dump(self.pretrain_result, f)

        if cnt_eval > 0:
            for seed in range(self.args.start_seed, self.args.start_seed + self.args.num_repeats):
                set_global_seed(seed)
                dataset = SharedPreprocessor.preprocess(self.args, seed)

                for mode in self.args.mode:
                    set_global_seed(seed)
                    Runner, methods = self.runner_maps.get(mode)
                    if Runner is None:
                        raise ValueError(f"Unknown mode: {mode}")
                        
                    runner = Runner(self.args, seed, dataset)
                    for m_ind in range(len(methods)):
                        if m_ind < len(methods) - 1:
                            getattr(runner, methods[m_ind])()
                        else:
                            if mode in ['variational']: 
                                res, train_elbo = getattr(runner, methods[m_ind])()
                                print(f"Train ELBO: {train_elbo}")
                            else:
                                res = getattr(runner, methods[m_ind])()
                            for m in ['train', 'test']:
                                for k in ['utility', 'uncertainty', 'fairness']:
                                    for i in range(NUM_RES[k]):
                                        self.result[mode][m][k][i].append(res[m][k][i])

    def print_res(self):
        for mode in self.args.mode:
            if mode != 'pretrain':
                print(f'\n+++ {mode} evaluation +++')
            for m in ['train', 'test']:
                for k in ['utility', 'uncertainty', 'fairness']:
                    for i in range(NUM_RES[k]):
                        if mode != 'pretrain':
                            self.result[mode][m][k][i] = np.stack(self.result[mode][m][k][i])
                            self.measure_table[mode][m].loc[MEASURES[k][i]] = [
                                np.mean(self.result[mode][m][k][i]), np.std(self.result[mode][m][k][i]) / np.sqrt(self.args.num_repeats)
                            ]
                if mode == 'pretrain':
                    for lmda in self.pretrain_result.keys():
                        for k in ['utility', 'uncertainty', 'fairness']:
                            for i in range(NUM_RES[k]):
                                self.pretrain_measure_table[lmda][m].loc[MEASURES[k][i]] = self.pretrain_result[lmda][m][k][i]
                
                pretrain_epochs = self.args.epochs_pre
                
                if mode == 'variational':
                    lmda = self.args.lmda_init
                else:
                    lmda = None
                if mode != 'pretrain':
                    print(f'+++ {m} +++')
                    for k in ['utility', 'uncertainty', 'fairness']:
                        for i in range(NUM_RES[k]):
                            print(f"{MEASURES[k][i]}: {self.measure_table[mode][m].loc[MEASURES[k][i], 'mean']:.4f}({self.measure_table[mode][m].loc[MEASURES[k][i], 'se']:.4f})")

def main():
    manager = ExperimentManager(args)
    manager.run()
    manager.print_res()


if __name__ == '__main__':
    main()