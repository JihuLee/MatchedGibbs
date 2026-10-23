import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import gc

from torch.optim import Adam
from src.loss import _fair_loss_dp, _fair_loss_mdp, _fair_loss_mmd
from torch.utils.data import DataLoader
from progress.bar import ChargingBar

from src.models import BayesianMLP
from src.eval import evaluate
from src.utils import make_path
from src.dataset import FairDataset, BalancedBatchSampler
from src.loss import get_batch_matching

class VariationalRunner:
    save_dir: str  
    constraints: str
    bar_string: str
    lmda: float
    best_by_val: bool
    n_models: int
    
    def __init__(self, args, seed, dataset):
        self.args = args
        self.seed = seed

        self.x_train, self.y_train, self.s_train = dataset['train']['x'], dataset['train']['y'], dataset['train']['s']
        self.x_test, self.y_test, self.s_test = dataset['test']['x'], dataset['test']['y'], dataset['test']['s']
        self.input_dim = self.x_train.shape[1]
        self.output_dim = 1
        self.n_models = 1
        self.lmda = args.lmda_init
        self.constraints = args.constraint_loss_func
        self.best_by_val = args.best_by_val
        self.save_dir = make_path(args, 'variational', None, self.n_models) + f"/seed={seed}/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.bar_string = "= Variational ="
        self.calc_prop = True


    def del_data(self):
        del self.dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def build_model(self, def_lmda=True):
        if def_lmda:
            lmda_init = self.lmda
        else: 
            lmda_init = None 
                
        model = BayesianMLP(
            num_layer=self.args.num_layer, 
            input_dim=self.input_dim, 
            rep_dim=self.args.rep_dim, 
            output_dim=self.output_dim, 
            lmda_init=lmda_init
        )
        return model
    

    def save_object(self, object, save_dir, filename):
        torch.save(object, save_dir+filename)


    def count_unique_elem(self, samples):
        return len(set(map(tuple, np.array(torch.stack(samples).cpu()))))


    def train(self):
        bar_string = self.bar_string
        
        train_dataset = FairDataset(features=self.x_train, labels=self.y_train, protected=self.s_train, to_torch=False)
        balanced_sampler = BalancedBatchSampler(
            protected=self.s_train,
            batch_size=self.args.batch_size
        )
        train_loader = DataLoader(train_dataset, batch_sampler=balanced_sampler)
        
        if self.args.task_loss_func == 'bce':
            criterion = nn.BCELoss()
        
        for i in range(self.n_models):
            model = self.build_model()
            model.to(self.args.device)
            
            optimizer = Adam(model.parameters(), lr=self.args.step_size)
            bar = ChargingBar(bar_string, max=self.args.variational_epochs)

            for epoch in range(1, self.args.variational_epochs + 1):
                model.train()
                total_loss = 0.0
                
                for batch in train_loader:
                    x, y, s = batch
                    batch_size = x.size(0)
                    optimizer.zero_grad()
                                        
                    logits_list = []
                    kl_list = torch.tensor(0.0, device=x.device)
                    for mc_run in range(self.args.mc_runs):
                        logits, kl = model(x)
                        logits_list.append(logits)
                        kl_list += kl

                    probs = torch.sigmoid(torch.cat(logits_list, dim=1)).mean(1)
                    probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
                    logits = torch.logit(probs)
                    kl = kl_list / self.args.mc_runs
                    
                    outputs = torch.sigmoid(logits)
                    
                    targets = y.unsqueeze(1).float()
                    outputs = outputs.view_as(targets)
                    
                    if self.args.task_loss_func == 'bce':
                        task_loss = criterion(outputs, targets)
                    
                    constraint_loss = torch.tensor(0.0, device=x.device)
                    if self.constraints == 'dp':
                        
                        constraint_loss = _fair_loss_dp(logits, s)
                    elif self.constraints == 'mmd':
                        
                        constraint_loss = _fair_loss_mmd(probs, s)
                    elif self.constraints == 'mdp':
                        
                        x_s0 = x[s == 0]
                        x_s1 = x[s == 1]
                        
                        y_s0 = y[s == 0]; y_s1 = y[s == 1]
                        matching = get_batch_matching(x_s0, x_s1, y_s0, y_s1, self.args.margin)
                        constraint_loss = _fair_loss_mdp(logits, matching, s, use_logits = True, use_clamp = False)
                        
                    elif self.constraints != 'none':
                        raise ValueError(f"Unknown constraint: {self.constraints}")
                    
                    if constraint_loss.dim() > 0:
                        constraint_loss = constraint_loss.mean()
                    
                    kl_term = (1.0 / (batch_size + 1e-8)) * kl
                    
                    loss = task_loss + kl_term + self.lmda * constraint_loss * batch_size
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                bar.suffix = f"[Epoch {epoch}] loss: {avg_loss:.4f}"
                bar.next()

            bar.finish()
            
            filename = f'/ep={self.args.variational_epochs},lr={self.args.step_size}_{i}.pt'
            self.save_object(model.state_dict(), self.save_dir, filename)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def calc_elbo(self):
        model = self.build_model().to(self.args.device)
        filename = f'/ep={self.args.variational_epochs},lr={self.args.step_size}_0.pt'
        model.load_state_dict(torch.load(self.save_dir + filename, map_location=self.args.device))
        model.eval()
        
        with torch.no_grad():
            logits, kl = model(self.x_train.to(self.args.device))
            
            if self.args.task_loss_func == 'bce':
                log_prob = -F.binary_cross_entropy_with_logits(
                    logits.squeeze(), 
                    self.y_train.to(self.args.device).squeeze(),
                    reduction='sum'
                )

            if self.constraints == 'dp':
                constraint_loss = _fair_loss_dp(logits, self.s_train.to(self.args.device))
            elif self.constraints == 'mmd':
                constraint_loss = _fair_loss_mmd(logits, self.s_train.to(self.args.device))
            elif self.constraints == 'mdp':
                x_s0 = self.x_train[self.s_train == 0].to(self.args.device)
                x_s1 = self.x_train[self.s_train == 1].to(self.args.device)
                if len(x_s0) > 0 and len(x_s1) > 0:
                    matching = get_batch_matching(x_s0, x_s1)
                    constraint_loss = _fair_loss_mdp(logits, matching, self.s_train.to(self.args.device))
                else:
                    constraint_loss = torch.tensor(0.0, device=self.args.device)
            else:  
                constraint_loss = torch.tensor(0.0, device=self.args.device)
            
            if constraint_loss.dim() > 0:
                constraint_loss = constraint_loss.mean()
            
            likelihood_term = log_prob
            kl_term = kl
            constraint_term = self.lmda * constraint_loss * self.x_train.size(0)  
            
            
            elbo = likelihood_term - kl_term - constraint_term
            
            return elbo.item()


    def eval(self, num_samples=5):
        preds_list_train, preds_list_test = [], []
        for i in range(self.n_models):
            model = self.build_model().to(self.args.device)
            if self.best_by_val:
                filename = f'/best_ep={self.args.variational_epochs},lr={self.args.step_size}_{i}.pt'
            else:
                filename = f'/ep={self.args.variational_epochs},lr={self.args.step_size}_{i}.pt'

            model.load_state_dict(torch.load(self.save_dir + filename, map_location=self.args.device))
            model.eval()
            
            with torch.no_grad():
                train_samples = []
                for _ in range(num_samples):
                    preds, _ = model(self.x_train.to(self.args.device))
                    preds = torch.sigmoid(preds).flatten()
                    train_samples.append(preds.unsqueeze(0))
                preds_train = torch.cat(train_samples).mean(0)
                preds_list_train.append(preds_train.cpu())
                
                test_samples = []
                for _ in range(num_samples):
                    preds, _ = model(self.x_test.to(self.args.device))
                    preds = torch.sigmoid(preds).flatten()
                    test_samples.append(preds.unsqueeze(0))
                preds_test = torch.cat(test_samples).mean(0)
                preds_list_test.append(preds_test.cpu())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        preds_train = torch.stack(preds_list_train).cpu()
        preds_test = torch.stack(preds_list_test).cpu()
        
        y_train_tensor = self.y_train.flatten().cpu().float()
        y_test_tensor = self.y_test.flatten().cpu().float()
        s_train_tensor = self.s_train.cpu()
        s_test_tensor = self.s_test.cpu()
        
        train_utility, train_uncertainty, train_fairness, _ = evaluate(
            preds_train,
            self.x_train.cpu(),
            y_train_tensor,
            s_train_tensor, 
            self.args,
            calc_prop=self.calc_prop
        )
        
        test_utility, test_uncertainty, test_fairness, _ = evaluate(
            preds_test,
            self.x_test.cpu(),
            y_test_tensor,
            s_test_tensor,
            self.args,
            calc_prop=self.calc_prop
        )
        
        train_elbo = self.calc_elbo()

        final_result = {
            'train': {
                'utility': train_utility, 
                'uncertainty': train_uncertainty, 
                'fairness': train_fairness,
            }, 
            'test': {
                'utility': test_utility, 
                'uncertainty': test_uncertainty, 
                'fairness': test_fairness,
            }
        }
            
        return final_result, train_elbo