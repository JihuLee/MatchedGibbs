import torch
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot    as plt
import seaborn              as sns
import ot
import torch.nn.functional  as F
import json
from torch.nn               import Sigmoid
from torch.optim            import Adam
from torch.utils.data       import DataLoader
from torch.nn.utils.stateless import functional_call
from progress.bar           import ChargingBar
from functools              import partial
from copy                   import deepcopy

from src.models             import MLP
from src.train              import _train
from src.eval               import evaluate
from src.utils              import make_path, select_indices_with_thinning
from src.dataset            import FairDataset, BalancedBatchSampler
from src.loss               import _fair_loss_mdp, _fair_loss_mmd, _fair_loss_dp

import hamiltorch.util      as hutil
import hamiltorch.samplers  as hsamplers

sns.set_theme('paper')


class BayesianRunner:
    save_dir: str      
    load_lmda: bool    
    calc_prop: bool    

    def __init__(self, args, seed, dataset, measures=None):
        self.args = args
        self.seed = seed
        self.args.pretrain_seed = seed
        self.x_train, self.y_train, self.s_train = dataset['train']['x'], dataset['train']['y'], dataset['train']['s']
        self.x_test, self.y_test, self.s_test = dataset['test']['x'], dataset['test']['y'], dataset['test']['s']
        
        if self.args.use_pretrained:
            self.pretrain_epochs = self.args.epochs_pre
        else:
            self.pretrain_epochs = 0

        self.pretrain_save_dir = make_path(self.args, 'pretrain', None) + f"epochs={self.args.epochs_pre}/seed={self.args.pretrain_seed}/"
        self.measures = measures
        self.num_samples_total = self.args.num_burn + self.args.num_samples * self.args.thin_interval

        if self.args.task_loss_func == 'bce':
            self.model_loss = 'binary_class_linear_output'
        self.input_dim = self.x_train.shape[1]
        self.output_dim = 1
        self.step_prop_lmda = self.args.step_size_lmda / self.args.step_size
   
    def build_model(self, def_lmda):
        if def_lmda:
            lmda_init = self.args.lmda_init
        else: 
            lmda_init = None
        model = MLP(
            num_layer=self.args.num_layer, 
            input_dim=self.input_dim, 
            rep_dim=self.args.rep_dim, 
           
            output_dim=self.output_dim, 
            lmda_init=lmda_init
        )
        return model
    
   
    def build_initials(self, model, predict=False):
        tau_list = []
        tau = 1.
        for _ in model.parameters():
            tau_list.append(tau)
        
        tau_list = torch.tensor(tau_list).to(self.args.device)

        if predict: 
            return tau_list
        else:
            params_init = hutil.flatten(model).to(self.args.device).clone()
            return tau_list, params_init 
       

   
    def pretrain(self):
       
        if not isinstance(self.s_train, torch.Tensor):
            s_train_tensor = torch.tensor(self.s_train.values if hasattr(self.s_train, 'values') else self.s_train)
        else:
            s_train_tensor = self.s_train
            
       
        balanced_sampler = BalancedBatchSampler(
            protected=s_train_tensor,
            batch_size=self.args.batch_size
        )
        
        train_dataset = FairDataset(features=self.x_train, labels=self.y_train, protected=self.s_train, to_torch=False)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=balanced_sampler,
           
        )
        self.args.lmda_grid[-1] = int(self.args.lmda_grid[-1])
        for lmda in np.linspace(*self.args.lmda_grid):
            lmda = round(lmda, 2)
            model = self.build_model(def_lmda=False)
            model.to(self.args.device)
            
            optim = Adam(model.parameters(), lr=self.args.lr)
            constraint_losses = []
            task_losses = []
            total_losses = []

            bar = ChargingBar(f'= Pretrain - lmda={lmda} =', max=self.args.epochs_pre)
            for epoch in range(1, self.args.epochs_pre+1):       
                task_loss, constraint = _train(
                    model=model, 
                    constraint_func=self.args.constraint_loss_func, 
                    dataloader=train_loader, 
                    optim=optim, 
                    args=self.args, 
                    lmda=lmda
                )
                bar.suffix = f"[Epoch {epoch}] loss: {task_loss + lmda * constraint:.8f}"
                bar.next()

                task_losses.append(task_loss)
                constraint_losses.append(constraint)
                total_losses.append(task_loss + lmda * constraint)

            bar.finish()
            
            filename = f'/ep={self.args.epochs_pre},lr={self.args.lr},lmda={lmda}.pt'
            self.save_object(model.state_dict(), self.pretrain_save_dir, filename)

    
    def load_pretrained_model(self, model, lmda, load_lmda=True):
        filename = f'ep={self.args.epochs_pre},lr={self.args.lr},lmda={lmda}.pt'
        model_dict = model.state_dict()
        if load_lmda:
            model_dict.update({'lmda': torch.tensor([lmda])})
        model_dict.update(torch.load(self.pretrain_save_dir + filename, map_location=self.args.device))
        model.load_state_dict(model_dict)
   

    def eval_pretrain(self):
        model = self.build_model(def_lmda=False).to(self.args.device)
        res = {round(lmda, 2): {'train': {}, 'test': {}} for lmda in np.linspace(*self.args.lmda_grid)}
        for lmda in np.linspace(*self.args.lmda_grid):
            lmda = round(lmda, 2)
            self.load_pretrained_model(model, lmda, load_lmda=False)

            model.eval()
            with torch.no_grad():
                logits = model(self.x_train.to(self.args.device))
                probs = torch.sigmoid(logits).squeeze().cpu()
                train_utility, train_uncertainty, train_fairness, _ = evaluate(
                    probs, 
                    self.x_train.cpu(), 
                    self.y_train.flatten().cpu(), 
                    self.s_train.cpu(), 
                    self.args
                )
                output_str = f"train - lmda={lmda}: "
                for i in range(len(train_utility)):
                    output_str += f"{self.measures['utility'][i]}: {train_utility[i]:.4f}, "
                for i in range(len(train_uncertainty)):
                    output_str += f"{self.measures['uncertainty'][i]}: {train_uncertainty[i]:.4f}, "
                for i in range(len(train_fairness)):
                    output_str += f"{self.measures['fairness'][i]}: {train_fairness[i]:.4f}, "
                print(output_str)
               
                logits = model(self.x_test.to(self.args.device))
                probs = torch.sigmoid(logits).squeeze().cpu()

                test_utility, test_uncertainty, test_fairness, _ = evaluate(
                    probs, 
                    self.x_test.cpu(), 
                    self.y_test.flatten().cpu(), 
                    self.s_test.cpu(), 
                    self.args
                )
        
                res[lmda]['train'] = {
                    'utility': train_utility, 
                    'uncertainty': train_uncertainty, 
                    'fairness': train_fairness
                }
                res[lmda]['test'] = {
                    'utility': test_utility, 
                    'uncertainty': test_uncertainty, 
                    'fairness': test_fairness
                }
        return res
   

    def sample(self):
        pass

   
    def calc_elbo(self):
        pass

   
    def _eval_samples(self):
        model = self.build_model(def_lmda=self.load_lmda).to(self.args.device)
        samples = torch.load(self.save_dir + 'params')
        tau_list = self.build_initials(model, predict=True)

        pred_list_train, _ = hsamplers.predict_model(
            model=model, 
            x=self.x_train, 
            y=self.y_train, 
            samples=samples[:], 
            model_loss=self.model_loss, 
            constraint_loss=None, 
            tau_out=1., 
            tau_list=tau_list
        )

        pred_list_test, _ = hsamplers.predict_model(
            model=model, 
            x=self.x_test, 
            y=self.y_test, 
            samples=samples[:], 
            model_loss=self.model_loss, 
            constraint_loss=None, 
            tau_out=1., 
            tau_list=tau_list
        )

        return pred_list_train, pred_list_test

    def eval(self):
        pred_list_train, pred_list_test = self._eval_samples()

        pred_list_train = Sigmoid()(pred_list_train)
        
        train_utility, train_uncertainty, train_fairness, _ = evaluate(
            pred_list_train.squeeze().cpu(), 
            self.x_train.cpu(), 
            self.y_train.flatten().cpu(), 
            self.s_train.cpu(), 
            self.args, 
            calc_prop=self.calc_prop
        )

        pred_list_test = Sigmoid()(pred_list_test)
        
        test_utility, test_uncertainty, test_fairness, _ = evaluate(
            pred_list_test.squeeze().cpu(), 
            self.x_test.cpu(), 
            self.y_test.flatten().cpu(), 
            self.s_test.cpu(), 
            self.args, 
            calc_prop=self.calc_prop
        )

        final_result = {
            'train': {
                'utility': train_utility, 
                'uncertainty': train_uncertainty, 
                'fairness': train_fairness
            }, 
            'test': {
                'utility': test_utility, 
                'uncertainty': test_uncertainty, 
                'fairness': test_fairness
            }
        }
        
        return final_result


    def save_object(self, object, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(object, save_dir+filename)


    def save_npy(self, object, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir+filename, object)


    def count_unique_elem(self, samples):
        return len(set(map(tuple, np.array(torch.stack(samples).cpu()))))
    


class GibbsRunner(BayesianRunner):
    def __init__(self, args, seed, dataset): 
        super().__init__(args, seed, dataset)
        self.save_dir = make_path(self.args, 'gibbs', self.pretrain_epochs) + f"seed={self.seed}/"
        self.load_lmda = False
        self.calc_prop = True
        if self.args.constraint_loss_func == 'mdp':
            if self.args.init_matching == 'ot':
                self.s0_indices, self.s1_indices, self.initial_matching = self.init_matching_ot()
        

    def init_matching_ot(self):
        s1_indices = torch.where(self.s_train == 1)[0]
        s0_indices = torch.where(self.s_train == 0)[0]
       
        x_s0 = self.x_train[s0_indices].to(self.args.device)
        x_s1 = self.x_train[s1_indices].to(self.args.device)
        
        y_s0 = self.y_train[s0_indices].to(self.args.device)
        y_s1 = self.y_train[s1_indices].to(self.args.device)

        s0_weight = torch.ones(x_s0.size(0), device=self.args.device) / x_s0.size(0)
        s1_weight = torch.ones(x_s1.size(0), device=self.args.device) / x_s1.size(0)
        
        M = torch.cdist(x_s0, x_s1, p=2) + self.args.margin * torch.cdist(y_s0, y_s1, p=1)
        G = ot.emd(s0_weight, s1_weight, M, numItermax=1000000)
        initial_matching = torch.argmax(G, dim=1)

        return s0_indices, s1_indices, initial_matching
    

    def sample(self):
        print("== Sample Gibbs ==")
        model = self.build_model(def_lmda=self.load_lmda)
        if self.args.use_pretrained:
            self.load_pretrained_model(model, self.args.lmda_init, self.load_lmda)

        tau_list, params_init = self.build_initials(model)
        if self.args.constraint_loss_func == 'mdp':
            if self.args.init_matching == 'ot':
                sample_func = partial(hsamplers.sample_gibbs_mdp_model, matching=self.initial_matching)
        else:
            sample_func = hsamplers.sample_gibbs_model

        params_sampled = sample_func(
            model=model, 
            x=self.x_train, 
            y=self.y_train, 
            s=self.s_train, 
            params_init=params_init, 
            lmda=self.args.lmda_init, 
            lr_gibbs=self.args.lr_gibbs, 
            model_loss=self.model_loss, 
            constraint_loss=self.args.constraint_loss_func, 
            num_samples=self.num_samples_total, 
            num_steps_per_sample=self.args.L, 
            step_size=self.args.step_size, 
            burn=self.args.num_burn, 
            sampler=hsamplers.Sampler.HMC, 
            tau_list=tau_list
        )

       
        if not self.args.constraint:
            n = self.args.num_samples * self.args.thin_interval
            params_sampled_final = [params_sampled[i] for i in np.arange(n) if i % self.args.thin_interval == self.args.thin_interval - 1]
        else:
            _, _, constraint_list = hsamplers.predict_model(
                model=model, 
                samples=params_sampled, 
                x=self.x_train, 
                y=self.y_train, 
                s=self.s_train, 
                model_loss=self.model_loss, 
                constraint_loss=self.args.constraint_loss_func, 
                tau_out=1., 
                tau_list=tau_list
            )
            constraint_list = np.array(constraint_list)
            ind_satisfied = np.where(constraint_list <= self.args.thres)[0]
            ind_selected = select_indices_with_thinning(ind_satisfied, self.args.thin_interval, self.args.num_samples)
            params_sampled_final = [params_sampled[i] for i in ind_selected]
            del constraint_list

       
        self.save_object(params_sampled_final, self.save_dir, 'params')
        del model, tau_list, params_init, params_sampled, params_sampled_final
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    
    def log_prob_theta(self, theta):
       
        model = self.build_model(def_lmda=self.load_lmda)
        if self.args.use_pretrained:
            self.load_pretrained_model(model, self.args.lmda_init, self.load_lmda)
        
       
        params_flattened_list = []
        for weights in model.parameters():
            params_flattened_list.append(weights.nelement())
        
       
        params = hutil.unflatten(model, theta)
        hutil.update_model_params_in_place(model, params)
        
       
        logits = model(self.x_train.to(self.args.device))
        if self.args.task_loss_func == 'bce':
            log_prob = -F.binary_cross_entropy_with_logits(
                logits.squeeze(), 
                self.y_train.to(self.args.device).squeeze(),
                reduction='sum'
            )
        
        if self.args.constraint_loss_func == 'dp':
            constraint = _fair_loss_dp(logits, self.s_train.to(self.args.device))
        elif self.args.constraint_loss_func == 'mmd':
            constraint = _fair_loss_mmd(logits, self.s_train.to(self.args.device))
        elif self.args.constraint_loss_func == 'mdp':
            constraint = _fair_loss_mdp(logits, self.initial_matching, self.s_train.to(self.args.device))
        else: 
            raise NotImplementedError()
        
       
        R_n = log_prob - self.args.lmda_init * self.x_train.size(0) * constraint
        
        return R_n

    def calc_elbo(self, num_samples=1000):
        model = self.build_model(def_lmda=self.load_lmda).to(self.args.device)
        samples = torch.load(self.save_dir + 'params')
        tau_list = self.build_initials(model, predict=True)
        dist_list = []

        with torch.no_grad():
            params_shape_list = []
            params_flattened_list = []
            for weights in model.parameters():
                params_shape_list.append(weights.shape)
                params_flattened_list.append(weights.nelement())
           
            log_likelihood_sum = 0.0
            mean_R_n = 0.0
            for s in samples:
                params = hutil.unflatten(model, s)
                hutil.update_model_params_in_place(model, params)
               
                with torch.no_grad():
                    logits = model(self.x_train.to(self.args.device))
                    
                    if self.args.task_loss_func == 'bce':
                        log_prob = -F.binary_cross_entropy_with_logits(
                            logits.squeeze(), 
                            self.y_train.to(self.args.device).squeeze(),
                            reduction='sum'
                        )
                
                    log_likelihood_sum += log_prob.item()
                    
                    R_n = self.log_prob_theta(s)
                    mean_R_n += R_n.item()
            
            log_likelihood_mean = log_likelihood_sum / len(samples)
            mean_R_n /= len(samples)

            for tau in tau_list:
                dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))
            
            r_n_list = []
            for _ in range(num_samples):
                sample = []
                for shape, dist in zip(params_shape_list, dist_list):
                    sample.append(dist.sample(shape).view(-1))
                flat_sample = torch.cat(sample)
                r_n_list.append(-self.log_prob_theta(flat_sample))
            
            r_n_tensor = torch.tensor(r_n_list, device='cpu')
            log_Z = torch.logsumexp(-r_n_tensor, dim=0) - torch.log(torch.tensor(num_samples))
            n_kl = mean_R_n + log_Z.item()
            elbo = log_likelihood_mean + n_kl
        
        return elbo
        

class MatchedGibbsRunner(BayesianRunner):
    def __init__(self, args, seed, dataset): 
        super().__init__(args, seed, dataset)
        self.save_dir = make_path(self.args, 'matched_gibbs', self.pretrain_epochs) + f"seed={self.seed}/"
        self.load_lmda = False
        self.calc_prop = True
        self.bar_string = "= Matched Gibbs ="
        self.s0_indices, self.s1_indices, self.initial_matching = self.init_matching()


    def init_matching(self):
        matching_path = self.args.save_dir + f'matched_gibbs/init_matching/seed={self.seed}/'
        try:
            data = torch.load(matching_path + 'matching.pt', map_location='cpu')
            s0_cached = data.get('s0_indices').to(self.args.device)
            s1_cached = data.get('s1_indices').to(self.args.device)
            init_cached = data.get('initial_matching').to(self.args.device)
            if s0_cached is not None and s1_cached is not None and init_cached is not None:
                return s0_cached, s1_cached, init_cached
        except Exception:
            pass
        
        s1_indices = torch.where(self.s_train == 1)[0]
        s0_indices = torch.where(self.s_train == 0)[0]
        
        x_s0 = self.x_train[s0_indices].to(self.args.device)
        x_s1 = self.x_train[s1_indices].to(self.args.device)
        y_s0 = self.y_train[s0_indices].to(self.args.device)
        y_s1 = self.y_train[s1_indices].to(self.args.device)

        s0_weight = torch.ones(x_s0.size(0), device=self.args.device) / x_s0.size(0)
        s1_weight = torch.ones(x_s1.size(0), device=self.args.device) / x_s1.size(0)
        
        M = torch.cdist(x_s0, x_s1, p=2) + self.args.margin * torch.cdist(y_s0, y_s1, p=1)
        G = ot.emd(s0_weight, s1_weight, M, numItermax=1000000)
        initial_matching = torch.argmax(G, dim=1)
       
        try:
            if not os.path.exists(matching_path):
                os.makedirs(matching_path)
            torch.save({
                's0_indices': s0_indices.cpu(),
                's1_indices': s1_indices.cpu(),
                'initial_matching': initial_matching.cpu(),
            }, matching_path + 'matching.pt')
        except Exception:
            pass

        return s0_indices, s1_indices, initial_matching

        
    def sample(self):
        print("== Sample Matched Gibbs ==")
        
        model = self.build_model(def_lmda=self.load_lmda)
        if self.args.use_pretrained:
            self.load_pretrained_model(model, self.args.lmda_init, self.load_lmda)
            
       
        tau_list, params_init = self.build_initials(model)
        dist_list = []
        for tau in tau_list:
            dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

       
        params_flattened_list = []
        for weights in model.parameters():
            params_flattened_list.append(weights.nelement())
        
        params_sampled = []
        matchings_sampled = []
        energy_list = []
        
        current_matching = deepcopy(self.initial_matching)

        bar = ChargingBar(self.bar_string, max=self.num_samples_total)
        accept_count = 0
        for i in range(self.num_samples_total):
            def log_prob_theta(theta):
                params = hutil.unflatten_dict(model, theta)
                logits = functional_call(model, params, (self.x_train.to(self.args.device),))
                
                log_prob = -F.binary_cross_entropy_with_logits(
                    logits.squeeze(), 
                    self.y_train.to(self.args.device).squeeze(),
                    reduction='sum'
                )
                
                log_prior = torch.tensor(0.0, device=theta.device, requires_grad=True)
                i_prev = 0
                for index, dist in zip(params_flattened_list, dist_list):
                    w = theta[i_prev:i_prev+index].view(-1)
                    log_prior = log_prior + dist.log_prob(w).sum()
                    i_prev += index
                    
                if self.args.constraint_loss_func == 'mdp':
                    constraint = _fair_loss_mdp(logits, current_matching, self.s_train.to(self.args.device))
                else: 
                    raise NotImplementedError()
                
                total_log_prob = log_prob + log_prior - self.args.lmda_init * self.x_train.size(0) * constraint
                
                return total_log_prob
            
           
            def grad_log_prob_theta(theta):
                theta = theta.detach().requires_grad_(True)
                log_prob = log_prob_theta(theta)
                grad = torch.autograd.grad(log_prob, theta)[0]
                return grad
            
           
            current_theta = params_init.clone()
            momentum = torch.randn_like(current_theta)
            theta = current_theta.clone()
            current_momentum = momentum.clone()
            grad_U = -grad_log_prob_theta(theta)
            current_momentum = current_momentum - 0.5 * self.args.step_size * grad_U

            for _ in range(self.args.L):
                theta = theta + self.args.step_size * current_momentum
                if _ < self.args.L - 1:
                    grad_U = -grad_log_prob_theta(theta)
                    current_momentum = current_momentum - self.args.step_size * grad_U
            grad_U = -grad_log_prob_theta(theta)

            current_momentum = current_momentum - 0.5 * self.args.step_size * grad_U
            current_momentum = -current_momentum
            current_U = -log_prob_theta(current_theta)
            current_K = 0.5 * torch.sum(momentum ** 2)
            proposed_U = -log_prob_theta(theta)
            proposed_K = 0.5 * torch.sum(current_momentum ** 2)
            
            if torch.rand(1).item() < torch.exp(current_U - proposed_U + current_K - proposed_K):
                current_theta = theta
                energy_list.append((proposed_U + proposed_K).item())
            else:
                energy_list.append((current_U + current_K).item())
            
            params_init = current_theta.detach().clone()
            params = hutil.unflatten(model, params_init)
            hutil.update_model_params_in_place(model, params)
            
            batch_size = len(self.s0_indices)
            x_s0_all = self.x_train[self.s0_indices]

            proposed_matching = current_matching.clone()
            permute_indices = torch.randperm(batch_size)[:self.args.permute_size]
            proposed_matching[permute_indices] = current_matching[permute_indices[torch.randperm(self.args.permute_size)]]
           
            x_s1_current = self.x_train[self.s1_indices[current_matching]] 
            x_s1_proposed = self.x_train[self.s1_indices[proposed_matching]] 
            
            with torch.no_grad():
                logits_s0 = model(x_s0_all) 
                logits_s1_current = model(x_s1_current) 
                logits_s1_proposed = model(x_s1_proposed) 
                
                current_log_joint = -F.binary_cross_entropy_with_logits(
                    logits_s1_current - logits_s0,
                    torch.zeros_like(logits_s0),
                    reduction='sum' 
                )
                
                proposed_log_joint = -F.binary_cross_entropy_with_logits(
                    logits_s1_proposed - logits_s0,
                    torch.zeros_like(logits_s0),
                    reduction='sum' 
                )
                
                current_distances = torch.norm(x_s0_all - x_s1_current, p=2, dim=1)
                current_prior = -current_distances.mean() / self.args.tau 
                
                proposed_distances = torch.norm(x_s0_all - x_s1_proposed, p=2, dim=1)
                proposed_prior = -proposed_distances.mean() / self.args.tau 
                
                log_accept_ratio = (proposed_log_joint - current_log_joint) + (proposed_prior - current_prior)
                
                if torch.log(torch.rand(1, device=current_matching.device)) < log_accept_ratio:
                    accept_count += 1
                    current_matching = proposed_matching
            
            if i >= self.args.num_burn and (i - self.args.num_burn) % self.args.thin_interval == 0:
                params_sampled.append(current_theta.detach().clone())
                matchings_sampled.append(current_matching.detach().clone())
            
            bar.suffix = f"[Iter {i+1}]"
            bar.next()
        
        bar.finish()
        print(f"Acceptance rate: {accept_count / self.num_samples_total}")
        
        self.save_object(params_sampled, self.save_dir, 'params')
        self.save_object(matchings_sampled, self.save_dir, 'matchings')
        self.save_npy(np.asarray(energy_list), self.save_dir, 'energy_list.npy')
        
        del model, tau_list, params_init, params_sampled, matchings_sampled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    
    def calc_R_n(self, theta, matching):
       
        model = self.build_model(def_lmda=self.load_lmda)
        if self.args.use_pretrained:
            self.load_pretrained_model(model, self.args.lmda_init, self.load_lmda)
        
        params_flattened_list = []
        for weights in model.parameters():
            params_flattened_list.append(weights.nelement())
        
        params = hutil.unflatten(model, theta)
        hutil.update_model_params_in_place(model, params)
        
        logits = model(self.x_train.to(self.args.device))
        if self.args.task_loss_func == 'bce':
            log_prob = -F.binary_cross_entropy_with_logits(
                logits.squeeze(), 
                self.y_train.to(self.args.device).squeeze(),
                reduction='sum'
            )
        
        constraint = _fair_loss_mdp(logits, matching, self.s_train.to(self.args.device))
        
        R_n = log_prob - self.args.lmda_init * self.x_train.size(0) * constraint
        
        return R_n

    def calc_elbo(self, num_samples=1000):
        model = self.build_model(def_lmda=self.load_lmda).to(self.args.device)
        samples = torch.load(self.save_dir + 'params')
        matchings = torch.load(self.save_dir + 'matchings')
        tau_list = self.build_initials(model, predict=True)
        dist_list = []

        with torch.no_grad():
            params_shape_list = []
            params_flattened_list = []
            for weights in model.parameters():
                params_shape_list.append(weights.shape)
                params_flattened_list.append(weights.nelement())
            
           
            log_likelihood_sum = 0.0
            mean_R_n = 0.0
            for s, m in zip(samples, matchings):
                params = hutil.unflatten(model, s)
                hutil.update_model_params_in_place(model, params)
               
                with torch.no_grad():
                    logits = model(self.x_train.to(self.args.device))
                    
                    if self.args.task_loss_func == 'bce':
                        log_prob = -F.binary_cross_entropy_with_logits(
                            logits.squeeze(), 
                            self.y_train.to(self.args.device).squeeze(),
                            reduction='sum'
                        )
                    
                    log_likelihood_sum += log_prob
                    
                    R_n = self.calc_R_n(s, m)
                    mean_R_n += R_n
            
           
            log_likelihood_mean = log_likelihood_sum / len(samples)
            mean_R_n /= len(samples)

            for tau in tau_list:
                dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))
            
            r_n_list = []
            for _ in range(num_samples):
                sample = []
                for shape, dist in zip(params_shape_list, dist_list):
                    sample.append(dist.sample(shape).view(-1))
                flat_sample = torch.cat(sample)
                r_n_list.append(-self.calc_R_n(flat_sample))
            
            r_n_tensor = torch.tensor(r_n_list, device='cpu')
            log_Z = torch.logsumexp(-r_n_tensor, dim=0) - torch.log(torch.tensor(num_samples))
            n_kl = mean_R_n + log_Z
            elbo = log_likelihood_mean + n_kl
        
        return elbo
