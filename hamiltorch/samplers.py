import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util

from src.loss import _fair_loss_dp, _fair_loss_mmd, _fair_loss_mdp, _fair_loss_wdp

class Sampler(Enum):
    HMC = 1
    RMHMC = 2
    HMC_NUTS = 3


class Integrator(Enum):
    EXPLICIT       = 1
    IMPLICIT       = 2
    S3             = 3
    SPLITTING      = 4
    SPLITTING_RAND = 5
    SPLITTING_KMID = 6


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3


def collect_gradients(log_prob, params, pass_grad = None):
    
    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    elif pass_grad is not None:
        if callable(pass_grad):
            params.grad = pass_grad(params)
        else:
            params.grad = pass_grad
    else:
        params.grad = torch.autograd.grad(log_prob,params)[0]
    return params


def fisher(params, log_prob_func=None, jitter=None, normalizing_const=1., softabs_const=1e6, metric=Metric.HESSIAN):

    log_prob, *_ = log_prob_func(params)
    def log_prob_func_first():
        return log_prob_func(params)[0]

    if util.has_nan_or_inf(log_prob):
        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
        raise util.LogProbError()
    if metric == Metric.JACOBIAN_DIAG:
        jac = util.jacobian(log_prob, params, create_graph=True, return_inputs=False)
        jac = torch.cat([j.flatten() for j in jac])
        fish = torch.matmul(jac.view(-1,1),jac.view(1,-1)).diag().diag()
    else:
        hess = torch.autograd.functional.hessian(log_prob_func_first, params, create_graph=True)
        fish = - hess
    if util.has_nan_or_inf(fish):
        print('Invalid hessian: {}, params: {}'.format(fish, params))
        raise util.LogProbError()
    if jitter is not None:
        params_n_elements = fish.shape[0]
        fish += (torch.eye(params_n_elements) * torch.rand(params_n_elements) * jitter).to(fish.device)
    if (metric is Metric.HESSIAN) or (metric is Metric.JACOBIAN_DIAG):
        return fish, None
    elif metric == Metric.SOFTABS:
        eigenvalues, eigenvectors = torch.linalg.eigh(fish, UPLO='L')
        abs_eigenvalues = (1./torch.tanh(softabs_const * eigenvalues)) * eigenvalues
        fish = torch.matmul(eigenvectors, torch.matmul(abs_eigenvalues.diag(), eigenvectors.t()))
        return fish, abs_eigenvalues
    else:
        raise ValueError('Unknown metric: {}'.format(metric))


def cholesky_inverse(fish, momentum):
    
    lower = torch.linalg.cholesky(fish)
    y = torch.linalg.solve_triangular(lower, momentum.view(-1, 1), upper=False, unitriangular=False)
    fish_inv_p = torch.linalg.solve_triangular(lower.t(), y, upper=True, unitriangular=False)
    return fish_inv_p


def gibbs(params, sampler=Sampler.HMC, log_prob_func=None, jitter=None, normalizing_const=1., softabs_const=None, mass=None, metric=Metric.HESSIAN):

    if sampler == Sampler.RMHMC:
        dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), fisher(params, log_prob_func, jitter, normalizing_const, softabs_const, metric)[0])
    elif mass is None:
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    else:
        if type(mass) is list:
            samples = torch.zeros_like(params)
            i = 0
            for block in mass:
                it = block[0].shape[0]
                dist = torch.distributions.MultivariateNormal(torch.zeros_like(block[0]), block)
                samples[i:it+i] = dist.sample()
                i += it
            return samples
        elif len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass ** 0.5)
    return dist.sample()


def leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, store_on_GPU = True, debug=False, pass_grad = None):

    params = params.clone(); momentum = momentum.clone()
    if sampler == Sampler.HMC and integrator != Integrator.SPLITTING and integrator != Integrator.SPLITTING_RAND and integrator != Integrator.SPLITTING_KMID:
        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob, *_ = log_prob_func(p)
            p = collect_gradients(log_prob, p, pass_grad)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return p.grad
        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum
            else:
                if type(inv_mass) is list:
                    i = 0
                    for block in inv_mass:
                        it = block[0].shape[0]
                        params[i:it+i] = params[i:it+i] + step_size * torch.matmul(block,momentum[i:it+i].view(-1,1)).view(-1)
                        i += it
                elif len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                else:
                    params = params + step_size * inv_mass * momentum
            p_grad = params_grad(params)
            momentum += step_size * p_grad
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
        return ret_params, ret_momenta
    elif sampler == Sampler.RMHMC and (integrator == Integrator.IMPLICIT or integrator == Integrator.S3):
        if integrator is not Integrator.S3:
            ham_func = None
        if pass_grad is not None:
            raise RuntimeError('Passing user-determined gradients not implemented for RMHMC')

        def fixed_point_momentum(params, momentum):
            momentum_old = momentum.clone()
            for i in range(fixed_point_max_iterations):
                momentum_prev = momentum.clone()
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                params = collect_gradients(ham, params)

                tries = 0
                while util.has_nan_or_inf(params.grad):
                    params = params.detach().requires_grad_()
                    ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                    params = collect_gradients(ham, params)
                    tries += 1
                    if tries > jitter_max_tries:
                        print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                        raise util.LogProbError()

                momentum = momentum_old - 0.5 * step_size * params.grad
                momenta_diff = torch.max((momentum_prev-momentum)**2)
                if momenta_diff < fixed_point_threshold:
                    break
            if debug == 1:
                print('Converged (momentum), iterations: {}, momenta_diff: {}'.format(i, momenta_diff))
            return momentum

        def fixed_point_params(params, momentum):
            params_old = params.clone()
            momentum = momentum.detach().requires_grad_()
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
            momentum = collect_gradients(ham,momentum)
            momentum_grad_old = momentum.grad.clone()
            for i in range(fixed_point_max_iterations):
                params_prev = params.clone()
                momentum = momentum.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                momentum = collect_gradients(ham,momentum)
                params = params_old + 0.5 * step_size * momentum.grad + 0.5 * step_size * momentum_grad_old
                params_diff = torch.max((params_prev-params)**2)
                if params_diff < fixed_point_threshold:
                    break
            if debug == 1:
                print('Converged (params), iterations: {}, params_diff: {}'.format(i, params_diff))
            return params
        ret_params = []
        ret_momenta = []
        for n in range(steps):
            momentum = fixed_point_momentum(params, momentum)
            params = fixed_point_params(params, momentum)

            params = params.detach().requires_grad_()
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
            params = collect_gradients(ham, params)

            tries = 0
            while util.has_nan_or_inf(params.grad):
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                    raise util.LogProbError()
            momentum -= 0.5 * step_size * params.grad

            ret_params.append(params)
            ret_momenta.append(momentum)
        return ret_params, ret_momenta

    elif sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
        if pass_grad is not None:
            raise RuntimeError('Passing user-determined gradients not implemented for RMHMC')

        leapfrog_hamiltonian_flag = Integrator.IMPLICIT
        def hamAB_grad_params(params,momentum):
            params = params.detach().requires_grad_()
            ham = hamiltonian(params, momentum.detach(), log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
            params = collect_gradients(ham, params)

            tries = 0
            while util.has_nan_or_inf(params.grad):
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum.detach(), log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                    raise util.LogProbError()

            return params.grad
        def hamAB_grad_momentum(params,momentum):
            momentum = momentum.detach().requires_grad_()
            params = params.detach().requires_grad_()
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
            momentum = collect_gradients(ham,momentum)
            return momentum.grad
        ret_params = []
        ret_momenta = []
        params_copy = params.clone()
        momentum_copy = momentum.clone()
        for n in range(steps):
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(params,momentum_copy)
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(params,momentum_copy)
            params = params + 0.5 * step_size * hamAB_grad_momentum(params_copy,momentum)
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(params_copy,momentum)
            c = torch.cos(torch.FloatTensor([2* explicit_binding_const * step_size])).to(params.device)
            s = torch.sin(torch.FloatTensor([2* explicit_binding_const * step_size])).to(params.device)
            params = 0.5 * ((params+params_copy) + c*(params-params_copy) + s*(momentum-momentum_copy))
            momentum = 0.5 * ((momentum+momentum_copy) - s*(params-params_copy) + c*(momentum-momentum_copy))
            params_copy = 0.5 * ((params+params_copy) - c*(params-params_copy) - s*(momentum-momentum_copy))
            momentum_copy = 0.5 * ((momentum+momentum_copy) + s*(params-params_copy) - c*(momentum-momentum_copy))


            params = params + 0.5 * step_size * hamAB_grad_momentum(params_copy,momentum)
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(params_copy,momentum)
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(params,momentum_copy)
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(params,momentum_copy)

            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        return [ret_params,params_copy], [ret_momenta, momentum_copy]

    elif sampler == Sampler.HMC and (integrator == Integrator.SPLITTING or integrator == Integrator.SPLITTING_RAND or Integrator.SPLITTING_KMID):
        if type(log_prob_func) is not list:
            raise RuntimeError('For splitting log_prob_func must be list of functions')
        if pass_grad is not None:
            raise RuntimeError('Passing user-determined gradients not implemented for splitting')

        def params_grad(p,log_prob_func):

            p = p.detach().requires_grad_()
            log_prob, *_ = log_prob_func(p)
            grad = torch.autograd.grad(log_prob,p)[0]
            del p, log_prob, log_prob_func
            torch.cuda.empty_cache()
            return grad

        params = params.detach()
        ret_params = []
        ret_momenta = []
        if integrator == Integrator.SPLITTING:
            M = len(log_prob_func)
            K_div = (M - 1) * 2
            if M == 1:
                raise RuntimeError('For symmetric splitting log_prob_func must be list of functions greater than length 1')
            for n in range(steps):
                for m in range(M):
                    grad = params_grad(params,log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m < M-1:
                            if inv_mass is None:
                                params += (step_size/K_div) * momentum
                            else:
                                if type(inv_mass) is list:
                                    pass
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size/K_div) * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                                else:
                                    params += (step_size/K_div) * inv_mass * momentum
                for m in reversed(range(M)):
                    grad = params_grad(params,log_prob_func[m])
                    with torch.no_grad():
                        momentum += 0.5 * step_size * grad
                        del grad
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if m > 0:
                            if inv_mass is None:
                                params += (step_size/K_div) * momentum
                            else:
                                if type(inv_mass) is list:
                                    pass
                                elif len(inv_mass.shape) == 2:
                                    params += (step_size/K_div) * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                                else:
                                    params += (step_size/K_div) * inv_mass * momentum

                if store_on_GPU:
                    ret_params.append(params.clone())
                    ret_momenta.append(momentum.clone())
                else:
                    ret_params.append(params.clone().cpu())
                    ret_momenta.append(momentum.clone().cpu())
        elif integrator == Integrator.SPLITTING_RAND:
            M = len(log_prob_func)
            idx = torch.randperm(M)
            for n in range(steps):
                for m in range(M):
                    momentum += 0.5 * step_size * params_grad(params, log_prob_func[idx[m]])
                    if inv_mass is None:
                        params += (step_size/M) * momentum
                    else:
                        if type(inv_mass) is list:
                            pass
                        elif len(inv_mass.shape) == 2:
                            params += (step_size/M) * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                        else:
                            params += (step_size/M) * inv_mass * momentum
                    momentum += 0.5 * step_size * params_grad(params,log_prob_func[idx[m]])

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())

        elif integrator == Integrator.SPLITTING_KMID:
            M = len(log_prob_func)
            if M == 1:
                raise RuntimeError('For symmetric splitting log_prob_func must be list of functions greater than length 1')
            for n in range(steps):
                for m in range(M):
                    momentum += 0.5 * step_size * params_grad(params,log_prob_func[m])

                if inv_mass is None:
                    params = params + (step_size) * momentum
                else:
                    if type(inv_mass) is list:
                        pass
                    elif len(inv_mass.shape) == 2:
                        params = params + (step_size) * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                    else:
                        params = params + (step_size) * inv_mass * momentum

                for m in reversed(range(M)):
                    momentum += 0.5 * step_size * params_grad(params,log_prob_func[m])

                ret_params.append(params.clone())
                ret_momenta.append(momentum.clone())

        return ret_params, ret_momenta

    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
    return float(-h_new + h_old)


def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):

    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0
    else:
        alpha = min(1.,float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10*torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1-(1/(t+t0)))*H_t + (1/(t+t0))*(desired_accept_rate - alpha)
    x_new = mu - (t**0.5)/gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new +  (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))

    return step_size, eps_bar, H_t


def rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=1e6, sampler=Sampler.HMC, integrator=Integrator.EXPLICIT, metric=Metric.HESSIAN):

    log_prob, *_ = log_prob_func(params)
    ndim = params.nelement()
    pi_term = ndim * torch.log(2.*torch.tensor(pi))

    fish, abs_eigenvalues = fisher(params, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric)

    if abs_eigenvalues is not None:
        if util.has_nan_or_inf(fish) or util.has_nan_or_inf(abs_eigenvalues):
            print('Invalid Fisher: {} , abs_eigenvalues: {}, params: {}'.format(fish, abs_eigenvalues, params))
            raise util.LogProbError()
    else:
        if util.has_nan_or_inf(fish):
            print('Invalid Fisher: {}, params: {}'.format(fish, params))
            raise util.LogProbError()

    if metric == Metric.SOFTABS:
        log_det_abs = abs_eigenvalues.log().sum()
    else:
        log_det_abs = torch.slogdet(fish)[1]
    fish_inverse_momentum = cholesky_inverse(fish, momentum)
    quadratic_term = torch.matmul(momentum.view(1, -1), fish_inverse_momentum)
    hamiltonian = - log_prob + 0.5 * pi_term + 0.5 * log_det_abs + 0.5 * quadratic_term
    if util.has_nan_or_inf(hamiltonian):
        print('Invalid hamiltonian, log_prob: {}, params: {}, momentum: {}'.format(log_prob, params, momentum))
        raise util.LogProbError()

    return hamiltonian


def hamiltonian(params, momentum, log_prob_func, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.EXPLICIT, metric=Metric.HESSIAN):

    if sampler == Sampler.HMC:
        if type(log_prob_func) is not list:
            log_prob, *_ = log_prob_func(params)

            if util.has_nan_or_inf(log_prob):
                print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
                raise util.LogProbError()

        elif type(log_prob_func) is list:
            log_prob = 0
            for elem_log_prob_func in log_prob_func:
                with torch.no_grad():
                    log_prob = log_prob + elem_log_prob_func(params)

                    if util.has_nan_or_inf(log_prob):
                        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
                        raise util.LogProbError()


        potential = -log_prob
        if inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)
        else:
            if type(inv_mass) is list:
                i = 0
                kinetic = 0
                for block in inv_mass:
                    it = block[0].shape[0]
                    kinetic = kinetic +  0.5 * torch.matmul(momentum[i:it+i].view(1,-1),torch.matmul(block,momentum[i:it+i].view(-1,1))).view(-1)
                    i += it
            elif len(inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
        hamiltonian = potential + kinetic
    elif sampler == Sampler.RMHMC and integrator == Integrator.IMPLICIT:
        hamiltonian = rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
    elif sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
        if type(params) is not list:
            hamiltonian = 2 * rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
        else:
            HA = rm_hamiltonian(params[0], momentum[1], log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
            HB = rm_hamiltonian(params[1], momentum[0], log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
            HC = (0.5 * torch.sum((params[0]-params[1])**2) + 0.5 * torch.sum((momentum[0]-momentum[1])**2))
            hamiltonian = HA + HB + explicit_binding_const * HC
    elif sampler == Sampler.RMHMC and integrator == Integrator.S3:
        log_prob, *_ = log_prob_func(params)
        ndim = params.nelement()
        pi_term = ndim * torch.log(2.*torch.tensor(pi))
        fish, abs_eigenvalues = fisher(params, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric)
        fish_inverse_momentum = cholesky_inverse(fish, momentum)
        quadratic_term = torch.matmul(momentum.view(1, -1), fish_inverse_momentum)
        hamiltonian = - log_prob + 0.5 * quadratic_term + ham_func(params)

        if util.has_nan_or_inf(hamiltonian):
            print('Invalid hamiltonian, log_prob: {}, params: {}, momentum: {}'.format(log_prob, params, momentum))
            raise util.LogProbError()
    else:
        raise NotImplementedError()
    return hamiltonian


def sample(
        log_prob_func, 
        params_init, 
        num_samples=10, 
        num_steps_per_sample=10, 
        step_size=0.1, 
        burn=0, 
        jitter=None, 
        inv_mass=None, 
        normalizing_const=1., 
        softabs_const=None, 
        explicit_binding_const=100, 
        fixed_point_threshold=1e-5, 
        fixed_point_max_iterations=1000, 
        jitter_max_tries=10, 
        sampler=Sampler.HMC, 
        integrator=Integrator.IMPLICIT, 
        metric=Metric.HESSIAN, 
        debug=False, 
        desired_accept_rate=0.8, 
        store_on_GPU = True, 
        pass_grad = None, 
        verbose = True
):

    device = params_init.device

    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples:
        raise RuntimeError('burn must be less than num_samples.')

    NUTS = False
    if sampler == Sampler.HMC_NUTS:
        if burn == 0:
            raise RuntimeError('burn must be greater than 0 for NUTS.')
        sampler = Sampler.HMC
        NUTS = True
        step_size_init = step_size
        H_t = 0.
        eps_bar = 1.

    mass = None
    if inv_mass is not None:
        if type(inv_mass) is list:
            mass = []
            for block in inv_mass:
                mass.append(torch.inverse(block))
        elif len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1 / inv_mass

    params = params_init.clone().requires_grad_()
    param_burn_prev = params_init.clone()
    if not store_on_GPU:
        ret_params = [params.clone().detach().cpu()]
    else:
        ret_params = [params.clone()]

    num_rejected = 0
    if verbose:
        util.progress_bar_init('Sampling ({}; {})'.format(sampler, integrator), num_samples, 'Samples')
    for n in range(num_samples):
        if verbose:
            util.progress_bar_update(n)
        try:
            momentum = gibbs(params, sampler=sampler, log_prob_func=log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric, mass=mass)

            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)

            leapfrog_params, leapfrog_momenta = leapfrog(params, momentum, log_prob_func, sampler=sampler, integrator=integrator, steps=num_steps_per_sample, step_size=step_size, inv_mass=inv_mass, jitter=jitter, jitter_max_tries=jitter_max_tries, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, metric=metric, store_on_GPU = store_on_GPU, debug=debug, pass_grad = pass_grad)
            if sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:

                ham = ham / 2

                params = leapfrog_params[0][-1].detach().requires_grad_()
                params_copy = leapfrog_params[-1].detach().requires_grad_()
                params_copy = params_copy.detach().requires_grad_()
                momentum = leapfrog_momenta[0][-1]
                momentum_copy = leapfrog_momenta[-1]

                leapfrog_params = leapfrog_params[0]
                leapfrog_momenta = leapfrog_momenta[0]

                new_ham = rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)

            else:
                params = leapfrog_params[-1].to(device).detach().requires_grad_()
                momentum = leapfrog_momenta[-1].to(device)
                new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)

            rho = min(0., acceptance(ham, new_ham))
            if debug == 1:
                print('Step: {}, Current Hamiltoninian: {}, Proposed Hamiltoninian: {}'.format(n,ham,new_ham))

            if rho >= torch.log(torch.rand(1)):
                if debug == 1:
                    print('Accept rho: {}'.format(rho))
                if n > burn:
                    if store_on_GPU:
                        ret_params.append(leapfrog_params[-1])
                    else:
                        ret_params.append(leapfrog_params[-1].cpu())
                else:
                    param_burn_prev = leapfrog_params[-1].to(device).clone()
            else:
                if n > burn:
                    num_rejected += 1
                    params = ret_params[-1].to(device)
                    if store_on_GPU:
                        ret_params.append(ret_params[-1].to(device))
                    else:
                        ret_params.append(ret_params[-1].cpu())
                else:
                    params = param_burn_prev.clone()
                if debug == 1:
                    print('REJECT')

            if NUTS and n <= burn:
                if n < burn:
                    step_size, eps_bar, H_t = adaptation(rho, n, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
                if n  == burn:
                    step_size = eps_bar
                    print('Final Adapted Step Size: ',step_size)

        except util.LogProbError:
            params = ret_params[-1].to(device)
            if n > burn:
                num_rejected += 1
                params = ret_params[-1].to(device)
                if store_on_GPU:
                    ret_params.append(ret_params[-1].to(device))
                else:
                    ret_params.append(ret_params[-1].cpu())
            else:
                params = param_burn_prev.clone()
            if debug == 1:
                print('REJECT')
            if NUTS and n <= burn:
                rho = float('nan')
                step_size, eps_bar, H_t = adaptation(rho, n, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
            if NUTS and n  == burn:
                step_size = eps_bar
                print('Final Adapted Step Size: ',step_size)

        if not store_on_GPU:
            momentum = None; leapfrog_params = None; leapfrog_momenta = None; ham = None; new_ham = None

            del momentum, leapfrog_params, leapfrog_momenta, ham, new_ham
            torch.cuda.empty_cache()

    if verbose:
        util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected / (num_samples - burn)))
    if NUTS and debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    elif debug == 2:
        return list(map(lambda t: t.detach(), ret_params)), num_samples - burn - num_rejected
    else:
        return list(map(lambda t: t.detach(), ret_params))


def define_model_log_prob(
    model, 
    model_loss, 
    x, 
    y, 
    params_flattened_list, 
    params_shape_list, 
    tau_list, 
    tau_out, 
    constraint_loss=None, 
    s=None, 
    normalizing_const=1., 
    predict=False, 
    prior_scale=1.0, 
    device='cpu'
):

    fmodel = util.make_functional(model)
    dist_list = []
    for tau in tau_list:
        dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

    def log_prob_func(params):
        params_unflattened = util.unflatten(model, params)

        i_prev = 0
        l_prior = torch.zeros_like( params[0], requires_grad=True)
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        if x is None:
            return l_prior/prior_scale

        x_device = x.to(device)
        y_device = y.to(device)

        output = fmodel(x_device, params=params_unflattened)

        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)
        elif model_loss == 'multi_class_linear_output':
            crit = nn.CrossEntropyLoss(reduction='sum')
            ll = - tau_out *(crit(output, y_device.long().view(-1)))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out *(torch.nn.functional.nll_loss(output, y_device.long().view(-1)))

        elif model_loss == 'regression':
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)

        elif model_loss == 'binary_class_linear_sigmoid_output': 
            crit = nn.BCELoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)

        elif callable(model_loss):
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()
        
        if constraint_loss is not None: 
            s_device = s.to(device)
            if constraint_loss == 'dp': 
                constraint = _fair_loss_dp(output, s_device)
            else: 
                raise NotImplementedError
            
            del s_device
        else: 
            constraint = None

        if torch.cuda.is_available():
            del x_device, y_device
            torch.cuda.empty_cache()

        if not predict: 
            output = None

        return (ll + l_prior / prior_scale), constraint, output

    return log_prob_func


def define_gibbs_model_log_prob(
    model, 
    model_loss, 
    constraint_loss, 
    x, 
    y, 
    s, 
    params_flattened_list, 
    params_shape_list, 
    tau_list, 
    tau_out, 
    lmda, 
    lr_gibbs, 
    normalizing_const=1., 
    predict=False, 
    prior_scale=1.0, 
    device='cpu'
):

    fmodel = util.make_functional(model)
    dist_list = []
    for tau in tau_list:
        dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

    def log_prob_func(params):
        params_unflattened = util.unflatten(model, params)

        i_prev = 0
        l_prior = torch.zeros_like(params[0], requires_grad=True)
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        if x is None:
            return l_prior / prior_scale

        x_device = x.to(device)
        y_device = y.to(device)
        s_device = s.to(device)
        n = x_device.shape[0]

        output = fmodel(x_device, params=params_unflattened)

        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)
        elif model_loss == 'multi_class_linear_output':
            crit = nn.CrossEntropyLoss(reduction='sum')
            ll = - tau_out * crit(output, y_device.long().view(-1))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out * torch.nn.functional.nll_loss(output, y_device.long().view(-1))

        elif model_loss == 'regression':
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)
        elif model_loss == 'binary_class_linear_sigmoid_output': 
            crit = nn.BCELoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)

        elif callable(model_loss):
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if constraint_loss == 'dp': 
            constraint = _fair_loss_dp(output, s_device)
        elif constraint_loss == 'mmd': 
            constraint = _fair_loss_mmd(output, s_device)
        elif constraint_loss == 'wdp':
            constraint = _fair_loss_wdp(output, s_device)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            del x_device, y_device, s_device
            torch.cuda.empty_cache()

        if not predict: 
            output = None

        return lr_gibbs * (ll - n * lmda * constraint) + l_prior / prior_scale, constraint, output

    return log_prob_func


def define_gibbs_mdp_model_log_prob(
    model, 
    model_loss, 
    constraint_loss, 
    x, 
    y, 
    s, 
    matching, 
    params_flattened_list, 
    params_shape_list, 
    tau_list, 
    tau_out, 
    lmda, 
    lr_gibbs, 
    normalizing_const=1., 
    predict=False, 
    prior_scale=1.0, 
    device='cpu'
):

    fmodel = util.make_functional(model)
    dist_list = []
    for tau in tau_list:
        dist_list.append(torch.distributions.Normal(torch.zeros_like(tau), tau**-0.5))

    def log_prob_func(params):
        params_unflattened = util.unflatten(model, params)

        i_prev = 0
        l_prior = torch.zeros_like(params[0], requires_grad=True)
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        if x is None:
            return l_prior / prior_scale

        x_device = x.to(device)
        y_device = y.to(device)
        s_device = s.to(device)
        n = x_device.shape[0]

        output = fmodel(x_device, params=params_unflattened)

        if model_loss == 'binary_class_linear_output':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)
        elif model_loss == 'multi_class_linear_output':
            crit = nn.CrossEntropyLoss(reduction='sum')
            ll = - tau_out * crit(output, y_device.long().view(-1))
        elif model_loss == 'multi_class_log_softmax_output':
            ll = - tau_out * torch.nn.functional.nll_loss(output, y_device.long().view(-1))

        elif model_loss == 'regression':
            ll = - 0.5 * tau_out * ((output - y_device) ** 2).sum(0)
        elif model_loss == 'binary_class_linear_sigmoid_output': 
            crit = nn.BCELoss(reduction='sum')
            ll = - tau_out * crit(output, y_device)

        elif callable(model_loss):
            ll = - model_loss(output, y_device).sum(0)
        else:
            raise NotImplementedError()

        if constraint_loss == 'mdp': 
            constraint = _fair_loss_mdp(output, matching, s_device)
        else:
            raise NotImplementedError()

        if torch.cuda.is_available():
            del x_device, y_device, s_device
            torch.cuda.empty_cache()

        if not predict: 
            output = None

        return lr_gibbs * (ll - n * lmda * constraint) + l_prior / prior_scale, constraint, output

    return log_prob_func


def sample_model(
        model, 
        x, 
        y, 
        params_init, 
        s=None, 
        model_loss='multi_class_linear_output', 
        num_samples=10, 
        num_steps_per_sample=10, 
        step_size=0.1, 
        burn=0, 
        inv_mass=None, 
        jitter=None, 
        normalizing_const=1., 
        softabs_const=None, 
        explicit_binding_const=100, 
        fixed_point_threshold=1e-5, 
        fixed_point_max_iterations=1000, 
        jitter_max_tries=10, 
        sampler=Sampler.HMC, 
        integrator=Integrator.IMPLICIT, 
        metric=Metric.HESSIAN, 
        debug=False, 
        tau_out=1., 
        tau_list=None, 
        store_on_GPU=True, 
        desired_accept_rate=0.8, 
        verbose = True
):

    device = params_init.device
    params_shape_list = []
    params_flattened_list = []
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(torch.tensor(1.))

    log_prob_func = define_model_log_prob(
        model, 
        model_loss, 
        x, 
        y, 
        params_flattened_list, 
        params_shape_list, 
        tau_list, 
        tau_out, 
        s=s, 
        normalizing_const=normalizing_const, 
        device=device
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample(
        log_prob_func, 
        params_init, 
        num_samples=num_samples, 
        num_steps_per_sample=num_steps_per_sample, 
        step_size=step_size, 
        burn=burn, 
        jitter=jitter, 
        inv_mass=inv_mass, 
        normalizing_const=normalizing_const, 
        softabs_const=softabs_const, 
        explicit_binding_const=explicit_binding_const, 
        fixed_point_threshold=fixed_point_threshold, 
        fixed_point_max_iterations=fixed_point_max_iterations, 
        jitter_max_tries=jitter_max_tries, 
        sampler=sampler, 
        integrator=integrator, 
        metric=metric, 
        debug=debug, 
        desired_accept_rate=desired_accept_rate, 
        store_on_GPU=store_on_GPU, 
        verbose=verbose
    )


def sample_gibbs_model(
        model, 
        x, 
        y, 
        s, 
        params_init, 
        lmda, 
        lr_gibbs=1., 
        model_loss='binary_class_linear_sigmoid_output', 
        constraint_loss='dp', 
        num_samples=10, 
        num_steps_per_sample=10, 
        step_size=0.1, 
        burn=0, 
        inv_mass=None, 
        jitter=None, 
        normalizing_const=1., 
        softabs_const=None, 
        explicit_binding_const=100, 
        fixed_point_threshold=1e-5, 
        fixed_point_max_iterations=1000, 
        jitter_max_tries=10, 
        sampler=Sampler.HMC, 
        integrator=Integrator.IMPLICIT, 
        metric=Metric.HESSIAN, 
        debug=False, 
        tau_out=1., 
        tau_list=None, 
        store_on_GPU=True, 
        desired_accept_rate=0.8, 
        verbose = True
):

    device = params_init.device
    params_shape_list = []
    params_flattened_list = []
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(torch.tensor(1.))

    log_prob_func = define_gibbs_model_log_prob(
        model=model, 
        model_loss=model_loss, 
        constraint_loss=constraint_loss, 
        x=x, 
        y=y, 
        s=s, 
        params_flattened_list=params_flattened_list, 
        params_shape_list=params_shape_list, 
        tau_list=tau_list, 
        tau_out=tau_out, 
        lmda=lmda, 
        lr_gibbs=lr_gibbs, 
        normalizing_const=normalizing_const, 
        device=device
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample(log_prob_func, params_init, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, jitter=jitter, inv_mass=inv_mass, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, jitter_max_tries=jitter_max_tries, sampler=sampler, integrator=integrator, metric=metric, debug=debug, desired_accept_rate=desired_accept_rate, store_on_GPU = store_on_GPU, verbose = verbose)


def sample_gibbs_mdp_model(
        model, 
        x, 
        y, 
        s, 
        matching, 
        params_init, 
        lmda, 
        lr_gibbs=1., 
        model_loss='binary_class_linear_sigmoid_output', 
        constraint_loss='dp', 
        num_samples=10, 
        num_steps_per_sample=10, 
        step_size=0.1, 
        burn=0, 
        inv_mass=None, 
        jitter=None, 
        normalizing_const=1., 
        softabs_const=None, 
        explicit_binding_const=100, 
        fixed_point_threshold=1e-5, 
        fixed_point_max_iterations=1000, 
        jitter_max_tries=10, 
        sampler=Sampler.HMC, 
        integrator=Integrator.IMPLICIT, 
        metric=Metric.HESSIAN, 
        debug=False, 
        tau_out=1., 
        tau_list=None, 
        store_on_GPU=True, 
        desired_accept_rate=0.8, 
        verbose = True
):

    device = params_init.device
    params_shape_list = []
    params_flattened_list = []
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(torch.tensor(1.))

    log_prob_func = define_gibbs_mdp_model_log_prob(
        model=model, 
        model_loss=model_loss, 
        constraint_loss=constraint_loss, 
        x=x, 
        y=y, 
        s=s, 
        matching=matching, 
        params_flattened_list=params_flattened_list, 
        params_shape_list=params_shape_list, 
        tau_list=tau_list, 
        tau_out=tau_out, 
        lmda=lmda, 
        lr_gibbs=lr_gibbs, 
        normalizing_const=normalizing_const, 
        device=device
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample(log_prob_func, params_init, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, jitter=jitter, inv_mass=inv_mass, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, jitter_max_tries=jitter_max_tries, sampler=sampler, integrator=integrator, metric=metric, debug=debug, desired_accept_rate=desired_accept_rate, store_on_GPU = store_on_GPU, verbose = verbose)


def predict_model(
        model, 
        samples, 
        x=None, 
        y=None, 
        s=None, 
        test_loader=None, 
        model_loss='binary_class_linear_sigmoid_output', 
        constraint_loss=None, 
        tau_out=1., 
        tau_list=None, 
        verbose=False
):

    with torch.no_grad():
        params_shape_list = []
        params_flattened_list = []
        build_tau = False
        if tau_list is None:
            tau_list = []
            build_tau = True
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau:
                tau_list.append(torch.tensor(1.))

        if x.device != samples[0].device:
            raise RuntimeError('x on device: {} and samples on device: {}'.format(x.device, samples[0].device))

        log_prob_func = define_model_log_prob(
            model, 
            model_loss, 
            x, 
            y, 
            params_flattened_list, 
            params_shape_list, 
            tau_list, 
            tau_out, 
            constraint_loss=constraint_loss, 
            s=s, 
            predict=True, 
            device=samples[0].device
        )

        pred_log_prob_list = []
        constraint_list = []
        pred_list = []
        for s in samples:
            lp, c, pred = log_prob_func(s)
            pred_log_prob_list.append(lp.detach())
            if constraint_loss is not None:
                constraint_list.append(c.detach().cpu())
            pred_list.append(pred.detach())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if constraint_loss is None:
        return torch.stack(pred_list), pred_log_prob_list
    else:
        return torch.stack(pred_list), pred_log_prob_list, constraint_list