import math
import torch
from torch.optim.optimizer import Optimizer


class Ranger(Optimizer):
    def __init__(
            self, params, lr=0.0005, alpha=0.5, k=5, N_sma_threshold=5, betas=(0.9, 0.999),
            eps=1e-5, weight_decay=0.000):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
        # @TODO Implement the above testing with AX ^

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas,
                        N_sma_threshold=N_sma_threshold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Since we keep LR the same for all param groups,
        # store it here for now for quick&easy access if we want to know it
        self.lr = lr

        # adjustable threshold
        self.N_sma_threshold = N_sma_threshold

        # look ahead params
        self.initial_lr = lr
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data.float()
                    if grad.is_sparse:
                        raise RuntimeError('Ranger optimizer does not support sparse gradients')

                    p_data_fp32 = p.data.float()

                    state = self.state[p]  # get state dict for this param

                    # On the first run initialize the dictionary for each weight group
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_data_fp32)
                        state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                        # look ahead weight storage now in state dict
                        state['slow_buffer'] = torch.empty_like(p.data)
                        state['slow_buffer'].copy_(p.data)
                    # @TODO Couldn't this branch happen after the if above is entered
                    # in thus replacing torch.zero_like) ??
                    else:
                        state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                        state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) /
                                              N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    # Find the interpolated weight between the slower buffer (the weight `k` steps ago)
                    # and the current weight, set that as the state for RAdam
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)

        return loss
