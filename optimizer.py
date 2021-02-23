import torch
import math

from torch.optim.optimizer import Optimizer


class MD_with_pnorm(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, dampening=0, p=3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 2.0 <= p:
            raise ValueError("Invalid p-norm value: {}".format(p))
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, p=p)
        super(MD_with_pnorm, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(MD_with_pnorm, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p = p[0, 0]
                grad = grad[0, 0]

                step_size = group['lr']
                
                q = group['p']/(group['p']-1)
                pw = 1/group['p']*(2/group['p']-1)
                qw = 1/q*(2/q-1)
                
                tilt_x = torch.sign(p.data)*torch.abs(p.data)**(group['p'] - 1)
                pow1 = torch.abs(p.data)**group['p']
                p_norm = torch.sum(pow1)**pw
                tilt_x = tilt_x*p_norm-step_size*grad
                q_norm = torch.pow(torch.sum(torch.pow(torch.abs(tilt_x), q)), qw)
                p.data = torch.sign(tilt_x)*torch.abs(tilt_x)**(q - 1)*q_norm
                # update = (group['p']) * (torch.abs(p.data) ** (group['p'] - 1)) * torch.sign(p.data) - step_size * grad
                # p.data = (torch.abs(update / (group['p'])) ** (1 / (group['p'] - 2))) * torch.sign(update)
        
        return loss

