import argparse
import torch
import torch.distributed as dist
import collections
from torch.optim.optimizer import Optimizer, required
import math
import matplotlib.pyplot as plt
import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from dgl.data.tree import SSTDataset
from torch.utils.data import DataLoader
from tree_lstm import TreeLSTM
from tqdm import tqdm

# NAG Optimizer
class NAG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super(NAG, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            lr_old = group.get("lr_old", lr)
            lr_correct = lr / lr_old if lr_old > 0 else lr

            for p in group["params"]:
                if p.grad is None:
                    continue

                p_data_fp32 = p.data
                if p_data_fp32.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                d_p = p.grad.data.float()
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.zeros_like(d_p)
                else:
                    param_state["momentum_buffer"] = param_state["momentum_buffer"].to(
                        d_p
                    )

                buf = param_state["momentum_buffer"]

                if weight_decay != 0:
                    p_data_fp32.mul_(1 - lr * weight_decay)
                p_data_fp32.add_(buf, alpha=momentum * momentum * lr_correct)
                p_data_fp32.add_(d_p, alpha=-(1 + momentum) * lr)

                buf.mul_(momentum * lr_correct).add_(d_p, alpha=-lr)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

            group["lr_old"] = lr

        return loss

# ProxSG
class ProxSG(Optimizer):
    def __init__(self, params, lr=required, lambda_=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if lambda_ is not required and lambda_ < 0.0:
            raise ValueError("Invalid lambda: {}".format(lambda_))

        defaults = dict(lr=lr, lambda_=lambda_)
        super(ProxSG, self).__init__(params, defaults)

    def calculate_d(self, x, grad_f, lambda_, lr):
        '''
            Calculate d for Omega(x) = ||x||_1
        '''
        trial_x = torch.zeros_like(x)
        pos_shrink = x - lr * grad_f - lr * \
            lambda_  # new x is larger than lr * lambda_
        neg_shrink = x - lr * grad_f + lr * \
            lambda_  # new x is less than -lr * lambda_
        pos_shrink_idx = (pos_shrink > 0)
        neg_shrink_idx = (neg_shrink < 0)
        trial_x[pos_shrink_idx] = pos_shrink[pos_shrink_idx]
        trial_x[neg_shrink_idx] = neg_shrink[neg_shrink_idx]
        d = trial_x - x

        return d

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad_f = p.grad.data

                if len(p.shape) > 1:  # weights
                    s = self.calculate_d(
                        p.data, grad_f, group['lambda_'], group['lr'])
                    p.data.add_(s, alpha=1)
                else:  # bias
                    p.data.add_(grad_f, alpha=-group['lr'])
        return loss

# SVRG Optimizer
class SVRG(Optimizer):
    r""" implement SVRG """ 

    def __init__(self, params, lr=required, freq =10):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, freq=freq)
        self.counter = 0
        self.counter2 = 0
        self.flag = False
        super(SVRG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('m', )

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            freq = group['freq']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                if 'large_batch' not in param_state:
                    buf = param_state['large_batch'] = torch.zeros_like(p.data)
                    buf.add_(d_p) #add first large, low variance batch
                    #need to add the second term in the step equation; the gradient for the original step!
                    buf2 = param_state['small_batch'] = torch.zeros_like(p.data)

                buf = param_state['large_batch']
                buf2 = param_state['small_batch']

                if self.counter == freq:
                    buf.data = d_p.clone() #copy new large batch. Begining of new inner loop
                    temp = torch.zeros_like(p.data)
                    buf2.data = temp.clone()
                    
                if self.counter2 == 1:
                    buf2.data.add_(d_p) #first small batch gradient for inner loop!

                #dont update parameters when computing large batch (low variance gradients)
                if self.counter != freq and self.flag != False:
                    p.data.add_((d_p - buf2 + buf), alpha=-group['lr'])

        self.flag = True #rough way of not updating the weights the FIRST time we calculate the large batch gradient
        
        if self.counter == freq:
            self.counter = 0
            self.counter2 = 0

        self.counter += 1    
        self.counter2 += 1

        return loss
    
class Prodigy(Optimizer):
    r"""
    Implements Adam with Prodigy step-sizes.
    Leave LR set to 1 unless you encounter instability.
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the Prodigy learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        beta3 (float):
            coefficients for computing the Prodidy stepsize using running averages.
            If set to None, uses the value of square root of beta2 (default: None).
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        safeguard_warmup (boolean):
            Remove lr from the denominator of D estimate to avoid issues during warm-up stage. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        d_coef (float):
            Coefficient in the expression for the estimate of d (default 1.0).
            Values such as 0.5 and 2.0 typically work as well. 
            Changing this parameter is the preferred way to tune the method.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """
    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True, 
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple and weight_decay > 0:
            print(f"Using decoupled weight decay")

       
        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use)
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
        else:
            bias_correction = 1

        dlr = d*lr*bias_correction
       
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True
               
                grad = p.grad.data
               
                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data).detach()
                    state['p0'] = p.detach().clone()
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
               
                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    d_numerator += (d / d0) * dlr * torch.dot(grad.flatten(), (p0.data - p.data).flatten()).item()

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1-beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1-beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(grad, alpha=((d / d0) * dlr))
                    d_denom += s.abs().sum().item()

            ######

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0:
            return loss
       
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                denom = exp_avg_sq.sqrt().add_(d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)


                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-dlr)

            group['k'] = k + 1

        return loss
    
class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False
                 ):

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

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']
                                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size*
                                         group['lr'])

                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "mask", "wordid", "label"]
)

def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(
            graph=batch_trees,
            mask=batch_trees.ndata["mask"].to(device),
            wordid=batch_trees.ndata["x"].to(device),
            label=batch_trees.ndata["y"].to(device),
        )

    return batcher_dev

def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    cuda = args.gpu >= 0
    device = th.device("cuda:{}".format(args.gpu)) if cuda else th.device("cpu")
    if cuda:
        th.cuda.set_device(args.gpu)

    trainset = SSTDataset()
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=batcher(device),
        shuffle=True,
        num_workers=0,
    )
    devset = SSTDataset(mode="dev")
    dev_loader = DataLoader(
        dataset=devset,
        batch_size=100,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0,
    )

    testset = SSTDataset(mode="test")
    test_loader = DataLoader(
        dataset=testset,
        batch_size=100,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0,
    )

    optimizers = [optim.SGD, optim.Adam, optim.RMSprop, optim.Adadelta, Ranger, Prodigy, NAG, ProxSG, SVRG]
    optimizer_names = ['SGD', 'Adam', 'RMSProp', 'Adadelta', 'Ranger', 'Prodigy', 'NAG', 'ProxSG', 'SVRG']

    # Initialize dictionaries to store the results
    train_loss = {}
    test_accuracy = {}
    test_root_accuracy = {}
    convergence_rate = {}

    # Loop over each optimizer
    for opt_fn, opt_name in zip(optimizers, optimizer_names):
        best_dev_acc = 0
        print(f"\nTraining with {opt_name} optimizer:")

        # Initialize your model and optimizer
        model = TreeLSTM(
            trainset.vocab_size,
            args.x_size,
            args.h_size,
            trainset.num_classes,
            args.dropout,
            cell_type="childsum" if args.child_sum else "nary",
            pretrained_emb=trainset.pretrained_emb,
        ).to(device)

        params_ex_emb = [
            x
            for x in list(model.parameters())
            if x.requires_grad and x.size(0) != trainset.vocab_size
        ]
        params_emb = list(model.embedding.parameters())

        for p in params_ex_emb:
            if p.dim() > 1:
                INIT.xavier_uniform_(p)

        if opt_name == 'Prodigy' or opt_name == 'NAG':
            optimizer = opt_fn(model.parameters(), lr=args.lr)
        elif opt_name == 'ProxSG':
            optimizer = opt_fn(model.parameters(), lr=args.lr, lambda_=0.0001)
        else:
            optimizer = opt_fn(
                [
                    {
                        "params": params_ex_emb,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                    },
                    {"params": params_emb, "lr": args.lr},
                ]
            )

        losses = []

        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
            for step, batch in enumerate(pbar):
                g = batch.graph.to(device)
                n = g.num_nodes()
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)

                logits = model(batch, g, h, c)
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp, batch.label, reduction="sum")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss =+ loss.item()
                pbar.set_postfix({'loss': epoch_loss / (step+1)})

            # Append the average loss for this epoch
            losses.append(epoch_loss / len(train_loader))

            # eval on dev set
            accs = []
            root_accs = []
            model.eval()
            for step, batch in enumerate(dev_loader):
                g = batch.graph.to(device)
                n = g.num_nodes()
                with th.no_grad():
                    h = th.zeros((n, args.h_size)).to(device)
                    c = th.zeros((n, args.h_size)).to(device)
                    logits = model(batch, g, h, c)

                pred = th.argmax(logits, 1)
                acc = th.sum(th.eq(batch.label, pred)).item()
                accs.append([acc, len(batch.label)])
                root_ids = [
                    i for i in range(g.num_nodes()) if g.out_degrees(i) == 0
                ]
                root_acc = np.sum(
                    batch.label.cpu().data.numpy()[root_ids]
                    == pred.cpu().data.numpy()[root_ids]
                )
                root_accs.append([root_acc, len(root_ids)])

                # Calculate the loss
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp, batch.label, reduction="sum")
                val_loss =+ loss.item()

            dev_acc = (
                1.0 * np.sum([x[0] for x in accs]) / np.sum([x[1] for x in accs])
            )
            dev_root_acc = (
                1.0
                * np.sum([x[0] for x in root_accs])
                / np.sum([x[1] for x in root_accs])
            )

            if dev_root_acc > best_dev_acc:
                best_dev_acc = dev_root_acc
                th.save(model.state_dict(), "best_{}.pkl".format(opt_name))

            print("Validation loss: ", val_loss / len(dev_loader))
            print("Validation accuracy: ", dev_acc)
            print("Validation root accuracy: ", dev_root_acc)

            # lr decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(1e-5, param_group["lr"] * 0.99)  # 10
                #print(param_group["lr"])

        # test
        model.load_state_dict(th.load("best_{}.pkl".format(opt_name)))
        model.eval()
        accs = []
        root_accs = []
        test_loss = 0
        model.eval()
        pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for step, batch in enumerate(pbar):
            g = batch.graph.to(device)
            n = g.num_nodes()
            with th.no_grad():
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)
                logits = model(batch, g, h, c)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])
            root_ids = [i for i in range(g.num_nodes()) if g.out_degrees(i) == 0]
            root_acc = np.sum(
                batch.label.cpu().data.numpy()[root_ids]
                == pred.cpu().data.numpy()[root_ids]
            )
            root_accs.append([root_acc, len(root_ids)])

            # Calculate the loss
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction="sum")
            test_loss =+ loss.item()
            pbar.set_postfix({'loss': test_loss / (step+1)})
            
        test_acc = 1.0 * np.sum([x[0] for x in accs]) / np.sum([x[1] for x in accs])
        test_root_acc = (
            1.0
            * np.sum([x[0] for x in root_accs])
            / np.sum([x[1] for x in root_accs])
        )
        print("Testing loss: ", test_loss / len(train_loader))
        print("Testing accuracy: ", test_acc)
        print("Testing root accuracy: ", test_root_acc)

        conv_rate = [losses[i] - losses[i-1] for i in range(1, len(losses))]

        # Store the results in the dictionaries
        train_loss[opt_name] = losses
        test_accuracy[opt_name] = test_acc*100
        test_root_accuracy[opt_name] = test_root_acc*100
        convergence_rate[opt_name] = conv_rate

    fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # Create 4 subplots

    # Plot losses
    for opt_name in optimizer_names:
        losses = train_loss[opt_name]
        axs[0].plot(losses, label=f'{opt_name}')
    axs[0].set_title('Training Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot test accuracies
    for opt_name in optimizer_names:
        accuracy = test_accuracy[opt_name]
        bar = axs[1].bar(opt_name, accuracy, label=f'{opt_name}')
        axs[1].text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), f'{accuracy:.2f}%', ha='center', va='bottom')
    axs[1].set_title('Test Accuracy')
    axs[1].set_xlabel('Optimizer')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()

    # Plot test root accuracies
    for opt_name in optimizer_names:
        root_accuracy = test_root_accuracy[opt_name]
        bar = axs[2].bar(opt_name, root_accuracy, label=f'{opt_name}')
        axs[2].text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(), f'{root_accuracy:.2f}%', ha='center', va='bottom')
    axs[2].set_title('Test Root Accuracy')
    axs[2].set_xlabel('Optimizer')
    axs[2].set_ylabel('Root Accuracy (%)')
    axs[2].legend()

    # Plot convergence rates
    for opt_name in optimizer_names:
        losses = train_loss[opt_name]
        convergence_rates = [losses[i] - losses[i-1] for i in range(1, len(losses))]
        axs[3].plot(range(1, len(losses)), convergence_rates, label=f'{opt_name}')
    axs[3].set_title('Convergence Rate')
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('Convergence Rate')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--child-sum", action="store_true")
    parser.add_argument("--x-size", type=int, default=300)
    parser.add_argument("--h-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    main(args)