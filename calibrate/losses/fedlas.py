from torch.nn import Module, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F



class STE_DCG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output[input > 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DCG(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, input_dim//2)
        self.fc_out = nn.Linear(input_dim//2, 2)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        out = self.fc_out(x)

        scaling_factor = F.softmax(out, dim=1)
        scaling_factor = scaling_factor[:, 1] - scaling_factor[:, 0]
        scaling_factor = STE_DCG.apply(scaling_factor)

        return scaling_factor

class NCI(Module):
    def __init__(self,
                 t_alpha=0.9,
                 ):
        super(NCI, self).__init__()
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)
        self.eps = 1e-3

    def forward(self, safe_norms):
        safe_norms = safe_norms.clone().detach()
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            if self.training:
                self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
                self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        return margin_scaler 
    

class FEDLAS(nn.Module):
    def __init__(self, beta=4, alpha=0.05, ignore_index=-100, reduction="mean", momentum=0.95,  logit_size=None):
        super(FEDLAS, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.DCG = DCG(logit_size)
        self.NCI = NCI(t_alpha=momentum)


    def forward(self, input, target, features):    

        n_classes = input.size()[-1]
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1)

    
        norms = features.norm(p=1,dim=1)
        scaled_norms = self.NCI(norms.detach())
        scaling_term = self.DCG(input.detach())

        margins = scaling_term*scaled_norms
        residual = 2*self.alpha*torch.sigmoid(self.beta*margins) 
        confidence = 1- residual

        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        loss =  (confidence * nll_loss + residual * smooth_loss) 

        if self.reduction == "mean":
            return loss.mean() 
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        

class FEDLASplus(nn.Module):
    def __init__(self, beta=0.5, alpha=0.1, ignore_index=-100, reduction="mean", momentum=0.95,margin=6, feature_size=None, logit_size=None):
        super(FEDLASplus, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.DCG = DCG(logit_size)
        self.NCI = NCI(t_alpha=momentum)
        self.margin = margin

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, target, features):   #Logits: pre-softmax outputs, Features: Global CLS Token or Globally Pooled Features. 

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1)

        norms = features.norm(p=1,dim=1)
        #do not update when evaluating the validation set, only update when training.
        scaled_norms = self.NCI(norms.detach())
        scaling_term = self.DCG(inputs.detach())

        margins = scaling_term*scaled_norms
        residual = 2*self.alpha*torch.sigmoid(self.beta*margins) 
        confidence = 1- residual
        
        diff = self.get_diff(inputs)
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        margin_loss = F.relu(diff-self.margin) #.detach()

        loss_ce = (confidence*(nll_loss)).mean() 
        loss_margin = (residual*(margin_loss.mean(dim=-1))).mean() 
        loss = loss_ce + loss_margin

        return loss
        
