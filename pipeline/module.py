import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CriterionWrapper(nn.Module):
    def __init__(self,
                 softlabel: torch.Tensor,
                 criterion: nn.Module, temperature: float,
                 w_dis: float, w_mse: float,
                 constrain: bool,
                 few_shot: np.ndarray):

        super().__init__()
        if w_mse + w_dis > 0:
            self.register_buffer('softlabel', softlabel)
            self.register_buffer('few_shot', torch.from_numpy(few_shot))
        
        self.criterion = criterion
        self.w_dis = w_dis
        self.w_mse = w_mse
        self.temperature = temperature
        self.constrain = constrain
    def weighted_mean(self, loss_per_sample: torch.Tensor, weights: torch.Tensor):
        weighted_loss = loss_per_sample * weights
        valid_sum = weights.sum()
        if valid_sum > 0:
            return weighted_loss.sum() / valid_sum
        else:
            return torch.tensor(0.0, device=loss_per_sample.device, dtype=loss_per_sample.dtype)

    def mask_for_batch(self, targets: torch.Tensor) -> torch.Tensor:
        batch_class_weight = torch.ones(targets.size(0), dtype=torch.long).to(targets.device)  # [B]
        if self.constrain:
            is_few_shot = torch.isin(targets, self.few_shot) # [B]
            batch_class_weight[is_few_shot] = 0.0
        return batch_class_weight
    
    def forward(self, outputs: dict|torch.Tensor, targets: torch.Tensor):
        if isinstance(outputs, dict):
            logits = outputs['logit']                # [B, C]
        else:
            logits = outputs

        loss_cls = self.criterion(logits, targets)
        loss_base = loss_cls
        loss_recorder = {'loss_cls': loss_cls.item()}

        if self.w_dis + self.w_mse > 0:
            batch_class_weight = self.mask_for_batch(targets)  # [B]

        if self.w_dis > 0:
            log_probs = F.log_softmax(logits / self.temperature, dim=1)
            kl_per_sample = F.kl_div(
                log_probs, self.softlabel[targets],
                reduction='none', log_target=False
            ).mean(dim=-1)
            kl_weighted = self.weighted_mean(kl_per_sample, batch_class_weight)
            loss_recorder['loss_soft'] = kl_weighted.item()
            loss_base += self.w_dis * kl_weighted

        if self.w_mse > 0:
            image_features = outputs['image_features']  # [B, D]
            text_features = outputs['text_features']    # [C, D]
            image_norm = F.normalize(image_features, dim=1)
            text_norm = F.normalize(text_features, dim=1)
            cos_sim = torch.mm(image_norm, text_norm.t())  # [B, C]

            mse_per_sample = F.mse_loss(
                F.softmax(cos_sim, dim=1), self.softlabel[targets], reduction='mean'
            ).mean(dim=-1)
            mse_weighted = self.weighted_mean(mse_per_sample, batch_class_weight)
            loss_recorder['loss_mse'] = mse_weighted.item()
            loss_base += self.w_mse * mse_weighted

        return loss_base, loss_recorder



class Aligner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.kaiming_normal_(self.linear.weight.data)

    def forward(self, x):
        x = self.ln(x)
        return self.linear(x)

