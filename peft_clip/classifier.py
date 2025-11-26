import torch
import torch.nn as nn
import torch.nn.functional as F



class BaseClassifier(nn.Module):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()


class LinearClassifier(BaseClassifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__(feat_dim, num_classes, dtype)
        nn.init.kaiming_normal_(self.weight.data)
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class CosineClassifier(BaseClassifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale


class L2NormedClassifier(BaseClassifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__(feat_dim, num_classes, dtype)

    def forward(self, x):
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)


class LayerNormedClassifier(BaseClassifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None):
        super().__init__(feat_dim, num_classes, dtype)
        self.ln = nn.LayerNorm(feat_dim, elementwise_affine=False, eps=1e-12, dtype=dtype)

    def forward(self, x):
        x = self.ln(x)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight)

def get_clip_classifier(name, dim, num_classes, dtype) -> BaseClassifier:
    if name == 'cosine':
        return CosineClassifier(dim, num_classes, dtype)
    elif name == 'l2':
        return L2NormedClassifier(dim, num_classes, dtype)
    elif name == 'ln':
        return LayerNormedClassifier(dim, num_classes, dtype)
    elif name == 'linear':
        return LinearClassifier(dim, num_classes, dtype)
    else:
        raise ValueError(f"Undefined head type: {name}")