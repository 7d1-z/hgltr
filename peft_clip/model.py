import torch
from torch import nn
import torch.nn.functional as F

import clip
from clip.model import CLIP
from peft_clip.peft_resnet import Peft_RN, RN_Tuner
from peft_clip.peft_vit import ViT_Tuner, Peft_ViT
from peft_clip.classifier import get_clip_classifier
from pipeline.module import Aligner
from util.util import get_hierarchy_text


class CLIP_Text(nn.Module): # totally copied from CLIP.encode_text
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).to(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).to(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def encode_text(encoder:nn.Module, text: torch.Tensor) -> torch.Tensor:
    try:
        text_features = encoder(text)
    except torch.OutOfMemoryError:
        text_split = torch.split(text, 1000)
        text_features = torch.cat([encoder(x) for x in text_split])
    return text_features

@torch.no_grad()
def init_fixed_text_features(module: nn.Module, text: torch.Tensor):
    assert hasattr(module, 'text_encoder'), "Module must have a text_encoder attribute."
    text_features = encode_text(module.text_encoder, text)
    return F.normalize(text_features, dim=-1)

def get_tokenized_prompt(classname: list[str], dataset: str) -> torch.Tensor:
    assert classname is None or dataset is None, "Only one of classname or dataset can be specified."
    if classname:
        prompts = [f"a photo of {c}" for c in classname]
    else:
        hierarchy_info = get_hierarchy_text(dataset)
        prompts = [f"a photo of {p}" for p in hierarchy_info]
    prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
    return prompts

def get_zero_shot_text_features(dataset: str, module: nn.Module, classname: list[str], device: torch.device):
    """
    register text_features buffer in module
    """
    prompts = get_tokenized_prompt(classname, dataset).to(device)
    module = module.to(device)
    return init_fixed_text_features(module, prompts)


class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def init_zero_shot_text_features(self, features: torch.Tensor):
        self.register_buffer("text_features", features)

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))

    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model: CLIP, num_classes):
        super().__init__()
        
        if cfg.backbone.startswith("CLIP-ViT"):
            self.image_encoder = Peft_ViT(clip_model.visual)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
        elif cfg.backbone.startswith("CLIP-RN"):
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")
        self.text_encoder = CLIP_Text(clip_model)
        self.feat_dim = self.image_encoder.out_dim
        self.dtype = self.image_encoder.dtype
        self.head = get_clip_classifier(cfg.head, self.feat_dim, num_classes, self.dtype)
        self.ensemble = cfg.ensemble
        self.clip_model = clip_model
        if self.ensemble:
            self.text_learner = Aligner(512, self.feat_dim)
            self.logit_scale = clip_model.logit_scale.exp()
        else:
            self.text_learner = None
            self.logit_scale = None

    def init_zero_shot_text_features(self, text_features: torch.Tensor):
        # text_features: [num_classes, 512], a fixed text prototype features for classes
        self.register_buffer("text_features", text_features.detach(), persistent=False)
    
    def forward(self, image, use_tuner=True, return_feature=False, ensemble=True):
        if self.ensemble and ensemble:
            return self.ensemble_forward(image, use_tuner)
        else:
            head = None if return_feature else self.head
            tuner = self.tuner if use_tuner else None
            return self.image_encoder(image, tuner, head)
    
    def ensemble_forward(self, image, use_tuner=True):
        """
        zero-shot classifier + linear classifier
        """
        tuner = self.tuner if use_tuner else None
        image_features = self.image_encoder(image, tuner, None)
        linear_logit = self.head(image_features)

        text_features = self.text_learner(self.text_features.detach())
        # text_features = F.normalize(text_features, dim=-1)
        # zero_logit = self.logit_scale * (image_features @ text_features.t())

        # logit = zero_logit + linear_logit
        return {"logit": linear_logit, "image_features": image_features, "text_features": text_features}