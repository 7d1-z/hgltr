import argparse

def parse_args():
    parse = argparse.ArgumentParser()
    #### dataset and dataloader parameters
    parse.add_argument('--batch_size', type=int, default=256)
    parse.add_argument('--num_workers', type=int, default=32)
    parse.add_argument('--seed', type=int, default=202506)
    parse.add_argument('--epochs', type=int, default=100)
    parse.add_argument('--lr', type=float, default=0.01)
    parse.add_argument('--optim', default='sgd', choices=['sgd', 'adamw'])
    parse.add_argument('--dataset', default='cifar100',
                       choices=['cifar10', 'cifar100', 'places', 'inaturalist', 'imagenet'])
    parse.add_argument("--r_imb", type=int, default=1)
    parse.add_argument("--sampler", type=str, default=None, choices=["balanced", None])
    parse.add_argument("--tte", action='store_true', default=False)

    #### model parameters
    parse.add_argument("--model", type=str, choices=["zero-shot", "peft"], default="peft")
    parse.add_argument("--clip", type=str, default="ViT-B/16",
                       choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32',
                                'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'])
    parse.add_argument("--backbone", type=str, choices=["CLIP-ViT", "ViT", "CLIP-RN"], default="CLIP-ViT")
    parse.add_argument("--head", type=str, choices=["cosine", "l2", "ln", "linear", "mine"], default="linear")
    parse.add_argument("--save", action='store_true', default=False)
    parse.add_argument("--ckpt", type=str, default=None)
    #### My experimental parameters
    parse.add_argument("--ensemble", action='store_true', default=False)
    parse.add_argument('--t_softlabel', type=float, default=1)
    parse.add_argument("--base_loss", type=str, default="ce", 
                       choices=["ce", "focal", "ldam", "cb", "adjust", "generalized", "lade"])
    parse.add_argument("--t_logits", type=float, default=1)
    parse.add_argument("--w_dis", type=float, default=0)
    parse.add_argument("--w_mse", type=float, default=0)
    parse.add_argument("--hierarchy_prompt", action='store_true', default=False)
    parse.add_argument("--constrain", action='store_true', default=False)
    parse.add_argument("--init_head", action='store_true', default=False)
    #### PEFT parameters
    parse.add_argument("--full_tuning", action='store_true', default=False)
    parse.add_argument("--bias_tuning", action='store_true', default=False)
    parse.add_argument("--ln_tuning", action='store_true', default=False)
    parse.add_argument("--vpt_shallow", action='store_true', default=False)
    parse.add_argument("--vpt_deep", action='store_true', default=False)
    parse.add_argument("--vpt_len", type=int, default=None)
    parse.add_argument("--adapter", action='store_true', default=False)
    parse.add_argument("--adapter_scale", action='store_true', default=False)
    parse.add_argument("--adapter_dim", type=int, default=64)
    parse.add_argument("--lora", action='store_true', default=False)
    parse.add_argument("--lora_mlp", action='store_true', default=False)
    parse.add_argument("--ssf_attn", action='store_true', default=False)
    parse.add_argument("--ssf_mlp", action='store_true', default=False)
    parse.add_argument("--ssf_ln", action='store_true', default=False)
    parse.add_argument("--mask", action='store_true', default=False)
    parse.add_argument("--mask_ratio", type=float, default=0.01)
    parse.add_argument("--partial", type=int, default=None)

    parse.add_argument("--tag", type=str, default='')
    return parse.parse_args()