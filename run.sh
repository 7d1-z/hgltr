# ImageNet-LT
accelerate launch --config_file config/gpu-12.yaml main.py \
    --epochs 10 --lr 0.05 \
    --dataset imagenet --base_loss adjust \
    --adapter --adapter_scale --ensemble --init_head \
    --t_softlabel 1 --w_dis 0.01 --w_mse 0.1 --constrain

# Places-LT
accelerate launch --config_file config/gpu-12.yaml main.py \
    --epochs 10 --lr 0.05 \
    --dataset places --base_loss adjust \
    --adapter --adapter_scale --ensemble --init_head \
    --t_softlabel 4 --w_mse 0.1 --constrain

# iNaturalist 2018
accelerate launch --config_file config/gpu-04.yaml main.py \
    --epochs 40 --lr 2e-4 --optim adamw \
    --dataset inaturalist --base_loss adjust \
    --adapter --adapter_scale --ensemble --init_head \
    --t_softlabel 1.5 --t_logits 1 --w_dis 0.01 --w_mse 0.1 \
    --head cosine --constrain