# Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer

Pytorch implementation of our Q-ViT accepted by NeurIPS2022.


<div align="center">
  <img src="pic.png" width="800"/>
</div>

## Tips

Any problem, please contact the first author (Email: yanjingli@buaa.edu.cn). 

Our code is heavily borrowed from DeiT (https://github.com/facebookresearch/deit).
## Dependencies
* Python 3.8
* Pytorch 1.7.1
* Torchvision 0.8.2
* timm 0.4.12

## Training:

### Train Q-ViT Deit-T 4bits:

We train the 2/3/4 bits Q-ViT Deit-T with 512 batchsize and 3e-4 lr. Please note that we use DistribuedSampler for Tiny models. 

When training 2/3 bits Q-ViT Deit-T, please change the model into 'twobits_deit_tiny_patch16_224/threebits_deit_tiny_patch16_224'

  > python -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 --use_env main.py --model fourbits_deit_tiny_patch16_224 --epochs 300 --warmup-epochs 0 --weight-decay 0. --batch-size 128  --data-path /mnt/lustre/share/images/ --lr 3e-4 --no-repeated-aug --output_dir ./dist_4bit_tiny_lamb_3e-4_300_512 --distillation-type hard --teacher-model vit_deit_tiny_distilled_patch16_224 --opt fusedlamb 

### Train Q-ViT Deit-S 2/3/4bits:

We train the 2/3/4 bits Q-ViT Deit-S with 512 batchsize and 3e-4 lr. Please note that we use RASampler for Small models. 

When training 2/3 bits Q-ViT Deit-S, please change the model into 'twobits_deit_small_patch16_224/threebits_deit_small_patch16_224'

  > python -m torch.distributed.launch --master_port=12345 --nproc_per_node=4 --use_env main.py --model fourbits_deit_small_patch16_224 --epochs 300 --warmup-epochs 0 --weight-decay 0. --batch-size 128  --data-path /mnt/lustre/share/images/ --lr 3e-4 --repeated-aug --output_dir ./dist_4bit_small_lamb_3e-4_300_512 --distillation-type hard --teacher-model vit_deit_small_distilled_patch16_224 --opt fusedlamb 


## Evaluation: 

  ### Eval Q-ViT Deit-S 2bits: (72.0% Top-1 Acc.):
    
    > python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 --use_env main.py --model twobits_deit_small_patch16_224 --weight-decay 0. --batch-size 64  --data-path /dataset/ImageNet --output_dir ./eval --resume ./best_checkpoint_2bit.pth --eval

  ### Eval Q-ViT Deit-S 3bits: (79.1% Top-1 Acc.):
    
    > python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 --use_env main.py --model threebits_deit_small_patch16_224 --weight-decay 0. --batch-size 64  --data-path /dataset/ImageNet --output_dir ./eval --resume ./best_checkpoint_3bit.pth --eval
    
## Checkpoints: 

  ### Q-ViT Deit-T

| Methods | Top-1 acc | Top-5 acc | Quantized model link |
|:-------:|:---------:|:---------:|:--------------------:|
| Q-Deit-T (4-bit)  |  74.3     |  91.6     | [Model](https://drive.google.com/file/d/1kRtJ0YkA5DiRYixZznV-93SMVqX4MVFQ/view?usp=share_link)  |

  ### Q-ViT Deit-S

| Methods | Top-1 acc | Top-5 acc | Quantized model link |
|:-------:|:---------:|:---------:|:--------------------:|
| Q-DeiT-S (3-bit)  |  79.1     |  94.3     | [Model](https://drive.google.com/file/d/1UbyrKB4h3fx8fsTQboz6IOZBy-utBlq3/view?usp=sharing)  |
| Q-Deit-S (2-bit)  |  72.0     |  90.3     | [Model](https://drive.google.com/file/d/1bcNpJ0Sqt19aJcyCrdmTgqOYCr79HL5f/view?usp=sharing)  |


Training codes and other models will be open-sourced successively.

