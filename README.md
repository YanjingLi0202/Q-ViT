# Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer
![pic](pic.png)

## Tips

Any problem, please contact the first author (Email: yanjingli@buaa.edu.cn). 

Our code is heavily borrowed from DeiT (https://github.com/facebookresearch/deit).
## Dependencies
* Python 3.8
* Pytorch 1.7.1
* Torchvision 0.8.2
* timm 0.4.12

## Evaluation: 

  ### Eval Q-ViT Deit-S 2bits: (72.0% Top-1 Acc.):
    
    > python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 --use_env main.py --model twobits_deit_small_patch16_224 --weight-decay 0. --batch-size 64  --data-path /dataset/ImageNet --output_dir ./eval --resume ./best_checkpoint_2bit.pth --eval

  ### Eval Q-ViT Deit-S 3bits: (79.1% Top-1 Acc.):
    
    > python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 --use_env main.py --model threebits_deit_small_patch16_224 --weight-decay 0. --batch-size 64  --data-path /dataset/ImageNet --output_dir ./eval --resume ./best_checkpoint_3bit.pth --eval
    

| Methods | Top-1 acc | Top-5 acc | Quantized model link |
|:-------:|:---------:|:---------:|:--------------------:|
| Q-DeiT-S (3-bit)  |  79.1     |  90.3     | [Model](https://drive.google.com/file/d/1UbyrKB4h3fx8fsTQboz6IOZBy-utBlq3/view?usp=sharing)  |
| Q-Deit-S (2-bit)  |  72.0     |  94.2     | [Model](https://drive.google.com/file/d/1bcNpJ0Sqt19aJcyCrdmTgqOYCr79HL5f/view?usp=sharing)  |


Training codes and other models will be open-sourced successively.

