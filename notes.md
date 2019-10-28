## Run on server

train

```
CUDA_VISIBLE_DEVICES=3 python my_uncon_train_mnist.py  \
    --image_batch 32 \
    --video_batch 32 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 10 \
    --dim_z_content 128 \
    --dim_z_motion 64 \
    /home/student/gyliu/data/data_yongz/mocogan/data/mnist_two_16f_gif.h5 \
    ~/data/data_yongz/mocogan/logs/mnist2_uncondition_fix_eval

CUDA_VISIBLE_DEVICES=2 python my_train.py  \
    --image_batch 64 \
    --video_batch 64 \
    --use_noise \
    --noise_sigma 0.02 \
    --image_discriminator ImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 256 \
    --dim_z_motion 128 \
    /home/student/gyliu/data/data_yongz/mocogan/data/mnist_single_16f_gif.h5 \
    ~/data/data_yongz/mocogan/logs/mnist_1025_158dbb4
```

TensorBoard
```
nohup tensorboard --logdir=~/data/data_yongz/mocogan/logs --port=8813 &
```


## Run local

```
python train.py  \
    --image_batch 2 \
    --video_batch 2 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator ImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    ../data/actions ../logs/actions

python my_train.py  \
    --image_batch 2 \
    --video_batch 2 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator ImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 128 \
    --dim_z_motion 64 \
    /Users/zhangyong/projects/JD_GANimation/GANimation_Story/data/mnist_single_16f/mnist_single_16f_gif.h5 ../logs/mnist_single
```

## Comments

* 在 Discriminator 加入 Noise 层有什么作用