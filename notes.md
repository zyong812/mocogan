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

CUDA_VISIBLE_DEVICES=2 python my_cond_train.py  \
    --image_batch 64 \
    --video_batch 64 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 128 \
    --dim_z_motion 64 \
    /home/student/gyliu/data/data_yongz/mocogan/data/mnist_two_16f_gif.h5 \
    ~/data/data_yongz/mocogan/logs/mnist2_first_frame_condition_fix_eval
```

TensorBoard
```
nohup tensorboard --logdir=~/data/data_yongz/mocogan/logs --port=8813 &
```


## Run local

```
python my_uncon_train.py  \
    --image_batch 2 \
    --video_batch 2 \
    --use_infogan \
    --use_noise \
    --use_categories \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator CategoricalVideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    --dim_z_category 4 \
    ../data/actions ../logs/actions

python my_cond_train.py  \
    --image_batch 2 \
    --video_batch 2 \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator VideoDiscriminator \
    --print_every 10 \
    --every_nth 2 \
    --dim_z_content 128 \
    --dim_z_motion 64 \
    ../data/actions ../logs/actions
```

## Comments

* PatchImageDiscriminator 据说效果更好，可以尝试下
* 在 Discriminator 加入 Noise 层有什么作用