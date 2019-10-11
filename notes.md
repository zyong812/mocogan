## Run on server

train

```
CUDA_VISIBLE_DEVICES=3 python train.py  \
    --image_batch 32 \
    --video_batch 32 \
    --use_infogan \
    --use_noise \
    --noise_sigma 0.1 \
    --image_discriminator PatchImageDiscriminator \
    --video_discriminator CategoricalVideoDiscriminator \
    --print_every 10 \
    --every_nth 10 \
    --dim_z_content 50 \
    --dim_z_motion 10 \
    --dim_z_category 4 \
    ../data/actions ~/data/data_yongz/mocogan/logs/actions_small_image_samplesize
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
```

## Comments

* PatchImageDiscriminator 据说效果更好，可以尝试下
* 在 Discriminator 加入 Noise 层有什么作用