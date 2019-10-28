"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    train.py [options] <dataset> <log_folder>

Options:
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 10]
    --video_batch=<count>           number of videos in video batch [default: 3]

    --image_size=<int>              resize all frames to this size [default: 64]

    --use_infogan                   when specified infogan loss is usedq:q
    --use_nocondition               use no condition

    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator

    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]

    --image_discriminator=<type>    specifies image disciminator type (see models.py for a
                                    list of available models) [default: PatchImageDiscriminator]

    --video_discriminator=<type>    specifies video discriminator type (see models.py for a
                                    list of available models) [default: CategoricalVideoDiscriminator]

    --video_length=<len>            length of the video [default: 16]
    --print_every=<count>           print every iterations [default: 1]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 4]
    --batches=<count>               specify number of batches to train [default: 100000]

    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 6]
"""

import os
import docopt
import PIL

import functools

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from logger import Logger
import models
from trainers import Trainer
from logger import Logger
import time
import mymodels
import json
from torch.autograd import Variable

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


def build_discriminator(type, **kwargs):
    discriminator_type = getattr(models, type)

    if 'Categorical' not in type and 'dim_categorical' in kwargs:
        kwargs.pop('dim_categorical')

    return discriminator_type(**kwargs)

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def gradient_penalty(real_data, generated_data, D, use_cuda=False, gp_weight=10.):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    if len(real_data.shape) == 5:
        alpha = torch.rand(batch_size, 1, 1, 1, 1)
    elif len(real_data.shape) == 4:
        alpha = torch.rand(batch_size, 1, 1, 1)

    alpha = alpha.expand_as(real_data)
    if use_cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    # interpolated.requires_grad_()
    interpolated = Variable(interpolated, requires_grad=True)
    if use_cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated, _ = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(
        prob_interpolated, interpolated, 
        grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(prob_interpolated.size()), 
        create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    wgp = gp_weight * ((gradients_norm - 1) ** 2).mean()

    return wgp


# if __name__ == "__main__":
args = docopt.docopt(__doc__)
print(args)
use_cuda = torch.cuda.is_available()
n_channels = int(args['--n_channels'])
video_length = int(args['--video_length'])
image_batch = int(args['--image_batch'])
video_batch = int(args['--video_batch'])

dim_z_content = int(args['--dim_z_content'])
dim_z_motion = int(args['--dim_z_motion'])
dim_z_category = int(args['--dim_z_category'])



# data
if 'mnist' in args['<dataset>']:
    import mnist_data as data
    voc = data.Vocabulary()
    data_file = args['<dataset>']
    video_clip_dataset_train = data.VideoClipDataset({'data_file': data_file}, voc, mode='train')
    video_loader = torch.utils.data.DataLoader(video_clip_dataset_train, batch_size=video_batch, drop_last=True, shuffle=True, num_workers=4)
elif 'action' in args['<dataset>'] or 'shape' in args['<dataset>']:
    import data
    dataset = data.VideoFolderDataset(args['<dataset>'], cache=os.path.join(args['<dataset>'], 'local.db'))
    video_clip_dataset_train = data.VideoClipDataset(dataset, 16, 2)
    video_loader = torch.utils.data.DataLoader(video_clip_dataset_train, batch_size=video_batch, drop_last=True, shuffle=True, num_workers=4)

# models
# generator = models.VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)
generator = mymodels.CondVideoGenerator(n_channels, dim_z_content, dim_z_motion, video_length)

image_discriminator = build_discriminator(args['--image_discriminator'], n_channels=n_channels*2,
                                            use_noise=args['--use_noise'], noise_sigma=float(args['--noise_sigma']))

video_discriminator = build_discriminator(args['--video_discriminator'], dim_categorical=dim_z_category,
                                            n_channels=n_channels, use_noise=args['--use_noise'],
                                            noise_sigma=float(args['--noise_sigma']))

if torch.cuda.is_available():
    generator.cuda()
    image_discriminator.cuda()
    video_discriminator.cuda()

opt_generator = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=0.00001)
opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)


# training
gan_criterion = nn.BCELoss()
logger = Logger(args['<log_folder>'])
logs = {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}
start_time = time.time()
generator.train()

p = json.dumps(args, sort_keys=True, indent=4)
logger.text_summary('args', p)

epoch_batch_num = len(video_loader)
for epoch in range(10000):
    video_sampler = enumerate(video_loader)
    for batch_ind in range(epoch_batch_num):
        batch_num = epoch * epoch_batch_num + batch_ind
        # prepare data
        _, real_videos_dict = next(video_sampler)
        real_videos = real_videos_dict['clip']
        real_images = real_videos_dict['sample_frame']
        first_frames = real_videos_dict['first_frame']
        if torch.cuda.is_available():
            real_videos = real_videos.cuda()
            real_images = real_images.cuda()
            first_frames = first_frames.cuda()

        if args['--use_nocondition']:
            first_frames = None
        fake_images, _ = generator.sample_images(image_batch, first_frames)
        fake_videos, _ = generator.sample_videos(video_batch, first_frames)

        # train video discriminator
        vd_real, _ = video_discriminator(real_videos)
        vd_fake, _ = video_discriminator(fake_videos.detach())
        vd_wgp = gradient_penalty(real_videos, fake_videos, video_discriminator)

        opt_video_discriminator.zero_grad()
        loss_video_dis = vd_fake.mean() - vd_real.mean() + vd_wgp
        loss_video_dis.backward()
        opt_video_discriminator.step()

        # train image discriminator
        id_real_input = torch.cat([first_frames, real_images], dim=1)
        id_fake_input = torch.cat([first_frames, fake_images.detach()], dim=1)
        id_real, _ = image_discriminator(id_real_input)
        id_fake, _ = image_discriminator(id_fake_input)
        id_wgp = gradient_penalty(id_real_input, id_fake_input, image_discriminator)

        opt_image_discriminator.zero_grad()
        loss_image_dis = id_fake.mean() - id_real.mean() + id_wgp
        loss_image_dis.backward()
        opt_image_discriminator.step()

        logs['l_image_dis'] += loss_image_dis.data.item()
        logs['l_video_dis'] += loss_video_dis.data.item()

        if batch_num % 5 == 0:
            # train generator
            opt_generator.zero_grad()
            ig_fake, _ = image_discriminator(torch.cat([first_frames, fake_images], dim=1))
            vg_fake, _ = video_discriminator(fake_videos)

            loss_gen = -ig_fake.mean()-vg_fake.mean()
            loss_gen.backward()
            opt_generator.step()
            logs['l_gen'] += loss_gen.data.item()

        if batch_num % 10 == 0:
            took_time = time.time() - start_time
            logs['l_gen'] = logs['l_gen'] / 2.
            logs['l_image_dis'] = logs['l_image_dis'] / 10.
            logs['l_video_dis'] = logs['l_video_dis'] / 10.

            print(f"Epoch/Batch [{epoch}/{batch_ind} ~ {batch_num}]: l_gen={logs['l_gen']:5.3f}, l_image_dis={logs['l_image_dis']:5.3f}, l_video_dis={logs['l_video_dis']:5.3f}. Took: {took_time:5.2f}")
            print(f"\t videoD: D_real={vd_real.mean():5.3f}, D_fake={vd_fake.mean():5.3f} || imageD: D_real={id_real.mean():5.3f}, D_fake={id_fake.mean():5.3f} || G: vid_fake={vg_fake.mean():5.3f}, img_fake={ig_fake.mean():5.3f}")

            for tag, value in logs.items():
                logger.scalar_summary(tag, value, batch_num)

            logs = {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}
            start_time = time.time()

            generator.eval()
            images, _ = generator.sample_images(image_batch, first_frames)
            logger.image_summary("Images", images_to_numpy(images), batch_num)
            videos, _ = generator.sample_videos(video_batch, first_frames)
            logger.video_summary("Videos", videos_to_numpy(videos), batch_num)
            generator.train()

