import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

class FrameEncoder(nn.Module):
    def __init__(self, img_embeding_size):
        super(FrameEncoder, self).__init__()
        ndf = 64
        self.ndf = ndf
        self.img_embeding_size = img_embeding_size

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=1, padding=0, bias=False),
            # state size. (ndf*16) x 1 x 1
        )
        self.linear = nn.Linear(self.ndf*16, img_embeding_size)


    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], self.ndf * 16)
        x = self.linear(x)
        return x


class CondVideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_motion, video_length, ngf=64):
        super(CondVideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_content

        self.frame_encoder = FrameEncoder(dim_z_content)
        self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def sample_z_m(self, num_samples, gen_length):
        h_t = [self.get_gru_initial_state(num_samples)]

        for _ in range(gen_length):
            e_t = self.get_iteration_noise(num_samples)
            h_t.append(self.recurrent(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        return z_m

    def sample_z_content(self, num_samples):
        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, self.video_length, axis=0)
        content = torch.from_numpy(content)
        if torch.cuda.is_available():
            content = content.cuda()
        return content

    def get_frame_encodings(self, inputs, gen_length):
        frame_encodings = self.frame_encoder(inputs)
        frame_encodings = frame_encodings.unsqueeze(1).repeat(1, gen_length, 1).view(-1, self.dim_z_content)
        return frame_encodings

    # todo: frame完全替代 z_content 还是部分替代？
    def prepare_inputs(self, num_samples, first_frames=None):
        if first_frames is None:
            z_content = self.sample_z_content(num_samples)
            z_motion  = self.sample_z_m(num_samples, self.video_length)
        else:
            z_content = self.get_frame_encodings(first_frames, self.video_length-1)
            z_motion  = self.sample_z_m(num_samples, self.video_length-1)

        z = torch.cat([z_content, z_motion], dim=1)
        return z

    def sample_videos(self, num_samples, first_frames=None):
        z = self.prepare_inputs(num_samples, first_frames)

        h = self.main(z.view(z.size(0), z.size(1), 1, 1))
        if first_frames is None:
            gen_length = self.video_length
            h = h.view(h.size(0) // gen_length, gen_length, self.n_channels, h.size(3), h.size(3))
            h = h.permute(0, 2, 1, 3, 4)
        else:
            gen_length = self.video_length - 1
            h = h.view(h.size(0) // gen_length, gen_length, self.n_channels, h.size(3), h.size(3))
            h = torch.cat([first_frames.unsqueeze(2), h.permute(0, 2, 1, 3, 4)], dim=2)

        return h, None

    def sample_images(self, num_samples, first_frames=None):
        z = self.prepare_inputs(num_samples, first_frames)

        gen_len = z.shape[0] // num_samples
        j = np.random.choice(gen_len, num_samples).astype(np.int64) + np.arange(num_samples) * gen_len

        z = z[j, ::]
        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.main(z)

        return h, None

    def get_gru_initial_state(self, num_samples):
        return T.FloatTensor(num_samples, self.dim_z_motion).normal_()

    def get_iteration_noise(self, num_samples):
        return T.FloatTensor(num_samples, self.dim_z_motion).normal_()
