import os
import numpy as np
import torch.utils.data
from torchvision import transforms
import functools
import PIL
import re
import h5py
import itertools
import pdb


EOS_token = 26
PAD_token = 27

class Vocabulary:
    def __init__(self):
        self.n_words = 30
        self.word2ind_dict = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'the': 10, 'digit': 11, 'and': 12, 'is':13, 'bouncing': 14, 'moving':15, 'here':16, 'there':17, 'around':18, 'jumping':19, 'left':20, 'right':21, '.':22, 'up':23, 'down':24, 'are': 25}
        self.ind2word_dict = {v: k for k, v in self.word2ind_dict.items()}

    def mnist_ind2sentence(self, inds):
        words = list(map(lambda x: self.ind2word_dict[x], inds))
        return ' '.join(words)

    def mnist_sentence2ind(self, sentence):
        words = sentence.split(' ')
        inds = list(map(lambda x: self.word2ind_dict[x], words)) + [EOS_token]
        return inds

    def input_preprocess(self, sentences=None, sent_inds=None, fillvalue=PAD_token, device=None):
        if sentences != None:
            indexes_batch = [self.mnist_sentence2ind(sentence) for sentence in sentences]
        elif sent_inds != None:
            indexes_batch = [x + [EOS_token] for x in sent_inds]

        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = list(itertools.zip_longest(*indexes_batch, fillvalue=fillvalue))
        padVar  = torch.LongTensor(padList)
        if device != None:
            padVar  = padVar.to(device)
            lengths = lengths.to(device)

        return padVar, lengths

class VideoClipDataset(torch.utils.data.Dataset):
    def __init__(self, params, voc, mode='train'):
        self.data_file = params['data_file']
        self.voc       = voc

        with h5py.File(self.data_file, 'r') as f:
            self.video_data = f['mnist_gif_%s' % mode].value
            self.caption_data = f['mnist_captions_%s' % mode].value


    # PIL.Image.fromarray(frame.astype('uint8'), 'RGB').show()
    # clip: BxHxWxC
    def _transform(self, clip):
        image_transforms = transforms.Compose([
            PIL.Image.fromarray,
            # transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        def video_transform(video, image_transform):
            vt = []
            for frame in video:
                vt.append(image_transform(frame))
            vt = torch.stack(vt)
            return vt

        video_transforms = functools.partial(video_transform, image_transform=image_transforms)
        transformed_clip = video_transforms(clip)
        return transformed_clip

    def __getitem__(self, item):
        caption = self.voc.mnist_ind2sentence(self.caption_data[item])
        video   = (self.video_data[item][:,0,:,:,None] * 255).astype('uint8').repeat(3, axis=3)
        frame_num = np.random.randint(video.shape[0])
        video   = self._transform(video)
        return {'vid': str(item),
                'first_frame': video[0],
                'sample_frame': video[frame_num], 
                'clip': video.permute(1,0,2,3),
                'desc': caption}

    def __len__(self):
        return len(self.caption_data)

if __name__ == "__main__":
    voc = Vocabulary()
    sentence_batch = ['the digit 3 is moving left and right .', 'the digit 0 is moving up and down .']
    input_variable, lengths = voc.input_preprocess(sentences=sentence_batch)
    print(input_variable)

    sentenceids_batch = [[1,2,3,4,6], [4,2,2,12,12,9,11,13,20]]
    input_variable, lengths = voc.input_preprocess(sent_inds=sentenceids_batch)
    print(input_variable)
