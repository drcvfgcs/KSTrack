import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader,opencv_loader
from ltr.admin.environment import env_settings
from glob import glob

class GTOT_TIR(BaseVideoDataset):
    """ GOTO dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=opencv_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__('GTOT', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        # if split is not None:
        #     if seq_ids is not None:
        #         raise ValueError('Cannot set both split_name and seq_ids.')
        #     ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        #     if split == 'train':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'gtot_train_split.txt')
        #     elif split == 'val':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'gtot_val_split.txt')
        #     else:
        #         raise ValueError('Unknown split name.')
        #     seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
        # elif seq_ids is None:
        #     seq_ids = list(range(0, len(self.sequence_list)))

        # self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()


    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'gtot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        # with open(os.path.join(self.root, 'list.txt')) as f:
        #     dir_list = list(csv.reader(f))
        # dir_list = [dir_name[0] for dir_name in dir_list]
        dir_list = os.listdir(self.root)
        dir_list = dir_list[::2]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundTruth_v.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=' ', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        gt[:,2] = gt[:,2]-gt[:,0]
        gt[:,3] = gt[:,3]-gt[:,1]
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        img_dir = sorted(glob(os.path.join(seq_path,'v','*')))[frame_id]
        return img_dir
        # return os.path.join(seq_path,'v', '{:05}v.png'.format(frame_id))    # frames start from 1
        # return os.path.join(seq_path,'i', '{:05}i.png'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_rgbt_frame(self,seq_path,frame_id):
        # img_vis_path = self.vis_frames_list[frame_id]
        img_tir_path = self.tir_frames_list[frame_id]
        # image_vis = self.image_loader(img_vis_path)
        image_tir = self.image_loader(img_tir_path)
        image = image_tir
        return image

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1]
        return raw_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)


        obj_class = self._get_class(seq_path)
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        vis_frames_path = '{}/v'.format(seq_path)
        self.vis_frames_list = sorted(glob(os.path.join(vis_frames_path,'*')))
        tir_frames_path = '{}/i'.format(seq_path)
        self.tir_frames_list = sorted(glob(os.path.join(tir_frames_path,'*')))

        # frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        frame_list = [self._get_rgbt_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, object_meta
