import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, List, Iterable
import itertools


def prepend_music_dir(music_dir: Path):
    def _prepend(obj):
        obj['lmel80'] = music_dir / obj['lmel80']
        obj['cfp'] = music_dir / obj['cfp']
        obj['chart'] = music_dir / obj['chart']
        return obj

    return _prepend


def _always(level: int):
    return True


class ChartGenSequenceDataset(Dataset):

    def __init__(self, dataset_path: Union[Path, str], window_length=200, window_step=200, seq_len=5,
                 *, level_predicate=_always, cfp_frq_bin_last=False):
        super(ChartGenSequenceDataset, self).__init__()
        if isinstance(dataset_path, str):
            self.dataset_root = Path(dataset_path)
        else:
            assert isinstance(dataset_path, Path)
            self.dataset_root = dataset_path
        self.window_length = window_length
        self.window_step = window_step
        self.seq_len = seq_len
        self.should_swap_cfp_dim_back = cfp_frq_bin_last
        dataset_meta_path = self.dataset_root / 'dataset_meta.json'
        if not dataset_meta_path.is_file():
            raise FileNotFoundError(f'{dataset_meta_path}')
        megameta = {}
        curr_idx = 0
        with dataset_meta_path.open(encoding='utf-8') as f:
            musics = json.load(f)['musics']
            for music in musics:
                music_id = music['id']
                music_subdir = self.dataset_root / f'{music_id:04d}'
                with (music_subdir / 'dataset_meta.json').open(encoding='utf-8') as f1:
                    subset_meta = json.load(f1)
                    audio_set = subset_meta['audio']
                    chart_set = subset_meta['chart']
                    max_frames = subset_meta['frames']
                    max_windows = (max_frames + ((-max_frames) % window_length) - window_length) // window_step - self.seq_len + 1
                    acprod = itertools.product(audio_set, chart_set)
                    combinations = list(map(prepend_music_dir(music_subdir), map(lambda x: {**x[0], **x[1]}, acprod)))
                    for combi in combinations:
                        if not level_predicate(combi['level']):
                            continue
                        combi['frames'] = max_frames
                        megameta[curr_idx] = combi
                        curr_idx += max_windows
        self.data_map = megameta
        self.dataset_length = curr_idx

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pick_single = False
        if not isinstance(idx, Iterable):
            idx = [idx]
            pick_single = True
        idx_order = np.argsort(idx)
        idx = np.sort(idx)
        item_list = []
        cache = {'idx': None, 'cfp': None, 'lmel80': None, 'chart': None}
        for i in idx:
            nearest_i = None
            for k in sorted(self.data_map):
                if k <= i:
                    nearest_i = k
                else:
                    break
            nearest_v = self.data_map[nearest_i]
            if cache['idx'] != nearest_i:
                cache['chart'] = np.load(nearest_v['chart']).astype('float32')
                cache['cfp'] = np.load(nearest_v['cfp']).astype('float32')
                cache['lmel80'] = np.load(nearest_v['lmel80']).astype('float32')
                cache['idx'] = nearest_i
            start = (i - nearest_i) * self.window_step
            pack = {'level': np.repeat(nearest_v['level'], self.seq_len)}
            for tag in ['chart','cfp', 'lmel80']:
                if tag != 'chart':
                    tagged_item = cache[tag][:, :, start:start + self.window_length * self.seq_len]
                    if tagged_item.shape[2] < self.window_length:
                        tagged_item = np.pad(tagged_item,
                                            [[0, 0], [0, 0], [0, (-tagged_item.shape[2]) % self.window_length]])
                    tagged_item = tagged_item.reshape([*tagged_item.shape[:2], self.seq_len, self.window_length])
                    tagged_item = np.moveaxis(tagged_item, 2, 0)
                else:
                    tagged_item = cache[tag][:, start:start + self.window_length * self.seq_len, :]
                    if tagged_item.shape[1] < self.window_length:
                        tagged_item = np.pad(tagged_item,
                                            [[0, 0], [0, (-tagged_item.shape[1]) % self.window_length], [0, 0]])
                    tagged_item = tagged_item.reshape([tagged_item.shape[0], self.seq_len, self.window_length, *tagged_item.shape[2:]])
                    tagged_item = np.moveaxis(tagged_item, 1, 0)
                pack[tag] = tagged_item
            item_list.append(pack)
        if pick_single:
            return item_list[0]
        return np.take(item_list, idx_order).tolist()


class ChartGenDataset(Dataset):

    def __init__(self, dataset_path: Union[Path, str], window_length=500, window_step=250, *, level_predicate=_always,
                 providing_data=('cfp', 'chart')):
        super(ChartGenDataset, self).__init__()
        if isinstance(dataset_path, str):
            self.dataset_root = Path(dataset_path)
        else:
            assert isinstance(dataset_path, Path)
            self.dataset_root = dataset_path
        for tag in providing_data:
            assert tag in [ 'cfp', 'chart']
        self.providing_data = providing_data
        self.window_length = window_length
        self.window_step = window_step
        dataset_meta_path = self.dataset_root / 'dataset_meta.json'
        if not dataset_meta_path.is_file():
            raise FileNotFoundError(f'{dataset_meta_path}')
        megameta = {}
        curr_idx = 0
        with dataset_meta_path.open(encoding='utf-8') as f:
            musics = json.load(f)['musics']
            for music in musics:
                music_id = music['id']
                music_subdir = self.dataset_root / f'{music_id:04d}'
                with (music_subdir / 'dataset_meta.json').open(encoding='utf-8') as f1:
                    subset_meta = json.load(f1)
                    audio_set = subset_meta['audio']
                    chart_set = subset_meta['chart']
                    max_frames = subset_meta['frames']
                    max_windows = (max_frames + ((-max_frames) % window_length) - window_length) // window_step
                    acprod = itertools.product(audio_set, chart_set)
                    combinations = list(map(prepend_music_dir(music_subdir), map(lambda x: {**x[0], **x[1]}, acprod)))
                    for combi in combinations:
                        if not level_predicate(combi['level']):
                            continue
                        combi['frames'] = max_frames
                        megameta[curr_idx] = combi
                        curr_idx += max_windows
        self.data_map = megameta
        self.dataset_length = curr_idx

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        pick_single = False
        if not isinstance(idx, Iterable):
            idx = [idx]
            pick_single = True
        idx_order = np.argsort(idx)
        idx = np.sort(idx)
        item_list = []
        cache = {'idx': None, 'cfp': None, 'lmel80': None, 'chart': None}
        for i in idx:
            nearest_i = None
            for k in sorted(self.data_map):
                if k <= i:
                    nearest_i = k
                else:
                    break
            nearest_v = self.data_map[nearest_i]
            if cache['idx'] != nearest_i:
                if 'chart' in self.providing_data:
                    cache['chart'] = np.load(nearest_v['chart'])
                if 'cfp' in self.providing_data:
                    cache['cfp'] = np.load(nearest_v['cfp'])
                cache['idx'] = nearest_i
            start = (i - nearest_i) * self.window_step
            pack = {'level': nearest_v['level']}
            for tag in self.providing_data:
                tagged_item = cache[tag][:, start:start + self.window_length, :]
                if tagged_item.shape[1] < self.window_length:
                    tagged_item = np.pad(tagged_item,
                                            [[0, 0], [0, (-tagged_item.shape[1]) % self.window_length], [0, 0]])
                pack[tag] = tagged_item
            item_list.append(pack)
        if pick_single:
            return item_list[0]
        return np.take(item_list, idx_order).tolist()
