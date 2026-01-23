# diffusion_policy/dataset/real_h5_image_dataset.py
from typing import Dict
import os, glob, copy
import h5py
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class RealH5ImageDataset(BaseImageDataset):
    def __init__(self,
            h5_glob_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()

        # 1) collect hdf5 paths
        paths = sorted(glob.glob(os.path.expanduser(h5_glob_path), recursive=True))
        assert len(paths) > 0, f"No hdf5 files matched: {h5_glob_path}"

        # 2) build replay buffer (numpy backend)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        for p in paths:
            with h5py.File(p, 'r') as f:
                img = f['obs']['camera_0'][...]                 # (T,84,84,3) uint8
                rot = f['obs']['robot_eef_rot'][...].astype(np.float32)   # (T,3)
                grip = f['obs']['gripper_open_state'][...]
                grip = grip.astype(np.float32)
                if grip.ndim == 1:
                    grip = grip[:, None]                        # (T,1)

                state = np.concatenate([rot, grip], axis=-1)    # (T,4)
                action = f['action']['target_pose'][...].astype(np.float32)  # (T,7)

                T = img.shape[0]
                assert state.shape == (T, 4)
                assert action.shape == (T, 7)

                self.replay_buffer.add_episode({
                    'img': img,
                    'state': state,
                    'action': action
                })

        # 3) train / val split by episode
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        image = np.moveaxis(sample['img'], -1, 1).astype(np.float32) / 255.0
        data = {
            'obs': {
                'image': image,              # (H,3,84,84)
                'state': sample['state']     # (H,4)
            },
            'action': sample['action']       # (H,7)
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
