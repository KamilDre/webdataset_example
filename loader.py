import os
import torch
import numpy as np
from PIL import Image

import webdataset as wds
from torch.utils.data import DataLoader


def get_loader(tar_pattern, batch_size=8, shuffle_buffer=1000, num_workers=4):
    """
    Returns a PyTorch DataLoader loading dictionaries from WebDataset tar shards.

    Args:
        tar_pattern (str): e.g. 'my_dataset/dataset_shard_{000000..000004}.tar'
        batch_size (int): batch size for DataLoader
        shuffle_buffer (int): buffer size for WebDataset shuffle
        num_workers (int): number of DataLoader workers

    Returns:
        DataLoader yielding dictionaries with keys:
        {
            'rgb': list of PIL Images or stacked numpy arrays,
            'depth': stacked numpy arrays,
            'seg': list of PIL Images,
            'intrinsic_matrix': stacked numpy arrays,
            'T_WC_opencv': stacked numpy arrays,
            'joint_angles': stacked numpy arrays,
            'instance_attribute_maps': list of dictionaries
        }
    """
    def convert_sample(sample):
            return {
                'rgb': torch.from_numpy(np.array(sample['rgb.png'])).permute(2, 0, 1).contiguous(),  # [C,H,W]
                'depth': torch.from_numpy(sample['depth.npy']).unsqueeze(0).contiguous(),            # [H,W]
                'seg': torch.from_numpy(np.array(sample['seg.npy'])).unsqueeze(0).contiguous(),      # [H,W]
                'intrinsic_matrix': sample['intrinsic_matrix.npy'],                                  # Leave as np.ndarray
                'T_WC_opencv': sample['t_wc_opencv.npy'],                                            # Leave as np.ndarray
                'joint_angles': torch.from_numpy(sample['joint_angles.npy']),                        # [2] tensor
                'instance_attribute_maps': sample['instance_attribute_maps.json']                    # Leave as list of dicts
            }

    dataset = (
        wds.WebDataset(tar_pattern, shardshuffle=3)  # prefetch 3 shards and shuffle them
        .decode("rgb")  # decode images to PIL format
        .shuffle(shuffle_buffer)
        .map(convert_sample)
    )

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers) # Ideally need custom collate_fn for segmentation.
    return loader
