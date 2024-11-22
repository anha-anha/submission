from tqdm import tqdm

import typing

import torch

from utils.utils import set_random_seed


class WmProvider():
    """
    Helps in housekeeping different wm schemes
    """
    def __init__(self,
                 latent_shape: typing.Tuple[int, int, int, int],
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cuda'),
                 **kwargs):
        """
        We always use the same list of keys. Can support up to 1000 images depending on batch_size in latent_shape.

        @param latent_shape: Shape of the latent tensor with batch dim either tuple or torch.shape. e.g. (1, 4, 64, 64)
        @param dtype: device
        @param device: device
        """
        # cast to torch.shape
        if isinstance(latent_shape, tuple):
            latent_shape = torch.Size(latent_shape)
        self.latent_shape = latent_shape
        self.batch_size = latent_shape[0]
        self.num_channels = latent_shape[1]
        self.latent_resolution = latent_shape[2]

        # ensure we have batch dim and deal with square images
        assert len(latent_shape) == 4 and latent_shape[-1] == latent_shape[-2]

        self.dtype = dtype
        self.device = device


    @classmethod
    def generate_providers(cls,
                           total_num_latents: int,
                           latent_shape: typing.Tuple[int, int, int, int],
                           batch_size: int,
                           w_seed: int = None,
                           target_start_index: int = 0,
                           target_end_index: int = 999999999,
                           use_diff_seed_per_batch: bool = False,
                           **provider_args: dict) -> typing.Generator:
        """
        Generate multiple providers to accomondate a number of generated latents, cutting into batch sizes, and keeping track of the correct offsets for crypto lists

        @param total_num_latents: total number of latents to generate
        @param latent_shape: shape of the latent tensor with batch dim either tuple or torch.shape. e.g. (1, 4, 64, 64)
        @param batch_size: batch size
        @param w_seed: seed for the provider
        @param target_start_index: starting index
        @param target_end_index: ending index
        @param increment_seed: increment seed

        @param provider_args: provider args

        @return: generator
        """
        print(target_start_index)
        target_start_index = target_start_index
        target_end_index = min(total_num_latents, target_end_index)
        for idx in tqdm(range(target_start_index,
                              target_end_index,
                              batch_size)):
            num_batches = (target_end_index - target_start_index) // batch_size

            size = min(batch_size, target_end_index - idx)

            # overwrite the batch_size of the latent shape for the provider
            latent_shape = (size,) + latent_shape[1:]

            # if there is offset in provider_args, we remove it
            if 'offset' in provider_args:
                provider_args.pop('offset')

            # control the seed and instantiate provider
            w_seed = w_seed + idx if w_seed is not None and use_diff_seed_per_batch else w_seed
            yield cls(latent_shape=latent_shape, offset=idx, w_seed=w_seed, **provider_args), idx, size, num_batches
