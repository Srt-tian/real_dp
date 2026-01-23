from diffusion_policy.dataset.real_h5_image_dataset import RealH5ImageDataset
if __name__ == "__main__":
    dataset = RealH5ImageDataset(
        "/home/tian/diffusion_policy/data/demo_real/episode_0008/episode.hdf5",
        horizon=16,
        pad_before=2,
        pad_after=0
    )

    x = dataset[0]
    print(x['obs']['image'].shape)   # (16,3,84,84)
    print(x['obs']['state'].shape)   # (16,4)
    print(x['action'].shape)         # (16,7)
    # print(x['obs']['state'][:3])
