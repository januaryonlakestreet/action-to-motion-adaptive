from torch.utils.data import DataLoader
from eval_scripts.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.motion_vae_motion_loader import MotionVAEGeneratedDataset


from motion_loaders.motion_vae_lie_motion_loader import MotionVAELieGeneratedDataset


def get_motion_loader(opt_path, num_motions, batch_size, device, ground_truth_motion_loader=None, label=None):
    opt = get_opt(opt_path, num_motions, device)

    if '/vae/' in opt_path:

        if 'lie' in opt.name:
            print('Generating %s ...' % opt.name)
            dataset = MotionVAELieGeneratedDataset(opt, num_motions, batch_size, device, ground_truth_motion_loader, label)
        else:
            print('Generating Adversaried Motion VAE Motion...')
            # print(label)
            dataset = MotionVAEGeneratedDataset(opt, num_motions, batch_size, device, label)


    else:
        raise NotImplementedError('Unrecognized model type')

    motion_loader = DataLoader(dataset, batch_size=128, num_workers=1)
    return motion_loader