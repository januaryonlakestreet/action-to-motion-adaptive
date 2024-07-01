from datetime import datetime
import numpy as np
import os
import torch

import utils.paramUtil as paramUtil
from get_opt import get_opt
from load_classifier import load_classifier, load_classifier_for_fid
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from motion_loaders.motion_loader import get_motion_loader
from fid import calculate_frechet_distance
from utils.matrix_transformer import MatrixTransformer as mt
from utils.plot_script import *

torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_accuracy(num_motions, gru_classifier, motion_loaders, dataset_opt, device, file):
    print('========== Evaluating Accuracy ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        accuracy = calculate_accuracy(motion_loader, len(dataset_opt.label_dec),
                                      gru_classifier, device)
        print(f'---> [{motion_loader_name}] Accuracy: {np.trace(accuracy) / num_motions:.4f}')
        print(f'---> [{motion_loader_name}] Accuracy: {np.trace(accuracy) / num_motions:.4f}', file=file, flush=True)


def calculate_accuracy(motion_loader, num_labels, classifier, device):
    print('Calculating Accuracies...')
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            batch_motion, batch_label = batch
            batch_motion = torch.clone(batch_motion).float().detach_().to(device)
            batch_label = torch.clone(batch_label).long().detach_().to(device)
            batch_prob, _ = classifier(batch_motion, None)
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch_label, batch_pred):
                # print(label.data, pred.data)
                confusion[label][pred] += 1

    return confusion


def evaluate_fid(ground_truth_motion_loader, gru_classifier_for_fid, motion_loaders, device, file):
    print('========== Evaluating FID ==========')
    ground_truth_activations, ground_truth_labels = \
        calculate_activations_labels(ground_truth_motion_loader, gru_classifier_for_fid, device)
    ground_truth_statistics = calculate_activation_statistics(ground_truth_activations)

    for motion_loader_name, motion_loader in motion_loaders.items():
        activations, labels = calculate_activations_labels(motion_loader, gru_classifier_for_fid, device)
        statistics = calculate_activation_statistics(activations)
        fid = calculate_fid(ground_truth_statistics, statistics)
        diversity = calc_diversity_only(activations, labels, len(dataset_opt.label_dec))

        print(f'---> [{motion_loader_name}] FID: {fid:.4f}')
        print(f'---> [{motion_loader_name}] FID: {fid:.4f}', file=file, flush=True)
        print(f'---> [{motion_loader_name}] Diversity: {diversity:.4f}')
        print(f'---> [{motion_loader_name}] Diversity: {diversity:.4f}', file=file, flush=True)


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_activations_labels(motion_loader, classifier, device):
    print('Calculating Activations...')
    activations = []
    labels = []

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            batch_motion, batch_label = batch
            batch_motion = torch.clone(batch_motion).float().detach_().to(device)

            activations.append(classifier(batch_motion, None))
            labels.append(batch_label)
        activations = torch.cat(activations, dim=0)
        labels = torch.cat(labels, dim=0)

    return activations, labels


def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calc_diversity_only(activations, labels, num_labels):
    print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity


def calculate_diversity_multimodality(activations, labels, num_labels):
    print('=== Evaluating Diversity ===')
    diversity_times = 200
    multimodality_times = 20
    labels = labels.long()
    num_motions = len(labels)

    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times

    print('=== Evaluating Multimodality ===')
    multimodality = 0
    labal_quotas = np.repeat(multimodality_times, num_labels)
    while np.any(labal_quotas > 0):
        # print(labal_quotas)
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx]
        if not labal_quotas[first_label]:
            continue

        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx]
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]

        labal_quotas[first_label] -= 1

        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)

    multimodality /= (multimodality_times * num_labels)

    return diversity, multimodality


def evaluation(log_file):
    with open(log_file, 'w') as f:
        for replication in range(20):
            motion_loaders = {}
            motion_loaders['ground truth'] = ground_truth_motion_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter(num_motions, device)
                motion_loaders[motion_loader_name] = motion_loader
            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            evaluate_accuracy(num_motions, gru_classifier, motion_loaders, dataset_opt, device, f)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            evaluate_fid(ground_truth_motion_loader, gru_classifier_for_fid, motion_loaders, device, f)

        print(f'Time: {datetime.now()}')
        print(f'Time: {datetime.now()}', file=f, flush=True)
        print(f'!!! DONE !!!')
        print(f'!!! DONE !!!', file=f, flush=True)



if __name__ == '__main__':
    # dataset_opt_path = './checkpoints/vae/ntu_rgbd_vibe/vae_velocS_f0001_t01_trj10_rela/opt.txt'
    # dataset_opt_path = './checkpoints/vae/humanact12/vae_velocR_f0001_t001_trj10_rela_fineG/opt.txt'
    dataset_opt_path = '../checkpoints/vae/humanact12/test1/opt.txt'
    label_spe = 3
    eval_motion_loaders = {
        'vanilla_vae_mse_kld01': lambda num_motions, device: get_motion_loader(
            '../checkpoints/vae/humanact12/test1/opt.txt',
            num_motions, 128, device, ground_truth_motion_loader, label_spe),

    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    num_motions = 3000
    # num_motions = 200

    dataset_opt = get_opt(dataset_opt_path, num_motions, device)
    # print(dataset_opt)
    gru_classifier_for_fid = load_classifier_for_fid(dataset_opt, device)
    gru_classifier = load_classifier(dataset_opt, device)

    ground_truth_motion_loader = get_dataset_motion_loader(dataset_opt, num_motions, device, label=label_spe)
    motion_loaders = {}
    # motion_loaders['ground_truth'] = ground_truth_motion_loader

    log_file = 'final_evaluation_mocap_veloc_label3_bk.log'
    evaluation(log_file)
