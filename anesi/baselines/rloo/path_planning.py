import argparse
import multiprocessing
import os
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb

from baselines.rloo import RLOOWrapper
from experiments.path_planning.anesi_sp import SPModel
from inference_models import NoPossibleActionsException
import ray
from experiments.path_planning.data.dataloader import get_datasets


def val_WC(x, label, weights, model, device):
    try:
        test_result = model.test(x, label, weights)
    except NoPossibleActionsException:
        print("No possible actions during testing")
        test_result = val_WC(x, label, weights, model, device)
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    acc_clauses = test_result[2].item()
    weights_acc = test_result[3].item()
    avg_dist = test_result[4].item()
    return acc, acc_prior, acc_clauses, weights_acc, avg_dist

if __name__ == '__main__':
    config = {
        # 12, 18, 24 or 30
        "N": 12,
        "DEBUG": False,
        "amount_samples": 10,
        "batch_size": 2,
        "batch_size_test": 100,
        "epochs": 300,
        "log_per_epoch": 10,
        "rloo_samples": 10,
        "perception_model": "small",
        "perception_lr": 0.001,
        "q_model": "small",
        "test": False,
        "use_cuda": True,
        "verbose": 1,
        "val_freq": 1,
        "use_ray": False,
        "neighbourhood_fn": "neighbours_8",
        "weight_types": [0.8, 1.2, 5.3, 7.7, 9.2],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="shortest-path", entity="nesy-gems")
        config = wandb.config
        print(config)
    else:
        run = wandb.init()
        config.update(wandb.config)
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    if config["use_ray"]:
        ray.init(num_cpus=multiprocessing.cpu_count())

    model = SPModel(config).to(device)
    train, val, test = get_datasets(config["N"], basepath="../../experiments/path_planning/data/")

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], shuffle=True)

    log_iterations = len(train_loader) // config["log_per_epoch"]
    if log_iterations == 0:
        log_iterations = 1

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)

    rloo = RLOOWrapper(model, config["rloo_samples"], config["perception_lr"])

    for epoch in range(1, config["epochs"] + 1):
        cum_hamming = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            grid, label, _ = batch

            x = grid.to(device)
            label = label.to(device)
            cum_hamming += rloo(x, label).item()

            if (i + 1) % log_iterations == 0:

                print(f"epoch: {epoch} "
                      f"Hamming: {cum_hamming / log_iterations:.4f} ")
                wandb.log({
                    # "epoch": epoch,
                    "RL_Hamming": cum_hamming / log_iterations,
                })
                cum_hamming = 0

        end_epoch_time = time.time()

        if epoch % config["val_freq"] != 0:
            continue

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc_prior = 0.
        val_acc_clauses = 0.
        val_acc_w = 0.
        val_avg_dist = 0.
        for i, batch in enumerate(val_loader):
            grid, label, weights = batch
            test_result = val_WC(grid.to(device), label.to(device), weights.to(device), model, device)
            val_acc_prior += test_result[1]
            val_acc_clauses += test_result[2]
            val_acc_w += test_result[3]
            val_avg_dist += test_result[4]

        val_accuracy_prior = val_acc_prior / len(val_loader)
        val_accuracy_clauses = val_acc_clauses / len(val_loader)
        val_accuracy_w = val_acc_w / len(val_loader)
        val_average_dist = val_avg_dist / len(val_loader)

        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy_prior} {prefix} clauses acc: {val_accuracy_clauses}"
                f" {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_accuracy_hamming": val_accuracy_clauses,
            f"{wdb_prefix}_accuracy_w": val_accuracy_w,
            f"{wdb_prefix}_average_dist": val_average_dist,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })


