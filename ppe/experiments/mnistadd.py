import argparse
import time

import yaml
from torch.utils.data import DataLoader
from experiments.data import addition
from experiments.nrm_mnist import MNISTAddModel
import torch
import wandb

SWEEP = True

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N": 1,
        "batch_size": 256,
        "amt_samples": 100,
        "nrm_lr": 1e-3,
        "nrm_loss": "bce",
        "policy": "both",
        "perception_lr": 1e-3,
        "perception_loss": "log-q",
        "epochs": 30,
        "log_per_epoch": 10,
        "hidden_size": 200,
        # "uniform_prob": 0.0,
        # "greedy_prob": 0.0,
        "prune": True,
        "dirichlet_init": 1,
        "dirichlet_lr": 0.1,
        "dirichlet_iters": 10,
        "K_beliefs": 100,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="test-project", entity="nesy-gems")
        config = wandb.config
        print(config)
    elif SWEEP:
        # TODO: I don't get how it's supposed to know what yaml file to open here.
        with open("sweeps/sweep.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config = wandb.config
        print(config)
    else:
        name = "addition_" + str(config["N"])
        wandb.init(
            project="test-project",
            entity="nesy-gems",
            name=name,
            notes="Test run",
            mode="disabled",
            tags=[],
            config=config,
        )

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_set = addition(config["N"], "train")
    val_set = addition(config["N"], "val")
    # test_set = addition(config["N"], "test")

    model = MNISTAddModel(config, device=device)

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    # if config["DEBUG"]:
    #     torch.autograd.set_detect_anomaly(True)

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        prob_sample_train = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            numb1, numb2, label = batch

            x = torch.cat([numb1, numb2], dim=1).to(device)
            label = label.to(device)
            loss_nrm, loss_percept = model.train(x, label)

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            prob_sample_train += model.test(x, label).item()

            if (i + 1) % log_iterations == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean()

                log_q_weight = torch.sigmoid(avg_alpha.log()).item()
                avg_alpha = avg_alpha.item()

                print(f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"nrm: {cum_loss_nrm / log_iterations:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"log_q_weight: {log_q_weight:.4f} ",
                      f"train_accuracy: {prob_sample_train / log_iterations:.4f}")

                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "nrm_loss": cum_loss_nrm / log_iterations,
                    "train_accuracy": prob_sample_train / log_iterations,
                    "avg_alpha": avg_alpha,
                    "log_q_weight": log_q_weight,
                })
                cum_loss_percept = 0
                cum_loss_nrm = 0
                prob_sample_train = 0

        end_epoch_time = time.time()

        print("----- VALIDATING -----")
        prob_sample = 0.
        for i, batch in enumerate(val_loader):
            numb1, numb2, label = batch
            x = torch.cat([numb1, numb2], dim=1)
            prob_sample += model.test(x.to(device), label.to(device)).item()

        val_accuracy = prob_sample / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time
        print("Validation accuracy: ", val_accuracy, "Epoch time: ", epoch_time, "Test time: ", test_time)

        wandb.log({
            # "epoch": epoch,
            "val_accuracy": val_accuracy,
            "val_time": test_time,
            "epoch_time": epoch_time,
        })
