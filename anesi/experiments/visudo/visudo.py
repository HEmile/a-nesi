import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb

from experiments.visudo.anesi_visudo import ViSudoModel
from experiments.visudo.parse_data import get_datasets
from inference_models import NoPossibleActionsException
from torchmetrics.classification import BinaryAUROC


SWEEP = True

def val_sudo(x, label, model, device):
    try:
        test_result = model.test(x, label, None)
    except NoPossibleActionsException:
        print("No possible actions during testing")
        test_result = val_sudo(x, label, model, device)
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    acc_clauses = test_result[2].item()
    prior_y = test_result[3]
    return acc, acc_prior, acc_clauses, prior_y

if __name__ == '__main__':
    config = {
        "N": 9,
        "DEBUG": False,
        "amt_samples": 100,
        "batch_size": 2,
        "batch_size_test": 100,
        "dirichlet_init": 0.01,
        "dirichlet_iters": 2,
        "dirichlet_L2": 100000.0,
        "dirichlet_lr": 0.1,
        "epochs": 300,
        "encoding": "pair",
        "fixed_alpha": None,
        "hidden_size": 200,
        "K_beliefs": 4,
        "layers": 1,
        "log_per_epoch": 1,
        "P_source": "both",
        "percept_loss_pref": 1.0,
        "perception_lr": 0.001,
        "perception_loss": "log-q",
        "policy": "off",
        "predict_only": True,
        "pretrain_epochs": 0,
        "prune": False,
        "q_lr": 0.001,
        "q_loss": "mse",
        "split": 11,
        "test": False,
        "train_negatives": True,
        "use_cuda": True,
        "verbose": 1,
        "val_freq": 10,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="visudo", entity="nesy-gems")
        config = wandb.config
        print(config)
    elif SWEEP:
        # TODO: I don't get how it's supposed to know what yaml file to open here.
        with open("sweeps/sweep.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)
    else:
        name = "visudo-" + str(config["N"])
        wandb.init(
            project=f"visudo",
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

    model = ViSudoModel(config).to(device)
    train, val, test = get_datasets(config["split"], dimension=config["N"], use_negative_train=config["train_negatives"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)


    model.perception.requires_grad_(False)

    for epoch in range(config["epochs"]):
        cum_loss_percept = 0
        cum_loss_q = 0

        start_epoch_time = time.time()

        if epoch > config["pretrain_epochs"]:
            model.perception.requires_grad_(True)

        for i, batch in enumerate(train_loader):
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            try:
                loss_nrm, loss_percept = model.train_all(x, label)
            except NoPossibleActionsException:
                print("No possible actions during training")
                continue

            cum_loss_percept += loss_percept.item()
            cum_loss_q += loss_nrm.item()

            if (i + 1) % log_iterations == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean()

                log_q_weight = torch.sigmoid((config['percept_loss_pref'] * avg_alpha).log()).item()
                avg_alpha = avg_alpha.item()

                print(f"epoch: {epoch} "
                      f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"nrm: {cum_loss_q / log_iterations:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"log_q_weight: {log_q_weight:.4f} ",)
                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "nrm_loss": cum_loss_q / log_iterations,
                    "avg_alpha": avg_alpha,
                    # "log_q_weight": log_q_weight,
                })
                cum_loss_percept = 0
                cum_loss_q = 0

        end_epoch_time = time.time()

        if epoch % config["val_freq"] != 0:
            continue

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc = 0.
        val_acc_prior = 0.
        val_acc_clauses = 0.
        prior_y = []
        labels = []
        for i, batch in enumerate(val_loader):
            grid, label = batch
            label = label.to(device)
            test_result = val_sudo(grid.to(device), label, model, device)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]
            val_acc_clauses += test_result[2]
            prior_y += [test_result[3]]
            labels += [label]

        val_accuracy = val_acc / len(val_loader)
        val_accuracy_prior = val_acc_prior / len(val_loader)
        val_accuracy_clauses = val_acc_clauses / len(val_loader)

        all_labels = torch.cat(labels, dim=0)
        all_prior_y = torch.cat(prior_y, dim=0)
        val_auroc = BinaryAUROC(thresholds=None)(all_prior_y, all_labels).item()

        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy_prior} {prefix} clauses acc: {val_accuracy_clauses}"
                f" {prefix} auroc: {val_auroc} {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_accuracy_clauses": val_accuracy_clauses,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
