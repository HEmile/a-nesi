import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb

from baselines.semantic_loss import SemanticLoss
from experiments.MNISTNet import MNIST_Net
from experiments.visudo.parse_data import get_datasets
from torchmetrics.classification import BinaryAUROC

from util import log1mexp, log_not

SWEEP = True

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "split": 1,
        "DEBUG": True,
        "N": 4,
        "lr": 1e-3,
        "test": False,
        "batch_size": 2,
        "batch_size_test": 10,
        "epochs": 300,
        "baseline": "SL",
        "log_per_epoch": 1,
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
        with open("sweep.yaml", 'r') as f:
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
            notes="Test run SL",
            mode="disabled",
            tags=[],
            config=config,
        )

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    perception = MNIST_Net(N=config["N"]).to(device)
    p_y_SL = SemanticLoss(f"sudoku_{config['N']}.sdd", f"sudoku_{config['N']}.vtree").to(device)
    train, val, test = get_datasets(config["split"], "../../../experiments/visudo", dimension=config["N"])

    optim = torch.optim.Adam(perception.parameters(), lr=config["lr"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    print(len(val_loader))

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)


    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            _P = perception(x)
            P = _P.view(_P.shape[0], -1)

            # Semantic loss returns log-probability of y=1 (sudoku is true)
            py_per_sample = p_y_SL(P)
            pgiven_y = torch.clone(py_per_sample)
            pgiven_y[label == 0] = log_not(pgiven_y[label == 0])
            loss = -pgiven_y.mean()

            if torch.isnan(loss):
                print("NAN LOSS")
                print(pgiven_y)
            else:
                loss.backward()

            optim.step()

            cum_loss += loss.item()

            if (i + 1) % log_iterations == 0:
                pr_loss = cum_loss / log_iterations
                print("Loss: ", pr_loss)
                wandb.log({"Loss": pr_loss,})
                cum_loss = 0

        end_epoch_time = time.time()

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc = 0.
        val_loss = 0.
        preds = []
        labels = []
        for i, batch in enumerate(val_loader):
            grid, label = batch
            label = label.to(device)
            x = grid.to(device)
            label = label.to(device)
            P = perception(x)
            P = P.view(P.shape[0], -1)

            log_py_per_sample = p_y_SL(P)
            pred = log_py_per_sample.exp() > 0.5
            preds += [pred]
            val_acc += (pred == label).float().mean().item()
            pgiven_y = torch.where(label == 1, log_py_per_sample, log_not(log_py_per_sample))
            val_loss += -pgiven_y.mean()
            labels += [label]

        val_accuracy = val_acc / len(val_loader)

        all_labels = torch.cat(labels, dim=0)
        all_preds_y = torch.cat(preds, dim=0).long()
        val_auroc = BinaryAUROC(thresholds=None)(all_preds_y, all_labels).item()

        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy} {prefix}"
                f" {prefix} auroc: {val_auroc} {prefix}"
                f" loss: {val_loss / len(val_loader)} {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_loss": val_loss / len(val_loader),
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
