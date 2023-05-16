import argparse
import time

import yaml
from torch.utils.data import DataLoader

from baselines.rloo import RLOOWrapper
from experiments.mnist_op import MNISTAddModel, MNISTMulModel
import torch
import wandb

from experiments.mnist_op.data import addition, multiplication
from inference_models import NoPossibleActionsException

def test(x, label, label_digits, model, device):
    label_digits_l = list(map(lambda d: d.to(device), label_digits[0] + label_digits[1]))
    try:
        test_result = model.test(x, label, label_digits_l)
    except NoPossibleActionsException:
        print("No possible actions during testing")
        test_result = test(x, label, label_digits, model, device)
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    explain_acc = test_result[2].item()
    digit_acc = test_result[3].item()
    return acc, acc_prior, explain_acc, digit_acc

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N": 1,
        "op": "add",
        "test": False,
        "batch_size": 16,
        "batch_size_test": 16,
        "rloo_samples": 100,
        "perception_lr": 1e-3,
        "epochs": 30,
        "log_per_epoch": 10,
        "y_encoding": "base10",
        # Need to encode these for the ANESI class
        "amount_samples": 100,
        "model": "full",
        "layers": 1,
        "hidden_size": 128,
        "prune": False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="mnist-add", entity="nesy-gems")
        config = wandb.config
        print(config)
    else:
        name = "addition_" + str(config["N"])
        wandb.init(
            project=f"mnist-{config['op']}",
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

    op = None
    model = None
    if config["op"] == "add":
        op = addition
        model = MNISTAddModel(config).to(device)
    elif config["op"] == "mul":
        op = multiplication
        model = MNISTMulModel(config).to(device)
    if config["test"]:
        train_set = op(config["N"], "full_train")
        val_set = op(config["N"], "test")
    else:
        train_set = op(config["N"], "train")
        val_set = op(config["N"], "val")

    rloo = RLOOWrapper(model, config["rloo_samples"], config["rloo_lr"])

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size_test"], False)

    print(len(val_loader))

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_hamming = 0
        train_acc = 0
        train_acc_prior = 0
        train_explain_acc = 0
        train_digit_acc = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            # label_digits is ONLY EVER to be used during testing!!!
            numb1, numb2, label, label_digits = batch

            x = torch.cat([numb1, numb2], dim=1).to(device)
            label = label.to(device)
            cum_hamming += rloo(x, label)

            test_result = test(x, label, label_digits, model, device)
            train_acc_prior += test_result[1]
            train_explain_acc += test_result[2]
            train_digit_acc += test_result[3]

            if (i + 1) % log_iterations == 0:
                print(f"Epoch {epoch + 1}/{config['epochs']}, "
                      f"Hamming: {cum_hamming / log_iterations:.4f}, "
                      f"train_acc_prior: {train_acc_prior / log_iterations:.4f}",
                      f"train_explain_acc: {train_explain_acc / log_iterations:.4f}",
                      f"train_digit_acc: {train_digit_acc / log_iterations:.4f}")

                wandb.log({
                    # "epoch": epoch,
                    "train_accuracy": train_acc / log_iterations,
                    "train_accuracy_prior": train_acc_prior / log_iterations,
                    "train_explain_accuracy": train_explain_acc / log_iterations,
                    "train_digit_accuracy": train_digit_acc / log_iterations,
                    "hamming": cum_hamming / log_iterations,
                })
                cum_hamming = 0
                train_acc_prior = 0
                train_explain_acc = 0
                train_digit_acc = 0

        end_epoch_time = time.time()

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc_prior = 0.
        val_explain_acc = 0.
        val_digit_acc = 0.
        for i, batch in enumerate(val_loader):
            numb1, numb2, label, label_digits = batch
            x = torch.cat([numb1, numb2], dim=1)

            test_result = test(x.to(device), label.to(device), label_digits, model, device)
            val_acc_prior += test_result[1]
            val_explain_acc += test_result[2]
            val_digit_acc += test_result[3]

        val_accuracy_prior = val_acc_prior / len(val_loader)
        val_digit_accuracy = val_digit_acc / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} Accuracy: {val_accuracy_prior}"
              f"{prefix} Digit: {val_digit_accuracy} Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_digit_accuracy": val_digit_accuracy,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
