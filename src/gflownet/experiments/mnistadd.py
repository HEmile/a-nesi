import sys
import torch
import wandb
from torch.utils.data import DataLoader

from gflownet.experiments.data import (
    addition,
)
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from gflownet.experiments.gfn_mnist import MNISTAddModel

LOG_ITER = 100

# This code is mostly based on the DeepProbLog MNIST example.
# It is modified to remove behavior unnecessary for the purposes of this experiment.
if __name__ == '__main__':
    # I suppose this is done to enumerate the possible configurations?
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    parameters = {
        "mc_method": ["gfnexact"],
        "grad_estim": ["vanilla-sf", "rao-blackwell", "hybrid-baseline"],
        "N": [1, 2, 3],
        "run": range(5),
        "batch_size": [11],
        "amt_samples": [7],
        "lr": [1e-3],
        "epochs": [20],
        "hidden_size": [200],
        "uniform_prob": [0.0],
        "greedy_prob": [0.0],
        "prune": [False],
        "loss": ['bce-tb', 'mse-tb'],
    }

    # TODO: Move hyperparameter sweep to wandb sweep
    args = get_configuration(parameters, i)

    name = "addition_" + config_to_string(args) + "_" + format_time_precise()

    train_set = addition(args["N"], "train")
    test_set = addition(args["N"], "test")

    model = MNISTAddModel(args)

    optimizer_p = torch.optim.Adam(model.parameters(), lr=args["lr"])

    loader = DataLoader(train_set, args["batch_size"], False)

    args = args if args else {}
    wandb.init(
        project="deepproblog-mc",
        entity="hemile",
        notes=name,
        tags=[args['mc_method']],
        config=args,
        mode="disabled"
    )

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args["epochs"]):
            print("----------------------------------------")
            print("NEW EPOCH", epoch)
            cum_loss_p = 0
            cum_loss_gfn = 0
            cum_partition = 0
            cum_prob = 0
            print(len(loader))
            for i, batch in enumerate(loader):
                optimizer_p.zero_grad()
                numb1, numb2, label = batch

                loss_p, loss_gfn, partition, succes_p = model([numb1], [numb2], label, args)

                # if epoch < 2:
                #     # First just explore and train the gfn before training the 'actor'
                #     loss_p *= 0

                cum_loss_p += loss_p.item()
                cum_loss_gfn += loss_gfn.item()
                cum_partition += partition.mean().item()
                cum_prob += succes_p.item()

                total_loss = loss_p + loss_gfn
                total_loss.backward()
                optimizer_p.step()
                if (i + 1) % LOG_ITER == 0:
                    total_cum_loss = cum_loss_p + cum_loss_gfn

                    print(f"actor: {cum_loss_p / LOG_ITER:.4f} gfn: {cum_loss_gfn / LOG_ITER:.4f} "
                          f"partition: {cum_partition / LOG_ITER:.4f} "
                          f"prob: {cum_prob / LOG_ITER:.4f}")

                    wandb.log({"loss": total_cum_loss / LOG_ITER, "succes_prob_train": cum_prob / LOG_ITER,
                               "sf_loss": cum_loss_p / LOG_ITER, "gfn_loss": cum_loss_gfn / LOG_ITER,
                               "partition": cum_partition / LOG_ITER},)
                    cum_loss_p = 0
                    cum_loss_gfn = 0
                    cum_partition = 0
                    cum_prob = 0

