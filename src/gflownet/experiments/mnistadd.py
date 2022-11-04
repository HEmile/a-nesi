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
        "method": ["gfn"],
        "grad_estim": ["vanilla-sf", "rao-blackwell", "hybrid-baseline"],
        "N": [1, 2, 3],
        "run": range(5),
        "batch_size": [11],
        "amt_samples": [60],
        "lr": [1e-3],
        # "mc_method": ["memory", "normal", "importance"],
        "epochs": [20],
        # "entropy_weight": [0]
    }

    # TODO: Move hyperparameter sweep to wandb sweep
    args = get_configuration(parameters, i)

    name = "addition_" + config_to_string(args) + "_" + format_time_precise()

    train_set = addition(args["N"], "train")
    test_set = addition(args["N"], "test")

    model = MNISTAddModel(args["N"], args["method"])

    optimizer_p = torch.optim.Adam(model.parameters(), lr=args["lr"])

    loader = DataLoader(train_set, args["batch_size"], False)

    args = args if args else {}
    wandb.init(
        project="deepproblog-mc",
        entity="hemile",
        notes=name,
        tags=[args['method']],
        config=args,
        mode="disabled"
    )

    for epoch in range(args["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss = 0
        cum_prob = 0
        for i, batch in enumerate(loader):
            optimizer_p.zero_grad()
            numb1, numb2, label = batch

            loss, succes_p = model(numb1, numb2, label, args["amt_samples"])

            cum_loss += loss.item()
            cum_prob += succes_p.mean().item()
            loss.backward()
            optimizer_p.step()
            if (i + 1) % LOG_ITER == 0:
                print(cum_loss / LOG_ITER, cum_prob / LOG_ITER)
                wandb.log({"loss": cum_loss / LOG_ITER, "succes_prob": cum_prob / LOG_ITER})
                cum_loss = 0
                cum_prob = 0

