import sys
import torch
import wandb
from torch.utils.data import DataLoader

from ppe.experiments.data import (
    addition,
)
from ppe.experiments.nrm_mnist import MNISTAddModel

LOG_ITER = 100

# This code is mostly based on the DeepProbLog MNIST example.
# It is modified to remove behavior unnecessary for the purposes of this experiment.
if __name__ == '__main__':
    # I suppose this is done to enumerate the possible configurations?
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    parameters = {
        "mc_method": "gfnexact",
        "N": 1,
        "batch_size": 16,
        "amt_samples": 100,
        "nrm_lr": 1e-3,
        "perception_lr": 1e-3,
        "epochs": 20,
        "hidden_size": 200,
        "uniform_prob": 0.0,
        "greedy_prob": 0.0,
        "prune": False,
        "loss": 'mse-tb',
        "dirichlet_init": 1,
        "dirichlet_lr": 0.1,
        "dirichlet_iters": 0,
        "K_beliefs": 100
    }

    # TODO: Move hyperparameter sweep to wandb sweep
    args = parameters

    name = "addition_" + str(args["N"])

    train_set = addition(args["N"], "train")
    test_set = addition(args["N"], "test")

    model = MNISTAddModel(args)

    loader = DataLoader(train_set, args["batch_size"], False)

    args = args if args else {}
    wandb.init(
        project="deepproblog-mc",
        entity="hemile",
        notes=name,
        tags=[args['mc_method']],
        config=args,
        mode="disabled",
    )

    # with torch.autograd.set_detect_anomaly(True):
    for epoch in range(args["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        for i, batch in enumerate(loader):
            numb1, numb2, label = batch

            x = torch.cat([numb1, numb2], dim=1)
            loss_nrm, loss_percept, P = model.train(x, label)

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            if (i + 1) % LOG_ITER == 0:
                print(f"actor: {cum_loss_percept / LOG_ITER:.4f} gfn: {cum_loss_nrm / LOG_ITER:.4f}" )

                wandb.log({"percept_loss": cum_loss_percept / LOG_ITER,
                           "nrm_loss": cum_loss_nrm / LOG_ITER},)
                cum_loss_percept = 0
                cum_loss_nrm = 0

