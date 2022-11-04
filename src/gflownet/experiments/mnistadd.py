import sys
import torch
import wandb
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from deepproblog.examples.MNIST.network import MNIST_Net
from gflownet.experiments import GFlowNetExact
from gflownet.experiments.data import (
    addition,
)
from deepproblog.utils import get_configuration, format_time_precise, config_to_string

LOG_ITER = 100

# This code is mostly based on the DeepProbLog MNIST example.
# It is modified to remove behavior unnecessary for the purposes of this experiment.
if __name__ == '__main__':
    # I suppose this is done to enumerate the possible configurations?
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    parameters = {
        "method": ["gfnexact"],
        "grad_estim": ["vanilla-sf", "rao-blackwell", "hybrid-baseline"],
        "N": [1, 2, 3],
        "run": range(5),
        "batch_size": [10],
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

    # The NN that will model p(x) (digit classification probabilities)
    network = MNIST_Net()

    optimizer_p = torch.optim.Adam(network.parameters(), lr=args["lr"])

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

    gfn = GFlowNetExact(args["N"])

    for epoch in range(args["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss = 0
        cum_prob = 0
        for i, batch in enumerate(loader):
            optimizer_p.zero_grad()
            numb1, numb2, label = batch

            # Predict the digit classification probabilities
            p1 = network(numb1)
            p2 = network(numb2)

            # Sample both positive and negative worlds to contrast using gflownet
            sample1_pos, sample2_pos = gfn.sample(p1, p2, label, 'positives', args["amt_samples"])

            # TODO: Generalize this to N > 1
            sample1_pos = sample1_pos[0].T
            sample2_pos = sample2_pos[0].T

            log_p_pos = (p1.log().gather(1, sample1_pos) + p2.log().gather(1, sample2_pos)).mean(1)

            # Smoothed mc succes probability estimate. Smoothed to ensure positive samples aren't ignored, but obv biased
            # Should maybe test if unbiased estimation works as well
            sample_r1 = Categorical(p1).sample((2*args["amt_samples"],))
            sample_r2 = Categorical(p2).sample((2*args["amt_samples"],))
            corr_counts = (sample_r1 + sample_r2 == label).float().sum(0)
            succes_p = corr_counts / (2*args["amt_samples"])
            succes_p_smooth = (corr_counts + 1) / (2*args["amt_samples"] + 2)

            # Use success probabilities as importance weights for the samples
            loss = (-log_p_pos * succes_p_smooth).mean()
            cum_loss += loss.item()
            cum_prob += succes_p.mean().item()
            loss.backward()
            optimizer_p.step()
            if i % LOG_ITER == 0:
                print(cum_loss / LOG_ITER, cum_prob / LOG_ITER)
                wandb.log({"loss": cum_loss / LOG_ITER, "succes_prob": cum_prob / LOG_ITER})
                cum_loss = 0
                cum_prob = 0

