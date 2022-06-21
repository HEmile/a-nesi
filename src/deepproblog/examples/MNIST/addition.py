import sys
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.engines.mc_engine import MCEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.network import MNIST_Net, MNIST_NETWORK_NAME
from deepproblog.examples.MNIST.data import (
    MNIST_train,
    MNIST_test,
    addition,
)
from deepproblog.examples.MNIST.sampler import AdditionSampler
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.sampling.grad_estim import factory_storch_method
from deepproblog.sampling.sampler import Sampler
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string


if __name__ == '__main__':
    # I suppose this is done to enumerate the possible configurations?
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    parameters = {
        "method": ["mc", "exact", "gm"],
        "grad_estim": ["rao-blackwell"],
        "N": [1, 2, 3],
        "pretrain": [0],
        "exploration": [False, True],
        "run": range(5),
        "batch_size": [13],
        "amt_samples": [4],
        "lr": [1e-3],
        "importance_sampling": [False]
    }


    args = get_configuration(parameters, i)

    torch.manual_seed(args["run"])

    name = "addition_" + config_to_string(args) + "_" + format_time_precise()

    train_set = addition(args["N"], "train")
    test_set = addition(args["N"], "test")

    network = MNIST_Net()

    pretrain = args["pretrain"]
    if pretrain is not None and pretrain > 0:
        network.load_state_dict(
            torch.load("models/pretrained/all_{}.pth".format(args["pretrain"]))
        )

    sampler = None
    if args["method"] == "mc":
        if args["importance_sampling"]:
            sampler = AdditionSampler(factory_storch_method(args["grad_estim"]), args["amt_samples"])
        else:
            sampler = Sampler(factory_storch_method(args["grad_estim"]), args["amt_samples"], 19)  # TODO: n classes target 19 depends on the amount of digits
    net = Network(network, MNIST_NETWORK_NAME, sampler=sampler, batching=True)
    model = Model("models/addition.pl", [net])

    net.optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"])
    if args["method"] == "exact":
        if args["exploration"] or args["N"] > 2:
            print("Not supported?")
            exit()
        model.set_engine(ExactEngine(model), cache=True)
    elif args["method"] == "gm":
        model.set_engine(
            ApproximateEngine(
                model, 1, geometric_mean, exploration=args["exploration"]
            )
        )
    elif args["method"] == "mc":
        model.set_engine(
            MCEngine(model)
        )
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, args["batch_size"], False)
    train = train_model(model, loader, 1,
                        log_iter=100,
                        profile=0,
                        test=lambda model: get_confusion_matrix(model, test_set, verbose=1).accuracy(),
                        run_note=name,
                        tags=[args['method']],
                        extra_config=args
                        )
    model.save_state("snapshot/" + name + ".pth")
    # train.logger.comment(dumps(model.get_hyperparameters()))
    # MAJOR TODO: Make sure it computes the confusion matrix accuracy!
    # train.logger.comment(
    #     "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    # )
    # train.logger.write_to_file("log/" + name)
