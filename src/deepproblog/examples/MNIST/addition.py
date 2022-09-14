import sys
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
from deepproblog.sampling.mapo_sampler import MemoryAugmentedDPLSampler
from deepproblog.sampling.memoizer import Memoizer
from deepproblog.sampling.sampler import Sampler, DefaultQueryMapper
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, config_to_string


if __name__ == '__main__':
    # I suppose this is done to enumerate the possible configurations?
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    parameters = {
        "method": ["gm", "mc", "exact", "gm"],
        "grad_estim": ["vanilla-sf", "rao-blackwell", "hybrid-baseline"],
        "N": [2, 1, 2, 3],
        "pretrain": [0],
        "exploration": [False, True],
        "run": range(5),
        "batch_size": [13],
        "amt_samples": [6],
        "lr": [1e-3],
        "mc_method": ["memory", "normal", "importance"],
        "epochs": [20],
        "entropy_weight": [0]
    }

    args = get_configuration(parameters, i)

    # torch.manual_seed(args["run"])

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
    memoizer = Memoizer(DefaultQueryMapper())
    n_classes = 2 * 10**args["N"] - 1
    if args["method"] == "mc":
        if args["mc_method"] == "importance":
            sampler = AdditionSampler(factory_storch_method(args["grad_estim"]), args["amt_samples"])
        elif args["mc_method"] == "memory":
            # TODO: Assumes we want equal amount of SWOR as sum-over.
            sampler = MemoryAugmentedDPLSampler(args["amt_samples"] - 1, 1, memoizer, n_classes, args["entropy_weight"])
        else:
            sampler = Sampler(factory_storch_method(args["grad_estim"]), args["amt_samples"], n_classes, args["entropy_weight"])  # TODO: n classes target 19 depends on the amount of digits
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
            MCEngine(model), memoizer
        )
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, args["batch_size"], False)
    train = train_model(model, loader, args["epochs"],
                        log_iter=100,
                        test_iter=10000,
                        profile=0,
                        test=lambda model: get_confusion_matrix(model, test_set, verbose=1).accuracy(),
                        run_note=name,
                        tags=[args['method']],
                        args=args
                        )
    model.save_state("snapshot/" + name + ".pth")
    # train.logger.comment(dumps(model.get_hyperparameters()))
    # MAJOR TODO: Make sure it computes the confusion matrix accuracy!
    # train.logger.comment(
    #     "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    # )
    # train.logger.write_to_file("log/" + name)
