from torch.utils.data import DataLoader
from experiments.data import addition
from experiments.nrm_mnist import MNISTAddModel
import torch
import wandb


if __name__ == '__main__':
    config = {
        "mc_method": "gfnexact",
        "N": 1,
        "batch_size": 16,
        "amt_samples": 100,
        "nrm_lr": 1e-3,
        "perception_lr": 1e-3,
        "epochs": 20,
        "log_iterations": 100,
        "hidden_size": 200,
        "uniform_prob": 0.0,
        "greedy_prob": 0.0,
        "prune": False,
        "loss": 'mse-tb',
        "dirichlet_init": 1,
        "dirichlet_lr": 0.1,
        "dirichlet_iters": 10,
        "K_beliefs": 100
    }

    # TODO: Setup hyperparameter sweep
    name = "addition_" + str(config["N"])

    train_set = addition(config["N"], "train")
    test_set = addition(config["N"], "test")

    model = MNISTAddModel(config)

    train_loader = DataLoader(train_set, config["batch_size"], False)
    test_loader = DataLoader(test_set, config["batch_size"], False)

    wandb.init(
        project="test-project",
        entity="nesy-gems",
        name=name,
        notes="Test run",
        tags=[],
        config=config,
    )

    train_set = addition(config["N"], "train")
    test_set = addition(config["N"], "test")

    model = MNISTAddModel(config)

    loader = DataLoader(train_set, config["batch_size"], False)

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0

        for i, batch in enumerate(train_loader):
            numb1, numb2, label = batch

            x = torch.cat([numb1, numb2], dim=1)
            loss_nrm, loss_percept = model.train(x, label)

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            if (i + 1) % config['log_iterations'] == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean().item()
                print(f"actor: {cum_loss_percept / config['log_iterations']:.4f} "
                      f"nrm: {cum_loss_nrm / config['log_iterations']:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f}")

                wandb.log({
                    "epoch": epoch,
                    "percept_loss": cum_loss_percept / config['log_iterations'],
                    "nrm_loss": cum_loss_nrm / config['log_iterations'],
                    "avg_alpha": avg_alpha,
                })
                cum_loss_percept = 0
                cum_loss_nrm = 0

        print("----- TESTING -----")
        prob_sample = 0.
        for i, batch in enumerate(test_loader):
            numb1, numb2, label = batch
            x = torch.cat([numb1, numb2], dim=1)
            prob_sample += model.test(x, label).item()

        print("Test accuracy: ", prob_sample / len(test_loader))
