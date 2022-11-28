from torch.utils.data import DataLoader
from experiments.data import addition
from experiments.nrm_mnist import MNISTAddModel
import torch
import wandb


if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "mc_method": "gfnexact",
        "N": 1,
        "batch_size": 11,
        "amt_samples": 7,
        "nrm_lr": 1e-3,
        "nrm_loss": "bce",
        "perception_lr": 1e-3,
        "perception_loss": "log-q",
        "epochs": 50,
        "log_iterations": 100,
        "hidden_size": 200,
        "uniform_prob": 0.0,
        "greedy_prob": 0.0,
        "prune": True,
        "dirichlet_init": 1,
        "dirichlet_lr": 0.1,
        "dirichlet_iters": 10,
        "K_beliefs": 100,
    }

    name = "addition_" + str(config["N"])

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_set = addition(config["N"], "train")
    val_set = addition(config["N"], "val")
    # test_set = addition(config["N"], "test")

    model = MNISTAddModel(config, device=device)

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size"], False)

    wandb.init(
        project="test-project",
        entity="nesy-gems",
        name=name,
        notes="Test run",
        tags=[],
        config=config,
    )

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        prob_sample_train = 0

        for i, batch in enumerate(train_loader):
            numb1, numb2, label = batch

            x = torch.cat([numb1, numb2], dim=1).to(device)
            label = label.to(device)
            loss_nrm, loss_percept = model.train(x, label)

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            prob_sample_train += model.test(x, label).item()

            if (i + 1) % config['log_iterations'] == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean().item()
                print(f"actor: {cum_loss_percept / config['log_iterations']:.4f} "
                      f"nrm: {cum_loss_nrm / config['log_iterations']:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"train_accuracy: {prob_sample_train / config['log_iterations']:.4f}")

                wandb.log({
                    "epoch": epoch,
                    "percept_loss": cum_loss_percept / config['log_iterations'],
                    "nrm_loss": cum_loss_nrm / config['log_iterations'],
                    "train_accuracy": prob_sample_train / config['log_iterations'],
                    "avg_alpha": avg_alpha,
                })
                cum_loss_percept = 0
                cum_loss_nrm = 0
                prob_sample_train = 0

        print("----- TESTING -----")
        prob_sample = 0.
        for i, batch in enumerate(val_loader):
            numb1, numb2, label = batch
            x = torch.cat([numb1, numb2], dim=1)
            prob_sample += model.test(x.to(device), label.to(device)).item()

        print("Test accuracy: ", prob_sample / len(val_loader))
        wandb.log({
            "epoch": epoch,
            "test_accuracy: ": prob_sample / len(val_loader),
        })
