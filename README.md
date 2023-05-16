# A-NeSI: Approximate Neurosymbolic Inference

We introduce **Approximate Neurosymbolic Inference (A-NeSI)** a new framework for Probabilistic Neurosymbolic Learning that uses neural networks for scalable approximate inference. A-NeSI 
1. performs approximate inference in polynomial time without relaxing the semantics of probabilistic logics; 
2. is trained using synthetic data generated by the background knowledge; 
3. can generate symbolic explanations of predictions; and 
4. can guarantee the satisfaction of logical constraints at test time. 

For more information, consult the papers listed below.

## Requirements

A-NeSI has the following requirements:

* [Python](https://www.python.org/) - v3.9.15
* [PyTorch](https://pytorch.org/) - v1.12.1
* [TorchVision](https://pytorch.org/vision/stable/index.html) - v0.13.1
* [Weights&Biases](https://wandb.ai/) - v0.13.2

Run the following:

1. Install the dependencies inside a new virtual environment: `bash setup_dependencies.sh`

2. Activate the virtual environment: `conda activate NRM`

3. Install the A-NeSI module: pip install -e .

## Experiments

The experiments are presented in the papers are available in the [anesi/experiments](anesi/experiments) directory. The experiments are organized with Weights&Biases. To reproduce the experiments from the paper, run
```bash
cd anesi/experiments
wandb sweep repeat/test_predict_only.yaml
wandb agent <sweep_id>
```
Note that you will need to update the entity and project parameters of wandb in the sweep files. 

