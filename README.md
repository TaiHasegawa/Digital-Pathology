# Graph Neural Network Exploiting Edge Features to Detect Basement Membrane in Oral Mucosal Tissue
A PyTorch implementation of the GNN architecture presented in "Graph Neural Network Exploiting Edge Features to Detect Basement Membrane in Oral Mucosal Tissue"

## Environment Settings
- Python >= 3.9.12, Pytorch >= 1.11.0.
- See  `requirements.txt`  for other packages.

## Run Experiments
Before running the codes, edit `config.json` to change the parameters and settings.
mode: $\in$ {train, val, test} for training, validation and test

Then run the code as follows:
> $ python main.py --json=config.json
