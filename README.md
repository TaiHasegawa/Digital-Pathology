# Graph Neural Network Exploiting Edge Features to Detect Basement Membrane in Oral Mucosal Tissue
A PyTorch implementation of the GNN architecture presented in "Graph Neural Network Exploiting Edge Features to Detect Basement Membrane in Oral Mucosal Tissue".
The GNN model takes distance and the gradients of density between nuclei.

It is not allowed to publish the tissue dataset due to the policy of [Swedish Ethical Review Authority](https://www.government.se/government-agencies/the-swedish-ethics-review-authority-etikprovningsmyndigheten/).
If your interested in our work, please contact [us](hasegawa.t.as@m.titech.ac.jp).

## Environment Settings
- Python >= 3.9.12, Pytorch >= 1.11.0.
- See  `requirements.txt`  for other packages.

## Run Experiments
Before running the codes, edit `config.json` to change the parameters and settings.

mode: $\in$ {train, val, test} for training, validation and test.

Then run the code as follows:
> $ python main.py --json=config.json
