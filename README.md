# partial-knowledge-distillation

## :book: Description

Student project as part of the Advanced NLP course of CentraleSupélec


## :busts_in_silhouette: Team Members

- Sacha Muller
- Guillaume Dugat


## :file_folder: Structure of the repository

The repository contains the following files : 
- `distill.py` : contains the training loop
- `dataset.py` : everything related to the dataset and dataloaders
- `models.py` : functions to load the models and tokenizer
- `loss.py` : contains the two losses used in our experiments
- `config.yaml` : contains all the parameters used in the experiment

There is also two notebooks that were used to plot graphs :
- `data_exploration.ipynb` : analyzes the dataset
- `experiments_exploration.ipynb` : analyzes the results of our experiments

## :hammer: Installation

This code can be launched using Python 3.9 or Python 3.10. You can install the requirements with : 
```
pip install -r requirements.txt
```


## :ferris_wheel: Usage

After modifying the `config.yaml` file (details below), you can launch the training with : 
```
python3 distill.py
```

## :wrench: Config file

The structure of the `config.yaml` file is as follows : 

- `paths` :
  - `data_folder` : name of the data folder
  - `token_dataset`: name of the file saving the tokenized dataset
  - `checkpoints`: name of the folder containing the checkpoints

- `distilled_model`:
  - `nb_layers`: number of layers in the distill model obtained with `model.get_untrained_distilgpt2`

- `training_parameters`:
  - `learning_rate`: learning rate value
  - `nb_epochs` : number of epochs
  - `batch_size` : size of the minibatches
  - `checkpoints_name_last`: name of the last checkpoint file
  - `checkpoints_name_best`: name of the best checkpoint file (see troubleshooting section)
  - `resume`: a boolean allowing to start an experiment from a previous checkpoint
  - `loss`: class of the loss to use along with its eventual parameters using the [configue syntax to load classes](https://github.com/illuin-tech/configue#instantiating-classes)

- `seed`: seed for randomness

## :japanese_ogre: Troubleshooting

- The saving of the best checkpoint during training has not been implemented yet, only the last checkpoint is saved. Also be aware that if you don't change the last checkpoint name in the config file, your results will be squashed when you launch another experiment.
- The evolution of the losses during training is only printed in the terminal. It would be better to use a logger.
- A nice feature would be to allow to pass the name of the config file as an argument, as it would allow to launch several experiments at the same time without having to edit the config file manually between each launch.
