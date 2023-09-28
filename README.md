# Bunny Party Bouncer Mini-Project
IMPORTANT TLDR: the main branch of this repo does not contain the results of relabeling, to see the final results, go to `demonstration` branch, and specifically to [demosntration notebook there](https://github.com/d-lowl/bunny-party/blob/demonstration/demonstration.ipynb)

## Project overview
PUBLIC NOTE: this project (and the task for it) is reworked from an exercise that I could not make public. Hence, I've reworked the dataset and the task, while keeping the implementation more or less the same. I will try marking clearly (with "PUBLIC NOTE") where the decisions were made due to the original task restrictions.

The overall idea behind making this mini-project public, is to show off how DVC and MLFlow can help with data versioning and reproducibility of ML experiments, in the setting of data modifications (relabeling in this case).

Project structure:
* `server/` -- ML model training server source code
* `tests/` -- unit tests (mainly for the server)
* `demonstration.ipynb` -- the demonstration notebook, which walks through the process of model training, data relabeling, and then retraining (doing evaluation, and exploring recorded experiments in between)
* `data/` -- DVC-managed data directory
* `models` -- Keras artifacts are saved here (ignored by git)

The versioned data is hosted on [DagsHub](https://dagshub.com/d-lowl/bunny-party)

## Task
PUBLIC NOTE: The following task is adapted from an exercise that I have done before, which I cannot publicise.

You were tasked to build a system that detects if someone is a bunny to let bunnies only to the **Bunny Party**. You are given a dataset of pictures of bunnies and not-bunnies. However, someone poisoned the dataset, and this dog with bunny ears is able to sneak into the party.

![lovely-pet-portrait-isolated.jpg](demo_preparation%2Fdata%2Foriginal%2Flovely-pet-portrait-isolated.jpg)

We need to prevent this from happening from now on. We want a system that would help us relabel incorrectly labeled images of this impostor dog in our dataset, so that the model actually behaves as intended. 

### Deliverables
Build a RESTful API server with these endpoints:
* Model Training: using the training dataset on the disk, train the model, and save the model to disk
* Model Evaluation: using the test dataset on the disk, evaluate the model, and return the evaluation
* Image Similarity Search: given an image ID, find similar images using an embedding (e.g. PCA), and return their IDs
* Image Relabeling: given an image ID and the desired class, relable it
PUBLIC NOTE: In the real world scenario I would offload the model training and evaluation to a training pipeline, and have similarity search and relabeling handled by a separate service or a library (depending on what kind of embedding is used, and how hard is it to compute it)

Write a notebook, that will do the following:
1. Train and evaluate the model, with the initial dataset
2. Search for mislabeled impostor images, using a seed image (that you find manually or otherwise)
3. Relabel mislabeled images
4. Train and evaluate again, to show the improvement in accuracy

Clearly separate the ML-related implementation and high-level logic (i.e. the server code and the notebook). Cover the code with tests.

PUBLIC NOTE: The original task was talking about "going the extra mile", for which I added the experiment tracking with MLFlow and data versioning with DVC (as it sounds important to version data when we do data relabeling).

## The Final Demonstration Notebook
There's a notebook in a separate branch that shows the process outlined above in the cell outputs, as well as the data is in `demonstration` branch. See the [demosntration notebook there](https://github.com/d-lowl/bunny-party/blob/demonstration/demonstration.ipynb) specifically.

## Prerequisites
* Python
* Pyenv
* Poetry

## Setup
Setup environment and install the dependencies:
```
pyenv local
poetry env use `pyenv which python`
poetry shell
poetry install
```

## Prepare demo dataset
The following will destroy the dataset, and recreate it from the original images:
```bash
rm -r data/* data/.gitignore
python demo_preparation/prepare_dataset.py
dvc add data/*
git add data/test.dvc data/class_name_to_id_mapping.txt.dvc data/train.dvc data/.gitignore
git commit -m "Initial data"
# Tag if necessary
```

## Start server
Start the uvicorn service
```
python server/server.py
```

## Test
```
pytest tests/
```

## Run demonstration notebook
Start Jupyter server and open `demonstration.ipynb`.
