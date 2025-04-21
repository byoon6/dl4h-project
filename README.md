# DL4H Final Project

This repository contains source code for reproducing the paper
"Temporal Supervised Contrastive Learning for Modeling Patient Risk Progression"
by Noroozizadeh S. et al.

To run the code, open `RunMain.ipynb` and follow the steps in the notebook.
The notebook has been thoroughly commented.

### Prerequisites

- Python 3.11.11 
- Jupyter Notebook 7.3.2 
- PyTorch 2.2.2
- scikit-learn 1.6.1
- NumPy 1.26.4

As the notebook uses synthetically generated data, it is not computationally heavy and there is no need for GPU.


### Contents

- `author/` contains (mostly unchanged) source code from [the authors' repository](https://github.com/Shahriarnz14/Temporal-Supervised-Contrastive-Learning).
- `byoon6/` contains Python code I implemented for this reproduction exercise.
- `RunMain.ipynb` is the main notebook that contains training and evaluation procedures.
- `ablation-*.ipynb` are the modified notebooks that contain results for the ablation studies.

