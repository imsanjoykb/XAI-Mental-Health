# Introduction
Code and data to accompany the TIFU (Transparency For Interpretability and Understandability) paper.

This repo contains the code and simulation data for generating the pedagogical examples presented in the paper.
The data is a simulated population using demographic and depression characteristics available in the literature, but we do not claim it is "realistic" or useful outside the context of the demonstrations in the accompanying paper.

Similarly, the notebooks provided implement the "toy" neural network and logistic regression model used in the paper for pedagogical purposes.  We did not spend time optimising either model for production/real world use cases, nor did we spend time pre-processing the data.

# 0 : Installation and Requirements
You'll need a working PyTorch (version 1.10.1), CUDA (version 10.2), Python 3 and Jupyter installation and access to a GPU (or, to run on a CPU, the code will require some modification).

You can inspect the data used to produce the plots in the paper (without the need to re-run/train the models).

* `code` -- this is the location for the scripts (see below)
* `sim-data` -- this contains a CSV file for the simulated population of patients with depression, varying demographics and treatment histories.  
* `net-states` -- neural network parameter states stored during training (for reproducibility -- avoiding the need to train the network)
* `out-data` -- a) CSV versions of neural network weight matrices and b) CSV dumps of the network's activations for the test set data (for reproducibility / visualisation without the need to train the network)

# 1 : Code
To experiment with the models in the TIFU paper use the following scripts:

* `log-reg-nn-model-depression-train-big-model.ipynb` is the notebook for training the 7 (linear) x 32 (ReLU) x 1 (sigmoid) densely-connected MLP example used in the paper
* `log-reg-nn-model-depression-test-big-model.ipynb` is the notebook to load the trained network state and "test" the model, then presenting each test-set patient and extracting/dumping the model's activations which are then used for visualisations -- the output from this notebook is in `out-data\test-acts-big-model.csv`
* `glm-model.ipynb` is the equivalent logistic regression model
* `datautils.py` is helper code to load the simulated patient data in a way compatible with PyTorch

Both the toy neural network and the logistic regression use the following variables in `sim-data/population-sample.csv`:

* `Age`, `Sex`, `MDD_hx`, `Rx_hx`, `Py_hx`, `Sev`, `Dur` -- representing (respectively) the simulated patients age, sex (0 = female, 1 = male), MDD history (0 = no previous episode, 1 = previous episodes), previous treatment with antidepressant medication (0 = no, 1 = yes), previous psychotherapy treatment (0 = no, 1 = yes), this MDD episode's severity (simulated MADRS score) and the current episode's duration in weeks.
* the target / output variable is `Offer_Rx` corresponding to 0 (do not offer medication) and 1 (offer medication)

The simulated data has a flag `train` = 0 (use for testing set) or = 1 (use as training data).

# 2 : Data
To explore/inspect the outputs of the model training / testing (without running the scriptes above), the following files are relevant:

* `big-net-weights-net-params-1.csv` = 7 x 32 matrix of weights connecting the input layer (7 nodes) to the hidden layer (32 nodes) presented in the TIFU paper's figures.
*  `big-net-weights-net-params-2.csv` = 32 x 1 weights from the input layer bias node to the hidden layer nodes
*  `big-net-weights-net-params-3.csv` = 1 x 32 matrix of weights connecting the hidden layer to the output node
*  `big-net-weights-net-params-4.csv` = 1 x 1 (scalar) weight from the hidden layer bias node to the output node

Further, to visualise the activations (outputs of the hidden layer), the file `test-acts-big-model.csv` contains one row (for each of the 1116 test-set patients) with:

* columns 1--7 = the input features (for convenience, replicates the same content in `population-sample.csv`)
* columns 8--39 = activations of the hidden layer (i.e. output of the ReLU hidden layer nodes)
* column 40 = activation (output) of the final single (sigmoid) output node (i.e. the probability of being offered antidepressant medication)
* column 41 = the training target value (i.e. the ground truth, to compare with column 40)







