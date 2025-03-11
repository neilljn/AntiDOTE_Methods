# AntiDOTE Methods

Contains code and data used for the paper 'Non-centering for discrete-valued state transition models: an application to ESBL-producing _E. coli_ transmission in Malawi', currently a preprint on arXiv: _add link here_.

The _function_scripts_ folder contains a function to simulate from the transmission model, and functions to perform Bayesian inference using three different methods for data-augmentation: the novel Rippler algorithm, a block update algorithm (move/add/delete), and the individual forward-filtering backwards-sampling algorithm. The _experiments_ folder contains simulation studies using the each of the three methods mentioned above, as well as applying the Rippler algorithm to the AntiDOTE dataset.

The Python code requires the following packages: numpy, jax, pandas, matplotlib, time, sys.
