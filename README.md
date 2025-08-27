# SmokingSomaticEvolutionModel

<!-- This is the public repository for the paper "Recovery of human upper airway epithelium after smoking cessation is driven by a slow-cycling stem cell population and immune surveillance". -->

In this repository, first-principles modelling built on the model from [Teixeira et al, 2013](https://doi.org/10.7554/eLife.00966) is used to attempt to explain two surprising results from [Yoshida et al, 2020](https://doi.org/10.1038/s41586-020-1961-1). The reasons for the results are unclear, so certain hypotheses have been conceived. The results, and some possible respective hypotheses, are as follows:

* a certain proportion of basal cells in smokers and ex-smokers have near-normal mutational burden, forming a roughly bimodal distribution
    * a quiescent population of cells, protected from mutation and able to repopulate the airway
    * alternatively, some cells could be inherently resistant to tobacco-induced mutagenesis
* this subpopulation of near-normal cells seems to increase in size after cessation of smoking
    * smoke could give driver mutations additional selectional advantage
    * immmune surveillance could remove mutated cells, but be hampered by the immunosuppressance of smoke

The aim of this project is to model each hypothesis modularly and use the clones data from two cohorts (published by Yoshida et al., Nature 2020 and Huang et al., Nature Genetics 2022) to assess the veracity of each combination of hypotheses.

## Quick start

To run a basic simulation, ensure a Julia installation and run `julia --project=ClonesModelling/FastSimulation ClonesModelling/FastSimulation/run_simulation_abc.jl`. Use the `--help` argument to see available parameters. This will produce output to stdout, in a format to be ingested in `ClonesModelling/simulation/output.py`, which can alternatively be stored by appending `> simulation_output.txt` to the above command.

## Repository contents

The central simulations are written using Julia 1.9.0 within the `ClonesModelling/FastSimulation` directory. All other code is written using Python 3.10.7 and separated into submodules by subdirectory of `ClonesModelling`:

* `simulation` contains code to call the Julia implementation within a subprocess and parse its outputs.
* `parameters` creates a class structure to encode the hierarchical structure of the inference: each hypothesis requires certain parameters, and collections of hypotheses form a hypothetical paradigm within which a simulation can be run. Prior distributions of parameters are defined and handled here.
* `visualisation` covers in-built data visualisation for monitoring of simulations and inference.
* `hpc` contains helper files for running simulations and inference procedures on the UCL CS cluster, using the Sun Grid Engine scheduler.
* `data` defines classes and helper functions for handling the modalities of data used in this project, all derived from single-cell whole genome sequencing data of healthy human upper airway basal stem cells:
    * Mutational burden distributions across the basal cells sequenced from a single sample
    * Gaussian mixture model embeddings of the above mutational burden distributions
    * Inferred phylogenetic trees denoted the relatedness of sequenced basal cells.
* `id_test` generates a large collection of simulated cohorts within each hypothetical paradigm, with parameter values randomly drawn from prior distributions, and for each of an array of biologically motivated distance metrics calculates the distance between each pair of simulated cohorts.
    * The `id_test/classifier` submodule generates an MDS embedding of each distance function's view of the dataset of simulated cohorts, concatenates them and trains machine learning classifiers in order to assess identifiability of the hypothetical paradigm from simulation outputs, and to apply this to the observed data.
    * The `id_test/aggregated_distance` submodule applies a simpler approach, summing all distance functions (after normalisation) and applying a *k*-nearest neighbours classification approach to the same identifiability and observed data classification questions.

In the `notebooks/python_pairs` folder are `jupytext` synchronised Python file equivalents of Jupyter notebooks, for compatibility with git. These are for more bespoke visualisation and bugfixing, and are included as a record. They can be reconstructed to `.ipynb` notebooks for re-use using the Python package `jupytext`, or used in their `.py` form.

The directory `notebooks/mdp_model_bootstrap` contains a simplified simulation structure modelling the size of a single clone with a given fitness, in both the spatial and non-spatial settings.

## Data required for inference

Simulations can be run and outputs visualised without reference to observed data.

The git-ignored directory `ClonesModelling/data/patient_data` contains publicly available data published by [Yoshida et al., Nature 2020](https://doi.org/10.1038/s41586-020-1961-1) and [Huang et al., Nature Genetics 2022](https://doi.org/10.1038/s41588-022-01035-w). These can be accessed [here](https://data.mendeley.com/datasets/b53h2kwpyy/2) and [here](http://vijglab.einsteinmed.org/static/vcf/lung_Huang.et.al.Naturegenetics.tar.gz) (note that the second link has stopped working since publication of that work).
