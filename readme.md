# Cost-Sensitive Trees for Interpretable Reinforcement Learning

This repository contains the source code accompanying the paper [Cost-Sensitive Trees for Interpretable Reinforcement Learning](https://dl.acm.org/doi/10.1145/3632410.3632443).

## Installation

The codebase uses slightly modified versions of scikit-learn and stable-baselines3 libraries. 
Building scikit-learn from source requires other dependencies. Please follow instructions from [here](https://scikit-learn.org/dev/developers/advanced_installation.html#building-from-source) to setup before installing these libraries.

Once this is setup, the libraries and all other dependencies can be installed using:

`pip install -r requirements.txt`

The code was written and tested on Python 3.7.9.

## BibTeX Citation

If you use CS-VIPER or CS-MoET in a scientific publication, we would appreciate using the following citation:

    @inproceedings{cost-sensitive-trees-rl,
	    author = {Nishtala, Siddharth and Ravindran, Balaraman},
	    title = {Cost-Sensitive Trees for Interpretable Reinforcement Learning},
	    year = {2024},
	    isbn = {9798400716348},
	    publisher = {Association for Computing Machinery},
	    address = {New York, NY, USA},
	    url = {https://doi.org/10.1145/3632410.3632443},
	    doi = {10.1145/3632410.3632443},
	    booktitle = {Proceedings of the 7th Joint International Conference on Data Science \& Management of Data (11th ACM IKDD CODS and 29th COMAD)},
	    pages = {91–99},
	    numpages = {9},
	    keywords = {interpretability, reinforcement learning, decision trees},
	    location = {Bangalore, India},
	    series = {CODS-COMAD '24}
    }
