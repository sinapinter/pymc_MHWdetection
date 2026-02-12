# Probabilistic Detection of Marine Heatwaves Using PyMC

This repository contains all code and analysis scripts associated with the paper:

**“Making the Most of Little Time: A Probabilistic Approach to Marine Heatwave Detection”**  
Sina Pinter¹², Matthew D. Rayson¹², and Nicole L. Jones¹²  

¹ School of Earth and Oceans, The University of Western Australia, Australia  
² Oceans Institute, The University of Western Australia, Australia  

---

## Overview

This repository provides the full computational workflow used in the paper, including:

- Bayesian temperature modelling using **PyMC**
- Probabilistic detection of marine heatwaves
- Markov Chain Monte Carlo (MCMC) diagnostics (trace plots, corner plots, ACF, ESS)
- Analysis across multiple sites and data record lengths
- Scripts used to generate all figures and tables in the manuscript and Supporting Information

The code is intended to ensure **reproducibility and transparency** of the results presented in the paper.

---

### Requirements

This analysis was developed in **Python 3.10**.  
The following Python packages are required to run the code:

- `numpy`
- `pandas`
- `arviz`
- `pymc`
- `matplotlib`
- `scipy`
- `seaborn`
- `xarray`
- `fastdtw`
- `nutpie`
- `corner`
- `marineHeatWaves` (Freely available on GitHub: Hobday et al. (2016) MHW detection method)

Standard Python libraries (`os`, `sys`, `datetime`, `collections`, `string`, `pickle`, etc.) do not require installation.


### Installation

The safest way to install everything is by creating a **new conda environment**:

```bash
# 1. Create a new environment
conda create -n pymc_env python=3.10 -c conda-forge
conda activate pymc_env
conda config --env --set channel_priority strict

# 2. Install core dependencies
conda install -c conda-forge \
numpy pandas arviz pymc matplotlib scipy seaborn xarray fastdtw \
pyarrow graphviz python-graphviz nutpie corner

# 3. Fix BLAS to avoid warnings and improve performance
conda install -c conda-forge "blas=*=openblas"

