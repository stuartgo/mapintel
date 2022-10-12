# MapIntel Research

This is the Research repository of the MapIntel project. This repository contains the results.py file which is responsible for the evaluation of the MapIntel system, and the main.tex file of the produced paper. 

In the *notebooks folder* you can find various jupyter notebooks, each with an experimental purpose described in the first cell of each file. Most of these notebooks depend on an Elasticsearch search instance running locally and connected to port 9200, which contains the documents used by the MapIntel system.

The MapIntel project repository, containing its codebase and instructions on how to use it, can be found at [github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel_project](https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel_project).

## Usage

Execute in this folder:
```
pip install -r requirements.txt
```
, to install the necessary dependencies (it is advisable to do it in an isolated environment).

Then run:
```
python results.py
```
, to execute the experiments. Mind that this is a very long process that takes several days to finish (to be more specific, it took a week to run in a ...)

To track and analyze the experiments' results we use MLflow. Launch the MLflow dashboard with:
```
mlflow ui --backend-store-uri ./artifacts/mlruns/
```
For a more in-depth analysis of the results you can consult the notebooks/results_exploration.ipynb file.

## Optional

To run the jupyter notebooks inside the notebooks folder, there is another requirements.txt file that should be installed.

**Requirements**: Some notebooks expect a running Open Distro for Elasticsearch instance with the MapIntel system documents at `https://localhost:9200`. 

## Project Organization

    ├── notebooks                   <- Jupyter notebooks: each with an experimental purpose described in the first cell.
    │
    ├── artifacts                   <- Artifacts (data, outputs, results, models, etc)
    │
    ├── docs
    │   │
    │   ├── assets                  <- Figures for the paper
    │   ├── main.tex                <- LaTeX document of the MapIntel paper
    │   └── bibliography.bib        <- Bibliography
    │
    ├── results.py                  <- Experiments script. Execution should produce experiments results.    
    ├── utils.py                    <- Auxiliary functions and classes to be used by results.py    
    ├── README.md
    └── requirements.txt            <- The requirements file for reproducing the results
