# Mapintel Experiments

This is the Experiments module of the Mapintel project. This module contains the results.py file which is responsible for the evaluation of the Mapintel system. 

In the *notebooks folder* you can find various jupyter notebooks, each with an experimental purpose described in the first cell of each file.

This module is independent of the UI application and therefore should not be deployed.

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
, to execute the experiments. Mind that this is a very long process that takes several days to finish.

## Optional

To run the jupyter notebooks inside the notebooks folder, there is another requirements.txt file that should be installed.

**Requirements**: Some notebooks expect a running Open Distro for Elasticsearch instance at `https://localhost:9200`. 
