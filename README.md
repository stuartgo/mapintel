# Mapintel

The project aims to explore new solutions in the area of text mining, more specifically the idea is to explore new vectorization techniques with unsupervised neural networks and produce an analytic visual environment to explore and access the text documents.

## Usage

Some components of the project require the existence of a **.env file** under the root folder of the project. This file holds all the private variables necessary for executing parts of the code. Because of its private nature, the file isn't versioned and needs to be created by the user. All variables should be stored as key-value pairs e.g. *VARIABLE_NAME = "variable_value"*. Below is a list of the existing variables and how each affects the project:

- NEWSAPIKEY: Holds the key for using the API to obtain the updated news-articles. A News API key can be obtained by [creating an account](https://newsapi.org/register) with News API. 

You can either run the different project components locally or in containers using the `docker-compose` tool. We advise to run them with docker as it provides a convenient and straightforward experience, allowing for easy reproduction of results. To launch the Mapintel UI application, run the following command from the root folder of the Mapintel repository:
```
docker-compose --profile api --profile ui up
```
To launch the Experiments container, run the following command from the root folder of the Mapintel repository:
```
docker-compose --profile experiments up
```

If you intend to run the project locally, then you will need to ensure every system and python dependency is satisfied. The requirements.txt file in the root folder contains all the python dependencies, while the system dependencies are scattered across the Dockerfiles in the same folders. Local reproducibility of results is something we intend to improve in the future and contribution in this area is much appreciated.

The project makes use of a CUDA-enabled GPU to improve its performance. Making the project compatible without this resource is something we intend to provide in the future. Any contributions in this aspect are appreciated.

Further usage information is present in the README files inside the *api*, *experiments* and *ui* folders.

## Project Organization

    ├── api                <- API based on FastAPI that connects the database with the rest of the application
    |
    ├── data               <- Stores any data produced and used by the project
    |   |
    |   ├── backups        <- Backups of NewsAPI documents
    │   ├── external       <- Data from third party sources
    │   ├── interim        <- Intermediate data that has been transformed
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    |
    ├── experiments        <- Performs experiments using data from the Open Distro for Elasticsearch instance
    |   |
    │   ├── notebooks      <- Jupyter notebooks: each with an experimental purpose described in the first cell
    |   ├── src            <- Source code for use in experiments
    |   └── setup.py       <- Makes src pip installable (pip install -e .) so src can be imported
    |
    ├── outputs            <- Trained and serialized models, model predictions, model summaries and other outputs
    |   |
    │   ├── figures        <- Figures resulting from experiments
    │   ├── ocr            <- Optical Character Recognition ouptuts
    |   ├── saved_embeddings    <- Saved document embeddings 
    │   └── saved_models   <- Saved machine learning models
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   |
    │   └── figures        <- Generated graphics and figures to be used in reporting
    |
    ├── ui                 <- UI based on Streamlit that allows interactive semantic searching and exploration of a large collection of news articles
    │
    ├── .env               <- Stores your secrets and config variables
    ├── docker-compose.yml <- The docker-compose file for reproducing the analysis environment using containers
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    └── requirements.txt   <- The requirements file for locally reproducing the analysis environment
