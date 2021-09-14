# Mapintel

The project aims to explore new solutions in the area of text mining, more specifically the idea is to explore new vectorization techniques with unsupervised neural networks and produce an analytic visual environment to explore and access the text documents.

## Usage

Some components of the project require the existence of a **.env file** under the root folder of the project. This file holds all the private variables necessary for executing parts of the code. Because of its private nature, the file isn't versioned and needs to be created by the user. All variables should be stored as key-value pairs e.g. *VARIABLE_NAME = "variable_value"*. Below is a list of the existing variables and how each affects the project:

- NEWSAPIKEY: Holds the key for using the API to obtain the updated news-articles. A News API key can be obtained by [creating an account](https://newsapi.org/register) with News API. 

You can either run the different project components locally or in containers using the `docker-compose` tool. We advise to run them with docker as it provides a convenient and straightforward experience, allowing for easy reproduction of results. The project has a default option to use of a CUDA-enabled GPU to improve its performance but we also made the project compatible without this resource. Please follow the options bellow:

To launch the Mapintel UI application with a CUDA-enabled GPU, run the following command from the root folder of the Mapintel repository:
```
docker-compose --profile api --profile ui up
```

To launch the Mapintel UI application with CPU only, run the following command from the root folder of the Mapintel repository:
```
docker-compose --profile api-cpu --profile ui-cpu up --build
```

To launch the Experiments container, run the following command from the root folder of the Mapintel repository:
```
docker-compose --profile experiments up
```

If you intend to run the project locally, then you will need to ensure every system and python dependency is satisfied. The requirements.txt file in the root folder contains all the python dependencies, while the system dependencies are scattered across the Dockerfiles in the same folders. Local reproducibility of results is something we intend to improve in the future and contribution in this area is much appreciated.

Further usage information is present in the README files inside the *api*, *experiments* and *ui* folders.

## Project Organization

    ├── api                         <- API based on FastAPI that connects the database with the rest of the application
    |   |
    │   ├── controller              <- Defines the FastAPI endpoints
    │   └── custom_components       <- Custom classes and functions
    |
    ├── data
    |   |
    │   ├── experiments             <- Experiments' data
    │   └── backups                 <- Backups of NewsAPI documents
    |
    ├── experiments                 <- Performs experiments using data from the Open Distro for Elasticsearch instance
    |   |
    │   └── notebooks               <- Jupyter notebooks: each with an experimental purpose described in the first cell
    |
    ├── outputs
    |   |
    │   ├── figures                 <- Figures
    |   ├── experiments             <- Experiments' outputs
    │   └── saved_models            <- Trained and serialized models
    │
    ├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    |
    ├── ui                          <- UI based on Streamlit that allows interactive semantic searching and exploration of a large collection of news articles
    |   |
    │   ├── ui_components           <- Defines UI related functions
    │   └── vis_components          <- Defines Visualization functions
    │
    ├── .env                        <- Stores your secrets and config variables
    ├── docker-compose.yml
    ├── LICENSE
    ├── README.md
    └── requirements.txt            <- The requirements file for locally reproducing the analysis environment
