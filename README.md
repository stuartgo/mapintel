Mapintel
==============================

The project aims to explore new solutions in the area of text mining, more specifically the idea is to explore new vectorization techniques with unsupervised neural networks and produce an analytic visual environment to explore and access the text documents.

Usage
------------

The project requires the existence of a **.env file** under the root of the project. This file holds all the private variables necessary for the execution of the code, such as NEWSAPIKEY, MONGOUSERNAME, MONGOPASSWORD and MONGODB. Each variable should be stored as a key-value pair e.g. *NEWSAPIKEY = "value_here"*. This file isn't versioned and therefore should be created by each user.

A conda environment should also be created to execute the code. A requirements.txt and a environment.yml file can be found on the project's root for this purpose. The Makefile can be used to create the environment using conda `make create_environment`. **The conda environment should be active before running any command related with the project (use `conda activate mapintel` for this purpose)**. After activating the environment, install the necessary dependencies with `make requirements`.

**The Makefile can be used to run the project:**

- `make all` will ensure the .env file exists, create all necessary directories, the dataset files, the model dumps and the evaluation scores file. Make sure the conda environment is active before running the command.

**Alternatively, the scripts below can be executed in the following order, given the .env and the necessary directories exist:**
1. *src/data/make_dataset_interim.py*

    Builds the cleaned (intermediate) csv file with documents from mongodb
2. *src/data/make_dataset_processed.py*

    Builds the processed (model ready) csv file with preprocessed documents
3. *src/features/vectorizer.py*

    Fits a BOW and a TF-IDF model to the preprocessed data and saves the fitted models for posterior use at models/saved_models
4. *src/features/doc2vec.py*

    Fits a set of Doc2vec models to the preprocessed data and saves the fitted models for posterior use at models/saved_models
5. *src/features/vectorizer_eval.py*

    Evaluates the BOW and TF-IDF embeddings and outputs the category predictive scores
6. *src/features/doc2vec_eval.py*

    Evaluates the Doc2vec embeddings and outputs the category predictive scores

MongoDB access
------------
Accessing the MongoDB database requires a username and password. To access the database with pymongo you can use the following expression: `pymongo.MongoClient(f"mongodb+srv://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb.e2na5.mongodb.net/{MONGODB}?retryWrites=true&w=majority")`, where MONGOUSERNAME, MONGOPASSWORD and MONGODB are variables with the username, password and database values respectively. These variables can be placed in the **.env file** and then imported as environment variables.

AWS Lambda setup
------------
To set up the AWS Lambda service you can use `make aws_set_lambdavars` to define the environment variables and `make aws_set_lambdafun` to deploy the lambda function and all necessary dependencies.
The scheduler trigger and the eventual destinations need to be set up manually. 

Project Organization
------------
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── .env               <- Stores your secrets and config variables
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── ocr_outputs
    │   └── saved_models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── mongodb_insertion  <- AWS lambda function package and related scripts. Used by AWS lambda
    │                         for regular insertion of NewsAPI articles into MongoDB database.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset_interim.py
    │   │   ├── make_dataset_processed.py
    │   │   └── text_preprocessing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── doc2vec.py
    │   │   ├── doc2vec_eval.py
    │   │   ├── vectorizer.py
    │   │   ├── vectorizer_eval.py
    │   │   └── embedding_eval.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── senteval_evaluation.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── embedding_space.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
