.PHONY: all data features evaluation clean lint aws_set_lambdavars aws_set_lambdafun create_environment requirements test_environment

# TODO: UPDATE MAKEFILE OR DELETE IF NOT NEEDED ANYMORE (CURRENTLY NOT WORKING)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL:=/bin/bash
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mapintel
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build the project
all: data features evaluation

## Make Dataset
data: data/interim/newsapi_docs.csv data/processed/newsapi_docs.csv outputs/saved_models/CorpusPreprocess.joblib

## Make Embeddings
features: outputs/saved_models/CountVectorizer.joblib outputs/saved_models/TfidfVectorizer.joblib outputs/saved_models/doc2vec*

## Make Embeddings Evaluation
evaluation: outputs/embedding_predictive_scores.csv

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Download the necessary pre-trained embeddings
pretrain_embeddings: data/external/GoogleNews-vectors-negative300.bin data/external/glove.840B.300d data/external/crawl-300d-2M.vec

## Download SentEval's transfer tasks dataset
transfer_tasks: src/senteval
	cd src/senteval/data/downstream && ./get_transfer_data.bash

# ## Upload Data to S3
# sync_data_to_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync data/ s3://$(BUCKET)/data/
# else
# 	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
# endif

# ## Download Data from S3
# sync_data_from_s3:
# ifeq (default,$(PROFILE))
# 	aws s3 sync s3://$(BUCKET)/data/ data/
# else
# 	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
# endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.7 --no-default-packages
else
	conda create --name $(PROJECT_NAME) python=2.7 --no-default-packages
endif
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Install Python Dependencies
requirements: test_environment
ifeq (True,$(HAS_CONDA))
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	# $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	conda env update --name $(PROJECT_NAME) --file environment.yml
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

.env:
	@echo ">>> .env file wasn't detected at root directory. Make sure to include there the environment variables specified in README"
	@false

data/interim/newsapi_docs.csv: src/data/make_dataset_interim.py .env
	$(PYTHON_INTERPRETER) src/data/make_dataset_interim.py

data/processed/newsapi_docs.csv outputs/saved_models/CorpusPreprocess.joblib: src/data/make_dataset_processed.py data/interim/newsapi_docs.csv
	$(PYTHON_INTERPRETER) src/data/make_dataset_processed.py

outputs/saved_models/doc2vec*: src/features/doc2vec.py data/processed/newsapi_docs.csv outputs/saved_models/CorpusPreprocess.joblib
	$(PYTHON_INTERPRETER) src/features/doc2vec.py

outputs/saved_models/CountVectorizer.joblib outputs/saved_models/TfidfVectorizer.joblib: src/features/vectorizer.py data/processed/newsapi_docs.csv outputs/saved_models/CorpusPreprocess.joblib
	$(PYTHON_INTERPRETER) src/features/vectorizer.py

# Double-colon rules provide a mechanism for cases in which the method used to update a target differs depending on which prerequisite files caused the update
outputs/embedding_predictive_scores.csv:: src/features/vectorizer_eval.py outputs/saved_models/CountVectorizer.joblib outputs/saved_models/TfidfVectorizer.joblib
	$(PYTHON_INTERPRETER) src/features/vectorizer_eval.py

# Double-colon rule
outputs/embedding_predictive_scores.csv:: src/features/doc2vec_eval.py outputs/saved_models/doc2vec*
	$(PYTHON_INTERPRETER) src/features/doc2vec_eval.py

data/external/GoogleNews-vectors-negative300.bin:
	@echo "--- Downloading Word2vec embeddings ---"
	wget -cP data/external/ https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz && gzip -d data/external/GoogleNews-vectors-negative300.bin.gz

data/external/glove.840B.300d:
	@echo "--- Downloading GloVe embeddings ---"
	curl -Lo data/external/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip && unzip data/external/glove.840B.300d.zip -d data/external && rm -f data/external/glove.840B.300d.zip

data/external/crawl-300d-2M.vec:
	@echo "--- Downloading FastText embeddings ---"
	curl -Lo data/external/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip && unzip data/external/crawl-300d-2M.vec.zip -d data/external && rm -f data/external/crawl-300d-2M.vec.zip

#################################################################################
# Self Documenting Commands                                                     #models
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
