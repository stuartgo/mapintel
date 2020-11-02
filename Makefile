.PHONY: all data features evaluation clean lint requirements sync_data_to_s3 sync_data_from_s3 aws_set_accesskeys aws_set_lambdavars aws_update_lambda

#################################################################################
# GLOBALS                                                                       #
#################################################################################

# SHELL:=/bin/bash
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = mapintel
PYTHON_INTERPRETER = python3
FOLDERS = data/external data/interim data/processed data/raw models/saved_models

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Build the project
all: directories data features evaluation

## Build Necessary Directory Structure
directories: $(FOLDERS)

## Make Dataset
data: data/interim/newsapi_docs.csv data/processed/newsapi_docs.csv models/saved_models/CorpusPreprocess.joblib

## Make Embeddings
features: models/saved_models/CountVectorizer.joblib models/saved_models/TfidfVectorizer.joblib models/saved_models/doc2vec*

## Make Embeddings Evaluation
evaluation: models/embedding_predictive_scores.csv

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set AWS lambda environment variables
aws_set_lambdavars: .env mongodb_insertion/get_lambda_vars.py
ifeq (default,$(PROFILE))
	cd mongodb_insertion && \
	aws lambda update-function-configuration --function-name newsapi_mongodb --environment $(shell $(PYTHON_INTERPRETER) mongodb_insertion/get_lambda_vars.py)
else
	cd mongodb_insertion && \
	aws lambda update-function-configuration --function-name newsapi_mongodb --environment $(shell $(PYTHON_INTERPRETER) mongodb_insertion/get_lambda_vars.py) --profile $(PROFILE)
endif

## Set AWS lambda function
aws_set_lambdafun: mongodb_insertion/lambda_function.py
	cd mongodb_insertion && \
	pip install --target ./python_package newsapi-python==0.2.6 pymongo==3.11.0 dnspython==1.16.0
	cd mongodb_insertion/python_package && \
	zip -r9 ../newsapi_mongodb.zip .
	cd mongodb_insertion && \
	zip -g newsapi_mongodb.zip lambda_function.py
ifeq (default,$(PROFILE))
	cd mongodb_insertion && \
	aws lambda update-function-code --function-name newsapi_mongodb --zip-file fileb://newsapi_mongodb.zip
else
	cd mongodb_insertion && \
	aws lambda update-function-code --function-name newsapi_mongodb --zip-file fileb://newsapi_mongodb.zip --profile $(PROFILE)
endif
	cd mongodb_insertion && \
	rm -rf python_package; \
	rm -rf newsapi_mongodb.zip

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3.7
else
	conda create --name $(PROJECT_NAME) python=2.7
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
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Creates folders (used with order-only-prerequisites)
$(FOLDERS):
	mkdir -p $@

.env: 
	@echo ">>> .env file wasn't detected at root directory. Make sure to include there the environment variables specified in README"
	@false

data/interim/newsapi_docs.csv: src/data/make_dataset_interim.py .env | data/interim
	$(PYTHON_INTERPRETER) src/data/make_dataset_interim.py

# This rule can have some problems when running a parallel make
data/processed/newsapi_docs.csv models/saved_models/CorpusPreprocess.joblib: src/data/make_dataset_processed.py data/interim/newsapi_docs.csv | data/processed models/saved_models
	$(PYTHON_INTERPRETER) src/data/make_dataset_processed.py

# This rule can have some problems when running a parallel make
models/saved_models/doc2vec*: src/features/doc2vec.py data/processed/newsapi_docs.csv models/saved_models/CorpusPreprocess.joblib | models/saved_models
	$(PYTHON_INTERPRETER) src/features/doc2vec.py

# This rule can have some problems when running a parallel make
models/saved_models/CountVectorizer.joblib models/saved_models/TfidfVectorizer.joblib: src/features/vectorizer.py data/processed/newsapi_docs.csv models/saved_models/CorpusPreprocess.joblib | models/saved_models
	$(PYTHON_INTERPRETER) src/features/vectorizer.py

# Double-colon rules provide a mechanism for cases in which the method used to update a target differs depending on which prerequisite files caused the update
models/embedding_predictive_scores.csv:: src/features/vectorizer_eval.py models/saved_models/CountVectorizer.joblib models/saved_models/TfidfVectorizer.joblib
	$(PYTHON_INTERPRETER) src/features/vectorizer_eval.py

# Double-colo rule
models/embedding_predictive_scores.csv:: src/features/doc2vec_eval.py models/saved_models/doc2vec*
	$(PYTHON_INTERPRETER) src/features/doc2vec_eval.py

#################################################################################
# Self Documenting Commands                                                     #
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
