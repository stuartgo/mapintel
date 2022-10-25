.DEFAULT_GOAL := help
SHELL := bash

DUTY = $(shell [ -n "${VIRTUAL_ENV}" ] || echo pdm run) duty

args = $(foreach a,$($(subst -,_,$1)_args),$(if $(value $a),$a="$($a)"))
check_quality_args = files
docs_serve_args = host port
release_args = version
test_args = match

BASIC_DUTIES = \
	changelog \
	check-dependencies \
	clean \
	coverage \
	docs \
	docs-deploy \
	docs-regen \
	docs-serve \
	format \
	release

.PHONY: help
help:
	@$(DUTY) --list

.PHONY: lock
lock:
	@pdm lock

.PHONY: $(BASIC_DUTIES)
$(BASIC_DUTIES):
	@$(DUTY) $@ $(call args,$@)

.PHONY: check-quality
check-quality:
	@nox --sessions check_quality

.PHONY: check-types
check-docs:
	@nox --sessions check_docs

.PHONY: check-types
check-quality:
	@nox --sessions check_types

.PHONY: check
check:
	@nox --sessions check
	@$(DUTY) check-dependencies

.PHONY: test
test:
	@nox --sessions test
