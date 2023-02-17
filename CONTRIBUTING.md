# Contributing

Contributions are welcome, and they are greatly appreciated.

## Environment setup

This project uses [PDM](https://github.com/pdm-project/pdm) to manage the various dependencies. To setup the development
environment you can follow the next steps:

- Install [PDM](https://github.com/pdm-project/pdm).

- Fork and clone the repository.

- `pdm install -dG :all` from the root of the project to install the main and development dependencies.

## Tasks

This project uses [nox](https://nox.thea.codes/en/stable/) to run tasks. Please check the `noxfile.py` at the root of the
project for more details. You can run any of the following commands and subcommands that corresponds to a particular task:

- Documentation:

  - `pdm docs serve` (or just `pdm docs`) to serve the documentation.

  - `pdm docs build` to build locally the documentation.

  - `pdm docs deploy` to build serve the documentation.

- Formatting:

  - `pdm formatting all` (or just `pdm formatting`) to format both the code and docstrings.

  - `pdm formatting code` to format only the code.

  - `pdm formatting docstrings` to format only the docstrings.

- Checks:

  - `pdm checks quality` to check code quality.

  - `pdm checks types` to check type annotations.

  - `pdm checks dependencies` to check for vulnerabilities in dependencies.

- Changelog:

  - `pdm changelog add` to add a news fragment to the changelog.

  - `pdm changelog build` to build the changelog.

- Release:

  - `pdm release` to release a new Python package with an updated version.

## Development

The next steps should be followed during development:

- `git checkout -b new-branch-name` to create a new branch and then modify the code.
- `pdm format-code` to auto-format the code.
- `pdm check-quality` to check code quality and fix any warnings.
- `pdm check-types` to check type annotations and fix any warnings.
- `pdm test` to run the tests.
- `pdm serve-docs` if you updated the documentation or the project dependencies to check that everything looks good.

## Pull Request guidelines

Link to any related issue in the Pull Request message. We also recommend using fixups:

```bash
git commit --fixup=SHA
```

Once all the changes are approved, you can squash your commits:

```bash
git rebase -i --autosquash master
```

And force-push:

```bash
git push -f
```
