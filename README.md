# MapIntel

[![ci](https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel/workflows/ci/badge.svg)](https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://NOVA-IMS-Innovation-and-Analytics-Lab.github.io/mapintel/)
[![pypi version](https://img.shields.io/pypi/v/mapintel.svg)](https://pypi.org/project/mapintel/)
[![gitpod](https://img.shields.io/badge/gitpod-workspace-blue.svg?style=flat)](https://gitpod.io/#https://github.com/NOVA-IMS-Innovation-and-Analytics-Lab/mapintel)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://gitter.im/mapintel/community)

MapIntel is a system for acquiring intelligence from vast collections of text data by representing
each document as a multidimensional vector that captures its own semantics. The system is designed
to handle complex Natural Language queries and visual exploration of the corpus.

The system searching module uses a retriever and re-ranker engine that first finds the closest
neighbors to the query embedding and then sifts the results through a cross-encoder model that
identifies the most relevant documents. The browsing module also leverages the embeddings by
projecting them onto two dimensions while preserving the multidimensional landscape, resulting in a
map where semantically related documents form topical clusters which we capture using topic
modeling. This map aims at promoting a fast overview of the corpus while allowing a more detailed
exploration and interactive information encountering process.

MapIntel can be used to explore many different types of corpora.

![MapIntel UI screenshot](./artifacts/ui.png)

## Installation

With `pip`:

```bash
pip install mapintel
```

With [`pipx`](https://github.com/pipxproject/pipx):

```bash
python -m pip install --user pipx
pipx install mapintel
```
