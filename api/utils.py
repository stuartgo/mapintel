import logging
import os
from pathlib import Path

from api.custom_components.custom_pipeline import CustomPipeline

logger = logging.getLogger(__name__)


PIPELINE_YAML_PATH = os.getenv("PIPELINE_YAML_PATH", "api/pipelines.yaml")
QUERY_PIPELINE_NAME = os.getenv("QUERY_PIPELINE_NAME", "query")
INDEXING_PIPELINE_NAME = os.getenv("INDEXING_PIPELINE_NAME", "indexing")


def load_pipeline_from_yaml(type: str):
    if type == "query":
        pipeline = CustomPipeline.load_from_yaml(
            Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME
        )
        logger.info(f"Loaded query pipeline with nodes: {pipeline.graph.nodes.keys()}")
    elif type == "indexing":
        pipeline = CustomPipeline.load_from_yaml(
            Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_PIPELINE_NAME
        )
        logger.info(
            f"Loaded indexing pipeline with nodes: {pipeline.graph.nodes.keys()}"
        )
    else:
        raise ValueError("type argument should be either 'query' or 'indexing'.")
    return pipeline


def load_component_from_yaml(name: str):
    component = CustomPipeline.get_node_from_yaml(Path(PIPELINE_YAML_PATH), name)
    logger.info(f"Loaded {name} component.")
    return component


def load_document_store():
    document_store = load_component_from_yaml("DocumentStore")
    return document_store


def load_retriever():
    retriever = load_component_from_yaml("Retriever")
    return retriever
