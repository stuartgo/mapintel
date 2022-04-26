from pathlib import Path

import yaml
from haystack import BaseComponent, Pipeline, logger

from api.custom_components.custom_nodes import *


class CustomPipeline(Pipeline):
    @classmethod
    def _load_or_get_component(cls, name: str, definitions: dict, components: dict):
        """
        Load a component from the definition or return if component object already present in `components` dict.
        :param name: name of the component to load or get.
        :param definitions: dict containing definitions of all components retrieved from the YAML.
        :param components: dict containing component objects.
        """
        if name in components.keys():  # check if component is already loaded.
            return components[name]
        component_params = definitions[name].get("params", {})
        component_type = definitions[name]["type"]

        try:
            logger.debug(
                f"Loading component `{name}` of type `{definitions[name]['type']}`"
            )

            for key, value in component_params.items():
                # Component params can reference to other components. For instance, a Retriever can reference a
                # DocumentStore defined in the YAML. All references should be recursively resolved.
                if (
                    isinstance(value, str) and value in definitions.keys()
                ):  # check if the param value is a reference to another component.
                    if (
                        value not in components.keys()
                    ):  # check if the referenced component is already loaded.
                        cls._load_or_get_component(
                            name=value, definitions=definitions, components=components
                        )
                    component_params[key] = components[
                        value
                    ]  # substitute reference (string) with the component object.

            instance = BaseComponent.load_from_args(
                component_type=component_type, **component_params
            )
            components[name] = instance
        except Exception as e:
            try:
                # Check if it is one of the defined custom nodes
                instance = globals()[component_type](**component_params)
                components[name] = instance
            except KeyError:
                raise Exception(f"Failed loading pipeline component '{name}': {e}")
        return instance

    @classmethod
    def get_node_from_yaml(
        cls, path: Path, node_name: str, overwrite_with_env_variables: bool = True
    ):
        with open(path, "r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        definitions = {}  # definitions of each component from the YAML.
        for definition in data["components"]:
            if overwrite_with_env_variables:
                cls._overwrite_with_env_variables(definition)
            name = definition.pop("name")
            definitions[name] = definition

        # Need to load every component because they can depend on each other
        components: dict = {}  # instances of component objects.
        for name in definitions.keys():
            cls._load_or_get_component(
                name=name, definitions=definitions, components=components
            )
        try:
            return components[node_name]
        except:
            raise KeyError(
                "Node name requested isn't defined in the pipeline yaml file."
            )
