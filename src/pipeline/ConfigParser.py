from typing import Type

from ..loader.IO.jsonfileloader import JsonFileLoader
from pathlib import Path
from enum import Enum
import inspect

from ..utils.dynamic_type_loader import load_type_dynamically_from_fqn

class ConfigParser:
    """
    Handles parsing of json config into typed dataclass
    """

    def __init__(self, datastructure_module_name: str = "src.datastructures") -> None:
        self.datastructure_module_name = datastructure_module_name

    def parse_config_from_file(self, config_path: Path):
        """
        loads a json config from the specified location and parses it into a typed object based on the type specified in 'type_name'
        """

        if not isinstance(config_path, Path):
            raise TypeError("'config_path' must be a Path")

        if not config_path.exists():
            raise ValueError(f"given config_path '{config_path}' does not exist")

        fileloader = JsonFileLoader(config_path)
        config_dict = fileloader.loadJsonFile()

        return self.parse_config(config_dict)

    def parse_config(self, config_dict: dict):
        """
        parse a config dict into a typed object based in on the type specified in 'type_name'
        """

        if not isinstance(config_dict, dict):
            raise TypeError("'config_dict' must be a dict")

        if "type_name" not in config_dict.keys():
            raise ValueError("'type_name' must be specified")

        current_type = load_type_dynamically_from_fqn(config_dict["type_name"])
        del config_dict["type_name"]  # should not be parsed

        return self.parse_config_into_typed_object(config_dict, current_type)

    def parse_config_into_typed_object(self, config_dict: dict, current_type: Type):
        """
        recursively converts a dict into the given dataclass type
        """

        if not isinstance(config_dict, dict) or not hasattr(current_type, "__dataclass_fields__"):
            return config_dict

        result_dict = {}
        current_fields = current_type.__dataclass_fields__

        for k, v in config_dict.items():

            if k not in current_fields:
                raise TypeError(f"unkown field name '{k}' for type '{current_type.__name__}'")

            field = current_fields[k]

            if v is None:
                result_dict[k] = None

            elif inspect.isclass(field.type) and issubclass(field.type, Enum):  # is Enum
                try:
                    result_dict[k] = field.type[v]
                except Exception as ex:
                    raise ValueError(f"value '{v}' is not valid for enum of type '{field.type}' ") from ex

            elif field.type.__module__.startswith(self.datastructure_module_name):  # complex type from the specificed module
                result_dict[k] = self.parse_config_into_typed_object(v, field.type)
            elif hasattr(field.type, "_name") and field.type._name == "List":

                if not isinstance(v, list):
                    raise TypeError(f"'{k}' must be a list.")

                list_element_type = field.type.__args__[0]

                parsed_list = []
                for el in v:
                    parsed_list.append(self.parse_config_into_typed_object(el, list_element_type))

                result_dict[k] = parsed_list
                
            elif hasattr(field.type, "_name") and field.type._name == "Dict":
                result_dict[k] = dict(v)
            else:
                result_dict[k] = field.type(v)

        return current_type(**result_dict)
