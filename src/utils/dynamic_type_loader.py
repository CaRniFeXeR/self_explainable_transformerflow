from typing import Type


def load_type_dynamically_from_fqn(type_fqn: str) -> Type:
    path = type_fqn.split(".")
    module_path = ".".join(path[0:-1])
    type_name = path[-1]

    return load_type_dynamically(module_path, type_name)


def load_type_dynamically(module_path: str, type_name: str) -> Type:

    try:
        module = __import__(module_path, fromlist=[type_name])
        current_type = getattr(module, type_name)
    except Exception as ex:
        raise TypeError(f"exception while dynamically loading type: '{module_path}.{type_name}'") from ex

    return current_type
