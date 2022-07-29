from ..datastructures.configs.modelfactoryconfig import ModelFactoryConfig
from ..pipeline.ConfigParser import ConfigParser
from ..utils.dynamic_type_loader import load_type_dynamically_from_fqn


class ModelFactory:
    """
    Factory that dynamically initialises a specified model with the given parameters.
    Model Type and Modelparameter's Type must be specified in the ModelFactoryConfig.
    """

    def __init__(self, model_factory_config: ModelFactoryConfig) -> None:
        self.config = model_factory_config

    def create_instance(self):
        """
        creates an instance of the model specified in the config.
        """

        model_class = load_type_dynamically_from_fqn(self.config.model_type)
        model_para_class = load_type_dynamically_from_fqn(self.config.params_type)

        parser = ConfigParser()
        model_params = parser.parse_config_into_typed_object(self.config.params, model_para_class)

        return model_class(model_params)
