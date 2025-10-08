from dataclasses import dataclass


@dataclass
class MetricConfig:
    """
    Base class for metric configuration.
    All metric configuration classes should inherit from this class.
    """

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create an instance of the configuration class from a dictionary.

        :param config_dict: A dictionary containing the configuration parameters.

        :return: An instance of the configuration class.
        """
        return cls(**config_dict)
