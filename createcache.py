from src.pipeline.ConfigParser import ConfigParser
from pathlib import Path

from src.pipeline.CacheCreation import CacheCreator


configParser = ConfigParser()


config = configParser.parse_config_from_file(Path(".\config\\cachecreation\\cache_creation_vie14_bln.json"))

print(config)

cacheCreator = CacheCreator(config)
cacheCreator.create_data_cache()
