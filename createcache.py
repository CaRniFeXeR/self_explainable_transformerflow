from cfgparser.json_config_parser import JSONConfigParser
from pathlib import Path

from src.pipeline.CacheCreation import CacheCreator


configParser = JSONConfigParser()


config = configParser.parse_config_from_file(Path(".\config\\cachecreation\\cache_creation_vie14_bln.json"))

print(config)

cacheCreator = CacheCreator(config)
cacheCreator.create_data_cache()
