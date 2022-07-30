from cfgparser.json_config_parser import JSONConfigParser
from pathlib import Path

from src.pipeline.Predicition import Prediction


configParser = JSONConfigParser()


config = configParser.parse_config_from_file(Path(".\config\\test\\test_bln.json"))

print(config)

prediction = Prediction(config)

res_per_gate = prediction.prediction_on_data()

