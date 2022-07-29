from src.pipeline.ConfigParser import ConfigParser
from pathlib import Path

from src.pipeline.Predicition import Prediction


configParser = ConfigParser()


config = configParser.parse_config_from_file(Path(".\config\\test\\workshop\\edna_train_vie14_test8.json"))

print(config)

prediction = Prediction(config)

res_per_gate = prediction.prediction_on_data()
# prediction.log_gate_comparision_to_file(res_per_gate)
# prediction.log_gate_comparision_to_file()

