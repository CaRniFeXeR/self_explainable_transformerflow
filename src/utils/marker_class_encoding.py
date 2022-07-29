from typing import List
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class MarkerClassEncoder:

    def __init__(self, possible_markers: List[str]) -> None:

        self.possible_markers = possible_markers
        self.encoder = LabelBinarizer()
        self.encoder.fit(self.possible_markers)

    def string_to_encoding(self, marker_class: str) -> np.ndarray:

        if marker_class not in self.possible_markers:
            raise ValueError(f"unkown '{marker_class}'. known classes: '{self.possible_markers}'")

        return self.encoder.transform([marker_class])[0]

    def encoding_to_string(self, encoded: np.array) -> str:

        return self.encoder.inverse_transform(encoded)[0]

    def prediction_to_string(self, prediction: np.array) -> str:

        max_idx = np.argmax(prediction)
        encoded = np.zeros(len(prediction))
        encoded[max_idx] = 1
        return self.encoding_to_string(np.array([encoded]))
