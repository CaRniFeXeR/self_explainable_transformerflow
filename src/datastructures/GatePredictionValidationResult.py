from dataclasses import dataclass
from .Gate import Gate


@dataclass
class GatePredictionValidationResult:
    sample_name : str
    gt_gate: Gate
    f1_score_ground_truth_events: float
    f1_score_predicted_events_sequential : float
    f1_score_ground_truth_labels : float
    n_events_gt : int
    n_events_pd : int
    n_events_label_gt : int
    mrd_true : float = -1.0
    mrd_predicted : float = -1.0
