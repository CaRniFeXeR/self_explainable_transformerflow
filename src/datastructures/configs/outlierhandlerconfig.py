from dataclasses import dataclass

@dataclass
class OutlierHandlerConfig:
    alpha : float = 0.05
    n_events_threshold : int = 200