from dataclasses import dataclass

@dataclass
class ArtificalGateDefinition:
    name : str
    original_name : str
    parent_name : str
    x_marker : str
    y_marker : str