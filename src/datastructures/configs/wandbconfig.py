from dataclasses import dataclass
from typing import List


@dataclass
class WandbConfig:
    entity: str
    prj_name: str
    notes: str
    tags: List[str]
    enabled: bool = True