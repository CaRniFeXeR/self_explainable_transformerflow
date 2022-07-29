from dataclasses import dataclass
from typing import List

import torch


@dataclass
class FlowSampleTensorData:
    sample_names: List[str]
    events: torch.FloatTensor
    labels: torch.FloatTensor
    padding_mask: torch.Tensor
    polygons: torch.Tensor

    def concat_data(self, to_concat):

        if not isinstance(to_concat, FlowSampleTensorData):
            raise TypeError("data_to_concat must be of type 'FlowSampleTensorData'")

        self.sample_names = self.sample_names + to_concat.sample_names

        if len(self.events.shape) == 2:
            self.events = torch.cat((self.events.unsqueeze(0), to_concat.events.unsqueeze(0)))
            self.labels = torch.cat((self.labels.unsqueeze(0), to_concat.labels.unsqueeze(0)))
            self.padding_mask = torch.cat((self.padding_mask.unsqueeze(0), to_concat.padding_mask.unsqueeze(0)))
            self.polygons = torch.cat((self.polygons.unsqueeze(0), to_concat.polygons.unsqueeze(0)))
        else:
            self.events = torch.cat((self.events, to_concat.events.unsqueeze(0)))
            self.labels = torch.cat((self.labels, to_concat.labels.unsqueeze(0)))
            self.padding_mask = torch.cat((self.padding_mask, to_concat.padding_mask.unsqueeze(0)))
            self.polygons = torch.cat((self.polygons, to_concat.polygons.unsqueeze(0)))

    def get_item(self, idx: int):

        return FlowSampleTensorData(self.sample_names[idx], self.events[idx], self.labels[idx], self.padding_mask[idx], self.polygons[idx])
