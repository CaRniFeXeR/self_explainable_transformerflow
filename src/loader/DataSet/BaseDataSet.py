
from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from ..FlowSample.flow_sample_base import FlowSampleBase
from ..FlowSample.file_sample_factory import FileSampleFactory
import random


class BaseDataSet():

    def __init__(self, name: str, rootfolder: Path, fileIndexes: List[str] = [], fileType=".xml", bloodsample_factory: FileSampleFactory = FileSampleFactory()):
        self.name = name.split("\\")[-1]
        self.rootfolder = rootfolder
        self.fileIndexes = fileIndexes
        self.currentPos = 0
        self.fileType = fileType
        self.bloodsample_factory = bloodsample_factory

    def __iter__(self):
        self.currentPos = 0
        return self

    def __next__(self) -> FlowSampleBase:
        return self.loadNextSample()

    def getNextIndex(self) -> str:

        if self.currentPos >= len(self.fileIndexes):
            raise StopIteration

        nextIndex = self.fileIndexes[self.currentPos]
        self.currentPos = self.currentPos + 1
        return nextIndex

    def loadNextSample(self) -> FlowSampleBase:
        index = self.getNextIndex()
        try:
            bloodSample = self.bloodsample_factory.init_blood_sample(self.getFilePath(index))
        except Exception as ex:
            print(f"error while loading sample of dataset {self.name} with path {self.getFilePath(index)}")
            print(ex)
            return self.loadNextSample()

        return bloodSample

    def getFirstSample(self) -> FlowSampleBase:
        index = self.fileIndexes[0]
        bloodSample = self.bloodsample_factory.init_blood_sample(self.getFilePath(index))

        return bloodSample

    def shuffle(self):

        random.shuffle(self.fileIndexes)

    def findSampleByCompareFunction(self, func) -> FlowSampleBase:

        filtered_indexes = list(filter(func, self.fileIndexes))

        if len(filtered_indexes) == 0:
            raise ValueError("sample not found!")

        index = filtered_indexes[0]
        return self.bloodsample_factory.init_blood_sample(self.getFilePath(index))

    def getFilePath(self, fileIndex: int) -> Path:
        fileName = str(fileIndex) + self.fileType
        return Path(self.rootfolder / Path(fileName))

    def getSingleSampleSplitedDataSets(self) -> List:
        resultDataSets = []
        for fileIndex in self.fileIndexes:
            singelFileDataSet = BaseDataSet(fileIndex, self.rootfolder, [
                                            fileIndex], fileType=self.fileType, bloodsample_factory=self.bloodsample_factory)
            resultDataSets.append(singelFileDataSet)

        return resultDataSets
