from pathlib import Path
from typing import List
from .BaseDataSet import BaseDataSet
from ..FlowSample.file_sample_factory import FileSampleFactory

class SplitableDataSet(BaseDataSet):

    def __init__(self, name: str, rootfolder: Path, fileIndexes: List[str] = [], fileType: str = ".xml", bloodsample_factory: FileSampleFactory = FileSampleFactory()):
        super(SplitableDataSet, self).__init__(name, rootfolder,
                                               fileIndexes, fileType, bloodsample_factory=bloodsample_factory)

    # todo implement dataset splits
    # def getSplits(self, n_splits: int = 5) -> List[TrainTestSplit]:
    #     kf = KFold(shuffle=True, n_splits=n_splits)

    #     cv_data = []
    #     i = 1
    #     fileIndexArray = np.array(self.fileIndexes)
    #     for train_indexes, test_indexes in kf.split(fileIndexArray):
    #         train_dataset = BaseDataSet(self.name + " train_fold" + str(
    #             i), self.rootfolder, fileIndexArray[train_indexes], self.fileType)
    #         test_dataset = BaseDataSet(self.name + " test_fold" + str(
    #             i), self.rootfolder, fileIndexArray[test_indexes], self.fileType)
    #         cv_data.append(TrainTestSplit(train_dataset, test_dataset))
    #         i = i + 1

    #     return cv_data
