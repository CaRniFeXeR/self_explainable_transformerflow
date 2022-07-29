from pathlib import Path
from typing import List

from ...utils.dynamic_type_loader import load_type_dynamically_from_fqn

from ...datastructures.configs.dataloaderconfig import DataLoaderConfig
from .SplitableDataSet import SplitableDataSet
from ..FlowSample.file_sample_factory import FileSampleFactory


class FileTypeFolderDataSet(SplitableDataSet):

    @staticmethod
    def init_from_config(config: DataLoaderConfig) -> SplitableDataSet:

        if not isinstance(config, DataLoaderConfig):
            raise TypeError(f"config must be from Type 'DataLoaderConfig' given {type(config)}")

        factory_type = load_type_dynamically_from_fqn(config.sample_factory_type)
        factory = factory_type()

        return FileTypeFolderDataSet(name=config.dataset_name,
                                     rootfolder=config.root_path,
                                     fileExtensionToUse=config.file_extension,
                                     fileNameMustContain=config.filename_must_contain,
                                     bloodsample_factory=factory)

    def __init__(self, name: str, rootfolder: Path, fileExtensionToUse: str = ".xml", fileNameMustContain: str = "", bloodsample_factory: FileSampleFactory = FileSampleFactory()):
        self.rootfolder = rootfolder
        self.fileExtensionToUse = fileExtensionToUse
        self.fileNameMustContain = fileNameMustContain
        fileIndexes = self.findFiles(self.rootfolder, [])
        fileIndexes.reverse()
        super(FileTypeFolderDataSet, self).__init__(name, rootfolder,
                                                    fileIndexes, "", bloodsample_factory=bloodsample_factory)
        print("init FileTypeFolderDataSet '" + name + "' with " +
              str(len(fileIndexes)) + " blood samples")

    def findFiles(self, folderPath: Path, fileIndexes: List):

        for entry in folderPath.iterdir():
            if entry.is_dir():
                fileIndexes = self.findFiles(entry, fileIndexes)
            elif self.fileExtensionToUse in entry.name and (self.fileNameMustContain == "" or self.fileNameMustContain in entry.name):
                try:
                    fileIndexes.append(entry.relative_to(self.rootfolder))
                except Exception as ex:
                    print(ex)
                    pass

        return fileIndexes

    def getFilePath(self, fileName: str) -> Path:
        return Path(self.rootfolder) / Path(fileName)
