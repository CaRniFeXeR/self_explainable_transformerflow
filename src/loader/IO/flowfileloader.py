from .fileloader import FileLoader


class FlowFileLoader(FileLoader):
    """
    Handles FlowMe FlowSample Loading. Flowme is import dynamically --> reference is not needed if only working on cached files.
    """

    def loadFlowMeData(self):
        fmp = __import__("flowme")
        return fmp.fcs(str(self.inputfile))
