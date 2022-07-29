import pandas as pd


class GateCollection:
    """
    GateCollection holds translations from different occuring gate names to the standarised names.
    """
    GATE_RENAME_DICT = {
        "CD7pos": "cd7+",
        "cd7pos": "cd7+",
        "cd7": "cd7+",
        "CD20 pos": "CD20",
        "blasten": "Blasts",
        "Blasten": "Blasts",
        "blasts": "Blasts",
        "blast": "Blasts",
        "Blast": "Blasts",
        "Syto pos": "Syto",
        "sytopos": "Syto",
        "Syto +": "Syto",
        "syto": "Syto",
        "Syto+": "Syto",
        "syto+": "Syto",
        "Syto +": "Syto",
        "CD19 pos": "CD19",
        "CD19+": "CD19",
        "CD19 +": "CD19",
        "cd19": "CD19",
        "cd19+": "CD19",
        "viable": "Intact",
        "intact": "Intact",
        "Viable": "Intact",
        "singlets": "Singlets",
        "mature B-cells": "mature B-Cells",
        "Plamsa cells": "Plasma Cells",
        "plasmacells": "Plasma Cells",
        "Plasmacells": "Plasma Cells"
    }

    @staticmethod
    def renameGates(gates: pd.DataFrame):
        renameDict = GateCollection.GATE_RENAME_DICT
        renameDict = {k: v for k, v in renameDict.items() if k in gates.columns and not v in gates.columns}
        if len(renameDict) > 0:
            gates.rename(columns=renameDict, inplace=True)

        return gates
