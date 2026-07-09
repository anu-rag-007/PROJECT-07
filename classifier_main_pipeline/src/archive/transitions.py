class StageTransitionEnforcer:
    def __init__(self):
        self.allowed = {
            "Wake": ["Wake", "N1"],
            "N1": ["Wake", "N1", "N2"],
            "N2": ["N1", "N2", "N3"],
            "N3": ["N2", "N3"],
            "REM": ["N2", "REM"]
        }
        self.prev = "Wake"

    def enforce(self, stage):
        if stage in self.allowed[self.prev]:
            self.prev = stage
        return self.prev
