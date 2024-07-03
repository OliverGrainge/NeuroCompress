class PruningConfig:
    def __init__(self, task: str, amount: int = 0.5):
        self.task = task
        self.amount = amount
