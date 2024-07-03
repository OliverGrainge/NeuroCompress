class DistillationConfig:
    def __init__(
        self,
        task: str,
        student_architecture: str,
        temperature: int = 1.0,
        alpha: int = 0.5,
    ):
        self.task = task
        self.student_architecture = student_architecture
        self.temperature = temperature
        self.alpha = alpha
