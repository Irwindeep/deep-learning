from dl.nn import Module

class Optimizer:
    def __init__(self):
        pass

    def step(self, model: Module) -> None:
        raise NotImplementedError(
            f"Optimizer step not defined"
        )
