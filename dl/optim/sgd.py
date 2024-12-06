from dl.nn import Module

class SGD:
    def __init__(
        self,
        lr: float = 0.01,
        beta: float = 0,
        unbiased: bool = False
    ) -> None:
        self.lr = lr
        self.beta = beta
        self.unbiased = unbiased

        self.unbiasing_beta = 0
        if self.unbiased: self.unbiasing_beta = self.beta

    def step(self, model: Module) -> None:
        for variable in model.variables():
            if variable.grad is None: continue
            variable.u = self.beta * variable.u + (1 - self.beta) * variable.grad
            variable -= self.lr * variable.u / (1 - self.unbiasing_beta)

        if self.unbiased: self.unbiasing_beta *= self.beta
