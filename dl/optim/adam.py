from dl.nn import Module

class Adam:
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        unbiased: bool = True,
        eps: float = 1e-8
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.unbiased = unbiased
        self.eps = eps

        self.unbiasing_beta1 = 0
        self.unbiasing_beta2 = 0
        if self.unbiased:
            self.unbiasing_beta1 = self.beta1
            self.unbiasing_beta2 = self.beta2

    def step(self, model: Module) -> None:
        for variable in model.variables():
            if variable.grad is None: continue
            variable.u = self.beta1 * variable.u + (1 - self.beta1) * variable.grad
            variable.v = self.beta2 * variable.v + (1 - self.beta2) * (variable.grad)**2
            variable -= self.lr * (variable.u / (1 - self.unbiasing_beta1))/((variable.v / (1 - self.unbiasing_beta2))**0.5 + self.eps)

        if self.unbiased:
            self.unbiasing_beta1 *= self.beta1
            self.unbiasing_beta2 *= self.beta2
