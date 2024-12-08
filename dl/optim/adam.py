from dl.nn import Module

class Adam:
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        unbiased: bool = True,
        eps: float = 1e-8
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.unbiased = unbiased
        self.eps = eps

        self.unbiasing_beta1 = 0.0
        self.unbiasing_beta2 = 0.0
        if self.unbiased:
            self.unbiasing_beta1 = self.beta1
            self.unbiasing_beta2 = self.beta2

    def step(self, model: Module) -> None:
        for variable in model.variables():
            if variable.grad is None: continue

            variable.grad.data += self.lr * self.weight_decay * variable.data

            variable.u = self.beta1 * variable.u + (1 - self.beta1) * variable.grad.data
            unbiased_u = variable.u/(1-self.unbiasing_beta1)

            variable.v = self.beta2 * variable.v + (1 - self.beta2) * (variable.grad.data)**2
            unbiased_v = variable.v/(1-self.unbiasing_beta2)

            variable.data -= self.lr * unbiased_u/((unbiased_v)**0.5 + self.eps)

        if self.unbiased:
            self.unbiasing_beta1 *= self.beta1
            self.unbiasing_beta2 *= self.beta2
