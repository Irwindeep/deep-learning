from dl.nn import Module

class SGD:
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        beta: float = 0.0,
        unbiased: bool = False
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.unbiased = unbiased

        self.unbiasing_beta = 0.0
        if self.unbiased: self.unbiasing_beta = self.beta

    def step(self, model: Module) -> None:
        for variable in model.variables():
            if variable.grad is None: continue

            variable.grad.data += self.weight_decay * variable.data

            variable.u = self.beta * variable.u + (1 - self.beta) * variable.grad.data
            unbiased_u = variable.u/(1 - self.unbiasing_beta)
            
            variable.data -= (self.lr * unbiased_u).data

        if self.unbiased: self.unbiasing_beta *= self.beta
