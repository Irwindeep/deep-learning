from typing import (
    Iterator,
    Dict,
    Optional,
    Any,
    Callable
)
import inspect
from dl.variable import Variable

__all__ = ["Module"]

class Module:
    _modules: Dict[str, 'Module']
    
    def __init__(self):
        self._modules = {}

    def variables(self) -> Iterator[Variable]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Variable): yield value
            elif isinstance(value, Module): yield from value.variables()

    def zero_grad(self) -> None:
        for variable in self.variables():
            variable.zero_grad()

    def forward(self, *input: Any) -> Any:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' pass"
        )

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)
