from typing import Iterator
import inspect
from dl.variable import Variable

class Module:
    def variables(self) -> Iterator[Variable]:
        for _, value in inspect.getmembers(self):
            if isinstance(value, Variable): yield value
            elif isinstance(value, Module): yield from value.variables()

    def zero_grad(self) -> None:
        for variable in self.variables():
            variable.zero_grad()
