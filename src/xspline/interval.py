"""
Interval Module
"""
from dataclasses import dataclass
from typing import List, Tuple, Union
from numbers import Number
from operator import le, lt, ge, gt

import numpy as np


@dataclass
class Interval:
    lb: float = -np.inf
    ub: float = np.inf
    lb_closed: bool = True
    ub_closed: bool = True

    def __post_init__(self):
        self.lb_closed = bool(self.lb_closed) and (not np.isinf(self.lb))
        self.ub_closed = bool(self.ub_closed) and (not np.isinf(self.ub))
        assert (self.lb < self.ub) or ((self.lb == self.ub) and
                                       (self.lb_closed and self.ub_closed))

    @property
    def size(self) -> float:
        return self.ub - self.lb

    def is_addable(self, invl: "Interval") -> bool:
        return (self.ub == invl.lb) and (self.ub_closed ^ invl.lb_closed)

    def is_andable(self, invl: "Interval") -> bool:
        return (self.ub > invl.lb) or ((self.ub == invl.lb) and
                                       (self.ub_closed and invl.lb_closed))

    def is_orable(self, invl: 'Interval') -> bool:
        return (self.ub > invl.lb) or ((self.ub == invl.lb) and
                                       (self.ub_closed or invl.lb_closed))

    def __add__(self, invl: "Interval") -> "Interval":
        assert self.is_addable(invl)
        return Interval(self.lb, invl.ub,
                        self.lb_closed, invl.ub_closed)

    def __radd__(self, invl: Union[int, "Interval"]) -> "Interval":
        return self if invl == 0 else self.__add__(invl)

    def __and__(self, invl: "Interval") -> "Interval":
        assert self.is_andable(invl)
        return Interval(invl.lb, self.ub,
                        invl.lb_closed, self.ub_closed)

    def __or__(self, invl: "Interval") -> "Interval":
        assert self.is_orable(invl)
        return Interval(self.lb, invl.ub,
                        self.lb_closed, invl.ub_closed)

    def __eq__(self, invl: "Interval") -> bool:
        if not isinstance(invl, Interval):
            raise ValueError("Can only compare to Interval instance.")
        return all([
            val == getattr(invl, key)
            for key, val in vars(self).items()
        ])

    def __invert__(self) -> Tuple[Union[None, "Interval"], Union[None, "Interval"]]:
        linvl = None if np.isinf(self.lb) else \
            Interval(-np.inf, self.lb, lb_closed=False, ub_closed=not self.lb_closed)
        rinvl = None if np.isinf(self.ub) else \
            Interval(self.ub, np.inf, lb_closed=not self.ub_closed, ub_closed=False)
        return linvl, rinvl

    def __contains__(self, num: Number) -> bool:
        assert isinstance(num, Number)
        lopt = ge if self.lb_closed else gt
        ropt = le if self.ub_closed else lt
        return lopt(num, self.lb) and ropt(num, self.ub)

    def __getitem__(self, index: int) -> float:
        assert index in [0, 1]
        return self.lb if index == 0 else self.ub

    def __repr__(self) -> str:
        lb_bracket = "[" if self.lb_closed else "("
        ub_bracket = "]" if self.ub_closed else ")"
        return f"{lb_bracket}{self.lb}, {self.ub}{ub_bracket}"
