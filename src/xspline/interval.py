"""
Interval Module
"""
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Union

import numpy as np


@dataclass
class BoundaryNumber:
    val: Number
    cld: bool = True

    def __post_init__(self):
        if not isinstance(self.val, Number):
            raise ValueError("`val` must be a number.")
        self.cld = bool(self.cld) and (not np.isinf(self.val))

    @classmethod
    def as_boundary_number(cls, num: Union[Number, Tuple]) -> "BoundaryNumber":
        if isinstance(num, cls):
            return num
        if isinstance(num, Number):
            num = cls(num)
        elif isinstance(num, Tuple):
            num = cls(*num)
        else:
            raise ValueError("Input has to be number, tuple or BoundaryNumber.")
        return num

    def __eq__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val == other.val) and (self.cld == other.cld)

    def __lt__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val < other.val) or ((self.val == other.val) and
                                          (self.cld < other.cld))

    def __le__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val < other.val) or ((self.val == other.val) and
                                          (self.cld <= other.cld))

    def __gt__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val > other.val) or ((self.val == other.val) and
                                          (self.cld > other.cld))

    def __ge__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val > other.val) or ((self.val == other.val) and
                                          (self.cld >= other.cld))

    def __and__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val < other.val) or ((self.val == other.val) and
                                          (self.cld and other.cld))

    def __or__(self, other: "BoundaryNumber") -> bool:
        other = BoundaryNumber.as_boundary_number(other)
        return (self.val < other.val) or ((self.val == other.val) and
                                          (self.cld or other.cld))

    def __invert__(self) -> "BoundaryNumber":
        return BoundaryNumber(self.val, not self.cld)

    def __repr__(self) -> str:
        lbracket, rbracket = ("[", "]") if self.cld else ("(", ")")
        return f"{lbracket}{self.val}{rbracket}"


@ dataclass
class Interval:
    lb: Union[Number, Tuple[Number, bool], BoundaryNumber] = BoundaryNumber(-np.inf)
    ub: Union[Number, Tuple[Number, bool], BoundaryNumber] = BoundaryNumber(np.inf)

    def __post_init__(self):
        self.lb = BoundaryNumber.as_boundary_number(self.lb)
        self.ub = BoundaryNumber.as_boundary_number(self.ub)
        if not self.lb & self.ub:
            raise ValueError(f"{self} is an empty interval.")

    @ property
    def size(self) -> float:
        return self.ub.val - self.lb.val

    def is_addable(self, invl: "Interval") -> bool:
        assert isinstance(invl, Interval)
        return self.ub == ~invl.lb

    def is_andable(self, invl: "Interval") -> bool:
        assert isinstance(invl, Interval)
        return max(self.lb, invl.lb) & min(self.ub, invl.ub)

    def is_orable(self, invl: "Interval") -> bool:
        assert isinstance(invl, Interval)
        return max(self.lb, invl.lb) | min(self.ub, invl.ub)

    def __add__(self, invl: "Interval") -> "Interval":
        assert self.is_addable(invl)
        return Interval(self.lb, invl.ub)

    def __radd__(self, invl: Union[int, "Interval"]) -> "Interval":
        return self if invl == 0 else self.__add__(invl)

    def __and__(self, invl: "Interval") -> "Interval":
        assert self.is_andable(invl)
        return Interval(max(self.lb, invl.lb), min(self.ub, invl.ub))

    def __or__(self, invl: "Interval") -> "Interval":
        assert self.is_orable(invl)
        return Interval(min(self.lb, invl.lb), max(self.ub, invl.ub))

    def __eq__(self, invl: "Interval") -> bool:
        assert isinstance(invl, Interval)
        return (self.lb == invl.lb) and (self.ub == invl.ub)

    def __invert__(self) -> Tuple[Union[None, "Interval"], Union[None, "Interval"]]:
        linvl = None if np.isinf(self.lb.val) else Interval(-np.inf, ~self.lb)
        rinvl = None if np.isinf(self.ub.val) else Interval(~self.ub, np.inf)
        return linvl, rinvl

    def __contains__(self, num: Number) -> bool:
        assert isinstance(num, Number)
        return self.lb <= num <= self.ub

    def __getitem__(self, index: int) -> float:
        if index >= 2:
            raise IndexError
        return self.lb.val if index == 0 else self.ub.val

    def __repr__(self) -> str:
        lbracket = "[" if self.lb.cld else "("
        rbracket = "]" if self.ub.cld else ")"
        return f"{lbracket}{self.lb.val}, {self.ub.val}{rbracket}"
