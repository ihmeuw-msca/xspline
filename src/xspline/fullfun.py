"""
Core Module
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

from xspline.funutils import check_fun_input, taylor_term
from xspline.interval import Interval


@dataclass
class FullFunction:
    domain: Interval = field(default_factory=Interval)
    support: Interval = field(default_factory=Interval)
    fun: Callable = field(default=None)

    def __call__(self, data: Iterable, order: int = 0) -> np.ndarray:
        if self.fun is None:
            return NotImplementedError()
        return self.fun(data, order)

    def __add__(self, rfun: "FullFunction") -> "FullFunction":
        domain = self.domain + rfun.domain
        support = self.support | rfun.support

        def fun(data: Iterable, order: int = 0) -> np.ndarray:
            data, order = check_fun_input(data, order)
            lx1 = data[-1] <= self.domain.ub
            rx1 = ~lx1
            val = np.zeros(data.shape[-1])
            if order >= 0:
                val[lx1] = self(data[:, lx1], order)
                val[rx1] = rfun(data[:, rx1], order)
            else:
                lx0 = data[0] <= self.domain.ub
                rx0 = ~lx0
                lboth = lx0 & lx1
                rboth = rx0 & rx1
                landr = lx0 & rx1

                val[lboth] = self(data[:, lboth], order)
                val[rboth] = rfun(data[:, rboth], order)

                ldata = data[:, landr].copy()
                rdata = data[:, landr].copy()
                ldata[1] = self.domain.ub.val
                rdata[0] = self.domain.ub.val
                for i in range(order + 1, 0):
                    val[landr] += self(ldata, i)*taylor_term(rdata, i - order)
                val[landr] += self(ldata, order) + rfun(rdata, order)
            return val

        return FullFunction(domain, support, fun)

    def __radd__(self, fun: Union[int, "FullFunction"]) -> "FullFunction":
        return self if fun == 0 else self.__add__(fun)
