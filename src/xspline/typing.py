from typing import Callable, Optional, Protocol

from numpy.typing import NDArray

VFunction = Callable[[NDArray], NDArray]
DFunction = Callable[[NDArray, int], NDArray]
IFunction = Callable[[NDArray, int], NDArray]

RawVFunction = Callable[[tuple, NDArray], NDArray]
RawDFunction = Callable[[tuple, NDArray, int], NDArray]
RawIFunction = Callable[[tuple, NDArray, int], NDArray]
RawFunction = RawVFunction | RawDFunction | RawIFunction

BoundaryPoint = tuple[float, bool]
IndiParams = tuple[BoundaryPoint, BoundaryPoint]
PolyParams = tuple[float, ...]
BsplParams = tuple[tuple[float, ...], int, int]


class XFunction(Protocol):

    def __call__(self, x: NDArray, order: int = 0, start: Optional[NDArray] = None, **kwargs) -> NDArray:
        ...
