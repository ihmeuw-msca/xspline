from typing import Annotated, Callable, Optional, Protocol

from numpy.typing import NDArray

PositiveInt = Annotated[int, "positive"]
NegativeInt = Annotated[int, "negative"]

VFunction = Callable[[NDArray], NDArray]
DFunction = Callable[[NDArray, PositiveInt], NDArray]
IFunction = Callable[[NDArray, NegativeInt], NDArray]

RawVFunction = Callable[[tuple, NDArray], NDArray]
RawDFunction = Callable[[tuple, NDArray, PositiveInt], NDArray]
RawIFunction = Callable[[tuple, NDArray, NegativeInt], NDArray]

BoundaryPoint = tuple[float, bool]
IndiParams = tuple[BoundaryPoint, BoundaryPoint]
PolyParams = tuple[float, ...]
BsplParams = tuple[tuple[float, ...], int, int]


class XFunction(Protocol):

    def __call__(self,
                 x: NDArray,
                 order: int = 0,
                 start: Optional[NDArray] = None,
                 **kwargs) -> NDArray:
        ...
