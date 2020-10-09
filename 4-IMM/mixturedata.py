from typing import (
    Collection,
    Generic,
    TypeVar,
    Union,
    Sequence,
    Any,
    List,
)

# from singledispatchmethod import singledispatchmethod  # pip install
from dataclasses import dataclass
import numpy as np

T = TypeVar("T")


@dataclass
class MixtureParameters(Generic[T]):
    __slots__ = ["weights", "components"]
    weights: np.ndarray
    components: Sequence[T]


# class Array(Collection[T], Generic[T]):
#     def __getitem__(self, key):
#         ...

#     def __setitem__(self, key, vaule):
#         ...


# @dataclass
# class MixtureParametersList(Generic[T]):
#     weights: np.ndarray
#     components: Array[Sequence[T]]

#     @classmethod
#     def allocate(cls, shape: Union[int, Tuple[int, ...]], component_type: T):
#         shape = (shape,) if isinstance(shape, int) else shape
#         # TODO
#         raise NotImplementedError

#     @singledispatchmethod
#     def __getitem__(self, key: Any) -> "MixtureParametersList[T]":
#         return MixtureParametersList(self.weights[key], self.components[key])

#     @__getitem__.register
#     def _(self, key: int) -> MixtureParameters:
#         return MixtureParameters(self.weights[key], self.components[key])

#     def __setitem__(
#         self,
#         key: Union[int, slice],
#         value: "Union[MixtureParameters[T], MixtureParametersList[T]]",
#     ) -> None:
#         self.weights[key] = value.weights
#         self.components[key] = value.components

#     def __len__(self):
#         return self.weights.shape[0]

#     def __iter__(self):
#         yield from (self[k] for k in range(len(self)))
