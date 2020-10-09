import sys
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union, overload

# %% Taken from https://github.com/numpy/numpy/tree/master/numpy/typing
from numpy import dtype, ndarray

if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

_Shape = Tuple[int, ...]

# Anything that can be coerced to a shape tuple
_ShapeLike = Union[int, Sequence[int]]

_DtypeLikeNested = Any  # TODO: wait for support for recursive types

if TYPE_CHECKING or HAVE_PROTOCOL:
    # Mandatory keys
    class _DtypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DtypeLikeNested]

    # Mandatory + optional keys
    class _DtypeDict(_DtypeDictBase, total=False):
        offsets: Sequence[int]
        # Only `str` elements are usable as indexing aliases, but all objects are legal
        titles: Sequence[Any]
        itemsize: int
        aligned: bool

    # A protocol for anything with the dtype attribute
    class _SupportsDtype(Protocol):
        dtype: _DtypeLikeNested

else:
    _DtypeDict = Any
    _SupportsDtype = Any


DtypeLike = Union[
    dtype, None, type, _SupportsDtype, str, Tuple[_DtypeLikeNested, int],
    Tuple[_DtypeLikeNested, _ShapeLike], List[Any], _DtypeDict,
    Tuple[_DtypeLikeNested, _DtypeLikeNested],
]


if TYPE_CHECKING or HAVE_PROTOCOL:
    class _SupportsArray(Protocol):
        @overload
        def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...
        @overload
        def __array__(self, dtype: DtypeLike = ...) -> ndarray: ...
else:
    _SupportsArray = Any


ArrayLike = Union[bool, int, float, complex, _SupportsArray, Sequence]

# %%
