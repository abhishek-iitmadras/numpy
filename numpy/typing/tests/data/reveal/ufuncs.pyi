from typing import Any, Literal, NoReturn, assert_type

import numpy as np
import numpy.typing as npt

i8: np.int64
f8: np.float64
AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]

assert_type(np.absolute.__doc__, str)
assert_type(np.absolute.types, list[str])

assert_type(np.absolute.__name__, Literal["absolute"])
assert_type(np.absolute.__qualname__, Literal["absolute"])
assert_type(np.absolute.ntypes, Literal[20])
assert_type(np.absolute.identity, None)
assert_type(np.absolute.nin, Literal[1])
assert_type(np.absolute.nin, Literal[1])
assert_type(np.absolute.nout, Literal[1])
assert_type(np.absolute.nargs, Literal[2])
assert_type(np.absolute.signature, None)
assert_type(np.absolute(f8), Any)
assert_type(np.absolute(AR_f8), npt.NDArray[Any])
assert_type(np.absolute.at(AR_f8, AR_i8), None)

assert_type(np.add.__name__, Literal["add"])
assert_type(np.add.__qualname__, Literal["add"])
assert_type(np.add.ntypes, Literal[22])
assert_type(np.add.identity, Literal[0])
assert_type(np.add.nin, Literal[2])
assert_type(np.add.nout, Literal[1])
assert_type(np.add.nargs, Literal[3])
assert_type(np.add.signature, None)
assert_type(np.add(f8, f8), Any)
assert_type(np.add(AR_f8, f8), npt.NDArray[Any])
assert_type(np.add.at(AR_f8, AR_i8, f8), None)
assert_type(np.add.reduce(AR_f8, axis=0), Any)
assert_type(np.add.accumulate(AR_f8), npt.NDArray[Any])
assert_type(np.add.reduceat(AR_f8, AR_i8), npt.NDArray[Any])
assert_type(np.add.outer(f8, f8), Any)
assert_type(np.add.outer(AR_f8, f8), npt.NDArray[Any])

assert_type(np.frexp.__name__, Literal["frexp"])
assert_type(np.frexp.__qualname__, Literal["frexp"])
assert_type(np.frexp.ntypes, Literal[4])
assert_type(np.frexp.identity, None)
assert_type(np.frexp.nin, Literal[1])
assert_type(np.frexp.nout, Literal[2])
assert_type(np.frexp.nargs, Literal[3])
assert_type(np.frexp.signature, None)
assert_type(np.frexp(f8), tuple[Any, Any])
assert_type(np.frexp(AR_f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])

assert_type(np.divmod.__name__, Literal["divmod"])
assert_type(np.divmod.__qualname__, Literal["divmod"])
assert_type(np.divmod.ntypes, Literal[15])
assert_type(np.divmod.identity, None)
assert_type(np.divmod.nin, Literal[2])
assert_type(np.divmod.nout, Literal[2])
assert_type(np.divmod.nargs, Literal[4])
assert_type(np.divmod.signature, None)
assert_type(np.divmod(f8, f8), tuple[Any, Any])
assert_type(np.divmod(AR_f8, f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])

assert_type(np.matmul.__name__, Literal["matmul"])
assert_type(np.matmul.__qualname__, Literal["matmul"])
assert_type(np.matmul.ntypes, Literal[19])
assert_type(np.matmul.identity, None)
assert_type(np.matmul.nin, Literal[2])
assert_type(np.matmul.nout, Literal[1])
assert_type(np.matmul.nargs, Literal[3])
assert_type(np.matmul.signature, Literal["(n?,k),(k,m?)->(n?,m?)"])
assert_type(np.matmul.identity, None)
assert_type(np.matmul(AR_f8, AR_f8), Any)
assert_type(np.matmul(AR_f8, AR_f8, axes=[(0, 1), (0, 1), (0, 1)]), Any)

assert_type(np.vecdot.__name__, Literal["vecdot"])
assert_type(np.vecdot.__qualname__, Literal["vecdot"])
assert_type(np.vecdot.ntypes, Literal[19])
assert_type(np.vecdot.identity, None)
assert_type(np.vecdot.nin, Literal[2])
assert_type(np.vecdot.nout, Literal[1])
assert_type(np.vecdot.nargs, Literal[3])
assert_type(np.vecdot.signature, Literal["(n),(n)->()"])
assert_type(np.vecdot.identity, None)
assert_type(np.vecdot(AR_f8, AR_f8), Any)

assert_type(np.bitwise_count.__name__, Literal["bitwise_count"])
assert_type(np.bitwise_count.__qualname__, Literal["bitwise_count"])
assert_type(np.bitwise_count.ntypes, Literal[11])
assert_type(np.bitwise_count.identity, None)
assert_type(np.bitwise_count.nin, Literal[1])
assert_type(np.bitwise_count.nout, Literal[1])
assert_type(np.bitwise_count.nargs, Literal[2])
assert_type(np.bitwise_count.signature, None)
assert_type(np.bitwise_count.identity, None)
assert_type(np.bitwise_count(i8), Any)
assert_type(np.bitwise_count(AR_i8), npt.NDArray[Any])

def test_absolute_outer_invalid() -> None:
    assert_type(np.absolute.outer(), NoReturn)
def test_frexp_outer_invalid() -> None:
    assert_type(np.frexp.outer(), NoReturn)
def test_divmod_outer_invalid() -> None:
    assert_type(np.divmod.outer(), NoReturn)
def test_matmul_outer_invalid() -> None:
    assert_type(np.matmul.outer(), NoReturn)

def test_absolute_reduceat_invalid() -> None:
    assert_type(np.absolute.reduceat(), NoReturn)
def test_frexp_reduceat_invalid() -> None:
    assert_type(np.frexp.reduceat(), NoReturn)
def test_divmod_reduceat_invalid() -> None:
    assert_type(np.divmod.reduceat(), NoReturn)
def test_matmul_reduceat_invalid() -> None:
    assert_type(np.matmul.reduceat(), NoReturn)

def test_absolute_reduce_invalid() -> None:
    assert_type(np.absolute.reduce(), NoReturn)
def test_frexp_reduce_invalid() -> None:
    assert_type(np.frexp.reduce(), NoReturn)
def test_divmod_reduce_invalid() -> None:
    assert_type(np.divmod.reduce(), NoReturn)
def test_matmul_reduce_invalid() -> None:
    assert_type(np.matmul.reduce(), NoReturn)

def test_absolute_accumulate_invalid() -> None:
    assert_type(np.absolute.accumulate(), NoReturn)
def test_frexp_accumulate_invalid() -> None:
    assert_type(np.frexp.accumulate(), NoReturn)
def test_divmod_accumulate_invalid() -> None:
    assert_type(np.divmod.accumulate(), NoReturn)
def test_matmul_accumulate_invalid() -> None:
    assert_type(np.matmul.accumulate(), NoReturn)

def test_frexp_at_invalid() -> None:
    assert_type(np.frexp.at(), NoReturn)
def test_divmod_at_invalid() -> None:
    assert_type(np.divmod.at(), NoReturn)
def test_matmul_at_invalid() -> None:
    assert_type(np.matmul.at(), NoReturn)
