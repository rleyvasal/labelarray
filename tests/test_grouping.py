"""Groupings and grouped reductions (``labelarray.grouped``).

A ``Grouping`` holds one code per position and the group labels, with the
group sizes and the positions sorted by group computed once. ``reduce``
gives every group's result bit-identical to reducing that group on its own,
whether the one-pass kernel (``ONE_PASS``) or the group-by-group path
computes it.
"""

import numpy as np
import pytest

from labelarray import Array, Bins, Grouping, factorize, grouped

OPS = ["sum", "mean", "min", "max", "any", "all", "std", "median", "count"]


def test_grouping_holds_plain_codes_and_labels_and_caches_its_derived_arrays():
    codes = np.array([2, 0, 2, 1, 0, 2])
    grouping = Grouping(codes, ["a", "b", "c"])
    assert isinstance(grouping.codes, np.ndarray) and grouping.codes.dtype.kind in "iu"
    assert grouping.labels == ["a", "b", "c"]
    assert len(grouping) == 3
    assert grouping.counts().tolist() == [2, 1, 3]
    order, starts, ends = grouping.blocks()
    assert order.tolist() == [1, 4, 3, 0, 2, 5]  # positions sorted by group, original order within a group
    assert starts.tolist() == [0, 2, 3] and ends.tolist() == [2, 3, 6]
    assert grouping.counts() is grouping.counts() and grouping.blocks() is grouping.blocks()
    empty = Grouping(np.array([], dtype=np.intp), [])
    assert len(empty) == 0 and empty.counts().tolist() == [] and empty.blocks()[0].tolist() == []


@pytest.mark.parametrize("groups", [3, 40])
@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "count"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_grouped_reductions_match_each_group_alone(groups, op, dtype):
    rng = np.random.default_rng(groups)
    codes = rng.integers(0, groups, 200)
    values = Array((rng.normal(size=(200, 2)) * 100).astype(dtype), ("point", "band"), {"band": ["r", "g"]})
    grouping = Grouping(codes, [f"g{i}" for i in range(groups)])
    got = grouped.reduce(values, op, grouping, "point", "group")
    assert got.dims == ("group", "band") and got.labels("group").tolist() == grouping.labels
    for code in range(groups):
        alone = values.isel(point=np.flatnonzero(codes == code)).reduce(op, ["point"])
        np.testing.assert_array_equal(got.data[code], alone.data)


def test_counts_and_broadcast_back_to_positions():
    grouping = Grouping(np.array([1, 0, 1, 1]), ["a", "b"])
    counts = grouped.counts(grouping, "key", per_position=2)
    assert counts.data.tolist() == [2, 6] and counts.labels("key").tolist() == ["a", "b"]
    stats = Array([10.0, 20.0], ("key",), {"key": ["a", "b"]})
    back = grouped.broadcast(stats, grouping, "point", "key", labels=np.array([5, 6, 7, 8]))
    assert back.dims == ("point",) and back.data.tolist() == [20.0, 10.0, 20.0, 20.0]
    assert back.labels("point").tolist() == [5, 6, 7, 8]


def _random_case(rng):
    groups = int(rng.integers(9, 80))
    sizes = rng.integers(1, 8, groups)
    if rng.random() < 0.4:  # some groups of 8 or more, reduced one by one
        sizes[rng.integers(groups, size=3)] = rng.integers(8, 40, 3)
    key = rng.permutation(np.repeat(rng.permutation(groups) * 3, sizes))
    dtype = str(rng.choice(["float32", "float64", "int16", "int64", "bool"]))
    layout = str(rng.choice(["1d", "first", "last"]))
    shape = (len(key),) if layout == "1d" else (len(key), 3)
    values = rng.normal(size=shape) * 10 ** rng.uniform(-3, 8, shape)
    if dtype.startswith("float"):
        values[rng.random(shape) < 0.05] = -0.0
        values[rng.random(shape) < 0.05] = np.nan
        values[rng.random(shape) < 0.01] = np.inf
    elif dtype == "bool":
        values = values > 0
    values = values.astype(dtype)
    if layout == "1d":
        array = Array(values, ("point",))
    elif layout == "first":
        array = Array(values, ("point", "feature"), {"feature": ["f1", "f2", "f3"]})
    else:
        array = Array(np.ascontiguousarray(values.T), ("feature", "point"), {"feature": ["f1", "f2", "f3"]})
    codes, labels = factorize(key)
    return array, Grouping(codes, labels)


def _same_bits(a: Array, b: Array) -> None:
    assert a.dims == b.dims and a.dtype == b.dtype
    assert a.data.tobytes() == b.data.tobytes()


@pytest.mark.parametrize("seed", range(200))
def test_one_pass_and_group_by_group_reductions_give_the_same_bits(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    array, grouping = _random_case(rng)
    op = str(rng.choice(OPS))
    na_rm = bool(rng.random() < 0.5)
    min_count = int(rng.integers(1, 4)) if op == "sum" and rng.random() < 0.3 else None
    with np.errstate(all="ignore"):
        fast = grouped.reduce(array, op, grouping, "point", "key", na_rm=na_rm, min_count=min_count)
        monkeypatch.setattr(grouped, "ONE_PASS", False)
        slow = grouped.reduce(array, op, grouping, "point", "key", na_rm=na_rm, min_count=min_count)
        _same_bits(fast, slow)
        for code in rng.choice(len(grouping), 5):
            alone = array.isel(point=np.flatnonzero(grouping.codes == code)).reduce(
                op, ["point"], na_rm=na_rm, min_count=min_count)
            assert fast.data[code].tobytes() == alone.data.tobytes()


def test_empty_bins_reduce_as_empty_groups(monkeypatch):
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    edges = np.concatenate([np.linspace(-3, 0, 20), np.linspace(10, 12, 5)])  # the last bins stay empty
    codes, labels = factorize(np.clip(values, -3, 0), Bins(edges))
    grouping = Grouping(codes, labels)
    array = Array(values, ("point",))
    for op in ("mean", "min", "count"):
        with np.errstate(all="ignore"):
            fast = grouped.reduce(array, op, grouping, "point", "bin")
            monkeypatch.setattr(grouped, "ONE_PASS", False)
            _same_bits(fast, grouped.reduce(array, op, grouping, "point", "bin"))
            monkeypatch.setattr(grouped, "ONE_PASS", True)
    assert grouped.counts(grouping, "bin").data[-4:].tolist() == [0, 0, 0, 0]


def test_label_arrays_are_kept_as_read_only_arrays():
    codes = np.array([1, 0, 1, 2])
    labels = np.array([10, 20, 30])
    grouping = Grouping(codes, labels)
    assert isinstance(grouping.labels, np.ndarray) and not grouping.labels.flags.writeable
    labels[0] = 99  # the caller's array is not the stored one
    assert grouping.labels.tolist() == [10, 20, 30] and len(grouping) == 3
    assert isinstance(Grouping(codes, [10, 20, 30]).labels, list)  # lists stay lists


@pytest.mark.parametrize("op", ["sum", "mean", "min", "count"])
def test_label_arrays_give_the_same_results_as_lists(op):
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 50, 400)
    array = Array(rng.normal(size=(400, 2)).astype("float32"), ("point", "band"))
    as_list = Grouping(codes, list(range(1, 51)))
    as_array = Grouping(codes, np.arange(1, 51))
    a = grouped.reduce(array, op, as_list, "point", "group")
    b = grouped.reduce(array, op, as_array, "point", "group")
    assert a.data.tobytes() == b.data.tobytes() and a.labels("group").tolist() == b.labels("group").tolist()
    assert grouped.counts(as_array, "group").labels("group").tolist() == list(range(1, 51))


@pytest.mark.parametrize("dtype, missing", [
    ("float32", "nan"), ("float16", "nan"), ("float64", "nan"), ("complex64", "nan"),
    ("datetime64[ns]", "NaT"), ("timedelta64[s]", "NaT"), ("int32", "float64"), ("bool", "float64"),
])
@pytest.mark.parametrize("op", ["min", "max"])
def test_an_empty_min_or_max_is_missing_in_the_values_own_type(dtype, missing, op):
    """A float, complex, or time type has its own missing value, so an empty group's min keeps the type: float32 stays
    float32, and times are NaT, which a float NaN cannot be stacked with. Integers and booleans have none."""
    values = (np.arange(6) if dtype.startswith(("datetime", "timedelta")) else np.arange(6) % 2).astype(dtype)
    array = Array(values, ("point",))
    nothing = array.isel(point=np.arange(0)).reduce(op, ["point"])
    want = np.dtype(np.float64) if missing == "float64" else np.dtype(dtype)
    assert nothing.data.dtype == want
    assert np.isnat(nothing.data) if missing == "NaT" else np.isnan(nothing.data)
    codes, labels = factorize(np.array([0.5, 0.5, 2.5, 2.5, 2.5, 0.5]), Bins([0, 1, 2, 3]))  # the middle bin is empty
    result = grouped.reduce(array, op, Grouping(codes, labels), "point", "bin")
    assert result.data.dtype == want
    assert (np.isnat if missing == "NaT" else np.isnan)(result.data[1])
    present = [array.isel(point=np.flatnonzero(codes == code)).reduce(op, ["point"]).data for code in (0, 2)]
    assert result.data[[0, 2]].astype(want).tolist() == np.array(present).astype(want).tolist()


@pytest.mark.parametrize("op, want", [("min", ["a", "c"]), ("max", ["b", "d"])])
def test_text_takes_min_and_max_as_objects(op, want):
    """NumPy's text types have no loop for min and max; text compares as Python objects, and the result is objects,
    which an empty group's NaN can sit among."""
    array = Array(np.array(list("bacd")), ("point",))
    whole = array.reduce(op, ["point"])
    assert whole.data.dtype == object and whole.data[()] == ("a" if op == "min" else "d")
    result = grouped.reduce(array, op, Grouping(np.array([0, 0, 1, 1]), [1, 2]), "point", "group")
    assert result.data.dtype == object and result.data.tolist() == want
    codes, labels = factorize(np.array([0.5, 0.5, 2.5, 2.5]), Bins([0, 1, 2, 3]))  # the middle bin is empty
    binned = grouped.reduce(array, op, Grouping(codes, labels), "point", "bin")
    assert binned.data.dtype == object and binned.data[0] == want[0] and binned.data[2] == want[1]
    assert binned.data[1] != binned.data[1]  # NaN


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "timedelta64[ns]"])
@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("na_rm", [False, True])
def test_a_quantile_is_numpy_s_with_its_type_and_empty_groups_missing(dtype, q, na_rm):
    rng = np.random.default_rng(2)
    values = (rng.normal(size=60) * 100).astype(dtype) if dtype != "timedelta64[ns]" else rng.integers(
        -1_000, 1_000, 60).astype(dtype)
    if dtype.startswith("float"):
        values[[3, 17]] = np.nan
    array = Array(values, ("point",))
    whole = array.reduce("quantile", ["point"], na_rm=na_rm, q=q)
    quantile = np.nanquantile if na_rm and dtype.startswith("float") else np.quantile
    want = quantile(values, q)
    assert whole.data.dtype == np.asarray(want).dtype and whole.data.tobytes() == np.asarray(want).tobytes()
    codes, labels = factorize(np.where(np.arange(60) < 30, 0.5, 2.5), Bins([0, 1, 2, 3]))  # the middle bin is empty
    result = grouped.reduce(array, "quantile", Grouping(codes, labels), "point", "bin", na_rm=na_rm, q=q)
    assert result.data.dtype == np.asarray(want).dtype
    for code, part in ((0, values[:30]), (2, values[30:])):
        assert result.data[code].tobytes() == np.asarray(quantile(part, q)).tobytes()
    empty = result.data[1]
    assert np.isnat(empty) if dtype.startswith("timedelta") else np.isnan(empty)


@pytest.mark.parametrize("q", [None, -0.1, 1.5, True])
def test_a_quantile_needs_q_between_0_and_1(q):
    with pytest.raises(ValueError, match="q between 0 and 1"):
        Array(np.arange(5.0), ("point",)).reduce("quantile", ["point"], q=q)
