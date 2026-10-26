"""Float32 sums and means: float64 accumulation, rounded once to float32.

Two ways, chosen by how many values each result adds:

- Every grouped reduction, and every ungrouped one adding ``FEW`` (8) or more
  values per result: the result is correctly rounded, the float32 nearest
  the exact sum or mean of the stored values, ties to even. A float64 sum
  and an error bound settle almost every result; where the bound cannot,
  the values are added exactly (``_exact``). A correctly rounded result is
  unique, so every memory layout, summation order, kernel (NumPy or the
  optional compiled one), thread count, and backend gives the same bits.
- Ungrouped reductions adding fewer than ``FEW`` values per result, such as
  ``sum(p**2, over="i")`` over x, y, and z: the values are added in float64
  in a fixed order, first to last along the reduced axes, and rounded once.
  The order does not depend on memory layout; the compiled kernel, when
  installed, adds in the same order in one pass. Checking each of these results
  would cost several times the sum itself (the float64 sum of a few float32
  values is often exact and exactly halfway between two float32 values, where
  a bound cannot decide), so cancellation among those few values is rounded
  in float64 as it goes: ``[2**53, 1, -2**53]`` sums to 0 here, not 1.

Data held in chunks (Dask) gets the same bits as data in memory, whatever the
chunking: ``chunk_estimate`` gives each chunk's float64 totals, sums of
magnitudes, counts, and the term count of its error bound, which combine
across chunks in any order; ``settle`` certifies the combined estimate;
``chunk_exact`` gives each chunk's exact sums, as integer buckets per
exponent, for the groups the bound could not settle, which add across chunks;
and ``from_chunks`` drives the two passes. A correctly rounded result is
unique, so these give the bits ``reduce`` gives.

Missing values: without ``na_rm`` a NaN makes its result NaN; with it, NaN
is skipped and a mean divides by the values present. +inf and -inf together
give NaN. A sum of no values is +0.0 and a mean NaN; a zero sum is +0.0, as
NumPy gives it. ``min_count`` makes a sum NaN where fewer values are present.
This module needs NumPy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

FEW = 8
COMPILED = True  # use the optional compiled kernels when Numba is installed
# From these many values; below them, calling compiled code costs more than it saves. Grouped sums: 34,688
# values took 201 µs compiled against 155, 300,000 took 322 against 1,231. Few-value sums: 180,000 values
# 89 against 78 µs, 300,000 103 against 118. Both paths give the same bits.
COMPILED_FROM = 100_000
IN_ORDER_COMPILED_FROM = 300_000

_U = 2.0**-53  # float64 unit roundoff
# Sums are made in chunks of positions, then the chunk totals added, so a result's error bound grows
# with the chunk length plus the number of chunks rather than with every position: about sqrt(n) chunks
# of sqrt(n) positions, with at most this many (chunk, group) bins at once.
_BINS = 1 << 22
_MAX32 = Fraction(float(np.finfo(np.float32).max))


def reduce(values: np.ndarray, op: str, axes: Sequence[int], *, na_rm: bool = False,
           min_count: int | None = None, codes: np.ndarray | None = None, groups: int = 0,
           sizes: np.ndarray | None = None) -> np.ndarray:
    """*op* (``"sum"`` or ``"mean"``) of float32 *values* over the axis numbers *axes*.

    Without *codes*, the result has the kept axes in their order. With
    *codes* (one group code from 0 to ``groups - 1`` per position along
    ``axes[0]``), it has one row per group first: ``(groups, *kept)``;
    *sizes*, positions per group along ``axes[0]``, may be passed when known.
    """
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError(f"float32 values expected, got {values.dtype}")
    axes = tuple(int(a) for a in axes)
    kept = tuple(a for a in range(values.ndim) if a not in axes)
    kept_shape = tuple(values.shape[a] for a in kept)
    positions = math.prod(values.shape[a] for a in axes)
    with np.errstate(all="ignore"):
        if codes is None and positions < FEW:
            compiled = (_compiled("in_order") if COMPILED and len(axes) == 1 and values.ndim <= 2
                        and values.size >= IN_ORDER_COMPILED_FROM else None)
            if compiled is not None:
                # A view: one reduced axis, at most one kept; the width is explicit, since -1 is ambiguous when empty.
                table = np.moveaxis(values, axes[0], 0).reshape(positions, math.prod(kept_shape))
                result, present = compiled(table, op == "mean", na_rm)
                if op == "sum" and min_count is not None:
                    result = np.where(present >= min_count, result, np.float32(np.nan))
                return result.reshape(kept_shape)
            return _in_order(values, axes, kept_shape, op, na_rm, min_count)
        table, rows, others = _table(values, axes, kept, codes)
        if codes is not None:
            if sizes is not None and others != 1:
                sizes = np.asarray(sizes) * others
            result = _settled(table, rows, groups, sizes, op, na_rm, min_count)
            return result.reshape((groups, *kept_shape))
        return _settled(table, None, 1, None, op, na_rm, min_count).reshape(kept_shape)


def _table(values, axes, kept, codes):
    """*values* as (positions along *axes*, positions along *kept*), each position's group code, and positions per code."""
    positions = math.prod(values.shape[a] for a in axes)
    width = math.prod(values.shape[a] for a in kept)
    table = np.transpose(values, axes + kept).reshape(positions, width)
    if codes is None:
        return table, None, 1
    others = math.prod(values.shape[a] for a in axes[1:])
    rows = np.asarray(codes) if others == 1 else np.repeat(np.asarray(codes), others)
    return table, rows, others


def _in_order(values, axes, kept_shape, op, na_rm, min_count) -> np.ndarray:
    """Few values per result: float64, added first to last along *axes*, rounded once.

    Each position along the reduced axes is a view of the input, added in
    place, so no copy is made whatever the layout; missing values are
    counted only when ``na_rm`` or ``min_count`` needs them.
    """
    moved = np.moveaxis(values, axes, tuple(range(len(axes))))
    total = np.zeros(kept_shape)
    counting = na_rm or (op == "sum" and min_count is not None)
    present = np.zeros(kept_shape, dtype=np.intp) if counting else None
    for index in np.ndindex(moved.shape[:len(axes)]):
        row = moved[index]
        if counting:
            missing = np.isnan(row)
            present += ~missing
            if na_rm:
                row = np.where(missing, np.float32(0), row)
        np.add(total, row, out=total)
    terms = math.prod(values.shape[a] for a in axes)
    return _finish(total, present, terms, op, na_rm, min_count)


def _finish(total, present, terms, op, na_rm, min_count) -> np.ndarray:
    if op == "mean":
        np.divide(total, present if na_rm else terms, out=total)
        return total.astype(np.float32)
    result = total.astype(np.float32)
    return np.where(present >= min_count, result, np.float32(np.nan)) if min_count is not None else result


# ── Correctly rounded: an estimate, a bound, and exact sums where needed ────


def _settled(table, rows, groups, sizes, op, na_rm, min_count) -> np.ndarray:
    """Correctly rounded results, one row per group (one row in all when *rows* is None)."""
    total, magnitude, present, values, terms, compensated = _estimate(table, rows, groups, sizes, na_rm)
    counts = present if na_rm else values
    result, sure = _certify(total, magnitude, terms, counts, op, compensated)
    unsure = ~sure
    for column in np.flatnonzero(unsure.any(axis=0)):
        chosen = np.flatnonzero(unsure[:, column])
        result[chosen, column] = _exact(table[:, column], rows, chosen, groups, op, na_rm)
    if op == "sum" and min_count is not None:
        result = np.where(present >= min_count, result, np.float32(np.nan))
    return result


def _estimate(table, rows, groups, sizes, na_rm, compiled=True):
    """Float64 sums, sums of magnitudes, values present, values, and terms for the bound, per group and column.

    The NumPy path adds chunks of positions (one ``np.bincount`` over
    (chunk, group) bins, or ``np.add.reduceat`` without groups) and then the
    chunk totals, so a result's error grows with the chunk length plus the
    number of chunks; "terms" is that effective count, for the bound only.
    """
    positions, width = table.shape
    chunks = max(1, min(math.isqrt(positions), _BINS // max(groups, 1)))
    length = -(-positions // chunks) if positions else 0
    if rows is None:
        missing = np.isnan(table)
        weights = np.where(missing, np.float32(0), table) if na_rm else table
        if positions:
            starts = np.arange(0, positions, length)
            total = np.add.reduceat(weights, starts, axis=0, dtype=np.float64).sum(axis=0)[None]
            magnitude = np.add.reduceat(np.abs(weights), starts, axis=0, dtype=np.float64).sum(axis=0)[None]
        else:
            total = magnitude = np.zeros((1, width))
        present = (positions - missing.sum(axis=0))[None]
        terms = np.full((1, width), min(positions, length + chunks))
        return total, magnitude, present, np.full((1, width), positions), terms, False
    if sizes is None:
        sizes = np.bincount(rows, minlength=groups)
    sizes = np.asarray(sizes)
    kernel = _compiled("grouped_estimate") if compiled and COMPILED and table.size >= COMPILED_FROM else None
    if kernel is not None:
        total, magnitude, present = kernel(table, np.ascontiguousarray(rows, dtype=np.intp), groups, na_rm)
        values = np.repeat(sizes[:, None], width, axis=1)
        return total, magnitude, present, values, values, True
    effective = np.minimum(sizes, length) + np.minimum(sizes, chunks)  # additions within a chunk, then of chunk totals
    terms = np.repeat(np.minimum(sizes, effective)[:, None], width, axis=1)
    key = rows if chunks == 1 else (np.arange(positions) // length) * groups + rows
    bins = chunks * groups
    total = np.empty((groups, width))
    magnitude = np.empty((groups, width))
    present = np.empty((groups, width), dtype=np.intp)
    for column in range(width):
        values = table[:, column]
        missing = np.isnan(values)
        anything_missing = bool(missing.any())
        weights = np.where(missing, np.float32(0), values) if na_rm and anything_missing else values
        total[:, column] = np.bincount(key, weights=weights, minlength=bins).reshape(chunks, groups).sum(axis=0)
        magnitude[:, column] = np.bincount(key, weights=np.abs(weights), minlength=bins).reshape(chunks, groups).sum(axis=0)
        present[:, column] = (sizes - np.bincount(rows, weights=missing, minlength=groups).astype(np.intp)
                              if anything_missing else sizes)
    return total, magnitude, present, np.repeat(sizes[:, None], width, axis=1), terms, False


def _certify(total, magnitude, terms, counts, op, compensated):
    """Results where the error bound proves the rounding, and where it does.

    A float64 sum of n values, in any order, is within gamma(n) * sum|x| of
    the exact sum, gamma(n) = n u / (1 - n u); a compensated one within
    (6u + 4 (n u)^2) * sum|x|. Where both ends of the interval round to the
    same float32, the exact value rounds there too.
    """
    n = terms.astype(np.float64)
    gamma = n * _U / (1 - n * _U)
    size = magnitude / (1 - gamma)  # the computed magnitude may itself be low by that factor
    bound = (6 * _U + 4 * (n * _U) ** 2) * size if compensated else gamma * size
    bound = bound * (1 + 8 * _U) + np.abs(total) * (2 * _U) + 1e-300  # the bound's and the ends' own rounding
    if op == "mean":
        centre = total / counts
        bound = bound / counts * (1 + 4 * _U) + np.abs(centre) * (4 * _U)
    else:
        centre = total
    low, high = (centre - bound).astype(np.float32), (centre + bound).astype(np.float32)
    sure = (low == high) & np.isfinite(bound) & np.isfinite(centre)
    if op == "mean":
        sure &= counts > 0
    return high, sure


def _exact(values, rows, chosen, groups, op, na_rm) -> np.ndarray:
    """Correctly rounded results for groups *chosen*, from exact sums of their values.

    Most such groups are small and hold values of nearby magnitudes (their
    exact mean lies on a float32 tie, where the bound cannot decide): their
    exact sums fit in 64-bit integers and are rounded with NumPy
    (``_rounded``). Groups spanning a wider range of exponents are summed and
    rounded in Python integers and fractions.
    """
    if rows is None:
        part, codes, count = values, np.zeros(len(values), dtype=np.intp), 1
    else:
        remap = np.full(groups, -1, dtype=np.intp)
        remap[chosen] = np.arange(len(chosen))
        codes = remap[rows]
        keep = codes >= 0
        part, codes, count = values[keep], codes[keep], len(chosen)
    buckets, low, present, nan, plus, minus = _exact_sums(part, codes, count)
    return _exact_finish(buckets, low, present, nan, plus, minus, op, na_rm)


def _exact_finish(buckets, low, present, nan, plus, minus, op, na_rm) -> np.ndarray:
    """Correctly rounded results from exact sums: *buckets* ``(count, width)`` per group and exponent from *low*."""
    count = len(buckets)
    out = np.full(count, np.nan, dtype=np.float32)
    undefined = (nan & (not na_rm)) | (plus & minus)
    infinite = (plus | minus) & ~undefined
    out[infinite] = np.where(plus[infinite], np.float32(np.inf), np.float32(-np.inf))
    regular = ~(undefined | infinite)
    if op == "mean":
        regular &= present > 0  # a mean of no values stays NaN
    divisor = present if op == "mean" else np.ones(count, dtype=np.int64)
    whole, first, fits = _whole(buckets)
    fast = regular & fits & (divisor < 2**35)  # so the integer quotient below stays within int64
    if fast.any():
        out[fast] = _rounded(whole[fast], low + first[fast] - 150, divisor[fast])
    for k in np.flatnonzero(regular & ~fits):
        row = buckets[k]
        total = sum(int(row[j]) << int(j) for j in np.flatnonzero(row)) * Fraction(2) ** (low - 150)
        out[k] = _round32(total / int(divisor[k]))
    return out


def _exact_sums(values, codes, count):
    """Exact sums as integer buckets per group and exponent, the lowest exponent, values present, and specials.

    A float32 is M * 2**(e - 150) with M a signed integer below 2**24 and e
    from 1 to 254. Integers M with the same e add exactly in float64 for
    fewer than 2**29 values, so they are summed per group and exponent with
    ``np.bincount``: group k's exact sum is the sum over j of
    ``buckets[k, j] * 2**(low + j - 150)``. Also, per group, the values
    present (not NaN), and whether a NaN, +inf, or -inf is among them.
    """
    bits = values.view(np.uint32)
    exponent = ((bits >> 23) & 0xFF).astype(np.intp)
    nan = np.isnan(values)
    infinite = np.isinf(values)
    finite = ~(nan | infinite)
    mantissa = (bits & 0x7FFFFF).astype(np.int64) | np.where(exponent > 0, 1 << 23, 0)
    mantissa = np.where(bits >> 31, -mantissa, mantissa)
    exponent = np.maximum(exponent, 1)
    present = np.bincount(codes, minlength=count) - np.bincount(codes, weights=nan, minlength=count).astype(np.intp)
    has_nan = np.bincount(codes, weights=nan, minlength=count) > 0
    plus = np.bincount(codes, weights=values == np.inf, minlength=count) > 0
    minus = np.bincount(codes, weights=values == -np.inf, minlength=count) > 0
    low, width = 1, 1
    if finite.any():
        low, high = int(exponent[finite].min()), int(exponent[finite].max())
        width = high - low + 1
    buckets = np.zeros(count * width, dtype=np.int64)
    if finite.any():
        key = codes[finite] * width + (exponent[finite] - low)
        weights = mantissa[finite].astype(np.float64)
        chunk = 1 << 28  # each bucket sum stays below 2**52, so float64 holds it exactly
        for start in range(0, len(key), chunk):
            buckets += np.bincount(key[start:start + chunk], weights=weights[start:start + chunk],
                                   minlength=count * width).astype(np.int64)
    return buckets.reshape(count, width), low, present, has_nan, plus, minus


def _whole(buckets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each group's exact sum as one integer, in units of its lowest nonzero bucket; that bucket; where the integer is below 2**52.

    Where it is, the shifted buckets and their sum stay within int64 and the
    sum is exact in float64 too; other groups need Python integers. Measured
    from each group's own lowest bucket, a group of any magnitude fits when
    its values span few exponents.
    """
    count, width = buckets.shape
    nonzero = buckets != 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), 0)
    place = first[:, None] + np.arange(width)
    aligned = np.where(place < width, np.take_along_axis(buckets, np.minimum(place, width - 1), axis=1), 0)
    shifts = np.arange(width)
    bound = (np.abs(aligned).astype(np.float64) * np.ldexp(1.0, shifts)).sum(axis=1)  # exact terms; rounding is small
    fits = bound < 2.0**52
    whole = np.zeros(count, dtype=np.int64)
    if fits.any():
        # Shifts of zero buckets beyond 62 bits are clamped; a nonzero bucket that far up does not fit.
        whole[fits] = np.left_shift(aligned[fits], np.minimum(shifts, 62)).sum(axis=1)
    return whole, first, fits


def _rounded(whole: np.ndarray, scale, divisor: np.ndarray) -> np.ndarray:
    """``whole * 2**scale / divisor`` correctly rounded to float32, ties to even, for |whole| below 2**52.

    The quotient is taken in integers to 27 or 28 significant bits, with its
    last bit set when anything was left over ("round to odd"); with at least
    two bits more than float32 keeps, the one rounding from float64 to
    float32 then rounds as the exact quotient would, subnormals included.
    """
    sign = np.where(whole < 0, -1.0, 1.0)
    size = np.abs(whole)
    divisor = divisor.astype(np.int64)
    zero = size == 0
    size = np.where(zero, 1, size)  # placeholders; zero sums are +0.0, as NumPy gives them
    size_bits = np.frexp(size.astype(np.float64))[1].astype(np.int64)  # bit lengths, exact below 2**53
    divisor_bits = np.frexp(divisor.astype(np.float64))[1].astype(np.int64)
    up = 27 + divisor_bits - size_bits  # shift so the quotient has 27 or 28 bits
    quotient = np.empty(len(size), dtype=np.int64)
    rest = np.empty(len(size), dtype=bool)
    left = up >= 0
    if left.any():  # size << up has 27 or 28 more bits than the divisor: below 2**63 for divisors below 2**35
        numerator = np.left_shift(size[left], up[left])
        quotient[left], remainder = np.divmod(numerator, divisor[left])
        rest[left] = remainder != 0
    right = ~left
    if right.any():
        whole_part, remainder = np.divmod(size[right], divisor[right])
        down = -up[right]
        quotient[right] = np.right_shift(whole_part, down)
        rest[right] = (remainder != 0) | ((whole_part & ((np.int64(1) << down) - 1)) != 0)
    quotient |= rest.astype(np.int64)
    result = np.ldexp(sign * quotient.astype(np.float64), scale - up).astype(np.float32)
    result[zero] = np.float32(0.0)
    return result


def _round32(value: Fraction) -> np.float32:
    """*value* correctly rounded to float32, ties to even."""
    if value == 0:
        return np.float32(0.0)
    if abs(value) >= _MAX32 + Fraction(2) ** 103:  # half a step above the largest float32
        return np.float32(math.inf if value > 0 else -math.inf)
    guess = np.float32(float(value))  # float() rounds correctly; the cast can round a second time
    best = guess
    for candidate in (np.nextafter(guess, np.float32(-np.inf)), np.nextafter(guess, np.float32(np.inf))):
        if not np.isfinite(candidate):
            continue
        gap, best_gap = abs(Fraction(float(candidate)) - value), abs(Fraction(float(best)) - value)
        if gap < best_gap or (gap == best_gap and int(candidate.view(np.uint32)) % 2 == 0):
            best = candidate
    return np.float32(best)


# ── Chunked data: pieces that combine in any order ──────────────────────────

PIECES = 7
TOTAL, MAGNITUDE, PRESENT, VALUES, PLUS, MINUS, TERMS = range(PIECES)
BUCKETS = 254 + 4  # one per float32 exponent, then values present, NaN, +inf, and -inf
EXACT_PAIRS = 1 << 14  # (group, column) pairs per exact pass: 33 MB of buckets per chunk


def chunk_estimate(values: np.ndarray, axes: Sequence[int], *, na_rm: bool = False,
                   codes: np.ndarray | None = None, groups: int = 0) -> np.ndarray:
    """One chunk's estimate of its float32 sums over *axes*: ``(PIECES, groups, *kept)`` in float64.

    The pieces, per group (one without *codes*) and kept position: the
    float64 total, the sum of magnitudes, the values present (not NaN), the
    values, the +inf and the -inf among them, and the term count of the
    total's error bound. Chunks combine in any order: every piece adds,
    except ``TERMS``, whose maximum is taken (``settle`` adds the number of
    chunks to it: Higham's lemma 3.3, where a float64 sum of totals each
    within gamma(n_i) of exact is within gamma(max n_i + k) of exact). The
    integers stay exact in float64 below 2**53 values. The NumPy path
    estimates every chunk: chunks run in parallel already, and the compiled
    kernel's own threads cannot be entered from several at once.
    """
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError(f"float32 values expected, got {values.dtype}")
    axes = tuple(int(a) for a in axes)
    kept = tuple(a for a in range(values.ndim) if a not in axes)
    kept_shape = tuple(values.shape[a] for a in kept)
    count = groups if codes is not None else 1
    with np.errstate(all="ignore"):
        table, rows, _ = _table(values, axes, kept, codes)
        total, magnitude, present, counted, terms, _ = _estimate(table, rows, count, None, na_rm, compiled=False)
        width = table.shape[1]
        plus = np.zeros((count, width))
        minus = np.zeros((count, width))
        if np.isinf(table).any():
            key = ((np.zeros(len(table), dtype=np.intp) if rows is None else rows)[:, None] * width
                   + np.arange(width)).ravel()
            plus = np.bincount(key, weights=(table == np.inf).ravel(), minlength=count * width).reshape(count, width)
            minus = np.bincount(key, weights=(table == -np.inf).ravel(), minlength=count * width).reshape(count, width)
    pieces = np.stack([total, magnitude, present, counted, plus, minus, terms]).astype(np.float64)
    return pieces.reshape((PIECES, count, *kept_shape))


def settle(pieces: np.ndarray, chunks: int, op: str, na_rm: bool) -> tuple[np.ndarray, np.ndarray]:
    """Results the combined *pieces* of *chunks* chunks settle, and where they do not: ``(result, unsure)``.

    Where the error bound proves the rounding, the result is the correctly
    rounded one; NaN, +inf and -inf follow the missing-value rules from the
    counts; a mean of no values is NaN. The rest, ``unsure``, need exact sums.
    """
    with np.errstate(all="ignore"):
        total, magnitude, present, counted, plus, minus, terms = pieces
        counts = present if na_rm else counted
        result, sure = _certify(total, magnitude, terms + chunks, counts, op, False)
        undefined = ((counted - present > 0) & (not na_rm)) | ((plus > 0) & (minus > 0))
        infinite = ((plus > 0) | (minus > 0)) & ~undefined
        result[undefined] = np.nan
        result[infinite] = np.where(plus[infinite] > 0, np.float32(np.inf), np.float32(-np.inf))
        sure |= undefined | infinite
        if op == "mean":
            empty = (counts == 0) & ~undefined
            result[empty] = np.nan
            sure |= empty
    return result, ~sure


def chunk_exact(values: np.ndarray, axes: Sequence[int], chosen: Sequence[int], *,
                codes: np.ndarray | None = None, groups: int = 0) -> np.ndarray:
    """One chunk's exact sums over *axes* for the groups *chosen*: ``(len(chosen), *kept, BUCKETS)`` in int64.

    Per chosen group and kept position: the exact sum as integer buckets, one
    per float32 exponent (``_exact_sums``), then the values present and the
    NaN, +inf and -inf among them. Chunks add. Without *codes*, *chosen* is
    ``[0]``. A bucket stays within int64 below 2**39 values of one exponent.
    """
    values = np.asarray(values)
    if values.dtype != np.float32:
        raise TypeError(f"float32 values expected, got {values.dtype}")
    axes = tuple(int(a) for a in axes)
    kept = tuple(a for a in range(values.ndim) if a not in axes)
    kept_shape = tuple(values.shape[a] for a in kept)
    chosen = np.asarray(chosen, dtype=np.intp)
    count = len(chosen)
    with np.errstate(all="ignore"):
        table, rows, _ = _table(values, axes, kept, codes)
        positions, width = table.shape
        if rows is None:
            part, pair_codes = table, np.zeros(positions, dtype=np.intp)
        else:
            remap = np.full(max(groups, 1), -1, dtype=np.intp)
            remap[chosen] = np.arange(count)
            pair_codes = remap[rows]
            keep = pair_codes >= 0
            part, pair_codes = table[keep], pair_codes[keep]
        pairs = (pair_codes[:, None] * width + np.arange(width)).ravel()
        buckets, low, present, nan, plus, minus = _exact_sums(np.ascontiguousarray(part).ravel(), pairs, count * width)
        out = np.zeros((count * width, BUCKETS), dtype=np.int64)
        out[:, low - 1:low - 1 + buckets.shape[1]] = buckets
        out[:, 254] = present
        out[:, 255] = nan
        out[:, 256] = plus
        out[:, 257] = minus
    return out.reshape((count, *kept_shape, BUCKETS))


def exact_results(buckets: np.ndarray, op: str, na_rm: bool) -> np.ndarray:
    """Correctly rounded results from exact sums added across chunks: ``(..., BUCKETS)`` in, float32 out."""
    buckets = np.asarray(buckets)
    shape = buckets.shape[:-1]
    flat = buckets.reshape(-1, BUCKETS)
    with np.errstate(all="ignore"):
        out = _exact_finish(flat[:, :254], 1, flat[:, 254], flat[:, 255] > 0, flat[:, 256] > 0, flat[:, 257] > 0,
                            op, na_rm)
    return out.reshape(shape)


def from_chunks(pieces: np.ndarray, chunks: int, op: str, exact, *, na_rm: bool = False,
                min_count: int | None = None) -> np.ndarray:
    """*op* from the combined *pieces* of *chunks* chunks, with exact sums where the bound cannot settle a result.

    *exact(chosen)* returns the chunks' exact sums for the groups *chosen*
    (``chunk_exact``, added across chunks); it is called once per batch of
    ``EXACT_PAIRS`` (group, kept position) pairs, and not at all where the
    bound settles everything. The result is ``(groups, *kept)`` in float32,
    the same bits as ``reduce`` gives in memory.
    """
    pieces = np.asarray(pieces, dtype=np.float64)
    result, unsure = settle(pieces, chunks, op, na_rm)
    count = len(result)
    width = result.size // count if count else 0
    rows = np.flatnonzero(unsure.reshape(count, width).any(axis=1)) if count else np.zeros(0, dtype=np.intp)
    batch = max(1, EXACT_PAIRS // max(width, 1))
    for start in range(0, len(rows), batch):
        chosen = rows[start:start + batch]
        done = exact_results(exact(chosen), op, na_rm)
        picked, mask = result[chosen], unsure[chosen]
        picked[mask] = done[mask]
        result[chosen] = picked
    if op == "sum" and min_count is not None:
        result = np.where(pieces[PRESENT] >= min_count, result, np.float32(np.nan))
    return result


# ── The optional compiled kernel ────────────────────────────────────────────

_KERNELS: dict = {}


def _compiled(name: str):
    """A compiled kernel from ``_compiled`` when Numba is installed (``labelarray[fast]``), else None."""
    if name not in _KERNELS:
        try:
            from labelarray import _compiled as module
        except ImportError:
            _KERNELS[name] = None
        else:
            _KERNELS[name] = getattr(module, name)
    return _KERNELS[name]
