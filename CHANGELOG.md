# Changelog

## Unreleased

- An estimate's total can be marked exact: `sums.EXACT` in its `TERMS`
  piece says no addition into the total rounded, which a producer such as
  a compiled kernel knows from each addition's error term
  (`chunk_estimate` never sets it). `sums.merge(held, piece)` combines two
  sets of pieces as before and keeps the mark only where adding the two
  totals was exact too (Knuth's two-sum error). `settle` rounds a marked
  result from its total directly, ties to even, with no error bound, so the
  ties a bound cannot decide (the mean of a few float32 values that lies
  between two of them) need no exact sums. On 10M points in 533,985 voxels,
  the 21,470 means the bound left to the exact pass all settle so.

## 0.1.6 — 2026-10-28

- `grouped.merge_sorted(first, second)`: two sorted arrays of distinct
  keys as one, their union and where each one's keys stand in it, as
  `np.union1d` and `np.searchsorted` give them, in one pass: a compiled
  two-pointer merge with `labelarray[fast]`, which releases the GIL, so
  callers' threads run it side by side, else a stable sort, which merges
  the two runs it is given. On 500,000 and 400,000 keys, 2.6 ms compiled
  and 6.8 ms on NumPy, against 100.5 ms. For keeping a table of groups as
  blocks of data merge into it.
- `sums.chunk_estimate` given a code per position and several values to
  each (axes beyond the first) sums each position's values first, then the
  positions per group, without repeating the position's code for each
  value; the error bound counts the position's additions and the group's,
  so the results stay correctly rounded.

## 0.1.5 — 2026-10-27

- `quantile`: `Array.reduce("quantile", axes, q=...)` and
  `grouped.reduce(..., "quantile", ..., q=...)` give NumPy's quantile
  (its default, linear rule), `np.nanquantile` with `na_rm`, in the type
  NumPy gives (float32 stays float32, integers give float64, durations
  durations); `q` must lie between 0 and 1. Of no positions, or of an
  empty group, it is the missing value of that type, where NumPy raises.

## 0.1.4 — 2026-10-27

- `min` and `max` of no positions, or of an empty group, give the missing
  value of the values' own type: a float or complex type's NaN (float32
  stays float32), a time type's NaT, NaN among objects. Only integers and
  booleans, which have no missing value, give float64's NaN, as before. A
  float NaN beside times could not be stacked, so `min` of times grouped
  with an empty bin failed.
- Text takes `min` and `max`: NumPy's text types have no loop for them, so
  text compares as Python objects, and the result is objects, which an
  empty group's NaN can sit among.

## 0.1.3 — 2026-10-26

- `Grouping` keeps labels given as a NumPy array as one (a read-only
  copy) instead of converting them to a list of Python values, which then
  had to become an array again for every result: for 97,000 groups, about
  3 ms per summary. Other sequences still become a list.
- Sums and means of fewer than 8 float32 values per result, without
  missing values to skip, convert those few rows to float64 at once and
  add them in the same order as before, writing the last addition straight
  into the float32 result: the same bits, about 10 µs less for three rows
  of 34,688 values.
- Tests: no warnings from combining chunks that hold infinities of both
  signs; `itertools.pairwise` in the chunked tests.

## 0.1.2 — 2026-10-26

- Float32 sums and means of data held in chunks (Dask) can be correctly
  rounded too, the same bits as `sums.reduce` gives in memory whatever the
  chunking. `sums.chunk_estimate` gives one chunk's float64 totals, sums of
  magnitudes, counts (values, values present, +inf, -inf), and the term count
  of its error bound, which combine across chunks in any order; `settle`
  certifies the combined estimate; `chunk_exact` gives one chunk's exact
  sums for the groups the bound could not settle, as int64 buckets per
  float32 exponent, which add across chunks; `exact_results` rounds them;
  `from_chunks` drives it all given a function returning the exact sums.
  Chunks take the NumPy estimate, since they already run in parallel.
- Exact sums are measured from each group's own lowest nonzero exponent, so
  a group whose values span few exponents fits in 64-bit integers whatever
  its magnitude; a group like `[2**100, 1, -2**100]` no longer goes through
  Python fractions. Results are unchanged, bit for bit.

## 0.1.1 — 2026-10-26

- Grouped float32 sums and means over many small groups are much faster:
  200,000 values in 96,846 groups (voxels holding one to a few points) now
  take 6 ms instead of 111. The exact mean of a few float32 values often
  lies exactly halfway between two float32 values, where the error bound
  cannot decide the rounding; about one group in ten went to an exact
  path computed in Python fractions. Groups whose values span a narrow
  range of exponents now have their exact sums computed as 64-bit integers
  and rounded with NumPy (integer division kept to 27 or 28 bits with the
  last bit set when anything is left over, then one rounding to float32),
  so the Python path is left for groups spanning a wide range. Results are
  unchanged, bit for bit: the tests compare them with exact rational
  arithmetic, and the new code gave the same bits as 0.1.0 on two million
  results.

## 0.1.0 — 2026-10-25

The first release on its own. labelarray was developed as the array layer
of tidy3a (`tidy3a.labelarray`) and leaves it unchanged in behaviour:

- `Array`: a NumPy array with named axes, exact labels, attributes, and a
  name. Selection by positions and labels (`isel`, `sel`), `transpose`,
  `rename`, reductions (`reduce`), and contraction (`dot`). NumPy's ufuncs,
  `np.where`, and Python's operators broadcast by axis name under the exact
  alignment rule.
- `Collection`: arrays sharing axes, one label array per axis, with
  selection, assignment, joins (`join`), nearest-label matching
  (`nearest`), and conversion to and from xarray.
- The exact label rule (`labels`) and label indexes: sorted, sort-once,
  and grid indexes, chosen by `index_for`, with `register_index` for other
  label types.
- Grouping: `Grouping`, `factorize`, the key transforms `Bins`,
  `Duration`, `Round`, and `GridCells`, and grouped reductions
  (`grouped.reduce`, `counts`, `broadcast`) bit-identical to reducing each
  group on its own.
- Float32 sums and means correctly rounded in any memory layout (`sums`),
  with an optional compiled path (`labelarray[fast]`) giving the same bits.
- One family of errors under `LabelArrayError`.
