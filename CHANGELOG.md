# Changelog

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
