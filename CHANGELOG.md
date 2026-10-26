# Changelog

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
