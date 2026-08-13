# cuasmR (development version)

## New features

* Added `hex64_bit_get()`, `hex64_bit_set()`, `hex64_nibble_get()` and
  `hex64_nibble_set()` for editing a 64-bit hex word **as digits**. `strtoi(x,
  16L)` returns an int32, so every 64-bit SASS word — and every 32-bit half
  whose leading digit is `>= 8` — silently becomes `NA`, and
  `sprintf("%016x", NA)` turns that into the literal `"0x              NA"`, a
  plausible-width token that passes downstream length checks and is written out
  as eight zero bytes. These helpers only ever hand `strtoi` a single character,
  so the overflow is impossible by construction (issues #198, #208).

* `capture_gpu_state()` now records the platform's GPU **power envelope**:
  `power_limit_w` (the enforced limit), `power_limit_default_w`,
  `power_limit_max_w`, and the derived `power_below_default` /
  `power_below_max`. All three limits ride the `nvidia-smi` invocation the
  function already made, so the per-sample cost is unchanged. Both
  references are kept deliberately: on the project's laptop the normal
  operating point is `enforced == max` with a lower default, so comparing
  against the default alone is blind to a documented fallback regime, while
  comparing against the max alone false-positives on hardware whose normal
  point *is* the default (issue #207).
* Added `capture_power_policy()`, which reads the Windows power-mode overlay
  that moves the enforced limit. Session-scoped and memoised (it costs a
  `powershell.exe` spawn); takes the observed `ac_state` because Windows
  keeps a separate overlay per power source and they differ in practice.
  Returns `source = "unavailable"` rather than a guess off Windows, on a
  failed query, or on a value that is not GUID-shaped.

## Bug fixes

* `decode_throttle()` no longer runs the 64-bit throttle mask through int32
  `strtoi()`. A mask with a bit at or above `2^31` returned `NA`, and the `NA`
  branch returned `character(0)` — the same value that means *no throttle* — so
  a throttled run decoded as a clean one and was compared against a baseline.
  An unparseable mask now returns `NA_character_`, which `classify_meta()`
  surfaces as an unrecognised reason and rejects (issue #208).


# cuasmR 0.2.0

## New features

* Added a packaged benchmark **measurement API**, migrated out of the
  project's one-off probe/bench R scripts (issue #134, PR-A). In pipeline
  order:
  `run_bench()` -> `parse_throughput()` -> `validate_sample()` ->
  `collect_valid_samples()` -> `report_median_metrics()` ->
  `check_regression()`.
* Added JSONL store helpers `append_jsonl_row()` and `read_jsonl()`
  (tolerant per-line reader).
* Added GPU-state helpers `capture_gpu_state()`, `classify_meta()`,
  `decode_throttle()`, and `summarise_meta()`.
* The WSL `LD_LIBRARY_PATH` guard now runs in `.onLoad()` rather than at
  source time.

## Internal

* `parse_throughput()` unifies four previously divergent parsers behind a
  GPU-free differential test suite (131 assertions across the new API).

# cuasmR 0.1.0

* Initial release: R-native SASS disassembler and patcher for NVIDIA
  sm_8x cubins. `cuasm_read()`, `cuasm_kernels()`, `cuasm_insns()`,
  `cuasm_set()`, `cuasm_write()`, `cuasm_save_cuasm()`,
  `cuasm_roundtrip_check()`, and `cuasm_sections()`. Decodes SASS via
  `nvdisasm`, supports byte-level instruction/control-word edits, and
  writes byte-identical roundtrips. Tracks nvdisasm output rather than
  internal cubin layout, so it survives CUDA major-version bumps.
