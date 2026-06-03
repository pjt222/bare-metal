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
