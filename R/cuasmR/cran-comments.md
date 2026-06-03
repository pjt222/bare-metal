## R CMD check results

0 errors | 0 warnings | 3 notes

* This is a new submission.

* "The Title field should be in title case ... 'Sm_8x Cubins'":
  `sm_8x` is the NVIDIA compute-architecture token (as used in
  `nvcc -arch=sm_86`), not an ordinary word, and is intentionally
  lower-case.

* "checking for future file timestamps ... unable to verify current
  time" and "Files 'README.md' or 'NEWS.md' cannot be checked without
  'pandoc'": both reflect the local check environment (no network time
  source / no pandoc installed), not the package.

## Test environments

* Windows 11, R 4.5.2 (local)

## SystemRequirements

The package shells out to `nvdisasm` (NVIDIA CUDA Toolkit 12.x/13.x),
declared in `SystemRequirements`. The CUDA/GPU-dependent tests
(`test-roundtrip.R`) self-skip when the cubin fixture or `nvdisasm`
is unavailable, so the package checks cleanly on machines without a
GPU or CUDA toolkit.
