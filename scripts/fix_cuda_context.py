import re
import sys
import glob
import os

def fix_cu_context(path):
    with open(path) as f:
        content = f.read()
    
    # Patch: skip files already using cuDevicePrimaryCtxRetain
    if 'cuDevicePrimaryCtxRetain' in content:
        return None  # already fixed

    # Pattern 1: CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    m = re.search(r'CHECK_CU\(cuCtxCreate\(&(\w+),\s*0,\s*(\w+)\)\);', content)
    if not m:
        # Pattern 2: CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, cu_dev));
        m = re.search(r'CUcontext\s+(\w+);\s*CHECK_CU\(cuCtxCreate\(&\1,\s*0,\s*(\w+)\)\);', content)
    if not m:
        return None

    ctx_var, dev_var = m.groups()

    # Replace create
    old_create = m.group(0)
    # Build replacement that keeps CUcontext decl if present, replaces just the create call
    if old_create.startswith('CUcontext'):
        new_create = old_create.replace(
            f'CHECK_CU(cuCtxCreate(&{ctx_var}, 0, {dev_var}));',
            f'CHECK_CU(cuDevicePrimaryCtxRetain(&{ctx_var}, {dev_var}));\n    CHECK_CU(cuCtxSetCurrent({ctx_var}));'
        )
    else:
        new_create = old_create.replace(
            f'CHECK_CU(cuCtxCreate(&{ctx_var}, 0, {dev_var}));',
            f'CHECK_CU(cuDevicePrimaryCtxRetain(&{ctx_var}, {dev_var}));\n    CHECK_CU(cuCtxSetCurrent({ctx_var}));'
        )
    content = content.replace(old_create, new_create, 1)

    # Replace destroy (all occurrences)
    content = content.replace(f'cuCtxDestroy({ctx_var});', f'cuDevicePrimaryCtxRelease({dev_var});')

    with open(path, 'w') as f:
        f.write(content)
    return (ctx_var, dev_var)

if __name__ == '__main__':
    files = sorted(glob.glob('phase*/**/bench.cu', recursive=True) +
                   glob.glob('phase*/**/bench_*.cu', recursive=True))
    for f in files:
        if 'bench_refactored' in f or '.bak' in f:
            continue
        result = fix_cu_context(f)
        if result:
            print(f'Fixed: {f} (ctx={result[0]}, dev={result[1]})')
