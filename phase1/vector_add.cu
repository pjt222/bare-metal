/*
 * vector_add.cu — Phase 1 Hello World kernel
 *
 * This is intentionally the simplest possible kernel.
 * We compile it to SASS, study every instruction, then hand-modify the SASS.
 *
 * extern "C" prevents C++ name mangling so the symbol name in the cubin
 * is exactly "vector_add" — easier to reference from the host.
 *
 * Compile to cubin:
 *   nvcc --cubin -arch=sm_86 -O1 -o vector_add.sm_86.cubin vector_add.cu
 *
 * Disassemble:
 *   cuobjdump -sass vector_add.sm_86.cubin
 *
 * Or use the build script:
 *   python ../scripts/build.py all vector_add.cu
 */

extern "C" __global__ void vector_add(
    const float * __restrict__ input_a,
    const float * __restrict__ input_b,
    float * __restrict__ output_c,
    int num_elements
) {
    int element_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (element_index < num_elements) {
        output_c[element_index] = input_a[element_index] + input_b[element_index];
    }
}
