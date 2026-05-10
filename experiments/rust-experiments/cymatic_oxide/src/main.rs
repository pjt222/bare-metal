// Cymatic gather-sum kernel ported from phase4/cymatic/bench_cymatic.cu.
//
// Mirrors the C++ kernel:
//   __global__ void gather_sum(const float *data, const int *idx,
//                              float *out, int n, int iters) {
//       int tid = blockIdx.x * blockDim.x + threadIdx.x;
//       int stride = blockDim.x * gridDim.x;
//       float s = 0;
//       for (int it = 0; it < iters; ++it)
//           for (int k = tid; k < n; k += stride) s += data[idx[k]];
//       if (tid == 0) atomicAdd(out, s);
//   }
//
// In Rust slice ABI, `n` is encoded in idx.len(), so it disappears as an
// explicit parameter. iters stays explicit. This is the smallest possible
// re-encoding; any extra Rust-side work (bounds checks, slice descriptor
// arithmetic) shows up as SASS bloat to compare against nvcc.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_launch;

#[kernel]
pub fn gather_sum(
    data: &[f32],
    idx: &[i32],
    mut out: DisjointSlice<f32>,
    iters: u32,
) {
    let block_idx = thread::blockIdx_x() as usize;
    let block_dim = thread::blockDim_x() as usize;
    let thread_idx = thread::threadIdx_x() as usize;
    let grid_dim = thread::gridDim_x() as usize;

    let tid = block_idx * block_dim + thread_idx;
    let stride = block_dim * grid_dim;
    let n = idx.len();

    let mut s: f32 = 0.0;
    let mut it: u32 = 0;
    while it < iters {
        let mut k = tid;
        while k < n {
            let gather_index = idx[k] as usize;
            s += data[gather_index];
            k += stride;
        }
        it += 1;
    }

    if tid == 0 {
        let out_ptr = out.as_mut_ptr();
        let atomic_out = unsafe { &*(out_ptr as *const DeviceAtomicF32) };
        atomic_out.fetch_add(s, AtomicOrdering::Relaxed);
    }
}

fn main() {
    println!("=== Cymatic gather-sum smoke test ===");

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    const ITERS: u32 = 1;

    let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let idx_host: Vec<i32> = (0..N as i32).rev().collect(); // gather reversed
    let out_host: Vec<f32> = vec![0.0];

    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
    let idx_dev = DeviceBuffer::from_host(&stream, &idx_host).unwrap();
    let mut out_dev = DeviceBuffer::from_host(&stream, &out_host).unwrap();

    let module = ctx
        .load_module_from_file("cymatic_gather.ptx")
        .expect("PTX load");

    cuda_launch! {
        kernel: gather_sum,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(data_dev), slice(idx_dev), slice_mut(out_dev), ITERS]
    }
    .expect("launch");

    let out_back = out_dev.to_host_vec(&stream).unwrap();
    let expected: f32 = (0..N).map(|i| i as f32).sum();
    let got = out_back[0];
    println!("expected sum = {}, got = {}", expected, got);
    if (got - expected).abs() < 1.0 {
        println!("OK");
    } else {
        eprintln!("MISMATCH");
        std::process::exit(1);
    }
}
