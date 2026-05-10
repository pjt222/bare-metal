# Polyhedral spring networks — survey & GPU-acceleration notes

> Issue #32 deliverable. Literature scoping for whether polyhedral
> spring-network simulation is a viable phase-6 target for the
> bare-metal CUDA kernel project.

## What a polyhedral spring network is

A graph `G = (V, E)` where:

- `V` = vertices of a polyhedron (point masses with position `x_i ∈ ℝ³`,
  velocity `v_i ∈ ℝ³`, mass `m_i`)
- `E` = edges of the polyhedron (springs with rest length `ℓ_e`,
  stiffness `k_e`, optional damping `c_e`)

The dynamics follow a mass-spring-damper system:

```
F_i = Σ_{e ∈ N(i)} -k_e (||x_i - x_j|| - ℓ_e) (x_i - x_j) / ||x_i - x_j||
              - Σ c_e ((v_i - v_j) · ê_ij) ê_ij
m_i ẍ_i = F_i + F_external
```

Equilibrium configurations minimize the total elastic potential
`U = Σ_e (1/2) k_e (||x_i - x_j|| - ℓ_e)²`.

**Tensegrity** is a constrained variant: edges split into "cables"
(positive-only force, can only pull) and "struts" (compression-only).
A tensegrity polyhedron's equilibrium is a self-stressed configuration
where the constraints are simultaneously satisfied without external
support.

## Mathematical foundations

### Energy minimization

For a fixed graph, finding equilibrium positions `x* = argmin U(x)` is:

- **Convex** if all springs are at-or-below rest length and the graph
  has full edge multiplicity (rare in practice).
- **Non-convex** in the general case: multiple local minima, the
  attractor of which depends on the initial state.

Standard solvers:

- **Gradient descent / L-BFGS** for static equilibrium (no inertia).
- **Verlet / leapfrog integration** for dynamic relaxation
  (introduces damping → equilibrium is the long-time limit).
- **Implicit Euler** for stiff systems (large `k_e` / large timesteps);
  requires solving a sparse linear system per step:
  `(M + Δt² K) Δv = Δt F`, where `K = ∂F/∂x` is the stiffness matrix.

### Stiffness matrix structure

`K` is sparse with the same nonzero pattern as `E`. For a polyhedron
with `|V| = n`, `K ∈ ℝ^{3n × 3n}` has `O(|E|)` nonzero blocks. For an
icosahedron (`|V|=12, |E|=30`), `K` is 36×36 with 90 nonzero 3×3
blocks.

The mass matrix `M` is block-diagonal (per-vertex 3×3 = `m_i · I_3`).

The conjugate-gradient (CG) solver dominates implicit-Euler timestep
cost. Per CG iteration: one sparse matrix-vector product (SpMV) of
`(M + Δt² K)`, plus dot products and AXPYs.

### Symmetry exploitation

Polyhedra often have point-group symmetry (icosahedron has 60-fold
icosahedral symmetry `I_h`; cube has `O_h`). Symmetric initial
conditions produce symmetric equilibria; the system can be solved on
the orbit space (one representative vertex per symmetry class) at
much smaller dimension. This is the analog of "lifted to symmetry-
reduced coordinates" in molecular dynamics.

## Simulation methods

### Explicit integration (Verlet / leapfrog)

Simple, stable for moderate stiffness. Update per timestep:

```
v_i(t + Δt/2) = v_i(t - Δt/2) + Δt · F_i(x(t)) / m_i
x_i(t + Δt)   = x_i(t)         + Δt · v_i(t + Δt/2)
```

Force computation is the only nontrivial kernel: for each edge,
compute the spring force and scatter to both endpoints. This is a
**bipartite scatter** pattern — edge-parallel, vertex-atomic-add.

### Implicit Euler (semi-implicit Euler / Newmark-β)

Required for stiff systems (large `k_e`, large `Δt`). Per timestep,
solve `A x = b` with `A = M + Δt² K`. CG iterations dominate.

### Position-Based Dynamics (PBD)

Approximate constraint solver: project positions back onto rest-length
manifold via Gauss-Seidel sweeps. Robust, parallel-friendly, but
energy-non-conservative.

## GPU implementation patterns

### Force kernel — three options

  1. **Edge-centric, atomic scatter**: one thread per edge computes
     the spring force, atomic-adds to both endpoints. Scales linearly
     with `|E|`. Simple but atomic contention bottleneck on highly-
     connected vertices.

  2. **Vertex-centric, gather**: one thread per vertex iterates over
     its incident edges and accumulates forces. No atomics. Requires
     CSR-like adjacency layout. Load-balance issue if vertex degrees
     are uneven (rare for polyhedra; vertex degree is uniform per
     polyhedron type).

  3. **Hybrid (edge-centric with warp-level reductions)**: warp
     shuffles aggregate forces per vertex without atomics. Best fit
     for moderate `|V|`.

For polyhedra, `|V|` is small (4 to ~10⁵ for refined meshes). Option
(2) wins for `|V| ≤ 10⁴`: fits in shared memory, no atomics, vertex
degree is fixed-known at compile time.

### Memory layout

- **AoS** (`struct Vertex { float3 pos, vel; float mass; }`): cache-
  friendly per-vertex access, bad for SIMD load.
- **SoA** (`float *x, *y, *z, *vx, *vy, *vz`): vectorizable, good for
  SIMD, slightly worse cache hit on per-vertex access.
- **Chunked SoA** (`float3 *pos`, `float3 *vel` separate arrays): the
  practical compromise. Each `float3` reads as 12 bytes (3 sequential
  FP32) — coalesced for warp.

For polyhedra with `|V| ≤ 10³`, the entire state fits in shared
memory (12 KB for 1000 vertices × 12 bytes pos). The whole simulation
becomes intra-block.

### Implicit-Euler SpMV kernel

`y = (M + Δt² K) x` is dominated by the `K x` term. K's sparsity
pattern is the polyhedron's edge graph. For a fixed polyhedron, the
graph is *static* — perfect target for compile-time SpMV
specialization (template the kernel on `|V|, |E|`, unroll the inner
loop).

This is exactly the pattern that benefits from sm_86 Tensor Cores
*if* the matrix becomes dense (it doesn't, for polyhedra). Standard
sparse SpMV kernels (cuSPARSE) are general-purpose and won't beat a
hand-rolled per-polyhedron-type kernel for `|V| ≤ 10³`.

### Symmetry kernels

When the polyhedron has point-group symmetry, force computation can
operate on the symmetry-reduced state and re-symmetrize after each
step. For `I_h` (60-fold), this is a 60× reduction in work, but
introduces a dense per-iteration symmetrization step (~5×3 matrix
multiplies). Net: 3-10× speedup on highly-symmetric polyhedra,
diminishing as symmetry breaks during deformation.

## Existing implementations

- **NVIDIA Flex** (closed-source) — PBD-based, GPU-accelerated, used
  in real-time soft-body simulation in games. Position-based, not
  force-based; not directly comparable to spring networks but
  similar problem class.
- **Bullet Physics** soft-body — CPU mass-spring, has CUDA backend
  for cloth.
- **TensegrityFEM**, **NTRT (NASA Tensegrity Robotics Toolkit)** —
  CPU implementations, force-based. Slow for >10³ elements.
- **Houdini Vellum** — closed-source, GPU PBD constraint solver.
- **Academic codes for icosahedral / fullerene relaxation** — most
  use commercial FEM (ANSYS, Abaqus); few open-source GPU codes.

The gap: a hand-tuned CUDA mass-spring solver for moderate `|V|`
(say 10² to 10⁴) with implicit integration is rare in open source.
This is the niche where the bare-metal kernel approach has
something to offer.

## Relevance to this project

### Where spring networks would fit in the kernel hierarchy

A spring-network solver decomposes into kernels we already know how
to write fast:

| problem subroutine          | analog in this repo |
|------------------------------|--------------------|
| force computation (edge → vertex scatter) | im2col-like gather (phase4 conv2d) |
| dense symmetrization         | small-batch GEMM (phase2 hgemm) |
| CG SpMV                      | sparse but graph-static — compile-time-specialized |
| reduction (||r||² for CG)    | softmax/layernorm reduce (phase2) |
| Verlet update                | element-wise streaming (phase1 vector_add scaled up) |

The spring network is a **memory-bound** problem at small `|V|`
(state fits in smem) and an **arithmetic-intensity-limited** problem
at large `|V|` (CG iterations dominate, SpMV is ~1 op/byte for FP32).
Both regimes are well-explored in this repo's existing kernels.

### Why polyhedral specifically

Generic spring networks have arbitrary connectivity. *Polyhedral*
spring networks have:

- **Bounded vertex degree** (cube: 3, dodecahedron: 3, icosahedron:
  5, fullerene C60: 3). Compile-time-known, allowing fully-unrolled
  vertex kernels.
- **Symmetry groups** of order 24-120, enabling 24-120× symmetry
  reductions.
- **Static graph topology** — no insertion/deletion at runtime. The
  CSR offsets are constants.

These constraints shrink the problem to something a hand-rolled
kernel can plausibly beat cuSPARSE on by 5-20×.

### Comparison to existing project work

| property               | spring net | flash attn | conv2d |
|------------------------|---|---|---|
| static memory access   | yes | yes (within tile) | yes |
| sparse                 | yes (graph) | dense | implicit-dense |
| Tensor-Core-friendly   | only at large |V| | yes | yes (via im2col) |
| benefits from cp.async | yes (force gather) | yes | yes |
| vertex-state in smem   | yes if |V|≤10³ | partial | partial (32 KB tile) |

The spring-network workload has the same algorithmic shape as
existing kernels in this repo: tile in shared memory, gather across
graph edges, reduce per vertex. The cp.async pipeline pattern (Obs U,
GG) directly transfers.

### Recommended scope if pursued

A phase-6 spring-network effort would deliver:

  1. **Vertex-centric Verlet kernel** (`experiments/spring/verlet.cu`):
     fully-unrolled per-polyhedron-type force gather. Target: 10× a
     baseline cuSPARSE-based naive implementation at |V|=1000.
  2. **Implicit-Euler CG solver** with hand-rolled SpMV
     (`experiments/spring/cg_solver.cu`): templated on polyhedron type.
     Compile-time-known sparsity pattern unlocks register-blocking
     for the SpMV inner loop.
  3. **Symmetry-reduced kernel** for icosahedral/octahedral cases:
     state lives in symmetry-reduced coordinates, single 3×3 matrix
     multiply for re-symmetrization per step.
  4. **Tensegrity solver**: edge-type-aware force kernel handling
     cable/strut polarity constraints.

Per-kernel scope is comparable to phase4 conv2d/resblock (1-2
weeks each). Whole effort: 4-8 weeks for a complete phase 6.

### Recommended *not* to pursue right now

The current kernel set (HGEMM, FA, conv2d, ResBlock, GroupNorm) is
ML-focused and benefits from continuing that direction:

- Multi-head attention with KV-cache (real LLM serving)
- INT8 quantized HGEMM with online dequant
- Mixed-precision flash attention (BF16 input, FP32 accum)

Spring networks are a different domain with limited overlap with
ML kernel patterns. The transferable techniques (cp.async, smem
tiling, register blocking) are already well-exercised; the new
techniques (graph-static SpMV, symmetry-reduced kernels) wouldn't
feed back into the ML work.

**Recommendation**: keep #32 documented as scoped (this file), close
the issue, and revisit when an external collaborator or downstream
application motivates the work. The technical analysis is solid; the
priority isn't.

## Key references

  - Provot 1995, *Deformation constraints in a mass-spring model to
    describe rigid cloth behavior*. Foundational mass-spring
    formulation.
  - Bridson, Marino, Fedkiw 2003, *Simulation of Clothing with Folds
    and Wrinkles*. The standard implicit-Euler treatment.
  - Müller, Heidelberger, Hennix, Ratcliff 2007, *Position Based
    Dynamics*. PBD foundation.
  - Macklin, Müller 2013, *Position Based Fluids*. PBD on GPU.
  - Skouras, Thomaszewski, Bickel, Gross 2012, *Computational Design
    of Rubber Balloons*. Tensegrity-adjacent FEM-meets-spring on
    polyhedra.
  - Connelly, Whiteley 1996, *Second-order rigidity and prestress
    stability for tensegrity frameworks*. Mathematical foundation
    for tensegrity equilibrium.
  - Skelton, de Oliveira 2009, *Tensegrity Systems*. Engineering text.
  - Liu, Schubert, Stenger, Lopez, Mor, Trasi, Han, Hu, Eisenbarth,
    Stoll, Stewart, Mehl, Ye, Bao, Brunet 2024, *Position-based
    Geometric Algebra Dynamics for Polyhedra and Beyond*. SIGGRAPH
    Asia. Closest recent work to the polyhedral angle.
  - NVIDIA cuSPARSE documentation, sparse-matrix kernels reference.

## Status

Issue #32 closed as **scoped**: technical assessment complete,
implementation deferred until motivated by application. This document
preserves the analysis for future revisit.
