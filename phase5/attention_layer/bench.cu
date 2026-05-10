/*
 * bench.cu — Attention layer pipeline benchmark (BenchDriver refactor)
 *
 * Build:
 *   nvcc -arch=sm_86 -O2 -o bench bench.cu -lcuda -I../../kernels/_common
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_fp16.h>

#include "../../kernels/_common/bench_driver.h"

static void cpu_layer(const float *X, const float *g, const float *b,
    const float *Wq, const float *Wk, const float *Wv, const float *Wo,
    float *res, int batch, int seq, int heads, int dh, float eps)
{
    int D = heads * dh, BS = batch * seq;
    float *nrm = (float*)malloc(BS * D * sizeof(float));
    float *Q = (float*)malloc(BS * D * sizeof(float));
    float *Kv = (float*)malloc(BS * D * sizeof(float));
    float *Vv = (float*)malloc(BS * D * sizeof(float));
    float *O = (float*)malloc(BS * D * sizeof(float));
    float *Pr = (float*)malloc(BS * D * sizeof(float));
    float *sc = (float*)malloc(seq * seq * sizeof(float));
    float sca = 1.0f / sqrtf((float)dh);

    for (int r = 0; r < BS; r++) {
        float mn = 0.0f, vr = 0.0f;
        for (int d = 0; d < D; d++) mn += X[r * D + d];
        mn /= D;
        for (int d = 0; d < D; d++) { float df = X[r * D + d] - mn; vr += df * df; }
        vr = 1.0f / sqrtf(vr / D + eps);
        for (int d = 0; d < D; d++) nrm[r * D + d] = g[d] * (X[r * D + d] - mn) * vr + b[d];
    }
    cpu_sgemm(BS, D, D, 1.0f, nrm, D, Wq, D, 0.0f, Q, D);
    cpu_sgemm(BS, D, D, 1.0f, nrm, D, Wk, D, 0.0f, Kv, D);
    cpu_sgemm(BS, D, D, 1.0f, nrm, D, Wv, D, 0.0f, Vv, D);

    for (int bi = 0; bi < batch; bi++)
        for (int h = 0; h < heads; h++)
            for (int qi = 0; qi < seq; qi++) {
                float rmx = -1e38f;
                for (int ki = 0; ki < seq; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < dh; d++)
                        dot += Q[((bi * seq + qi) * heads + h) * dh + d] *
                               Kv[((bi * seq + ki) * heads + h) * dh + d];
                    sc[qi * seq + ki] = dot * sca;
                    rmx = fmaxf(rmx, sc[qi * seq + ki]);
                }
                float sm = 0.0f;
                for (int ki = 0; ki < seq; ki++) { sc[qi * seq + ki] = expf(sc[qi * seq + ki] - rmx); sm += sc[qi * seq + ki]; }
                float rc = 1.0f / sm;
                for (int d = 0; d < dh; d++) {
                    float ac = 0.0f;
                    for (int ki = 0; ki < seq; ki++)
                        ac += sc[qi * seq + ki] * rc * Vv[((bi * seq + ki) * heads + h) * dh + d];
                    O[((bi * seq + qi) * heads + h) * dh + d] = ac;
                }
            }

    cpu_sgemm(BS, D, D, 1.0f, O, D, Wo, D, 0.0f, Pr, D);
    for (int i = 0; i < BS * D; i++) res[i] = Pr[i] + X[i];
    free(nrm); free(Q); free(Kv); free(Vv); free(O); free(Pr); free(sc);
}

static void f2h_host(const float *s, unsigned short *d, int n) {
    for (int i = 0; i < n; i++) { __half h = __float2half(s[i]); memcpy(&d[i], &h, 2); }
}

static int ugrid(int n) { return (n + 1023) / 1024; }

int main(int argc, char **argv) {
    int batch = (argc > 1) ? atoi(argv[1]) : 1;
    int seq   = (argc > 2) ? atoi(argv[2]) : 256;
    int heads = (argc > 3) ? atoi(argv[3]) : 8;
    int D     = (argc > 4) ? atoi(argv[4]) : 512;
    int dh    = 64;

    if (D != heads * dh) { fprintf(stderr, "d_model != heads*64\n"); return 1; }
    if (seq % 64 != 0) { fprintf(stderr, "seq must be multiple of 64\n"); return 1; }

    int BS = batch * seq;
    float eps = 1e-5f, attn_sc = 1.0f / sqrtf((float)dh);
    size_t bsd = (size_t)BS * D, dd = (size_t)D * D;

    printf("=== Attention Layer Pipeline (BenchDriver) ===\n");
    printf("batch=%d seq=%d heads=%d d_model=%d\n\n", batch, seq, heads, D);

    BenchDriver driver;
    driver.init_context();

    // Load modules manually (multiple fns per module needed)
    CUmodule mod_ln, mod_hg, mod_fl, mod_fu, mod_ut;
    auto ld = [&](const char* p) { CUmodule m; if (cuModuleLoad(&m,p)!=CUDA_SUCCESS){fprintf(stderr,"Cannot load %s\n",p);exit(1);} return m; };
    mod_ln = ld("../../kernels/reductions/layernorm/layernorm.sm_86.cubin");
    mod_hg = ld("../../kernels/gemm/hgemm/hgemm.sm_86.cubin");
    mod_fl = ld("../../phase3/flash_attention/flash_br16.sm_86.cubin");
    mod_fu = ld("../../phase3/flash_attention/flash_fused.sm_86.cubin");
    mod_ut = ld("utils.sm_86.cubin");

    CUfunction fn_ln, fn_hg, fn_fl, fn_fu, fn_f2h, fn_tr_bshd, fn_tr_bhsd, fn_res;
    CHECK_CU(cuModuleGetFunction(&fn_ln, mod_ln, "layernorm_block"));
    CHECK_CU(cuModuleGetFunction(&fn_hg, mod_hg, "hgemm_wmma"));
    CHECK_CU(cuModuleGetFunction(&fn_fl, mod_fl, "flash_attn_br16"));
    CHECK_CU(cuModuleGetFunction(&fn_fu, mod_fu, "flash_attn_fused"));
    CHECK_CU(cuModuleGetFunction(&fn_f2h,   mod_ut, "fp32_to_fp16"));
    CHECK_CU(cuModuleGetFunction(&fn_tr_bshd, mod_ut, "transpose_bshd"));
    CHECK_CU(cuModuleGetFunction(&fn_tr_bhsd, mod_ut, "transpose_bhsd"));
    CHECK_CU(cuModuleGetFunction(&fn_res,   mod_ut, "residual_add"));

    size_t fsmem = 2 * 64 * 64 * 2 + 64 * 64 * 4 + 64 * 64 * 4;
    CHECK_CU(cuFuncSetAttribute(fn_fl, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)fsmem));
    CHECK_CU(cuFuncSetAttribute(fn_fu, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)fsmem));

    // Host data
    auto h_X  = driver.host_alloc<float>(bsd);
    auto h_g  = driver.host_alloc<float>(D);
    auto h_b  = driver.host_alloc<float>(D);
    auto h_Wq = driver.host_alloc<float>(dd);
    auto h_Wk = driver.host_alloc<float>(dd);
    auto h_Wv = driver.host_alloc<float>(dd);
    auto h_Wo = driver.host_alloc<float>(dd);
    auto h_r  = driver.host_alloc<float>(bsd);
    auto h_rf = driver.host_alloc<float>(bsd);
    fill_random(h_X.get(), bsd, 42);
    fill_random(h_Wq.get(), dd, 50);
    fill_random(h_Wk.get(), dd, 51);
    fill_random(h_Wv.get(), dd, 52);
    fill_random(h_Wo.get(), dd, 53);
    for (int i = 0; i < D; i++) { h_g[i] = 1.0f; h_b[i] = 0.0f; }

    auto h_Wqf = driver.host_alloc<unsigned short>(dd);
    auto h_Wkf = driver.host_alloc<unsigned short>(dd);
    auto h_Wvf = driver.host_alloc<unsigned short>(dd);
    auto h_Wof = driver.host_alloc<unsigned short>(dd);
    f2h_host(h_Wq.get(), h_Wqf.get(), (int)dd);
    f2h_host(h_Wk.get(), h_Wkf.get(), (int)dd);
    f2h_host(h_Wv.get(), h_Wvf.get(), (int)dd);
    f2h_host(h_Wo.get(), h_Wof.get(), (int)dd);

    // CPU ref (small sizes)
    bool run_cpu = (BS <= 512 && seq <= 256);
    if (run_cpu) {
        printf("Computing CPU ref... "); fflush(stdout);
        cpu_layer(h_X.get(), h_g.get(), h_b.get(), h_Wq.get(), h_Wk.get(), h_Wv.get(), h_Wo.get(),
                  h_r.get(), batch, seq, heads, dh, eps);
        printf("done.\n\n");
    } else {
        printf("CPU ref skipped (too large).\n\n");
    }

    // Device data
    auto d_X  = driver.device_alloc<float>(bsd);
    auto d_g  = driver.device_alloc<float>(D);
    auto d_b  = driver.device_alloc<float>(D);
    auto d_Wq = driver.device_alloc<unsigned short>(dd);
    auto d_Wk = driver.device_alloc<unsigned short>(dd);
    auto d_Wv = driver.device_alloc<unsigned short>(dd);
    auto d_Wo = driver.device_alloc<unsigned short>(dd);
    auto d_res = driver.device_alloc<float>(bsd);
    driver.copy_h2d(d_X, h_X, bsd * sizeof(float));
    driver.copy_h2d(d_g, h_g, D * sizeof(float));
    driver.copy_h2d(d_b, h_b, D * sizeof(float));
    driver.copy_h2d(d_Wq, h_Wqf, dd * 2);
    driver.copy_h2d(d_Wk, h_Wkf, dd * 2);
    driver.copy_h2d(d_Wv, h_Wvf, dd * 2);
    driver.copy_h2d(d_Wo, h_Wof, dd * 2);

    // Intermediates
    auto d_xn   = driver.device_alloc<float>(bsd);
    auto d_xnh  = driver.device_alloc<unsigned short>(bsd);
    auto d_qf   = driver.device_alloc<float>(bsd);
    auto d_kf   = driver.device_alloc<float>(bsd);
    auto d_vf   = driver.device_alloc<float>(bsd);
    auto d_qh   = driver.device_alloc<unsigned short>(bsd);
    auto d_kh   = driver.device_alloc<unsigned short>(bsd);
    auto d_vh   = driver.device_alloc<unsigned short>(bsd);
    auto d_ob   = driver.device_alloc<float>(bsd);
    auto d_of   = driver.device_alloc<float>(bsd);
    auto d_ofh  = driver.device_alloc<unsigned short>(bsd);
    auto d_out  = driver.device_alloc<float>(bsd);

    printf("VRAM: ~%.1f MB\n\n", (4*dd*2 + bsd*4*5 + bsd*2*4)/1e6f);

    int nb = (int)bsd;
    int ug = ugrid(nb);
    int hgx = (D + 31) / 32, hgy = (BS + 31) / 32;
    int fgx = seq / 64;
    auto trg = (nb + 255) / 256;

    auto pipe_orig = [&]() {
        void *a_ln[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_xn.ptr, &BS, &D, &eps };
        CHECK_CU(cuLaunchKernel(fn_ln, BS, 1, 1, 128, 1, 1, 0, nullptr, a_ln, nullptr));

        void *a_f2h[] = { &d_xn.ptr, &d_xnh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_f2h, nullptr));

        void *aq[] = { &d_xnh.ptr, &d_Wq.ptr, &d_qf.ptr, &BS, &D, &D };
        void *ak[] = { &d_xnh.ptr, &d_Wk.ptr, &d_kf.ptr, &BS, &D, &D };
        void *av[] = { &d_xnh.ptr, &d_Wv.ptr, &d_vf.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, aq, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, ak, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, av, nullptr));

        // reuse d_ofh as tmp
        void *a_qa[] = { &d_qf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_qa, nullptr));
        void *a_trq[] = { &d_ofh.ptr, &d_qh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, a_trq, nullptr));
        void *a_ka[] = { &d_kf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_ka, nullptr));
        void *a_trk[] = { &d_ofh.ptr, &d_kh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, a_trk, nullptr));
        void *a_va[] = { &d_vf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_va, nullptr));
        void *a_trv[] = { &d_ofh.ptr, &d_vh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, a_trv, nullptr));

        void *a_fl[] = { &d_qh.ptr, &d_kh.ptr, &d_vh.ptr, &d_ob.ptr, &seq, &heads, &attn_sc };
        CHECK_CU(cuLaunchKernel(fn_fl, fgx, heads, batch, 128, 1, 1, (unsigned)fsmem, nullptr, a_fl, nullptr));

        void *a_tr[] = { &d_ob.ptr, &d_of.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bhsd, trg, 1, 1, 256, 1, 1, 0, nullptr, a_tr, nullptr));

        void *a_ofh[] = { &d_of.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_ofh, nullptr));

        void *a_wo[] = { &d_ofh.ptr, &d_Wo.ptr, &d_out.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, a_wo, nullptr));

        void *a_rs[] = { &d_out.ptr, &d_X.ptr, &d_res.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_res, ug, 1, 1, 256, 1, 1, 0, nullptr, a_rs, nullptr));
    };

    auto pipe_fused = [&]() {
        void *a_ln[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_xn.ptr, &BS, &D, &eps };
        CHECK_CU(cuLaunchKernel(fn_ln, BS, 1, 1, 128, 1, 1, 0, nullptr, a_ln, nullptr));

        void *a_f[] = { &d_xn.ptr, &d_xnh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_f, nullptr));

        void *aq[] = { &d_xnh.ptr, &d_Wq.ptr, &d_qf.ptr, &BS, &D, &D };
        void *ak[] = { &d_xnh.ptr, &d_Wk.ptr, &d_kf.ptr, &BS, &D, &D };
        void *av[] = { &d_xnh.ptr, &d_Wv.ptr, &d_vf.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, aq, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, ak, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, av, nullptr));

        void *a_fq[] = { &d_qf.ptr, &d_qh.ptr, &nb };
        void *a_fk[] = { &d_kf.ptr, &d_kh.ptr, &nb };
        void *a_fv[] = { &d_vf.ptr, &d_vh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_fq, nullptr));
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_fk, nullptr));
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_fv, nullptr));

        void *a_fu[] = { &d_qh.ptr, &d_kh.ptr, &d_vh.ptr, &d_of.ptr, &seq, &heads, &attn_sc };
        CHECK_CU(cuLaunchKernel(fn_fu, fgx, heads, batch, 128, 1, 1, (unsigned)fsmem, nullptr, a_fu, nullptr));

        void *a_ofh[] = { &d_of.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a_ofh, nullptr));

        void *a_wo[] = { &d_ofh.ptr, &d_Wo.ptr, &d_out.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, a_wo, nullptr));

        void *a_rs[] = { &d_out.ptr, &d_X.ptr, &d_res.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_res, ug, 1, 1, 256, 1, 1, 0, nullptr, a_rs, nullptr));
    };

    // Correctness
    if (run_cpu) {
        pipe_orig();
        CHECK_CU(cuCtxSynchronize());
        auto h_out = driver.host_alloc<float>(bsd);
        driver.copy_d2h(h_out, d_res, bsd * sizeof(float));
        auto r = check_fp32(h_out.get(), h_r.get(), bsd, 0.5f, 0.5f);
        print_check_result("original pipeline", r);

        pipe_fused();
        CHECK_CU(cuCtxSynchronize());
        driver.copy_d2h(h_out, d_res, bsd * sizeof(float));
        auto rf = check_fp32(h_out.get(), h_r.get(), bsd, 0.5f, 0.5f);
        print_check_result("fused pipeline", rf);
        printf("  FP16 accumulation → loose tolerance expected\n\n");
    }

    // Perf
    int wu = 5, bn = 20;
    printf("Performance (avg of %d runs, %d warmup):\n\n", bn, wu);

    for (int i = 0; i < wu; i++) pipe_orig();
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t1; t1.start();
    for (int i = 0; i < bn; i++) pipe_orig();
    CHECK_CU(cuCtxSynchronize());
    float ms_o = t1.stop_ms() / bn;

    for (int i = 0; i < wu; i++) pipe_fused();
    CHECK_CU(cuCtxSynchronize());
    BenchTimer t2; t2.start();
    for (int i = 0; i < bn; i++) pipe_fused();
    CHECK_CU(cuCtxSynchronize());
    float ms_f = t2.stop_ms() / bn;

    printf("  ORIGINAL (11 steps)       %7.3f ms  (100%%)\n", ms_o);
    printf("  FUSED (no transpose)      %7.3f ms  (%+.1f%%)\n\n",
           ms_f, 100.0f * (ms_f - ms_o) / ms_o);

    // Stage timing helper
    auto time_stage = [&](const char* name, auto fn) {
        for (int i = 0; i < wu; i++) fn();
        CHECK_CU(cuCtxSynchronize());
        BenchTimer t; t.start();
        for (int i = 0; i < bn; i++) fn();
        CHECK_CU(cuCtxSynchronize());
        float ms = t.stop_ms() / bn;
        printf("  %-28s %6.3f ms  (%4.1f%%)\n", name, ms, 100.0f * ms / ms_o);
        return ms;
    };

    // Original stages
    time_stage("LayerNorm", [&]() {
        void *a[] = { &d_X.ptr, &d_g.ptr, &d_b.ptr, &d_xn.ptr, &BS, &D, &eps };
        CHECK_CU(cuLaunchKernel(fn_ln, BS, 1, 1, 128, 1, 1, 0, nullptr, a, nullptr));
    });
    time_stage("f2h (x_norm)", [&]() {
        void *a[] = { &d_xn.ptr, &d_xnh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
    });
    time_stage("3x HGEMM QKV", [&]() {
        void *aq[] = { &d_xnh.ptr, &d_Wq.ptr, &d_qf.ptr, &BS, &D, &D };
        void *ak[] = { &d_xnh.ptr, &d_Wk.ptr, &d_kf.ptr, &BS, &D, &D };
        void *av[] = { &d_xnh.ptr, &d_Wv.ptr, &d_vf.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, aq, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, ak, nullptr));
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, av, nullptr));
    });
    time_stage("f2h + transpose Q,K,V", [&]() {
        void *qf[] = { &d_qf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, qf, nullptr));
        void *tq[] = { &d_ofh.ptr, &d_qh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, tq, nullptr));
        void *kf[] = { &d_kf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, kf, nullptr));
        void *tk[] = { &d_ofh.ptr, &d_kh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, tk, nullptr));
        void *vf[] = { &d_vf.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, vf, nullptr));
        void *tv[] = { &d_ofh.ptr, &d_vh.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bshd, trg, 1, 1, 256, 1, 1, 0, nullptr, tv, nullptr));
    });
    time_stage("Flash Attention", [&]() {
        void *a[] = { &d_qh.ptr, &d_kh.ptr, &d_vh.ptr, &d_ob.ptr, &seq, &heads, &attn_sc };
        CHECK_CU(cuLaunchKernel(fn_fl, fgx, heads, batch, 128, 1, 1, (unsigned)fsmem, nullptr, a, nullptr));
    });
    time_stage("transpose + f2h O", [&]() {
        void *a[] = { &d_ob.ptr, &d_of.ptr, &batch, &seq, &heads, &dh };
        CHECK_CU(cuLaunchKernel(fn_tr_bhsd, trg, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
        void *b[] = { &d_of.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, b, nullptr));
    });
    time_stage("HGEMM output proj", [&]() {
        void *a[] = { &d_ofh.ptr, &d_Wo.ptr, &d_out.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, a, nullptr));
    });
    time_stage("Residual add", [&]() {
        void *a[] = { &d_out.ptr, &d_X.ptr, &d_res.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_res, ug, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
    });
    printf("\n  Fused pipeline stages:\n");
    time_stage("fused: 3x f2h Q,K,V", [&]() {
        void *fq[] = { &d_qf.ptr, &d_qh.ptr, &nb };
        void *fk[] = { &d_kf.ptr, &d_kh.ptr, &nb };
        void *fv[] = { &d_vf.ptr, &d_vh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, fq, nullptr));
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, fk, nullptr));
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, fv, nullptr));
    });
    time_stage("fused: Flash BSHD", [&]() {
        void *a[] = { &d_qh.ptr, &d_kh.ptr, &d_vh.ptr, &d_of.ptr, &seq, &heads, &attn_sc };
        CHECK_CU(cuLaunchKernel(fn_fu, fgx, heads, batch, 128, 1, 1, (unsigned)fsmem, nullptr, a, nullptr));
    });
    time_stage("fused: f2h O", [&]() {
        void *a[] = { &d_of.ptr, &d_ofh.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_f2h, ug, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
    });
    time_stage("fused: HGEMM out", [&]() {
        void *a[] = { &d_ofh.ptr, &d_Wo.ptr, &d_out.ptr, &BS, &D, &D };
        CHECK_CU(cuLaunchKernel(fn_hg, hgx, hgy, 1, 64, 2, 1, 0, nullptr, a, nullptr));
    });
    time_stage("fused: Residual", [&]() {
        void *a[] = { &d_out.ptr, &d_X.ptr, &d_res.ptr, &nb };
        CHECK_CU(cuLaunchKernel(fn_res, ug, 1, 1, 256, 1, 1, 0, nullptr, a, nullptr));
    });

    cuModuleUnload(mod_ln); cuModuleUnload(mod_hg);
    cuModuleUnload(mod_fl); cuModuleUnload(mod_fu); cuModuleUnload(mod_ut);
    return 0;
}
