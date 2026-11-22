#pragma once

// exp for the NSA block selection (ggml_nsa_select), shared by the CPU and CUDA kernels.
//
// The selection is a discrete top-k, so a 1-ulp difference in one softmax score can change the selected blocks. The
// CUDA backend is built with -use_fast_math (expf becomes the approximate __expf) and the CPU calls libm expf, so the
// two backends would not give the same bits. This version uses only explicit fmaf and an integer add to the exponent
// field. Both are exact IEEE operations on CPU and GPU, and there is no a*b + c left for -ffp-contract / --fmad to
// fuse, so both backends get the same bits. Max error 0.94 ulp over every float in [-64, 0] (libm expf: 0.50).
//
// x < -64 returns 0. The callers pass x = logit - max <= 0, where such a term cannot change the float softmax sum.
// Flushing it also keeps the following divisions and sums in the normal range, which the CUDA build (-ftz) needs to
// match the CPU bit for bit.

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_NSA_HD __host__ __device__
#else
#define GGML_NSA_HD
#endif

static inline GGML_NSA_HD float ggml_nsa_expf(float x) {
    if (x != x) {
        return x; // NaN
    }
    if (x < -64.0f) {
        return 0.0f;
    }
    if (x > 88.0f) {
        return INFINITY;
    }
    // n = round(x / ln 2): adding 1.5*2^23 rounds to an integer (to nearest even), subtracting it again is exact
    const float magic = 12582912.0f;
    const float n = fmaf(x, 1.442695040888963f, magic) - magic;
    // r = x - n*ln 2 with ln 2 split in a high part (15 significant bits) and a low part, |r| <= 0.35
    float r = fmaf(n, -0.693145751953125f, x);
    r = fmaf(n, -1.428606765330187e-06f, r);
    // exp(r): Taylor series to degree 7 (truncation error < 1e-8), Horner with fmaf
    float p = 1.984126984e-4f;
    p = fmaf(p, r, 1.388888889e-3f);
    p = fmaf(p, r, 8.333333333e-3f);
    p = fmaf(p, r, 4.166666667e-2f);
    p = fmaf(p, r, 1.666666667e-1f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    // p * 2^n: p is in [0.70, 1.42] and n in [-92, 127], so adding n to the exponent field is exact
    int32_t bits;
    memcpy(&bits, &p, sizeof(bits));
    bits += (int32_t) n * (1 << 23);
    float res;
    memcpy(&res, &bits, sizeof(res));
    return res;
}
