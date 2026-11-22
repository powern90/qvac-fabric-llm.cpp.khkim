#include "nsa.cuh"
#include "ggml-nsa-exp.h"

#include <climits>

#define NSA_MAX_DV 128
#define NSA_MAX_BS 128
#define NSA_SELECT_CHUNK 64
#define NSA_SPLIT_CTAS_PER_SM 3 // split-K target: ~3 CTAs per SM (measured on B200: T=16/64 best at 2-4)

static __device__ __forceinline__ float nsa_to_f32(const float x) { return x; }
static __device__ __forceinline__ float nsa_to_f32(const half  x) { return __half2float(x); }

// quantized K/V cache types (block-contiguous rows along the head dim)
template <typename T> struct nsa_kv_quant : std::false_type {};
template <> struct nsa_kv_quant<block_q8_0> : std::true_type {};
template <> struct nsa_kv_quant<block_q4_0> : std::true_type {};

// K/V type combinations with a kernel: F16/F32 pairs, and pairs of F16/Q8_0/Q4_0
static bool nsa_kv_types_ok(const ggml_type tk, const ggml_type tv) {
    auto fp = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_F32; };
    auto lp = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0; };
    return (fp(tk) && fp(tv)) || (lp(tk) && lp(tv));
}

// ggml_nsa_select: one CTA per (query, kv head), one warp per q head of the group
// the float op order matches the CPU reference: fmaf dot over d, lane partial sums + xor butterfly, head sum in order
static __global__ void nsa_select_kernel(
        const char * __restrict__ q, const char * __restrict__ kc, const char * __restrict__ pos, int32_t * __restrict__ dst,
        const int D, const int G, const int Hkv, const int NB, const int bs, const int n_sel, const float scale,
        const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2, const size_t nbp0) {
    const int DK = D + 1; // padded k_cmp row: lanes read different rows without bank conflicts
    const int CH = min(NB, NSA_SELECT_CHUNK);

    // no static shared memory: the dynamic size limit is raised to smpbo
    extern __shared__ float smem[];
    float * P     = smem;                // [G][NB] logits, then softmax probs of each head
    float * score = P + (size_t) G*NB;   // [NB + 3] candidate scores
    float * Qs    = score + NB + 3;      // [G][D]
    float * Kt    = Qs + G*D;            // [CH][D + 1] k_cmp chunk
    int   * extra = (int *) (Kt + CH*DK); // [3] ids of the forced blocks without a complete block
    int   & n_cand_s = extra[3];

    const int t    = blockIdx.x;
    const int h    = blockIdx.y;
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int nw   = blockDim.y;
    const int tid  = w*WARP_SIZE + lane;
    const int nth  = nw*WARP_SIZE;

    int32_t * out = dst + ((int64_t) t*Hkv + h)*n_sel;

    const int p = *(const int32_t *) (pos + t*nbp0);
    if (p < 0) {
        for (int i = tid; i < n_sel; i += nth) {
            out[i] = -1;
        }
        return;
    }
    const int cur   = p / bs;
    const int n_cmp = (int) min((int64_t) NB, ((int64_t) p + 1) / bs);

    const char * kh = kc + h*nbk1;

    for (int e = tid; e < G*D; e += nth) {
        const int g = e / D;
        const int d = e - g*D;
        Qs[e] = *(const float *) (q + (int64_t) (h*G + g)*nbq1 + (int64_t) t*nbq2 + d*sizeof(float));
    }

    // logits, k_cmp staged in shared memory in chunks of CH blocks
    for (int c0 = 0; c0 < n_cmp; c0 += CH) {
        const int nc = min(CH, n_cmp - c0);
        __syncthreads();
        for (int e = tid; e < nc*D; e += nth) {
            const int bb = e / D;
            const int d  = e - bb*D;
            Kt[bb*DK + d] = *(const float *) (kh + (int64_t) (c0 + bb)*nbk2 + d*sizeof(float));
        }
        __syncthreads();
        for (int g = w; g < G; g += nw) {
            const float * qg = Qs + g*D;
            for (int bb = lane; bb < nc; bb += WARP_SIZE) {
                const float * kb = Kt + bb*DK;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot = fmaf(qg[d], kb[d], dot);
                }
                P[(size_t) g*NB + c0 + bb] = scale*dot;
            }
        }
    }
    ggml_cuda_syncwarp();

    for (int g = w; g < G; g += nw) {
        float * Pg = P + (size_t) g*NB;

        float mx = -INFINITY;
        for (int b = lane; b < n_cmp; b += WARP_SIZE) {
            mx = fmaxf(mx, Pg[b]);
        }
        mx = warp_reduce_max(mx);

        float part = 0.0f;
        for (int b = lane; b < n_cmp; b += WARP_SIZE) {
            const float e = ggml_nsa_expf(Pg[b] - mx); // same bits as the CPU kernel, see ggml-nsa-exp.h
            Pg[b] = e;
            part += e;
        }
        const float sum = warp_reduce_sum(part);
        for (int b = lane; b < n_cmp; b += WARP_SIZE) {
            Pg[b] = __fdiv_rn(Pg[b], sum);
        }
    }
    __syncthreads();

    for (int b = tid; b < n_cmp; b += nth) {
        float acc = 0.0f;
        for (int g = 0; g < G; ++g) {
            acc += P[(size_t) g*NB + b];
        }
        score[b] = (b == 0 || b == cur - 1 || b == cur) ? 1.0f : __fdiv_rn(acc, (float) G);
    }
    // forced blocks without a complete block go after the complete blocks, in order
    if (tid == 0) {
        int n = n_cmp;
        auto force = [&](int b) {
            if (b >= n_cmp) {
                extra[n - n_cmp] = b;
                score[n++] = 1.0f;
            }
        };
        force(0);
        if (cur >= 2) {
            force(cur - 1);
        }
        if (cur >= 1) {
            force(cur);
        }
        n_cand_s = n;
    }
    __syncthreads();

    if (w != 0) {
        return;
    }

    // n_sel rounds of warp argmax, ties to the lower block index; taken entries get score -1
    const int n_cand = n_cand_s;
    for (int i = 0; i < n_sel; ++i) {
        float bv = -1.0f;
        int   bc = INT_MAX;
        for (int c = lane; c < n_cand; c += WARP_SIZE) {
            if (score[c] > bv) {
                bv = score[c];
                bc = c;
            }
        }
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, off, WARP_SIZE);
            const int   oc = __shfl_xor_sync(0xffffffff, bc, off, WARP_SIZE);
            if (ov > bv || (ov == bv && oc < bc)) {
                bv = ov;
                bc = oc;
            }
        }
        if (lane == 0) {
            if (bc == INT_MAX) {
                out[i] = -1;
            } else {
                out[i] = bc < n_cmp ? bc : extra[bc - n_cmp];
                score[bc] = -1.0f;
            }
        }
        ggml_cuda_syncwarp();
    }
}

static size_t nsa_select_smem(int64_t G, int64_t NB, int64_t D) {
    const int64_t CH = std::min<int64_t>(NB, NSA_SELECT_CHUNK);
    return sizeof(float)*(G*NB + NB + 3 + G*D + CH*(D + 1)) + 4*sizeof(int);
}

void ggml_cuda_op_nsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * kc  = dst->src[1];
    const ggml_tensor * pos = dst->src[2];

    GGML_ASSERT(ggml_is_contiguous(dst));

    const int D   = q->ne[0];
    const int Hq  = q->ne[1];
    const int T   = q->ne[2];
    const int Hkv = kc->ne[1];
    const int NB  = kc->ne[2];
    const int G   = Hq / Hkv;

    const int   bs    = ggml_get_op_params_i32(dst, 0);
    const int   n_sel = ggml_get_op_params_i32(dst, 1);
    const float scale = ggml_get_op_params_f32(dst, 2);

    if (T == 0) {
        return;
    }

    const int    nw   = std::min(G, WARP_SIZE);
    const size_t smem = nsa_select_smem(G, NB, D);

    CUDA_SET_SHARED_MEMORY_LIMIT(nsa_select_kernel, ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo);

    const dim3 grid(T, Hkv);
    const dim3 block(WARP_SIZE, nw);
    nsa_select_kernel<<<grid, block, smem, ctx.stream()>>>(
        (const char *) q->data, (const char *) kc->data, (const char *) pos->data, (int32_t *) dst->data,
        D, G, Hkv, NB, bs, n_sel, scale, q->nb[1], q->nb[2], kc->nb[1], kc->nb[2], pos->nb[0]);
}

bool ggml_cuda_nsa_select_supported(int device, const ggml_tensor * op) {
    const ggml_tensor * q   = op->src[0];
    const ggml_tensor * kc  = op->src[1];
    const ggml_tensor * pos = op->src[2];

    if (q->type != GGML_TYPE_F32 || kc->type != GGML_TYPE_F32 || pos->type != GGML_TYPE_I32 || !ggml_is_contiguous(op)) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || kc->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t G = q->ne[1] / kc->ne[1];
    return nsa_select_smem(G, kc->ne[2], q->ne[0]) <= ggml_cuda_info().devices[device].smpbo;
}

// 32 bit load from a 2 byte aligned address (the 18/34 byte q4_0/q8_0 blocks are only 2 byte aligned)
static __device__ __forceinline__ int nsa_get_int_b2(const void * x, const int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) (x16[2*i32 + 0] | ((uint32_t) x16[2*i32 + 1] << 16));
}

// quantized [nj][n] tile, row jj (n/32 consecutive blocks) at src + jj*nb_row, as F32 into dst with row stride ld
// one work item = 4 quant bytes of one block: 4 values (q8_0) or 8 values (q4_0)
// the loads of R items are issued before any of them is decoded (one memory latency per R*nth items, like the
// 16 byte loads of nsa_load_tile_fp)
// the dequantization is the float math of the CPU to_float (dequantize_row_q8_0/q4_0): (float) q * d
template <typename T>
static __device__ __forceinline__ void nsa_load_tile_q(
        float * dst, const int ld, const char * src, const size_t nb_row, const int nj, const int n,
        const int tid, const int nth) {
    constexpr int QK  = 32;
    constexpr int CPB = sizeof(T::qs)/4; // work items per block
    constexpr int R   = 8;               // items in flight per thread (measured on B200: 8 beats 4 for both types)
    static_assert(sizeof(T::qs) % 4 == 0, "bad block");
    const int ipr = (n/QK)*CPB;
    const int nit = nj*ipr;
    for (int e0 = 0; e0 < nit; e0 += R*nth) {
        int  q4[R];
        half dh[R];
        int  oo[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int e = e0 + tid + r*nth;
            if (e < nit) {
                const int jj = e / ipr;
                const int rr = e - jj*ipr;
                const int ib = rr / CPB;
                const int c  = rr - ib*CPB;
                const T * x = (const T *) (src + jj*nb_row) + ib;
                dh[r] = x->d;
                q4[r] = nsa_get_int_b2(x->qs, c);
                oo[r] = jj*ld + ib*QK + 4*c;
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (e0 + tid + r*nth >= nit) {
                break;
            }
            const float d = __half2float(dh[r]);
            float *     o = dst + oo[r];
            if constexpr (std::is_same_v<T, block_q8_0>) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[i] = (float) (int8_t) (q4[r] >> (8*i)) * d;
                }
            } else {
                static_assert(std::is_same_v<T, block_q4_0>, "unsupported quant type");
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int b = (q4[r] >> (8*i)) & 0xFF;
                    o[i]        = (float) ((b & 0x0F) - 8) * d;
                    o[i + QK/2] = (float) ((b >>   4) - 8) * d;
                }
            }
        }
    }
}

// load a [nj][n] tile (row jj at src + jj*nb_row, element d at d*nb_el) as F32 into dst with row stride ld
// vec: 16 byte loads along rows (nb_el == sizeof(T)) or along tokens (nb_row == sizeof(T), full blocks only)
template <typename T>
static __device__ __forceinline__ void nsa_load_tile_fp(
        float * dst, const int ld, const char * src, const size_t nb_el, const size_t nb_row,
        const int nj, const int n, const int bs, const bool vec, const int tid, const int nth) {
    constexpr int EPC = 16/sizeof(T);
    constexpr int R   = 4;
    if (vec && nb_el == sizeof(T)) {
        const int cpr = n / EPC;
        const int nch = nj*cpr;
        for (int e0 = 0; e0 < nch; e0 += R*nth) {
            int4 buf[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = e0 + tid + r*nth;
                if (e < nch) {
                    const int jj = e / cpr;
                    buf[r] = *(const int4 *) (src + jj*nb_row + (e - jj*cpr)*16);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = e0 + tid + r*nth;
                if (e < nch) {
                    const int jj = e / cpr;
                    const int d0 = (e - jj*cpr)*EPC;
                    const T * x = (const T *) &buf[r];
#pragma unroll
                    for (int i = 0; i < EPC; ++i) {
                        dst[jj*ld + d0 + i] = nsa_to_f32(x[i]);
                    }
                }
            }
        }
    } else if (vec && nb_row == sizeof(T) && nj == bs && bs % EPC == 0) {
        const int cpt = bs / EPC;
        const int nch = n*cpt;
        for (int e0 = 0; e0 < nch; e0 += R*nth) {
            int4 buf[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = e0 + tid + r*nth;
                if (e < nch) {
                    const int d = e / cpt;
                    buf[r] = *(const int4 *) (src + d*nb_el + (e - d*cpt)*16);
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int e = e0 + tid + r*nth;
                if (e < nch) {
                    const int d   = e / cpt;
                    const int jj0 = (e - d*cpt)*EPC;
                    const T * x = (const T *) &buf[r];
#pragma unroll
                    for (int i = 0; i < EPC; ++i) {
                        dst[(jj0 + i)*ld + d] = nsa_to_f32(x[i]);
                    }
                }
            }
        }
    } else if (nb_el == sizeof(T)) {
        for (int e = tid; e < nj*n; e += nth) {
            const int jj = e / n;
            const int d  = e - jj*n;
            dst[jj*ld + d] = nsa_to_f32(*(const T *) (src + jj*nb_row + d*sizeof(T)));
        }
    } else {
        for (int e = tid; e < nj*n; e += nth) {
            const int d  = e / nj;
            const int jj = e - d*nj;
            dst[jj*ld + d] = nsa_to_f32(*(const T *) (src + jj*nb_row + d*nb_el));
        }
    }
}

// quantized T: rows only (nb_el is the block size, vec is ignored)
template <typename T>
static __device__ __forceinline__ void nsa_load_tile(
        float * dst, const int ld, const char * src, const size_t nb_el, const size_t nb_row,
        const int nj, const int n, const int bs, const bool vec, const int tid, const int nth) {
    if constexpr (nsa_kv_quant<T>::value) {
        GGML_UNUSED_VARS(nb_el, bs, vec);
        nsa_load_tile_q<T>(dst, ld, src, nb_row, nj, n, tid, nth);
    } else {
        nsa_load_tile_fp<T>(dst, ld, src, nb_el, nb_row, nj, n, bs, vec, tid, nth);
    }
}

// ggml_nsa_sel_attn: one CTA per (query, kv head), one warp per HPW q heads of the group
// the CTA loads the K/V tile of each selected block into shared memory once, all q heads of the group use it
// HPW > 1 reuses each K/V value loaded from shared memory for HPW heads (shared memory bandwidth is the limit)
// D_/DV_/BS_ != 0 are compile-time sizes, 0 means use the runtime value
// split-K (SPLIT, small T*Hkv): CTA z handles the slots [z*spc, (z+1)*spc) of the S selected blocks and writes
// the unnormalized partial (o[DV], max, sum) of each q head to part, nsa_sel_attn_combine merges the gridDim.z partials
#define NSA_SEL_ATTN_MAX_WARPS 8

template <typename TK, typename TV, int D_, int DV_, int BS_, int HPW, bool SPLIT>
static __global__ void __launch_bounds__(NSA_SEL_ATTN_MAX_WARPS*WARP_SIZE) nsa_sel_attn_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const int32_t * __restrict__ idx, const char * __restrict__ pos, float * __restrict__ dst,
        float * __restrict__ part, const int spc,
        const int D_rt, const int DV_rt, const int BS_rt, const int N, const int S, const int G, const int Hkv,
        const size_t nbq1, const size_t nbq2,
        const size_t nbk0, const size_t nbk1, const size_t nbk2,
        const size_t nbv0, const size_t nbv1, const size_t nbv2,
        const size_t nbp0, const float scale, const int window, const bool vec_k, const bool vec_v) {
    static_assert(HPW == 1 || HPW == 2 || HPW == 4, "bad HPW");
    constexpr int NV = ((DV_ ? DV_ : NSA_MAX_DV) + WARP_SIZE - 1)/WARP_SIZE; // output dims per lane
    constexpr int NJ = ((BS_ ? BS_ : NSA_MAX_BS) + WARP_SIZE - 1)/WARP_SIZE; // block tokens per lane

    const int D  = D_  ? D_  : D_rt;
    const int DV = DV_ ? DV_ : DV_rt;
    const int bs = BS_ ? BS_ : BS_rt;
    const int DK = D + 1; // padded K row: lanes read different rows without bank conflicts

    const int t    = blockIdx.x;
    const int h    = blockIdx.y;
    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int nw   = blockDim.y;
    const int tid  = w*WARP_SIZE + lane;
    const int nth  = nw*WARP_SIZE;

    extern __shared__ float4 smem4[];
    float * smem = (float *) smem4;
    const int qoff = (bs*DK + bs*DV + 3) & ~3;
    float * Ks = smem;                                 // [bs][D + 1]
    float * Vs = Ks + bs*DK;                           // [bs][DV]
    float * Qw = smem + qoff + w*D*HPW;                // [D][HPW] q of the heads of this warp
    float * Pw = smem + qoff + nw*D*HPW + w*bs*HPW;    // [bs][HPW] probs of the current block

    const int p    = *(const int32_t *) (pos + t*nbp0);
    const int jend = (int) min((int64_t) N, (int64_t) p + 1);
    const int64_t jmin = window > 0 ? max(0, p - window + 1) : 0; // first visible token

    const int32_t * irow = idx + ((int64_t) t*Hkv + h)*S;
    const char    * kh   = k + h*nbk1;
    const char    * vh   = v + h*nbv1;

    for (int g0 = 0; g0 < G; g0 += nw*HPW) {
        const int  gw     = g0 + w*HPW;
        const bool active = gw < G;

        if (active) {
            for (int e = lane; e < D*HPW; e += WARP_SIZE) {
                const int d = e / HPW;
                const int g = gw + e % HPW;
                Qw[e] = g < G ? scale*((const float *) (q + (int64_t) (h*G + g)*nbq1 + (int64_t) t*nbq2))[d] : 0.0f;
            }
        }

        float m[HPW];
        float l[HPW];
        float acc[HPW][NV];
#pragma unroll
        for (int i = 0; i < HPW; ++i) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
#pragma unroll
            for (int c = 0; c < NV; ++c) {
                acc[i][c] = 0.0f;
            }
        }

        const int s0 = SPLIT ? blockIdx.z*spc : 0;
        const int s1 = SPLIT ? min(S, s0 + spc) : S;
        for (int s = s0; s < s1; ++s) {
            const int b = irow[s];
            if (b < 0) {
                continue;
            }
            const int64_t j0 = (int64_t) b*bs;
            const int     nj = (int) min((int64_t) bs, jend - j0);
            const int     jlo = (int) max((int64_t) 0, jmin - j0); // tokens [jlo, nj) of the block are visible
            if (nj <= 0 || jlo >= nj) {
                continue;
            }

            __syncthreads();
            nsa_load_tile<TK>(Ks, DK, kh + j0*nbk2, nbk0, nbk2, nj, D,  bs, vec_k, tid, nth);
            nsa_load_tile<TV>(Vs, DV, vh + j0*nbv2, nbv0, nbv2, nj, DV, bs, vec_v, tid, nth);
            __syncthreads();

            if (!active) {
                continue;
            }

            // lane owns tokens lane + j*WARP_SIZE; rows >= nj hold stale data and are masked below
            float sj[HPW][NJ];
            const float * kr[NJ];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                kr[j] = Ks + min(lane + j*WARP_SIZE, bs - 1)*DK;
#pragma unroll
                for (int i = 0; i < HPW; ++i) {
                    sj[i][j] = 0.0f;
                }
            }
#pragma unroll 8
            for (int d = 0; d < D; ++d) {
                float qd[HPW];
                if constexpr (HPW == 4) {
                    const float4 t4 = *(const float4 *) (Qw + d*4);
                    qd[0] = t4.x; qd[1] = t4.y; qd[2] = t4.z; qd[3] = t4.w;
                } else if constexpr (HPW == 2) {
                    const float2 t2 = *(const float2 *) (Qw + d*2);
                    qd[0] = t2.x; qd[1] = t2.y;
                } else {
                    qd[0] = Qw[d];
                }
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const float kv = kr[j][d];
#pragma unroll
                    for (int i = 0; i < HPW; ++i) {
                        sj[i][j] += qd[i]*kv;
                    }
                }
            }

            // online softmax update per head
#pragma unroll
            for (int i = 0; i < HPW; ++i) {
                float mb = -INFINITY;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int jj = lane + j*WARP_SIZE;
                    sj[i][j] = jj < nj && jj >= jlo ? sj[i][j] : -INFINITY;
                    mb = fmaxf(mb, sj[i][j]);
                }
                mb = warp_reduce_max(mb);

                const float m_new = fmaxf(m[i], mb);
                const float corr  = expf(m[i] - m_new);
                float ls = 0.0f;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int jj = lane + j*WARP_SIZE;
                    if (jj < bs) {
                        const float pj = expf(sj[i][j] - m_new);
                        Pw[jj*HPW + i] = pj;
                        ls += pj;
                    }
                }
                l[i] = l[i]*corr + warp_reduce_sum(ls);
                m[i] = m_new;
#pragma unroll
                for (int c = 0; c < NV; ++c) {
                    acc[i][c] *= corr;
                }
            }
            ggml_cuda_syncwarp();

            for (int jj = 0; jj < nj; ++jj) {
                float pj[HPW];
                if constexpr (HPW == 4) {
                    const float4 t4 = *(const float4 *) (Pw + jj*4);
                    pj[0] = t4.x; pj[1] = t4.y; pj[2] = t4.z; pj[3] = t4.w;
                } else if constexpr (HPW == 2) {
                    const float2 t2 = *(const float2 *) (Pw + jj*2);
                    pj[0] = t2.x; pj[1] = t2.y;
                } else {
                    pj[0] = Pw[jj];
                }
                const float * vr = Vs + jj*DV;
#pragma unroll
                for (int c = 0; c < NV; ++c) {
                    const int d = lane + c*WARP_SIZE;
                    const float vv = d < DV ? vr[d] : 0.0f;
#pragma unroll
                    for (int i = 0; i < HPW; ++i) {
                        acc[i][c] += pj[i]*vv;
                    }
                }
            }
        }

        if (active) {
#pragma unroll
            for (int i = 0; i < HPW; ++i) {
                const int g = gw + i;
                if (g >= G) {
                    break;
                }
                const int64_t row = (int64_t) t*G*Hkv + h*G + g;
                if constexpr (SPLIT) {
                    // partial layout: [row][n_split][DV] o, then [row][n_split][2] (max, sum)
                    const int64_t r = row*gridDim.z + blockIdx.z;
                    float * o = part + r*DV;
#pragma unroll
                    for (int c = 0; c < NV; ++c) {
                        const int d = lane + c*WARP_SIZE;
                        if (d < DV) {
                            o[d] = acc[i][c];
                        }
                    }
                    if (lane == 0) {
                        float * ml = part + (int64_t) gridDim.x*G*Hkv*gridDim.z*DV + 2*r;
                        ml[0] = m[i];
                        ml[1] = l[i];
                    }
                    continue;
                }
                float * o = dst + row*DV;
                const float inv = l[i] > 0.0f ? 1.0f/l[i] : 0.0f;
#pragma unroll
                for (int c = 0; c < NV; ++c) {
                    const int d = lane + c*WARP_SIZE;
                    if (d < DV) {
                        o[d] = acc[i][c]*inv;
                    }
                }
            }
        }
        __syncthreads();
    }
}

// merge the n_split partials of each output row: one CTA per row, one thread per output dim
// splits without any token have sum 0 and are skipped, a row without any token gets 0
static __global__ void nsa_sel_attn_combine(const float * __restrict__ part, float * __restrict__ dst,
        const int DV, const int n_split, const int64_t n_rows) {
    const int64_t row = blockIdx.x;
    const int     d   = threadIdx.x;
    if (d >= DV) {
        return;
    }
    const float * o  = part + row*n_split*DV + d;
    const float * ml = part + n_rows*n_split*DV + row*n_split*2;

    float M = -INFINITY;
    for (int z = 0; z < n_split; ++z) {
        M = fmaxf(M, ml[2*z]);
    }
    float L   = 0.0f;
    float acc = 0.0f;
    if (M != -INFINITY) {
        for (int z = 0; z < n_split; ++z) {
            const float l = ml[2*z + 1];
            if (l > 0.0f) {
                const float w = expf(ml[2*z] - M);
                L   += l*w;
                acc += o[z*DV]*w;
            }
        }
    }
    dst[row*DV + d] = L > 0.0f ? acc/L : 0.0f;
}

static int nsa_sel_attn_hpw(int64_t G) {
    return G % 4 == 0 ? 4 : 1;
}

// at most NSA_SEL_ATTN_MAX_WARPS warps: the kernel uses up to ~160 registers per thread, so larger blocks do not
// fit the register file (the g0 loop covers the remaining heads)
static int64_t nsa_sel_attn_nw(int64_t G, int hpw) {
    return std::min<int64_t>((G + hpw - 1)/hpw, NSA_SEL_ATTN_MAX_WARPS);
}

static size_t nsa_sel_attn_smem(int64_t D, int64_t DV, int64_t bs, int64_t nw, int hpw) {
    const int64_t qoff = (bs*(D + 1) + bs*DV + 3) & ~3;
    return sizeof(float)*(qoff + nw*D*hpw + nw*bs*hpw);
}

// 16 byte loads need 16 byte aligned rows (or token runs for a transposed layout)
static bool nsa_vec_ok(const ggml_tensor * t, size_t row_bytes) {
    const size_t ts = ggml_type_size(t->type);
    if ((uintptr_t) t->data % 16 != 0 || t->nb[1] % 16 != 0) {
        return false;
    }
    if (t->nb[0] == ts) {
        return row_bytes % 16 == 0 && t->nb[2] % 16 == 0;
    }
    return t->nb[2] == ts && t->nb[0] % 16 == 0;
}

template <typename TK, typename TV, int D_, int DV_, int BS_, int HPW>
static void nsa_sel_attn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    const ggml_tensor * pos = dst->src[4];

    const int D   = q->ne[0];
    const int Hq  = q->ne[1];
    const int T   = q->ne[2];
    const int DV  = v->ne[0];
    const int Hkv = k->ne[1];
    const int N   = k->ne[2];
    const int S   = idx->ne[0];
    const int G   = Hq / Hkv;

    const int   bs     = ggml_get_op_params_i32(dst, 0);
    const float scale  = ggml_get_op_params_f32(dst, 1);
    const int   window = ggml_get_op_params_i32(dst, 2);

    const int    nw   = nsa_sel_attn_nw(G, HPW);
    const size_t smem = nsa_sel_attn_smem(D, DV, bs, nw, HPW);

    const int device = ggml_cuda_get_device();
    CUDA_SET_SHARED_MEMORY_LIMIT((nsa_sel_attn_kernel<TK, TV, D_, DV_, BS_, HPW, false>), ggml_cuda_info().devices[device].smpbo);
    CUDA_SET_SHARED_MEMORY_LIMIT((nsa_sel_attn_kernel<TK, TV, D_, DV_, BS_, HPW, true>),  ggml_cuda_info().devices[device].smpbo);

    // split-K over the selected blocks when the T*Hkv CTAs do not fill the GPU, aiming at ~NSA_SPLIT_CTAS_PER_SM*nsm CTAs
    const int64_t nsm     = ggml_cuda_info().devices[device].nsm;
    const int64_t n_ctas  = (int64_t) T*Hkv;
    int           n_split = 1;
    int           spc     = S;
    if (S > 1 && n_ctas < nsm) {
        const int64_t want = std::min<int64_t>(S, (NSA_SPLIT_CTAS_PER_SM*nsm + n_ctas - 1)/n_ctas);
        spc     = (int) ((S + want - 1)/want);
        n_split = (S + spc - 1)/spc;
    }

    const int64_t n_rows = (int64_t) T*Hq;
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (n_split > 1) {
        part.alloc(n_rows*n_split*(DV + 2));
    }

    const dim3 grid(T, Hkv, n_split);
    const dim3 block(WARP_SIZE, nw);
    auto kernel = n_split > 1 ? nsa_sel_attn_kernel<TK, TV, D_, DV_, BS_, HPW, true> : nsa_sel_attn_kernel<TK, TV, D_, DV_, BS_, HPW, false>;
    kernel<<<grid, block, smem, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        (const int32_t *) idx->data, (const char *) pos->data, (float *) dst->data,
        part.get(), spc,
        D, DV, bs, N, S, G, Hkv,
        q->nb[1], q->nb[2],
        k->nb[0], k->nb[1], k->nb[2],
        v->nb[0], v->nb[1], v->nb[2],
        pos->nb[0], scale, window,
        !nsa_kv_quant<TK>::value && nsa_vec_ok(k, D*sizeof(TK)), !nsa_kv_quant<TV>::value && nsa_vec_ok(v, DV*sizeof(TV)));

    if (n_split > 1) {
        const int nt = (DV + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE;
        nsa_sel_attn_combine<<<n_rows, nt, 0, ctx.stream()>>>(part.get(), (float *) dst->data, DV, n_split, n_rows);
    }
}

template <typename TK, typename TV, int HPW>
static void nsa_sel_attn_dispatch_dims(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int D  = dst->src[0]->ne[0];
    const int DV = dst->src[2]->ne[0];
    const int bs = ggml_get_op_params_i32(dst, 0);

    if (D == 64 && DV == 64 && bs == 64) {
        nsa_sel_attn_launch<TK, TV, 64, 64, 64, HPW>(ctx, dst);
    } else {
        nsa_sel_attn_launch<TK, TV, 0, 0, 0, HPW>(ctx, dst);
    }
}

template <typename TK, typename TV>
static void nsa_sel_attn_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t G = dst->src[0]->ne[1] / dst->src[1]->ne[1];
    if (nsa_sel_attn_hpw(G) == 4) {
        nsa_sel_attn_dispatch_dims<TK, TV, 4>(ctx, dst);
    } else {
        nsa_sel_attn_dispatch_dims<TK, TV, 1>(ctx, dst);
    }
}

// V type for a given K type, only the combinations of nsa_kv_types_ok are instantiated
template <typename TK>
static void nsa_sel_attn_dispatch_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr bool k_fp = std::is_same_v<TK, half> || std::is_same_v<TK, float>;
    constexpr bool k_lp = std::is_same_v<TK, half> || nsa_kv_quant<TK>::value;
    const ggml_type tv = dst->src[2]->type;
    if (tv == GGML_TYPE_F16) {
        nsa_sel_attn_dispatch<TK, half>(ctx, dst);
        return;
    }
    if constexpr (k_fp) {
        if (tv == GGML_TYPE_F32) {
            nsa_sel_attn_dispatch<TK, float>(ctx, dst);
            return;
        }
    }
    if constexpr (k_lp) {
        if (tv == GGML_TYPE_Q8_0) {
            nsa_sel_attn_dispatch<TK, block_q8_0>(ctx, dst);
            return;
        }
        if (tv == GGML_TYPE_Q4_0) {
            nsa_sel_attn_dispatch<TK, block_q4_0>(ctx, dst);
            return;
        }
    }
    GGML_ABORT("unsupported V type %s", ggml_type_name(tv));
}

void ggml_cuda_op_nsa_sel_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(dst->src[3]));

    if (dst->src[0]->ne[2] == 0) {
        return;
    }

    GGML_ASSERT(nsa_kv_types_ok(k->type, v->type));
    switch (k->type) {
        case GGML_TYPE_F16:  nsa_sel_attn_dispatch_v<half>(ctx, dst);       break;
        case GGML_TYPE_F32:  nsa_sel_attn_dispatch_v<float>(ctx, dst);      break;
        case GGML_TYPE_Q8_0: nsa_sel_attn_dispatch_v<block_q8_0>(ctx, dst); break;
        case GGML_TYPE_Q4_0: nsa_sel_attn_dispatch_v<block_q4_0>(ctx, dst); break;
        default: GGML_ABORT("unsupported K type %s", ggml_type_name(k->type));
    }
}

bool ggml_cuda_nsa_sel_attn_supported(int device, const ggml_tensor * op) {
    const ggml_tensor * q   = op->src[0];
    const ggml_tensor * k   = op->src[1];
    const ggml_tensor * v   = op->src[2];
    const ggml_tensor * idx = op->src[3];
    const ggml_tensor * pos = op->src[4];

    if (q->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32 || pos->type != GGML_TYPE_I32) {
        return false;
    }
    if (!nsa_kv_types_ok(k->type, v->type)) {
        return false;
    }
    // quantized K/V: block-contiguous rows of whole blocks (a transposed quantized V cannot exist)
    for (const ggml_tensor * t : { k, v }) {
        if (ggml_is_quantized(t->type) && (t->nb[0] != ggml_type_size(t->type) || t->ne[0] % ggml_blck_size(t->type) != 0 ||
                t->nb[1] % 2 != 0 || t->nb[2] % 2 != 0)) {
            return false;
        }
    }
    if (q->nb[0] != sizeof(float) || !ggml_is_contiguous(idx) || !ggml_is_contiguous(op)) {
        return false;
    }

    const int64_t D  = q->ne[0];
    const int64_t DV = v->ne[0];
    const int64_t bs = ggml_get_op_params_i32(op, 0);
    if (DV > NSA_MAX_DV || bs > NSA_MAX_BS) {
        return false;
    }

    const int64_t G   = q->ne[1] / k->ne[1];
    const int     hpw = nsa_sel_attn_hpw(G);
    return nsa_sel_attn_smem(D, DV, bs, nsa_sel_attn_nw(G, hpw), hpw) <= ggml_cuda_info().devices[device].smpbo;
}
