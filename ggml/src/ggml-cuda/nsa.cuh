#include "common.cuh"

void ggml_cuda_op_nsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_nsa_sel_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_nsa_select_supported(int device, const ggml_tensor * op);
bool ggml_cuda_nsa_sel_attn_supported(int device, const ggml_tensor * op);
