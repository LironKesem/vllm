# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
import torch

from vllm._custom_ops import (
    cutlass_scaled_fp4_mm,
    scaled_fp4_quant,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
# TODO: for now i am importing it like any other external repo(flash-infer), the todo is to port it to helion repo!
#      -- but we will need to decided /or if to avoid the customop
sys.path.insert(0, '/home/redhat-et/src/lkesem/helion/examples')
from nvfp4_gemm import  nvfp4_matmul
from nvfp4_gemm import nvfp4_matmul, quantize_fp4_e2m1, pack_fp4,swizzle_fp8_scales,unpack_and_dequantize_fp4

class HelionNvFp4W4A16LinearKernel(NvFp4LinearKernel):
    """NVFP4 GEMM via the vLLM Helion kernel."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if cutlass_fp4_supported():
            return True, None
        return False, "Helion FP4 kernels not available"

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            raw_weights = layer.weight.data
            raw_scales = layer.weight_scale.data
            
            # 1. Capture the exact original packed layout from memory
            # If raw_weights is [2048, 2048], then row_count = 2048, col_count = 2048
            orig_rows, orig_cols = raw_weights.shape
            
            print(f"\n[HELION-SHAPE] Entering processor with Weight: {raw_weights.shape}, Scale: {raw_scales.shape}")

            # 2. Unpack vLLM's weights back to float space to fix the sub-byte scrambling
            w_unpacked = unpack_and_dequantize_fp4(raw_weights)
            
            # 3. Explicitly enforce [K, N] positioning based on what Helion's packer expects
            # For Helion's packer, Dimension 0 MUST be the true K (which is row_count * 2)
            # Let's inspect the unpacked shape to place K cleanly as rows
            expected_K = orig_rows * 2
            
            if w_unpacked.shape[0] != expected_K:
                # If K isn't the rows, flip the matrix to guarantee [K, N] configuration
                w_unpacked = w_unpacked.T.contiguous()
            else:
                w_unpacked = w_unpacked.contiguous()

            # 4. Re-quantize and repack vertically along the K axis
            w_quantized = quantize_fp4_e2m1(w_unpacked)
            w_packed = pack_fp4(w_quantized).view(torch.float4_e2m1fn_x2)

            # 5. Swizzle the scale factors to match the newly generated weight layout
            if raw_scales.shape[0] != w_packed.shape[1]:
                raw_scales = raw_scales.T.contiguous()
            else:
                raw_scales = raw_scales.contiguous()
                
            layer.weight_scale = torch.nn.Parameter(
                swizzle_fp8_scales(raw_scales), requires_grad=False
            )

            # 6. Run final padding check and pass it to the layer
            padded_weight, weights_padding_cols = pad_nvfp4_weight_for_cutlass(w_packed)
            
            layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)
            layer.weights_padding_cols = weights_padding_cols
            
            print(f"[HELION-SHAPE] Exiting processor. Final Weight Shape: {layer.weight.data.shape}\n")
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:

        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)
        
        x_2d = x.reshape(-1, x.shape[-1])
        #print(f" gemv_helion path: M={x_fp4.shape[0]},N={layer.weight.shape[0]},K={ x_fp4.shape[1]} with backend {backend}")
        alpha_float = float(layer.alpha)
        out = nvfp4_matmul(x_2d, layer.weight, layer.weight_scale, alpha=alpha_float)

        if out.dtype != output_dtype:
            out = out.to(output_dtype)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
