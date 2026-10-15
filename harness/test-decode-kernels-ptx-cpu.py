"""Compile the S8 Triton kernels (moe/pregate.py, moe/shared_epilogue.py) for sm_120 without a GPU and
check the PTX: no ``.shared::cluster`` anywhere; the epilogue keeps the torch roundings (IEEE div.rn,
round-to-nearest bf16 conversion, the bf16 multiply and add not contracted into one fma).
"""
import unittest
from unittest.mock import patch

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from freetoken.moe import pregate as P
from freetoken.moe import shared_epilogue as SE

TARGET = GPUTarget("cuda", 120, 32)
POINTERS = {"res_ptr": "*bf16", "norm_w_ptr": "*bf16", "gate_w_ptr": "*bf16", "logits_ptr": "*fp32",
            "slot_row_ptr": "*i32", "query_ptr": "*i32", "routed_ptr": "*bf16", "shared_ptr": "*bf16",
            "gate_ptr": "*bf16", "out_ptr": "*bf16", "x_ptr": "*bf16", "eps": "fp32"}


def ptx(fn, constexprs, num_warps, fusion):
    sig = {name: ("constexpr" if name in constexprs else POINTERS.get(name, "i32")) for name in fn.arg_names}
    src = ASTSource(fn=fn, signature=sig,
                    constexprs={(fn.arg_names.index(k),): v for k, v in constexprs.items()})
    try:
        kernel = triton.compile(src, target=TARGET, options={"num_warps": num_warps, "enable_fp_fusion": fusion})
    except Exception as exc:  # noqa: BLE001 -- no bundled ptxas/LLVM in this environment
        raise unittest.SkipTest(f"triton cannot compile for sm_120 here: {exc}")
    return kernel.asm["ptx"]


class EpiloguePtx(unittest.TestCase):
    def test_fusion_off_keeps_two_roundings(self):
        text = ptx(SE._epilogue_kernel, {"BLOCK": 1024}, 4, False)
        self.assertNotIn("fma.rn.bf16", text)
        self.assertIn("div.rn.f32", text)
        self.assertIn("cvt.rn.bf16.f32", text)
        self.assertNotIn("shared::cluster", text)
        table = ptx(SE._sigmoid_table_kernel, {"BLOCK": 1024}, 4, False)
        self.assertIn("div.rn.f32", table)

    def test_default_fusion_would_contract(self):
        # the negative control for the flag: Triton's default contracts routed + shared * sig
        self.assertIn("fma.rn.bf16", ptx(SE._epilogue_kernel, {"BLOCK": 1024}, 4, True))

    def test_launches_disable_fusion(self):
        seen = []

        class Kernel:
            def __getitem__(self, grid):
                return lambda *a, **kw: seen.append(kw)

        with patch.object(SE, "_epilogue_kernel", Kernel()):
            SE._launch(SE.torch.zeros(1, 8, dtype=SE.torch.bfloat16), SE.torch.zeros(1, 8, dtype=SE.torch.bfloat16),
                       SE.torch.zeros(1, 1, dtype=SE.torch.bfloat16), SE.torch.zeros(1, 8, dtype=SE.torch.bfloat16))
        self.assertIs(seen[0]["enable_fp_fusion"], False)
        source = (SE.__file__ and open(SE.__file__, encoding="utf-8").read())
        call = source[source.index("_sigmoid_table_kernel[("):]
        self.assertIn("enable_fp_fusion=False", call[:call.index(")\n")])


class PregatePtx(unittest.TestCase):
    def test_compiles_without_cluster_shared(self):
        for fn, const, warps in ((P._pregate_logits_kernel, {"H": 2048, "BLOCK_H": 2048, "BLOCK_E": 8}, 8),
                                 (P._pregate_select_kernel, {"TOP_K": 8, "Q": 2, "BLOCK": 256}, 1)):
            self.assertNotIn("shared::cluster", ptx(fn, const, warps, True), fn.__name__)


if __name__ == "__main__":
    unittest.main()
