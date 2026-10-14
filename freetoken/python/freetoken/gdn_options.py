"""CPU-only optional GDN configuration; no kernel or model imports."""

def fla_variant(cuda_gdn, value):
    if value in (None, '', '0'):
        return None
    if value not in ('1', 'wu', 'ho'):
        raise ValueError('QWEN36_CUDA_GDN_FLA must be unset, 0, 1, wu or ho')
    if not cuda_gdn:
        raise ValueError('QWEN36_CUDA_GDN_FLA requires QWEN36_CUDA_GDN=1')
    return 'ho' if value == 'ho' else 'wu'
