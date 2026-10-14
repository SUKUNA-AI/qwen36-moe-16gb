"""Native MTP qualification: exact rollback and independent BF16 chunk gate."""
import importlib.util
import json
import pathlib

ROOT=pathlib.Path('/home/USER/qwen36-api/optimization')


def main():
    path=pathlib.Path(__file__).with_name('diagnose-mtp.py')
    spec=importlib.util.spec_from_file_location('qwen36_mtp_diagnosis',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.diagnose()
    # Preserve historical fields while identifying what each comparison tests.
    result['prefill_vs_incremental_relative_l2']=result['full_vs_incremental']['relative_l2']
    result['rollback_relative_l2']=result['historical_mixed_chunk_vs_full']['relative_l2']
    result['rollback_vs_incremental_relative_l2']=result['historical_mixed_chunk_vs_incremental']['relative_l2']
    (ROOT/'reports/mtp-kernel.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({key:result[key] for key in ['native_tensors','native_weight_bytes',
          'prefill_vs_incremental_relative_l2','rollback_relative_l2','exact_rollback_pass',
          'chunk_numerical_guard_pass','quality_acceptance_claim','pass']}))
    assert result['exact_rollback_pass'],'Same-state/same-shape rollback contract failed'
    assert result['chunk_numerical_guard_pass'],'BF16 chunk numerical gate remains failed; threshold was not relaxed'


if __name__=='__main__':main()
