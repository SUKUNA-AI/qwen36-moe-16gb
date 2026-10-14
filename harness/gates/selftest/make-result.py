import datetime
import hashlib
import json
import pathlib

H = pathlib.Path('/path/to/qwen36-workspace/work/agents/final-gates')
PFX = '/path/to/qwen36-workspace/'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
scripts = ['final-gates.sh', 'final-install.sh', 'final-prep.sh', 'final-helpers.py', 'parent-lockwrap.py', 'plan.md']
cfg = {}
for mode in ('in', 'out'):
    for name in ('main-release', 'long256-release'):
        p = H / 'selftest' / f'gather-{mode}' / f'{name}.json'
        cfg.setdefault(f'gather_{mode}', {})[name + '.json'] = {'text': p.read_text(), 'sha256': sha(p), 'bytes': p.stat().st_size}
result = {
    'schema': 1, 'agent': 'final-gates',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'status': 'DRAFT (CPU only; no GPU step, sync, qualify, matrix, soak, release or live step was run)',
    'decision_inputs': {'gather': 'in (parent ~12:27Z: r3 teacher 125/125, decode +5.6%/+4.8%, VRAM gate PASS)'},
    'scripts': {s: str(H / s).replace(PFX, '') for s in scripts},
    'sha256': {s: sha(H / s) for s in scripts},
    'usage': {
        'prep': 'bash work/agents/final-gates/final-prep.sh --gather in --apply   # only P2 (allowlist) left; P1/P3 done by the parent',
        'dry_run': 'bash work/agents/final-gates/final-gates.sh --gather in --dry-run',
        'gates': 'bash work/agents/final-gates/final-gates.sh --gather in --run-id f1   # A,B,D,C,E,Z; ~3h25m wall; run in background',
        'install': 'bash work/agents/final-gates/final-install.sh --run-id f1 --gather in --decision <reviewed decision.json> '
                   '[--accept-inconclusive-speed] [--check-only | --no-live | --live-extra]'},
    'needed_allowlist_names': ['teacher-main-release', 'teacher-long256-release'],
    'allowlist_state_at_handoff': "ABSENT in work/FreeToken-opt/python/freetoken/engine/qualification.py; A0 stops until added "
                                  "(insert after the anchor 'teacher-main-combined',)",
    'needed_configs': {
        'derivation': {
            'main-release.json': 'r2-attn.json (pinned in final-helpers.PINNED_R2ATTN; refuses if it changed) + {QWEN36_DECODE_GATHER: tuned, '
                                 'TVM_FFI_CUDA_ARCH_LIST: "12.0"} when gather in (cross-checked == main-candidate-r3.json) + '
                                 '{FREETOKEN_DISABLE_JIT: "1"}; compact JSON (separators , :), no trailing newline, O_EXCL, never overwritten',
            'long256-release.json': 'main-release.json with context 262144 and memory_ratio 0.73 (identical env: the L4 identity check vs '
                                    'arm main-release allows only context/memory_ratio/qualified to differ)',
            'created_by': 'final-gates.sh A1 (final-helpers.py configs --gather in --opt outputs/qwen36-api/optimization --write); synced '
                          'with --extra-config main-release.json --extra-config long256-release.json'},
        **cfg},
    'profiles_json': {
        'builder': 'final-helpers.py profiles: release-tooling staging/profiles-draft.json (sha dce3a145) + layer_major_attn_batch: 2 '
                   '(int; patch 01 VALUE_FIELDS key) in main128/long256; decode_gather kept (gather in) or deleted (gather out); '
                   'qualified=true exactly for the decided profiles',
        'canonical_sha256_gather_in': {
            'main128': '542e0fe6adfbfb90d75c09606788e6afc7e21c0559122280b4dcde23cbaad1a9',
            'long256': 'c6f8ad912082e61ac8987773aea5323619542d660fc8ddcf5211f941bdf3b2a3',
            'baseline64': '982cd95851eaa86e69394ab17337c414c3c098df8006392d2fb6a634baca7662'},
        'matches_parent_assembly': True},
    'make_qualification_evidence': {
        'consumed_by': 'make-qualification.py build_receipt (patched 6398c923): CUDA_DECISION_EVIDENCE {cuda_nvfp4_prefill: '
                       'cuda_nvfp4_evidence, cuda_gdn: cuda_gdn_evidence, cuda_fp8_gemv: cuda_fp8_gemv_evidence}, each '
                       '{run_receipts, bitexact_reports} non-empty and listed in evidence; run receipts -> check_run_receipt '
                       '(configuration -> profile semantics, staged runtime source_files, service files, packages, native library '
                       'sha256); bitexact reports -> check_bitexact_report (compare-quality report: expected==observed>0, '
                       'max_abs==max_relative_l2==0, forced targets, quality_gate_pass, no new functional failures, baseline teacher-bf128)',
        'produced_by': {
            'bitexact_reports': 'reports/quality-<cand>.json (qualify.py -> compare-quality.py)',
            'run_receipts(cuda)': 'reports/final-<run>-qrun-teacher-<cand>.json = run object of reports/teacher-<cand>.json '
                                  '(teacher diagnostic env allowed by RUN_DIAGNOSTIC_ENV)',
            'run_receipts(profile)': 'main128: final-<run>-qrun-functional-main-release.json, final-<run>-c128-aK-run.json, '
                                     'soak-final-<run>-aK-run.json; long256: final-<run>-qrun-functional-long256-release.json, '
                                     'final-<run>-l256-aK-run.json; baseline64: final-<run>-b64-aK-run.json'},
        'decision_draft': 'final-gates.sh Z writes reports/final-gates-<run>-decision-draft.json (O_EXCL) with evidence, run_receipts '
                          'and cuda_* filled; known_limitations start with a TODO line that final-install.sh refuses'},
    'gates': {
        'order': 'A, B, D, C, E, Z (C after D: C0/L4/F1 identity and the soak --expected-identity bind to the D matrix manifest)',
        'freeze_exceptions_allowed': [
            'outputs/qwen36-api/profile-runtime.py 0f30ee09c751', 'outputs/qwen36-api/optimization/release.py 6d8328c3d48e',
            'outputs/qwen36-api/optimization/make-qualification.py 6398c923d3a7',
            'outputs/qwen36-api/optimization/run-experiment.sh a06b0b29a178', 'outputs/qwen36-api/optimization/bench.py 204d28710ef0',
            'outputs/qwen36-api/optimization/screen-chunk.py 798a2728f941'],
        'matrix_arms': ['bf128 (baseline: checklist C1, compare-ab default, final3/final4 baseline)', 'main-release'],
        'per_profile_gates': {
            'main128': 'A, B.main-release, D.pilot, D.matrix=PASS, D.compare in {PASS, INCONCLUSIVE}, C.c0 in {PASS, SKIPPED_BY_PARENT}, '
                       'E.pilot, E.soak=PASS',
            'long256': 'A, B.long256-release, D.manifest, C.l4=PASS',
            'baseline64': 'A, D.manifest, C.f1 in {PASS, PASS_WITH_KNOWN_LIMITATION}'}},
    'estimates': {
        'gpu_minutes': {
            'A_sync_apply': 0.3, 'B_qualify_main_release': '4-6 (main-final 5.4, r2 4.6, attn 3.2)',
            'B_qualify_long256_release': '7-10 [UNSURE: no prior 256K qualify]', 'D_tool_pilot': 2,
            'D_matrix_10_arms': '35-42 (final4-r1b 3.5 min/arm, final3-v5 4.2)', 'D_compare': '<1', 'C0_main128_gates': '6 [UNSURE]',
            'C_L4_long256': '6-8 (r1b 7.4)', 'C_F1_baseline64': 4, 'E_soak_pilot': 2.5, 'E_soak_7200s': '123-126 (r1b 122)'},
        'cpu_minutes': {'A_total': '~5', 'Z': '<1'},
        'total_gpu_minutes': '~196 (range 180-220)', 'total_wall': '~3h25m + lock waits',
        'final_install': '25-45 min (F0-F4 ~5-10, live 15-25, +10-15 with --live-extra)'},
    'selftests_cpu': {
        'syntax': 'bash -n OK (3 scripts); ast parse OK (2 helpers); no CRLF',
        'mapping': 'PASS gather in/out vs the staged and the patched profile-runtime + make-qualification; negative controls FAIL as '
                   'expected (gather mismatch, FLA r2 candidate, main-combined without attention batch)',
        'helpers': 'qcheck PASS on quality-main-candidate-r2/main-final; extract-run PASS r2 functional+teacher, FAIL on a wrong config; '
                   'provenance PASS attn1, FAIL when gather required (absent then); summary/decision-check on synthetic steps (TODO refused)',
        'final_prep_check_mode': '12:28Z: P1 applied (8/8), P2 pending (2 names), P3 already done (83 modules prebuilt incl. gather)',
        'final_gates_dry_run': '12:29Z --gather in: all preconditions OK except the 2 allowlist names; mapping PASS; freeze exactly the '
                               '6 allowed MISMATCH; sync-v5 --dry-run rc 0 (0 freeze mismatches); current native provenance: gather '
                               'unknown (expected before the final sync)',
        'artifacts': 'work/agents/final-gates/selftest/'},
    'open_questions': [
        'Order deviation: C runs after D (identity reference = D matrix manifest).',
        'Matrix baseline arm = bf128; add main-final as a 3rd arm (+~20 min, family 18) only if a vs-previous-candidate claim is wanted.',
        'Tool pilot on bf128 (harness-v6 runbook step 7) rather than main-release.',
        'QWEN36_BENCH_WARMUP_S not set for the matrix (V6 default 0); r2/r3 screens used 60.',
        'C1/C2 functional reference = functional-main-release.json (driver4 used main-final functional for baseline64); falling back to '
        'functional-bf128.json is an owner call.',
        'C0 (main128 131072/120000 context+boundary gates) is an addition for checklist C5.1; --skip-c0 makes it non-blocking.',
        'long256-release VRAM at .73 with FP8 GEMV + attention batching + gather unmeasured; budget-calc preflight models R1b mechanics '
        'only; L4 enforces the 0.5 GiB floor.',
        'Windows interop (cmd.exe qwen36 / powershell.exe from WSL) not exercised; supervisor lifetime when started via interop '
        'unverified; fallback --no-live + ops-live-runbook in PowerShell.',
        'Known-limitation wording in the decision is the parent/owner job; INCONCLUSIVE speed needs --accept-inconclusive-speed.',
        'Sync label final can be applied once; workspace drift after it needs --sync-label final2 + a new --run-id (qualify reports are '
        'overwritten on re-qualify).',
        'network-acceptance assumed to return exactly 60 rows (runbook); other counts are flagged FAIL.',
        'F1 rewrites outputs/qwen36-api/profiles.json from the draft (same canonical content as the parent assembly, different '
        'formatting, qualified flags set); backup kept in work/agents/final-gates/backup/.',
        'Native kernel provenance shows the gather module as unknown until the final sync; experiments started before A4 see that state.'],
    'not_run': [
        'all GPU steps (A4 apply, B, D, C, E)', 'final-gates.sh real mode', 'final-prep.sh --apply',
        'final-install.sh (profiles/receipt/release install/rollback check/Windows release/live ops/network acceptance/forge check)',
        'no STATE.md entry, no GitHub action'],
}
(H / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps(result['sha256'], indent=1))
