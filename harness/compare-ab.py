"""Strict frozen-receipt paired comparison (harness V6) of whole-request benchmarks.

Owner decisions (2026-10-13), V6:
  * The ONLY gate is the full request time (elapsed) of every priority case:
    fresh at every manifest length (64/4096/16384/65536/120000), history_continuation,
    last_message_edit, tool_result_edit (`elapsed_s`) and the natural-EOS tool cycle
    (`task_elapsed_s`). ttft_content_s and decode_client_tps are reported, never gating.
  * Statistics over exactly five pairs: x_i = log(baseline_elapsed_i / candidate_elapsed_i),
    Student t with df=4 on mean(x) with se = sd(x, n-1)/sqrt(5):
      non-inferiority  lower one-sided 97.5% bound exp(mean - t(.975,4)*se) >= 1/1.03
                       for EVERY gating case;
      regression       upper one-sided 97.5% bound exp(mean + t(.975,4)*se) < 1/1.03
                       for ANY gating case -> FAIL;
      gain             lower bound exp(mean - t(1-.025/F,4)*se) >= 1.05 for at least one
                       gating case, F = gating cases x candidates of the matrix family
                       (len(manifest profiles)-1), Bonferroni;
      gain excluded    upper one-sided 97.5% bound < 1.05 for EVERY gating case -> FAIL
                       (intersection-union; V5 rule kept).
  * Per arm: launcher identity (files + start environment) bound to the manifest and equal
    for both arms; service-sampled process VRAM footprint <= total - 1 GiB (bench.vram_budget; above: INVALID).
Verdicts:
  PASS          complete valid data, non-inferior on every gating case and gain established
  FAIL          complete valid data, a >3% regression or a 5% gain is excluded
  INCONCLUSIVE  missing pairs/reports/gating metrics, or intervals decide neither way
  INVALID       contract violation: wrong run/source/library/manifest/launcher, changed
                output budget/sampling/input, failed tool cycle, VRAM floor, etc.
Only PASS sets performance_gate_pass. Exit codes: PASS 0, FAIL 1, INCONCLUSIVE 2,
INVALID 3, 4 = internal/usage error or an existing comparison file (no verdict),
75 = GPU-exclusive lock busy (no verdict). The comparison file is created with O_EXCL
and never overwritten. Native runtime/config/harness/launcher are re-hashed against the
manifest while this process holds the GPU-exclusive lock (non-blocking), so no matrix,
pilot, sync-v5 --apply or GPU job can change native files during the check.
"""
import argparse,json,math,os,statistics,sys
import bench
from bench import (DEFAULT_LENGTHS,HARNESS_VERSION,BENCH_FIXTURE_VERSION,LAUNCHER_FILES,WINDOWS_RESERVE_BYTES,GpuLockBusy,
                   acquire_gpu_lock,parse_lengths,validate_receipt,current_run_spec,expected_cases,valid_sha,
                   cache_requirement,canonical_hash,vram_budget)
import tool_fixture_v5 as tool_fixture

PAIRS=5
T_DF=PAIRS-1
NONINFERIORITY=1/1.03
GAIN=1.05
ALPHA_ONE_SIDED=.025
GATING_METRIC='elapsed_s'
REPORTED_METRICS=('ttft_content_s','elapsed_s','decode_client_tps')
EXIT_CODES={'PASS':0,'FAIL':1,'INCONCLUSIVE':2,'INVALID':3}
EXIT_INTERNAL=4
EXIT_GPU_LOCK_BUSY=75


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    """Usage errors must not exit 2 (= INCONCLUSIVE)."""
    def error(self,message):raise UsageError(message)


def t_quantile(p,df=T_DF):
    """Upper quantile of Student t with 4 degrees of freedom (closed-form CDF, bisection)."""
    if df!=4:raise ValueError('Only df=4 (five pairs) is implemented')
    if not (type(p) in (int,float) and .5<=p<1):raise ValueError('p must be in [0.5, 1)')
    # F(t)=1/2+(3/8)s(1-s^2/12) with s=2t/sqrt(4+t^2) in [0,2) (monotone).
    lo,hi=0.,2.
    for _ in range(200):
        mid=(lo+hi)/2
        if .5+.375*mid*(1-mid*mid/12)<p:lo=mid
        else:hi=mid
    s=(lo+hi)/2
    return 2*s/math.sqrt(4-s*s)


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def summarize(ratios,family=None):
    """t-interval (df=4) of the geometric-mean paired ratio; family!=None adds the Bonferroni gain bound."""
    if len(ratios)!=PAIRS or not all(positive(v) for v in ratios):
        raise ValueError('Exactly five positive paired ratios required')
    logs=[math.log(v) for v in ratios]
    mean=statistics.fmean(logs)
    se=statistics.stdev(logs)/math.sqrt(PAIRS)
    t=t_quantile(1-ALPHA_ONE_SIDED)
    out={'geomean':math.exp(mean),'log_mean':mean,'log_se':se,'t_975':t,
         'lower_975':math.exp(mean-t*se),'upper_975':math.exp(mean+t*se)}
    if family is not None:
        if type(family) is not int or family<1:raise ValueError('Invalid comparison family size')
        tg=t_quantile(1-ALPHA_ONE_SIDED/family)
        out.update(gain_family_size=family,gain_t=tg,gain_lower_bonferroni=math.exp(mean-tg*se))
    return out


def stats_row(case,length,metric,ratios,gating,family):
    row={'case':case,'length':length,'metric':metric,'gating':gating,'pairs':len(ratios),'ratios':ratios}
    if len(ratios)==PAIRS:row.update(summarize(ratios,family if gating else None))
    return row


def load(path,failures,label):
    """Missing evidence is INCONCLUSIVE ('missing_' codes); corrupt evidence is INVALID."""
    try:return json.loads(path.read_text())
    except FileNotFoundError:failures.append('missing_'+label)
    except (OSError,ValueError):failures.append('unreadable_'+label)
    return None


def as_dict(value):
    return value if isinstance(value,dict) else {}


def compare_tool_tasks(root,name,baseline,candidate,manifest,fixed_runs,failures,observations):
    ratios=[]
    def require(condition,code):
        if not condition:failures.append(code)
        return bool(condition)
    expected_fixture=manifest.get('tool_fixture')
    for repeat in range(PAIRS):
        pair={}
        for arm in (baseline,candidate):
            stem=f'{name}-{repeat}-{arm}'
            doc=load(root/'reports'/f'{stem}-tool-task.json',failures,f'tool_report_{repeat}_{arm}')
            if doc is None:continue
            run=as_dict(doc.get('run'))
            try:validate_receipt(run,arm,manifest['arms'][arm])
            except (ValueError,KeyError,TypeError):require(False,f'tool_receipt_{repeat}_{arm}')
            require(doc.get('schema_version')==2 and doc.get('mode')=='matrix' and doc.get('name')==stem
                    and doc.get('repeat')==repeat and doc.get('fixture')==expected_fixture
                    and doc.get('source_sha256')==manifest['harness_source_files']['tool-task-bench.py'],f'tool_report_binding_{repeat}_{arm}')
            if (repeat,arm) in fixed_runs:
                require(run.get('run_id')==fixed_runs[(repeat,arm)],f'tool_run_binding_{repeat}_{arm}')
            else:failures.append(f'missing_fixed_report_for_tool_{repeat}_{arm}')
            row=as_dict(doc.get('row'))
            observations['tool_rows'][(repeat,arm)]=row
            if not require(doc.get('status')=='complete' and row.get('correct') is True and row.get('failure_codes')==[],
                           f'tool_cycle_failed_{repeat}_{arm}'):
                codes=doc.get('failure_codes') if isinstance(doc.get('failure_codes'),list) else []
                observations['tool_failures'].append({'repeat':repeat,'arm':arm,
                    'failure_codes':[code for code in codes if code in tool_fixture.FAILURE_CODES]})
                continue
            steps=row.get('steps') if isinstance(row.get('steps'),list) else []
            require(row.get('case')=='tool_cycle' and row.get('repeat')==repeat and row.get('mode')=='matrix'
                    and isinstance(expected_fixture,dict) and row.get('fixture_sha256')==expected_fixture.get('sha256')
                    and row.get('task_fixture_sha256')==expected_fixture.get('task_fixture_sha256')
                    and row.get('tool_result_matches_call_id') is True and row.get('safe_tool_result')==tool_fixture.EXPECTED_RESULT
                    and valid_sha(row.get('returned_call_id_sha256')) and len(steps)==2,f'tool_correct_cycle_{repeat}_{arm}')
            pair[arm]=row
            for index,step in enumerate(steps[:2]):
                step=as_dict(step);usage=as_dict(step.get('usage'));sampling=step.get('sampling');engine=as_dict(step.get('engine'))
                metrics=(step.get('engine_metrics_status')=='joined' and bool(engine) and isinstance(step.get('peaks'),dict)
                         and bool(step['peaks']) and type(step.get('cached_prompt_tokens')) is int)
                if not metrics:failures.append(f'missing_tool_metrics_{repeat}_{arm}_{index}')
                require(step.get('status')==200 and step.get('reasoning_chars')==0
                        and (not metrics or step.get('serving_run_id')==engine.get('run_id')==run.get('run_id'))
                        and step.get('finish_reason')==('tool_calls' if index==0 else 'stop')
                        and sampling=={**tool_fixture.SAMPLING,'max_tokens':tool_fixture.STEP_LIMITS[index],'thinking':False,'ignore_eos':False}
                        and all(type(usage.get(k)) is int and usage[k]>0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
                        and usage.get('total_tokens')==usage.get('prompt_tokens',0)+usage.get('completion_tokens',0),
                        f'tool_step_contract_{repeat}_{arm}_{index}')
                observations['tool_cache'].append({'repeat':repeat,'arm':arm,'step':index,
                    'cached_prompt_tokens':step.get('cached_prompt_tokens'),'cache_hit_observed':step.get('cache_hit_observed')})
            timed=positive(row.get('task_elapsed_s')) and len(steps)==2 and all(positive(as_dict(s).get('elapsed_s')) for s in steps)
            if not timed:failures.append(f'missing_tool_timing_{repeat}_{arm}')
            else:require(math.isclose(row['task_elapsed_s'],sum(s['elapsed_s'] for s in steps),rel_tol=1e-9),f'tool_timing_sum_{repeat}_{arm}')
        if len(pair)!=2:continue
        b,c=pair[baseline],pair[candidate]
        for key in ('task_fixture_sha256','second_history_normalized_sha256'):
            require(valid_sha(b.get(key)) and b.get(key)==c.get(key),f'tool_same_{key}_{repeat}')
        counts=lambda row:[{key:as_dict(as_dict(step).get('usage')).get(key) for key in ('prompt_tokens','completion_tokens','total_tokens')}
                           for step in row.get('steps',[])]
        require(counts(b)==counts(c),f'tool_equal_actual_tokens_{repeat}')
        if positive(b.get('task_elapsed_s')) and positive(c.get('task_elapsed_s')):
            ratios.append(b['task_elapsed_s']/c['task_elapsed_s'])
    return ratios


def verdict_of(failures,complete,rows):
    invalid=[code for code in failures if not code.startswith('missing_')]
    if invalid:return 'INVALID',['contract_violation']
    if failures or not complete:return 'INCONCLUSIVE',['missing_data']
    gating=[r for r in rows if r['gating']]
    if not gating:return 'INCONCLUSIVE',['missing_data']
    noninferior=all(r['lower_975']>=NONINFERIORITY for r in gating)
    gain=any(r['gain_lower_bonferroni']>=GAIN for r in gating)
    regressions=[r for r in gating if r['upper_975']<NONINFERIORITY]
    if regressions:return 'FAIL',[f'regression_over_3pct_established:{r["case"]}_{r.get("length")}_{r["metric"]}' for r in regressions]
    if noninferior and gain:return 'PASS',['noninferior_all_gating_cases','gain_at_least_5pct_bonferroni']
    gain_excluded=all(r['upper_975']<GAIN for r in gating)
    if gain_excluded:return 'FAIL',['five_pct_gain_excluded']
    return 'INCONCLUSIVE',[*(['noninferiority_not_established'] if not noninferior else []),
                           *(['gain_not_established'] if not gain else [])]


def check_launcher(report,manifest,label,require):
    """Bench-report launcher identity: manifest files, clean static contract, self-consistent hash."""
    launcher=as_dict(report.get('launcher'))
    env=launcher.get('environment')
    ok=require(launcher.get('schema_version')==1 and launcher.get('files')==manifest.get('launcher_files')
               and isinstance(env,dict) and bool(env) and launcher.get('environment_sha256')==canonical_hash(env)
               and launcher.get('problems')==[],f'launcher_identity_{label}')
    return launcher.get('environment_sha256') if ok else None


def compare(root,name,candidate,baseline,lengths=DEFAULT_LENGTHS,output=128,verify_current=True):
    lengths=parse_lengths(lengths)
    failures=[];groups={};positions=[]
    observations={'cache':[],'tool_cache':[],'tool_failures':[],'content_differs':0,'content_compared':0,
                  'report_metric_gaps':[],'vram':[],'tool_rows':{}}
    def require(condition,code):
        if not condition:failures.append(code)
        return bool(condition)
    def result(**extra):
        complete=extra.pop('complete',False);verdict_rows=extra.pop('verdict_rows',[])
        verdict,reasons=verdict_of(failures,complete,verdict_rows)
        base={'candidate':candidate,'baseline':baseline,'name':name,'harness_version':HARNESS_VERSION,
              'manifest_harness_version':manifest.get('harness_version') if isinstance(manifest,dict) else None,
              'verdict':verdict,'verdict_reasons':reasons,'contract_pass':not failures,'failures':failures,
              'rows':[],'cache_observations':observations['cache'],'tool_cache_observations':observations['tool_cache'],
              'tool_failures':observations['tool_failures'],'report_metric_gaps':observations['report_metric_gaps'],
              'vram_observations':observations['vram'],'drift_keys':observations.get('drift_keys',[]),
              'output_content':{'compared':observations['content_compared'],'differs':observations['content_differs'],
                                'note':'observation only; quality is gated separately'},
              'performance_gate_pass':verdict=='PASS'}
        base.update(extra);return base
    manifest=load(root/'reports'/f'{name}-manifest.json',failures,'manifest')
    if not isinstance(manifest,dict):
        if manifest is not None:failures.append('manifest_schema')
        return result()
    expected_keys=expected_cases(lengths)
    require(manifest.get('harness_version')==HARNESS_VERSION,'manifest_not_harness_v6')
    require(manifest.get('schema_version')==1,'manifest_schema')
    require(manifest.get('pairs')==PAIRS,'manifest_requires_five_pairs')
    require(manifest.get('lengths')==list(lengths),'manifest_lengths_mismatch')
    require(manifest.get('output')==output,'manifest_output_mismatch')
    require(manifest.get('bench_fixture_version')==BENCH_FIXTURE_VERSION,'manifest_bench_fixture_version')
    require(manifest.get('tool_fixture')==tool_fixture.describe(),'manifest_tool_fixture_mismatch')
    require(isinstance(manifest.get('tool_pilot'),dict) and valid_sha(manifest['tool_pilot'].get('report_sha256')),'manifest_tool_pilot_missing')
    require(baseline!=candidate and baseline in manifest.get('profiles',[]) and candidate in manifest.get('profiles',[]),'manifest_profiles')
    launcher_files=manifest.get('launcher_files')
    require(isinstance(launcher_files,dict) and set(launcher_files)==set(LAUNCHER_FILES)
            and all(valid_sha(v) for v in launcher_files.values()),'manifest_launcher_files')
    require(manifest.get('windows_reserve_bytes')==WINDOWS_RESERVE_BYTES,'manifest_windows_reserve')
    if failures:return result()
    if verify_current:
        current=current_run_spec(root,manifest['profiles'],lengths,output,PAIRS)
        drift=sorted(key for key,value in current.items() if manifest.get(key)!=value)
        require(not drift,'current_runtime_or_config_differs_from_manifest')
        if drift:observations['drift_keys']=drift
    expected_arms=manifest['arms']
    common_keys=('source_sha256','service_source_files','checkpoint','checkpoint_revision','packages')
    require(all(expected_arms[baseline][key]==expected_arms[candidate][key] for key in common_keys),'cross_arm_common_source_mismatch')
    schedule=manifest.get('schedule',[])
    from matrix import make_schedule
    require(schedule==make_schedule(manifest['profiles'],PAIRS),'manifest_schedule_incomplete_or_unbalanced')
    matrix=load(root/'reports'/f'{name}-matrix.json',failures,'matrix_report')
    matrix=matrix if isinstance(matrix,list) else []
    fixed_runs={};arm_vram={};arm_pre_start={};launcher_envs=set()
    for repeat in range(PAIRS):
        indexed={}
        for arm in (baseline,candidate):
            stem=f'{name}-{repeat}-{arm}'
            report=load(root/'reports'/f'{stem}.json',failures,f'report_{repeat}_{arm}')
            if not isinstance(report,dict):continue
            receipt=as_dict(report.get('run'))
            try:validate_receipt(receipt,arm,expected_arms[arm])
            except (ValueError,KeyError,TypeError):require(False,f'receipt_{repeat}_{arm}')
            fixed_runs[(repeat,arm)]=receipt.get('run_id')
            launcher_env=check_launcher(report,manifest,f'{repeat}_{arm}',require)
            if launcher_env is not None:launcher_envs.add(launcher_env)
            benchmark=as_dict(report.get('benchmark'))
            require(report.get('name')==stem,f'report_name_{repeat}_{arm}')
            require(benchmark.get('schema_version')==1 and benchmark.get('harness_version')==HARNESS_VERSION
                    and benchmark.get('fixture_version')==BENCH_FIXTURE_VERSION
                    and benchmark.get('output')==output and benchmark.get('lengths')==list(lengths)
                    and benchmark.get('repeat_start')==repeat and benchmark.get('repeats')==1
                    and benchmark.get('source_sha256')==manifest['harness_source_files']['bench.py'],f'benchmark_receipt_{repeat}_{arm}')
            completed=[r for r in matrix if r.get('repeat')==repeat and r.get('profile')==arm and r.get('status')=='complete']
            planned=[r for r in schedule if r.get('repeat')==repeat and r.get('profile')==arm]
            if not completed:failures.append(f'missing_matrix_entry_{repeat}_{arm}')
            elif require(len(completed)==len(planned)==1,f'matrix_entry_{repeat}_{arm}'):
                entry=completed[0];vram=as_dict(entry.get('vram'))
                require(entry.get('schedule_position')==planned[0].get('schedule_position')
                        and entry.get('pair_position')==planned[0].get('pair_position')
                        and entry.get('harness_version')==HARNESS_VERSION
                        and entry.get('run_id')==receipt.get('run_id'),f'matrix_run_binding_{repeat}_{arm}')
                require(as_dict(entry.get('launcher')).get('environment_sha256')==as_dict(report.get('launcher')).get('environment_sha256')
                        and as_dict(entry.get('launcher')).get('files')==manifest.get('launcher_files'),f'matrix_launcher_binding_{repeat}_{arm}')
                require(entry.get('arm_verdict')=='VALID' and vram.get('gate') in ('PASS','UNAVAILABLE')
                        and vram.get('windows_reserve_bytes')==WINDOWS_RESERVE_BYTES,f'matrix_arm_invalid_{repeat}_{arm}')
                arm_pre_start[(repeat,arm)]=vram.get('pre_start_gpu')
                observations['vram'].append({'repeat':repeat,'arm':arm,'matrix_process_peak_bytes':vram.get('process_peak_bytes'),
                    'matrix_min_gpu_free_bytes':vram.get('min_gpu_free_bytes'),
                    'foreign_gpu_used_bytes_pre_start':vram.get('foreign_gpu_used_bytes_pre_start'),'gate':vram.get('gate')})
                positions.append({k:planned[0][k] for k in ('repeat','profile','schedule_position','pair_position')})
            all_rows=[r for r in report.get('rows',[]) if isinstance(r,dict)] if isinstance(report.get('rows'),list) else []
            arm_vram[(repeat,arm)]=all_rows
            rows=[r for r in all_rows if r.get('case')!='warmup']
            keys=[(r.get('case'),r.get('length')) for r in rows]
            require(len(keys)==len(set(keys)),f'duplicate_case_{repeat}_{arm}')
            require(set(keys)<=expected_keys,f'unexpected_case_{repeat}_{arm}')
            if not expected_keys<=set(keys):failures.append(f'missing_case_{repeat}_{arm}')
            indexed[arm]=dict(zip(keys,rows))
            for row in rows:
                label=f'{repeat}_{arm}_{row.get("case")}_{row.get("length")}'
                usage=as_dict(row.get('usage'));sampling=as_dict(row.get('sampling'));engine=as_dict(row.get('engine'))
                metrics=(row.get('engine_metrics_status')=='joined'
                         and all(type(engine.get(k)) in (int,float) and math.isfinite(engine[k]) and engine[k]>=0
                                 for k in ('prefill_gpu_ms','decode_gpu_ms','engine_span_s'))
                         and isinstance(row.get('peaks'),dict) and bool(row['peaks'])
                         and type(row.get('cached_prompt_tokens')) is int and row['cached_prompt_tokens']>=0
                         and type(row.get('cache_hit_observed')) is bool)
                if not metrics:failures.append(f'missing_metrics_{label}')
                elif not positive(row['peaks'].get('gpu_total_used_bytes')):failures.append(f'missing_gpu_used_sample_{label}')
                require(row.get('repeat')==repeat and (not metrics or (row.get('serving_run_id')==receipt.get('run_id')
                        and engine.get('run_id')==receipt.get('run_id'))),f'row_run_binding_{label}')
                require(row.get('status')==200 and usage.get('completion_tokens')==sampling.get('max_tokens')==output
                        and sampling.get('ignore_eos') is True and sampling.get('thinking') is False
                        and row.get('reasoning_chars')==0 and row.get('finish_reason')=='length',f'output_contract_{label}')
                require(all(type(usage.get(k)) is int and usage[k]>0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
                        and usage.get('total_tokens')==usage.get('prompt_tokens',0)+usage.get('completion_tokens',0),f'usage_schema_{label}')
                require(row.get('cache_requirement')==cache_requirement(row.get('case')),f'cache_semantics_{label}')
                if metrics:require(row['cache_hit_observed']==(row['cached_prompt_tokens']>0),f'actual_cache_label_{label}')
        if len(indexed)!=2:continue
        for key in sorted(expected_keys):
            b=indexed[baseline].get(key);row=indexed[candidate].get(key)
            if b is None or row is None:continue
            label=f'{repeat}_{key[0]}_{key[1]}'
            require(row.get('sampling')==b.get('sampling'),f'same_sampling_{label}')
            require(valid_sha(row.get('messages_sha256')) and row.get('messages_sha256')==b.get('messages_sha256'),f'same_input_{label}')
            require(valid_sha(row.get('request_fixture_sha256')) and row.get('request_fixture_sha256')==b.get('request_fixture_sha256'),f'same_request_fixture_{label}')
            if key[0] in ('last_message_edit','tool_result_edit'):
                for parent,arm in ((b,baseline),(row,candidate)):
                    primer=as_dict(parent.get('primer'));run_id=parent.get('serving_run_id')
                    primer_metrics=primer.get('engine_metrics_status')=='joined' and bool(as_dict(primer.get('engine')))
                    if not primer_metrics:failures.append(f'missing_primer_metrics_{label}_{arm}')
                    require(primer.get('status')==200 and (not primer_metrics or primer.get('serving_run_id')==as_dict(primer.get('engine')).get('run_id')==run_id)
                            and as_dict(primer.get('usage')).get('completion_tokens')==output
                            and primer.get('sampling')==parent.get('sampling'),f'primer_contract_{label}_{arm}')
                require(valid_sha(as_dict(row.get('primer')).get('request_fixture_sha256'))
                        and row['primer']['request_fixture_sha256']==as_dict(b.get('primer')).get('request_fixture_sha256')
                        and as_dict(row['primer'].get('usage')).get('prompt_tokens')==as_dict(as_dict(b.get('primer')).get('usage')).get('prompt_tokens'),
                        f'same_primer_{label}')
            require(as_dict(row.get('usage')).get('prompt_tokens')==as_dict(b.get('usage')).get('prompt_tokens'),f'same_prompt_tokens_{label}')
            counts=[r.get('cached_prompt_tokens') for r in (b,row)]
            if all(type(v) is int for v in counts):
                # Fresh inputs must not hit. History/edit tasks record observed values on
                # both sides; a zero-token miss is valid evidence and never labelled a hit.
                if key[0]=='fresh':require(counts==[0,0],f'fresh_cache_{label}')
            observations['cache'].append({'repeat':repeat,'case':key[0],'length':key[1],
                'cache_requirement':cache_requirement(key[0]),
                'baseline_cached_tokens':counts[0],'candidate_cached_tokens':counts[1],
                'baseline_hit':b.get('cache_hit_observed'),'candidate_hit':row.get('cache_hit_observed')})
            if valid_sha(b.get('content_sha256')) and valid_sha(row.get('content_sha256')):
                observations['content_compared']+=1;observations['content_differs']+=b['content_sha256']!=row['content_sha256']
            for metric in REPORTED_METRICS:
                if not (positive(row.get(metric)) and positive(b.get(metric))):
                    # Only the gating full-request time can make the comparison INCONCLUSIVE.
                    if metric==GATING_METRIC:failures.append(f'missing_timing_{metric}_{label}')
                    else:observations['report_metric_gaps'].append(f'{metric}_{label}')
                    continue
                ratio=row[metric]/b[metric] if metric=='decode_client_tps' else b[metric]/row[metric]
                groups.setdefault((key,metric),[]).append(ratio)
    task_ratios=compare_tool_tasks(root,name,baseline,candidate,manifest,fixed_runs,failures,observations)
    # VRAM budget per arm over every service-sampled request (bench rows incl. warmup/primers + tool steps).
    for (repeat,arm),bench_rows in sorted(arm_vram.items()):
        tool=observations['tool_rows'].get((repeat,arm))
        budget=vram_budget(bench_rows+([tool] if tool else []),arm_pre_start.get((repeat,arm)))
        if budget['gate']=='UNAVAILABLE':failures.append(f'missing_vram_budget_sample_{repeat}_{arm}')
        else:require(budget['gate']!='INVALID',f'process_vram_over_budget_{repeat}_{arm}')
        observations['vram'].append({'repeat':repeat,'arm':arm,'report_process_peak_bytes':budget['process_peak_bytes'],
                                     'budget_bytes':budget['budget_bytes'],'report_min_gpu_free_bytes':budget['min_gpu_free_bytes']})
    require(len(launcher_envs)<=1,'launcher_environment_differs_between_arms')
    candidates=max(1,len(manifest['profiles'])-1)
    gating_cases=len(expected_keys)+1
    family=gating_cases*candidates
    out=[]
    for key in sorted(expected_keys):
        for metric in REPORTED_METRICS:
            ratios=groups.get((key,metric),[])
            gating=metric==GATING_METRIC
            out.append(stats_row(key[0],key[1],metric,ratios,gating,family))
    task=stats_row('tool_cycle',None,'task_elapsed_s',task_ratios,True,family)
    gating_rows=[r for r in out if r['gating']]+[task]
    complete=len(gating_rows)==gating_cases and all(r['pairs']==PAIRS for r in gating_rows)
    noninferior=complete and all(r['lower_975']>=NONINFERIORITY for r in gating_rows)
    gain=complete and any(r['gain_lower_bonferroni']>=GAIN for r in gating_rows)
    return result(complete=complete,verdict_rows=out+[task],rows=out,tool_task=task,pair_positions=positions,five_pairs=complete,
                  gating_cases=[f'{r["case"]}_{r["length"]}_{r["metric"]}' for r in gating_rows],
                  no_regression_over_3pct_full_request_975=noninferior,
                  whole_request_gain_at_least_5pct_bonferroni=gain,
                  gain_comparison_family_size=family,gain_family={'gating_cases':gating_cases,'candidates':candidates},
                  launcher_environment_sha256=sorted(launcher_envs),windows_reserve_bytes=WINDOWS_RESERVE_BYTES,
                  method=('Student t (df=4) on log paired ratios of exactly five pairs; one-sided 97.5% bounds; '
                          'gain bound Bonferroni over gating cases x candidates'),
                  ratio_units='Latency: baseline/candidate; throughput: candidate/baseline. 1/1.03 means at most 3% longer full request.',
                  limits=['Five pairs; serial/autocorrelated system drift is not modelled beyond the balanced schedule.',
                          'Only full-request elapsed time gates; ttft_content_s and decode_client_tps are reported only.',
                          'decode_client_tps is an SSE/usage estimate, not engine-only or kernel throughput.',
                          'Missing evidence or failure to exclude a 3% regression is not a passing performance gate.',
                          'history_continuation/edit cached tokens are observations; a zero-token miss is not a hit.',
                          'Minimum GPU free is sampled by the service during requests (not during model load).'])


def comparison_path(root,name,candidate,baseline):
    return root/'reports'/f'{name}-{candidate}-vs-{baseline}-comparison.json'


def write_new(path,report):
    """O_EXCL: an existing comparison is never overwritten (FileExistsError)."""
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as target:target.write(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main(argv=None,lock_path=None):
    try:
        p=Parser();p.add_argument('--name',required=True);p.add_argument('--candidate',required=True)
        p.add_argument('--baseline',default='bf128');p.add_argument('--output',type=int,default=128)
        p.add_argument('--lengths',default=','.join(map(str,DEFAULT_LENGTHS)));a=p.parse_args(argv)
        lengths=parse_lengths(a.lengths)
    except (UsageError,ValueError) as error:
        print(json.dumps({'result':'usage_error','message':str(error)[:300]}),file=sys.stderr,flush=True)
        return EXIT_INTERNAL
    path=comparison_path(bench.ROOT,a.name,a.candidate,a.baseline)
    if path.exists() or path.is_symlink():
        print(json.dumps({'result':'refused','reason':'comparison_exists','report':path.name}),file=sys.stderr,flush=True)
        return EXIT_INTERNAL
    try:lock=acquire_gpu_lock(lock_path)
    except GpuLockBusy as error:
        print(json.dumps({'result':'refused','reason':'gpu_lock_busy','message':str(error)}),file=sys.stderr,flush=True)
        return EXIT_GPU_LOCK_BUSY
    try:
        try:
            report=compare(bench.ROOT,a.name,a.candidate,a.baseline,lengths,a.output)
            # Only a V6 manifest owns a comparison file (never a V5/legacy name).
            if report.get('manifest_harness_version')==HARNESS_VERSION:
                write_new(path,report);report['report']=path.name
        except Exception as error:  # no verdict: never confused with FAIL/INCONCLUSIVE
            print(json.dumps({'result':'internal_error','error_type':type(error).__name__,'message':str(error)[:300]}),
                  file=sys.stderr,flush=True)
            return EXIT_INTERNAL
    finally:os.close(lock)
    print(json.dumps(report))
    return EXIT_CODES[report['verdict']]


if __name__=='__main__':raise SystemExit(main())
