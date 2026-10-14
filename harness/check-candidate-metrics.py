"""Check numerical placement evidence from the completed functional fixture."""
import argparse,json
from bench import ROOT

p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args()
data=json.loads((ROOT/'reports'/('functional-'+a.candidate+'.json')).read_text())
configuration=data['run']['configuration'];env=configuration.get('env',{})
rows=[]
for row in data['rows']:
    if not row['case'].startswith('long-facts-'):continue
    e=row.get('engine',{})
    checks={
      'metrics_joined':row.get('engine_metrics_status')=='joined',
      'source_and_engine_run_match':e.get('run_id')==data['run']['run_id'],
      'full_host_experts_pinned':e.get('host_expert_bytes',0)>16*1024**3 and e.get('host_expert_bytes')==e.get('host_pinned_expert_bytes'),
      'gpu_experts_in_use':e.get('gpu_expert_bytes',0)>0 and e.get('valid_gpu_expert_slots',0)>0,
      'h2d_payload_measured':e.get('expert_counter_deltas',{}).get('prefill_h2d_bytes',0)>0,
      'kv_bf16':e.get('kv_dtype')=='torch.bfloat16',
      'component_inventory_valid':e.get('model_cuda_state_storage',{}).get('bytes',0)>0 and e.get('allocator_bytes_outside_named_components',-1)>=0,
      'request_peaks_valid':e.get('request_torch_peak_allocated_bytes',0)>=e.get('torch_allocated_bytes',1),
      'attention_gate_loaded':e.get('fused_attention_gate_initialized_layers')==10 if env.get('QWEN36_FUSED_ATTN_GATE')=='1' else True,
      'direct_fp32_snapshot':e.get('fp32_gdn_snapshot_enabled') is True if env.get('QWEN36_FP32_GDN_SNAPSHOT')=='1' else True,
      'retained_snapshot_enabled_and_donated':(e.get('retain_chunk_snapshot_enabled') is True and
          e.get('prefix_snapshot_counters_total',{}).get('retained_donations',0)>0)
          if env.get('QWEN36_RETAIN_CHUNK_SNAPSHOT')=='1' else True,
      'unchanged_gdn_pool':e.get('gdn_bytes')==579502080,
    }
    rows.append({'case':row['case'],'checks':checks})
report={'rows':rows,'pass':len(rows)==2 and all(all(r['checks'].values()) for r in rows)}
(ROOT/'reports'/('placement-'+a.candidate+'.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report));assert report['pass']
