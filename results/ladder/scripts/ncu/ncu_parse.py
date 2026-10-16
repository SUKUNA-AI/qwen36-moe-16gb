"""ncu_parse.py DETAILS.csv RAW.csv ANALYZER.py -> one JSON line, generic names only (kernel symbols never leave)."""
import csv, importlib.util, json, sys
spec = importlib.util.spec_from_file_location('a', sys.argv[3]); a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
WANT = {'Duration': 'duration_ns', 'DRAM Throughput': 'dram_throughput_pct', 'Memory Throughput': 'memory_throughput_pct',
        'Compute (SM) Throughput': 'sm_throughput_pct', 'Achieved Occupancy': 'achieved_occupancy_pct',
        'Theoretical Occupancy': 'theoretical_occupancy_pct', 'Registers Per Thread': 'registers_per_thread'}
def num(x):
    try: return float(str(x).replace(',', ''))
    except ValueError: return None
out = {'metrics': {}, 'category': None}
for row in csv.DictReader(open(sys.argv[1], newline='')):
    out['category'] = out['category'] or a.category(row.get('Kernel Name', ''))
    out['grid'], out['block'] = row.get('Grid Size'), row.get('Block Size')
    key = WANT.get(row.get('Metric Name'))
    if key and key not in out['metrics']: out['metrics'][key] = num(row.get('Metric Value'))
try:
    rows = list(csv.reader(open(sys.argv[2], newline='')))
    head = rows[0]; vals = rows[2] if len(rows) > 2 else []
    get = lambda n: num(vals[head.index(n)]) if n in head and len(vals) > head.index(n) else None
    t, i = get('smsp__inst_executed_pipe_tensor.sum'), get('smsp__inst_executed.sum')
    out['metrics']['tensor_inst_share_pct'] = 100 * t / i if t is not None and i else None
    tc, ce = get('sm__pipe_tensor_cycles_active.avg'), get('sm__cycles_elapsed.avg')
    out['metrics']['tensor_pipe_active_pct_of_elapsed'] = 100 * tc / ce if tc is not None and ce else None
    out['metrics']['dram_bytes_read'] = get('dram__bytes_op_read.sum'); out['metrics']['dram_bytes_write'] = get('dram__bytes_op_write.sum')
except (OSError, ValueError, IndexError): pass
print(json.dumps(out))
