"""Run the frozen native screen-chunk.py main() while the CALLER already holds the real GPU lock (blocking
`flock -w` around session.sh). screen-chunk's own non-blocking lock is pointed at a private dummy lock file
(same pattern as work/agents/parent/parent-screen.py); no harness file is modified."""
import importlib.util
import sys

N = '/home/USER/qwen36-api/optimization'
sys.path.insert(0, N)
spec = importlib.util.spec_from_file_location('screen_chunk', N + '/screen-chunk.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
raise SystemExit(mod.main(sys.argv[1:], lock_path='/tmp/qwen36-synclean-refill-screen-dummy.lock'))
