"""Explicit single GPU component test under the experiment's build environment."""
import os,pathlib,runpy,sys
root=pathlib.Path('/home/USER/qwen36-api')
os.environ.update(CUDA_HOME=str(root/'cuda-13.4'),CC='/usr/bin/gcc-15',CXX='/usr/bin/g++-15',
                  CUDAHOSTCXX='/usr/bin/g++-15',NVCC_CCBIN='/usr/bin/g++-15',MAX_JOBS='4')
sys.path[:0]=[str(root/'optimization/python'),str(root/'optimization')]
os.environ.setdefault('QWEN36_DATA_ROOT',str(root/'optimization'))
name=sys.argv.pop(1)
if name not in ('test-mtp.py','diagnose-mtp.py','test-complement.py','test-kv.py','test-spec-gpu.py'):raise SystemExit('Unknown component')
runpy.run_path(str(root/'optimization'/name),run_name='__main__')
