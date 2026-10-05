"""One-command reproduction; stops at any failing data/model/geometry gate."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');parser.add_argument('--workers',type=int,default=24);parser.add_argument('--threads',type=int,default=4);a=parser.parse_args();cache=a.cache.resolve()
    stages=[('geometry.py',[]),('download_data.py',['--cache',str(cache),'--workers',str(a.workers)]),('train.py',['--cache',str(cache),'--threads',str(a.threads)]),('check_ig.py',['--cache',str(cache)]),('evaluate.py',['--cache',str(cache),'--threads',str(a.threads)]),('summarize.py',[]),('baseline_sensitivity.py',['--cache',str(cache)])]
    for script,args in stages:
        print(f'Running {script}',flush=True);subprocess.run([sys.executable,str(ROOT/script),*args],check=True)

if __name__=='__main__':main()
