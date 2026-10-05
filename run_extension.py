"""Reproduce the exploratory extension; requires original manifest/checkpoints."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'data_cache');a=p.parse_args();cache=str(a.cache.resolve())
    stages=[('recheck_feedback.py',[]),('train_extension.py',['--cache',cache]),('evaluate_extension.py',['--cache',cache,'--architecture','both']),('positive_control.py',['--cache',cache]),('fitted_positive_control.py',['--cache',cache]),('summarize_extension.py',[]),('summarize_control_grid.py',[])]
    for script,args in stages:
        print('Running',script,flush=True);subprocess.run([sys.executable,str(ROOT/script),*args],check=True)

if __name__=='__main__':main()
