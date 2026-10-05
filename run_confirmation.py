"""Reproduce the frozen fresh-cohort confirmation with bundled model states."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=ROOT.parent/'confirmation_cache');p.add_argument('--source-cache',type=Path,default=ROOT.parent/'metadata_cache');a=p.parse_args()
    for command in [['download_confirmation.py','--cache',str(a.cache),'--source-cache',str(a.source_cache)],['evaluate_confirmation.py','--cache',str(a.cache)],['summarize_confirmation.py'],['verify_confirmation.py']]:subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
if __name__=='__main__':main()
