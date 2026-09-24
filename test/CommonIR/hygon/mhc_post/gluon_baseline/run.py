"""Run an exactly captured specialization through the selected implementation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"common"))
from entry import main
if __name__=="__main__":main('mhc_post','baseline')
