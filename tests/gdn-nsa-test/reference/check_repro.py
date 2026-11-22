# Cross-process reproducibility of the kernel reference: recompute logits in a fresh process and compare to saved.
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np, torch
from run_ref import load_model, load_tokens, HERE
m = load_model()
for name in sys.argv[1:]:
    ids = torch.tensor([load_tokens(name)], device='cuda')
    with torch.no_grad():
        lg = m(ids, use_cache=False).logits[0].float().cpu().numpy()
    ref = np.load(os.path.join(HERE, f'logits_{name}.npy'))
    print(name, 'fresh-process max|diff| vs saved kernel logits:', float(np.abs(lg - ref).max()), flush=True)
