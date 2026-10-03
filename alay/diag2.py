import time
import numpy as np
import torch
from model import SignalNet

X = torch.from_numpy(np.load("data/rml_prepared.npz")["X"][:256]).cuda()

def try_setting(name, cudnn_on):
    torch.backends.cudnn.enabled = cudnn_on
    net = SignalNet().cuda()
    print(f"\n--- {name} ---", flush=True)
    for i in range(10):
        t = time.time()
        net(X).sum().backward()
        torch.cuda.synchronize()
        print(f"step {i + 1}: {time.time() - t:.3f}s", flush=True)

try_setting("cuDNN OFF", cudnn_on=False)
try_setting("cuDNN ON", cudnn_on=True)