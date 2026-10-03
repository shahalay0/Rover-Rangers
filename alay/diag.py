import time
t0 = time.time()
def step(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

step("starting, importing torch...")
import numpy as np
import torch
step(f"torch {torch.__version__}, GPU found: {torch.cuda.is_available()}")
step(f"GPU name: {torch.cuda.get_device_name(0)}")

d = np.load("data/rml_prepared.npz")
X = d["X"]
step(f"loaded flashcards {X.shape}")

Xg = torch.from_numpy(X).cuda()
torch.cuda.synchronize()
step("flashcards copied to GPU")

from model import SignalNet
net = SignalNet().cuda()
step("brain built on GPU")

out = net(Xg[:256])
torch.cuda.synchronize()
step("first guess (forward pass) done")

out.sum().backward()
torch.cuda.synchronize()
step("first knob calculation (backward pass) done")

for _ in range(100):
    net(Xg[:256]).sum().backward()
torch.cuda.synchronize()
step("100 more practice steps done")