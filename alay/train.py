import time
import numpy as np
import torch
import torch.nn as nn
from model import SignalNet

# Use the GPU if available
torch.backends.cudnn.enabled = False   # cuDNN is very slow on this laptop; use PyTorch's built-in method
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

# Load the flashcards and put them on the GPU (about 225 MB, fits easily)
d = np.load("data/rml_prepared.npz")
X = torch.from_numpy(d["X"]).to(device)
Y = torch.from_numpy(d["Y"]).long().to(device)
train_ids = torch.from_numpy(d["train"]).to(device)
val_ids = torch.from_numpy(d["val"]).to(device)

# The student, the "how wrong" measurer, and the knob-nudger
net = SignalNet().to(device)
loss_fn = nn.CrossEntropyLoss()                            # computes the loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # lr = size of each nudge

def quiz(ids):
    """Score the student on cards it doesn't learn from."""
    net.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(ids), 2048):
            batch = ids[start:start + 2048]
            guesses = net(X[batch]).argmax(dim=1)
            correct += (guesses == Y[batch]).sum().item()
    return correct / len(ids)

EPOCHS = 20      # 20 trips through the whole practice pile
BATCH = 256      # cards per handful
best = 0.0

for epoch in range(1, EPOCHS + 1):
    net.train()                                   # practice mode (Dropout on)
    t0 = time.time()
    total_loss = 0.0
    shuffled = train_ids[torch.randperm(len(train_ids), device=device)]  # reshuffle each time

    for start in range(0, len(shuffled), BATCH):
        batch = shuffled[start:start + BATCH]
        scores = net(X[batch])                    # 1. guess
        loss = loss_fn(scores, Y[batch])          # 2. how wrong?
        optimizer.zero_grad()                     # 3. clear old nudge directions
        loss.backward()                           # 4. work out which way to turn each knob
        optimizer.step()                          # 5. nudge the knobs
        total_loss += loss.item() * len(batch)
        n_done = start // BATCH + 1
        if n_done % 100 == 0:
            print(f"   epoch {epoch}: {n_done} handfuls done, current loss {loss.item():.3f}", flush=True)
    score = quiz(val_ids)
    note = ""
    if score > best:
        best = score
        torch.save(net.state_dict(), "best_model.pt")   # save the knob settings
        note = "  <- best so far, saved"
    print(f"epoch {epoch:2d} | loss {total_loss / len(shuffled):.3f} | "
          f"quiz {score * 100:.1f}% | {time.time() - t0:.0f}s{note}")

print(f"Done! Best quiz score: {best * 100:.1f}%")