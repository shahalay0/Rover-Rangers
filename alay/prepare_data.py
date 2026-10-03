import pickle
import numpy as np

# 1. Open the original dataset
with open("data/RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# 2. Give each game a number: 8PSK=0, AM-DSB=1, ... WBFM=10 (alphabetical)
mods = sorted({key[0] for key in data})
print("Game numbers:", {name: number for number, name in enumerate(mods)})

# 3. Pour all the boxes into one big stack of cards
X, Y, S = [], [], []          # X = card fronts (dots), Y = answers, S = fog level
for (mod, snr), recordings in data.items():
    X.append(recordings)                                   # 1000 cards
    Y.append(np.full(len(recordings), mods.index(mod)))    # 1000 copies of the answer
    S.append(np.full(len(recordings), snr))                # 1000 copies of the fog level
X = np.concatenate(X).astype(np.float32)
Y = np.concatenate(Y)
S = np.concatenate(S)
print("Card fronts:", X.shape, " Answers:", Y.shape, " Fog levels:", S.shape)

# 4. Shuffle, then split into 70% practice / 15% quiz / 15% final exam
rng = np.random.default_rng(0)        # seed 0 = same shuffle every time we run
order = rng.permutation(len(X))
n_train = int(0.70 * len(X))
n_val = int(0.15 * len(X))
train = order[:n_train]
val = order[n_train:n_train + n_val]
test = order[n_train + n_val:]
print("Practice:", len(train), " Quiz:", len(val), " Final exam:", len(test))

# 5. Save everything into one file
np.savez("data/rml_prepared.npz", X=X, Y=Y, S=S, mods=np.array(mods),
         train=train, val=val, test=test)
print("Saved to data/rml_prepared.npz")