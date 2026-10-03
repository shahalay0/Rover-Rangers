import torch.nn as nn

class SignalNet(nn.Module):
    def __init__(self, n_games=11):
        super().__init__()

        # Stage 1: magnifying glasses, from small patterns to bigger ones
        self.pattern_finder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),    # 32 glasses, each looks at 7 photos
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),   # 64 glasses look at what the first ones found
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                               # shrink: 128 steps -> 64
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),                               # shrink: 64 steps -> 32
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
        )

        # Stage 2: summarize everything found, then vote
        self.decider = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),   # average each pattern over the whole card
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),                         # randomly ignore some clues while practicing
            nn.Linear(64, n_games),                  # 11 scores, one per game
        )

    def forward(self, x):
        return self.decider(self.pattern_finder(x))