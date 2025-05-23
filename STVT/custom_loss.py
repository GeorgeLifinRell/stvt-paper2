import torch
import torch.nn as nn
import torch.nn.functional as F

class STVTCustomLoss(nn.Module):
    def __init__(self):
        super(STVTCustomLoss, self).__init__()
        # Learnable penalty weights with positivity via softplus
        self.lambda1 = nn.Parameter(torch.tensor(1.0))  # For smoothness
        self.lambda2 = nn.Parameter(torch.tensor(1.0))  # For semantic
        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, target, features=None):
        ce_loss = self.ce(output, target)

        probs = F.softmax(output, dim=1)[:, 1]  # Importance score
        smoothness_loss = torch.mean((probs[1:] - probs[:-1]) ** 2)

        if features is not None:
            selected = (target == 1)
            if selected.sum() > 0:
                mu_summary = features[selected].mean(dim=0)
                mu_video = features.mean(dim=0)
                semantic_loss = F.mse_loss(mu_summary, mu_video)
            else:
                semantic_loss = torch.tensor(0.0, device=features.device)
        else:
            semantic_loss = torch.tensor(0.0, device=output.device)

        total_loss = (
            ce_loss +
            F.softplus(self.lambda1) * smoothness_loss +
            F.softplus(self.lambda2) * semantic_loss
        )

        return total_loss
