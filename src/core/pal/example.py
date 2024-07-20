import torch
import torch.nn as nn
import torch.optim as optim


# Define the ProbSeenModel
class ProbSeenModel(nn.Module):
    def __init__(self, input_dim):
        super(ProbSeenModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos):
        x = self.fc(pos)
        return self.sigmoid(x)


# Define the pCTRModel
class pCTRModel(nn.Module):
    def __init__(self, input_dim):
        super(pCTRModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


# Define the PAL framework's loss function
class PALLoss(nn.Module):
    def __init__(self):
        super(PALLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prob_seen, pctr, targets):
        bctr = prob_seen * pctr
        loss = self.criterion(bctr, targets)
        return loss


# Example usage
input_dim_pos = 1  # Example input dimension for position
input_dim_ctr = 10  # Example input dimension for features

# Create the models
prob_seen_model = ProbSeenModel(input_dim_pos)
pctr_model = pCTRModel(input_dim_ctr)

# Create the custom loss function
pal_loss = PALLoss()

# Example data
batch_size = 32
pos = torch.randn(batch_size, input_dim_pos)  # Example position features
x = torch.randn(batch_size, input_dim_ctr)  # Example item features
targets = torch.randint(0, 2, (batch_size, 1)).float()  # Example targets (0 or 1)

# Forward pass
prob_seen = prob_seen_model(pos)
pctr = pctr_model(x)

# Compute the loss
loss = pal_loss(prob_seen, pctr, targets)

# Print the loss
print(f"Loss: {loss.item()}")

# Backward pass and optimization
optimizer = optim.Adam(list(prob_seen_model.parameters()) + list(pctr_model.parameters()), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
