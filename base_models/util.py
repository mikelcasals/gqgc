import torch

class QuantumLossFunction(torch.nn.Module):
    def __init__(self):
        super(QuantumLossFunction, self).__init__()
        # Initialize any parameters or sub-modules here

    def forward(self, predictions, targets):
        # Implement the loss computation here
        return quantum_loss_function(predictions, targets)
    
def quantum_loss_function(class_output, y):
    y = y.float()
    class_output = class_output.float()

    
    return torch.nn.BCELoss(reduction='mean')(class_output, y)
