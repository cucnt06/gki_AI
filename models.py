from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim




def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        import torch
import torch.nn as nn
from torch.nn import Module

class DigitClassificationModel(Module):
    """
    Neural network for MNIST digit classification.
    Input: (batch_size, 784)
    Output: (batch_size, 10)
    """

    def __init__(self):
        super().__init__()

        self.layer1 = Linear(784, 512)
        self.layer2 = Linear(512, 256)
        self.layer3 = Linear(256, 128)
        self.layer4 = Linear(128, 10)

    

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        return self.layer4(x)
