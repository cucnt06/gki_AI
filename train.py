from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"


def train_digitclassifier(model, dataset):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    epochs = 30

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00001,
    )

    for epoch in range(epochs):
        model.train()

        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch["x"])
            loss = digitclassifier_loss(logits, batch["label"])
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with no_grad():
            accuracy = dataset.get_validation_accuracy()

        print(f"Epoch {epoch + 1}/{epochs} - Validation accuracy: {accuracy:.2%}")


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 30

    for epoch in range(epochs):
        model.train()

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()

            y_pred = model(x)
            loss = digitconvolution_Loss(y_pred, y)

            loss.backward()
            optimizer.step()

        model.eval()

        with no_grad():
            accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Validation accuracy: {accuracy:.2%}"
        )