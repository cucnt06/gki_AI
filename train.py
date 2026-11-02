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





    
def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """

    batch_size = 128
    epochs = 35
    patience = 8

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    best_accuracy = 0.0
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        """ YOUR CODE HERE """

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            x = batch["x"]
            y = batch["label"]

            optimizer.zero_grad()

            y_pred = model(x)

            loss = digitclassifier_loss(y_pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predicted = y_pred.argmax(dim=1)
            labels = y.argmax(dim=1)

            correct += (
                predicted == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_accuracy = correct / total
        avg_loss = total_loss / len(dataloader)

        model.eval()

        with no_grad():
            accuracy = dataset.get_validation_accuracy()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_accuracy:.2%} | "
            f"Validation: {accuracy:.2%}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            best_weights = {
                name: weight.detach().clone()
                for name, weight in model.state_dict().items()
            }

            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2%}"
    )



