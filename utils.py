from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def train_and_eval(
    model,
    criterion,
    optimizer,
    max_epoch,
    train_loader,
    train_dataset,
    validation_loader,
    validation_dataset,
    device
):
    training_losses = []
    validation_losses = []

    training_accuracies = []
    validation_accuracies = []

    training_f1macros = []
    training_f1s = []
    training_precisions = []
    training_recalls = []

    validation_f1macros = []
    validation_f1s = []
    validation_precisions = []
    validation_recalls = []

    for epoch in range(max_epoch):
        print(f"Epoch: {epoch + 1:02}/{max_epoch}", end='')
        # Training
        model.train()

        training_loss = 0  # compute the cumulative loss within an epoch
        training_accuracy = 0

        training_predictions = []
        training_trues = []

        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # multiply by batch size to get the total loss for this batch
            training_loss += loss.item() * x.size(0)

            predictions = torch.argmax(logits, dim=1)

            training_predictions.extend(predictions.detach().cpu().numpy())
            training_trues.extend(y.cpu().numpy())

            training_accuracy += (predictions == y).sum().item()

        training_loss /= len(train_dataset)  # len(dataset) == len(tensor)
        training_accuracy /= len(train_dataset)

        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)

        training_f1macro = f1_score(training_trues, training_predictions, average="macro", zero_division=0)
        training_f1macros.append(training_f1macro)

        training_f1 = f1_score(training_trues, training_predictions, average="weighted", zero_division=0)
        training_f1s.append(training_f1)

        training_precision = precision_score(training_trues, training_predictions, average="weighted", zero_division=0)
        training_precisions.append(training_precision)

        training_recall = recall_score(training_trues, training_predictions, average="weighted", zero_division=0)
        training_recalls.append(training_recall)

        # Validation
        model.eval()

        validation_loss = 0
        validation_accuracy = 0

        validation_predictions = []
        validation_trues = []

        with torch.inference_mode():
            for i, (x, y) in enumerate(validation_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)

                loss = criterion(logits, y)

                validation_loss += loss.item() * x.size(0)

                predictions = torch.argmax(logits, dim=1)

                validation_predictions.extend(predictions.detach().cpu().numpy())
                validation_trues.extend(y.cpu().numpy())

                validation_accuracy += (predictions == y).sum().item()

        validation_loss /= len(validation_dataset)
        validation_accuracy /= len(validation_dataset)

        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        validation_f1macro = f1_score(validation_trues, validation_predictions, average="macro", zero_division=0)
        validation_f1macros.append(validation_f1macro)

        validation_f1 = f1_score(validation_trues, validation_predictions, average="weighted", zero_division=0)
        validation_f1s.append(validation_f1)

        validation_precision = precision_score(validation_trues, validation_predictions, average="weighted", zero_division=0)
        validation_precisions.append(validation_precision)

        validation_recall = recall_score(validation_trues, validation_predictions, average="weighted", zero_division=0)
        validation_recalls.append(validation_recall)

        print("\tDone!")

    return (
        training_losses,
        validation_losses,

        training_accuracies,
        validation_accuracies,

        training_f1macros,
        training_f1s,
        training_precisions,
        training_recalls,

        validation_f1macros,
        validation_f1s,
        validation_precisions,
        validation_recalls,

        training_trues,
        training_predictions,

        validation_trues,
        validation_predictions
    )

def test(
    model,
    test_loader,
    device
):
    testing_predictions = []
    testing_trues = []

    misclassified_predictions = []

    model.eval()

    with torch.inference_mode():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            # TODO
            # probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            testing_predictions.extend(predictions.detach().cpu().numpy())
            testing_trues.extend(y.cpu().numpy())

    testing_accuracy = accuracy_score(testing_trues, testing_predictions)

    print(testing_accuracy)

    return (
        testing_predictions,
        testing_trues,
        misclassified_predictions
    )