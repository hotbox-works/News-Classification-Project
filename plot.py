import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_figtext(hyperparams):
    lr = hyperparams.get("learning_rate", None)
    batch = hyperparams.get("batch_size", None)
    result = f"Learning Rate: {lr} / Batch Size: {batch}"

    momentum = hyperparams.get("momentum", None)
    wd = hyperparams.get("weight_decay", None)

    if momentum is not None and wd is not None:
        result += f" / Momentum: {momentum} / Weight Decay: {wd}"

    return result

def plot_losses(epochs, training_losses, validation_losses, model_name, hyperparams):
    plt.figure(figsize=(10, 7))

    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(f"Training & Validation Losses ({model_name})")

    figtext = get_figtext(hyperparams)
    plt.figtext(1, 0, figtext, ha="right")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_accuracies(epochs, training_accuracies, validation_accuracies, model_name, hyperparams):
    plt.figure(figsize=(10, 7))

    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(f"Training & Validation Accuracies ({model_name})")

    figtext = get_figtext(hyperparams)
    plt.figtext(1, 0, figtext, ha="right")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_f1s(epochs, training_f1macros, validation_f1macros, training_f1s, validation_f1s, model_name, hyperparams):
    plt.figure(figsize=(10, 7))

    plt.plot(
        epochs,
        training_f1macros,
        label="Training Macro-F1"
    )

    plt.plot(
        epochs,
        validation_f1macros,
        label="Validation Macro-F1"
    )

    plt.plot(
        epochs,
        training_f1s,
        label="Training F1 (Weighted)"
    )

    plt.plot(
        epochs,
        validation_f1s,
        label="Validation F1 (Weighted)"
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")

    plt.title(f"Training & Validation Macro-F1/F1 Scores ({model_name})")

    figtext = get_figtext(hyperparams)
    plt.figtext(1, 0, figtext, ha="right")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_precisions_and_recalls(epochs, training_precisions, validation_precisions, training_recalls, validation_recalls, model_name, hyperparams):
    plt.figure(figsize=(10, 7))

    plt.plot(
        epochs,
        training_precisions,
        label="Training Precision"
    )

    plt.plot(
        epochs,
        validation_precisions,
        label="Validation Precision"
    )

    plt.plot(
        epochs,
        training_recalls,
        label="Training Recall"
    )

    plt.plot(
        epochs,
        validation_recalls,
        label="Validation Recall"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Training & Validation Precision/Recall ({model_name})")

    figtext = get_figtext(hyperparams)
    plt.figtext(1, 0, figtext, ha="right")

    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, validation_trues, validation_predictions, model_name, hyperparams):
    cm = confusion_matrix(
        validation_trues,
        validation_predictions
    )

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # overriding display_labels to show the actual labels
    cm_display.plot(values_format='d')

    plt.title(f"Validation Confusion Matrix ({model_name})")

    figtext = get_figtext(hyperparams)
    plt.figtext(1, 0, figtext, ha="right")

    plt.tight_layout()
    plt.show()
