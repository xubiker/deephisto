"""This is a prototype, should be checked and rewritten"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from patch_samplers.region_samplers import RegionRandomBatchedDataset


def _img_ano_paths(sample: str = "train"):
    ds_path = Path("/Users/xubiker/dev/PATH-DT-MSU.WSS2")
    img_paths = [
        p
        for p in (ds_path / "images" / sample).iterdir()
        if p.is_file() and p.suffix == ".psi"
    ]
    anno_paths = [
        ds_path / "annotations" / sample / f"{p.stem}.json" for p in img_paths
    ]
    return list(zip(img_paths, anno_paths))


if __name__ == "__main__":

    # Configuration
    device = torch.device(
        "mps"
        if torch.has_mps
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    image_layer = 2
    batch_size = 8
    num_epochs = 50
    augment_factor = 5
    learning_rate = 1e-4
    output_dir = Path("./output")  # Directory to save plots

    output_dir.mkdir(parents=True, exist_ok=True)

    # Data Transformations and Loaders
    inputs_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2).contiguous()),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
        ]
    )

    data_augmentations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            inputs_transform,
        ]
    )

    train_dataset = RegionRandomBatchedDataset(
        _img_ano_paths(),
        patch_size=224,
        batch_size=batch_size,
        layer=image_layer,
        patches_from_one_region=8,
    )

    num_classes = len(train_dataset.classes)

    # Load Pretrained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(
        model.fc.in_features, num_classes
    )  # Adjust the final layer
    model = model.to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_balanced_accuracies, val_balanced_accuracies = [], []

    best_val_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        y_true_train, y_pred_train = [], []

        its = len(train_dataset) // batch_size * augment_factor

        for inputs, labels, _ in tqdm(
            train_dataset.generator_torch(
                its,
                data_augmentations,
            ),
            total=its,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        train_loss /= its
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_bal_acc = balanced_accuracy_score(y_true_train, y_pred_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_balanced_accuracies.append(train_bal_acc)

        # Validation Loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        y_true_val, y_pred_val = [], []

        its_val = 50
        with torch.no_grad():
            for inputs, labels, _ in train_dataset.generator_torch(
                its_val, transforms=inputs_transform
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        val_loss /= its_val
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_balanced_accuracies.append(val_bal_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Bal Acc: {train_bal_acc:.4f} "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    # Plot Loss and Metrics
    def save_plot(values, title, ylabel, filename):
        plt.figure()
        plt.plot(values, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(output_dir / filename)
        plt.close()

    save_plot(train_losses, "Loss During Training", "Loss", "loss_plot.jpg")
    save_plot(
        train_accuracies,
        "Accuracy During Training",
        "Accuracy",
        "accuracy_plot.jpg",
    )
    save_plot(
        train_balanced_accuracies,
        "Balanced Accuracy During Training",
        "Balanced Accuracy",
        "balanced_accuracy_plot.jpg",
    )
