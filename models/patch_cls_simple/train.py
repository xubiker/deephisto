from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import models.patch_cls_simple.utils as utils
from models.patch_cls_simple.model import get_model
from patch_samplers.region_samplers import (
    AnnoRegionRndSampler,
    extract_and_save_subset,
)
from utils import get_img_ano_paths


def save_plot(train_values, val_values, test_values, title, filename):
    plt.figure()
    plt.plot(train_values, label="train")
    plt.plot(val_values, label="val")
    plt.plot(test_values, label="test")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(Path(cfg["training"]["out_dir"]) / filename)
    plt.close()


def prepare_test_patches(cfg):
    img_anno_paths_test = get_img_ano_paths(
        ds_folder=Path(cfg["dataset"]["folder"]), sample="test"
    )

    extract_and_save_subset(
        img_anno_paths=img_anno_paths_test,
        out_folder=Path(cfg["test"]["dir"]),
        patch_size=cfg["dataset"]["patch_size"],
        layer=cfg["dataset"]["layer"],
        patches_per_class=cfg["test"]["samples_per_class"],
    )


def train(cfg):

    device = utils.get_device()
    print(f"Using device: {device}")

    # # prepare directories
    save_dir = Path(cfg["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["training"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # # data transformations
    inputs_transform = transforms.Compose(
        [transforms.Lambda(lambda x: x.permute(0, 3, 1, 2).contiguous())]
    )
    data_augmentations = transforms.Compose(
        [
            inputs_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(10),
        ]
    )

    # train_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomVerticalFlip(),
    #     ]
    # )

    test_transform = transforms.Compose([transforms.ToTensor()])

    img_anno_paths_train = get_img_ano_paths(
        ds_folder=Path(cfg["dataset"]["folder"]), sample="train"
    )

    train_val_dataset = AnnoRegionRndSampler(
        img_anno_paths_train,
        patch_size=cfg["dataset"]["patch_size"],
        layer=cfg["dataset"]["layer"],
        patches_from_one_region=cfg["dataset"]["patches_from_one_region"],
        one_image_for_batch=cfg["training"]["one_image_for_batch"],
    )

    # train_val_dataset = ImageFolder(
    #     root=Path("./data/"), transform=train_transform
    # )

    test_dataset = ImageFolder(
        root=Path(cfg["test"]["dir"]), transform=test_transform
    )

    # load model
    model = get_model(cfg["model"]["n_classes"]).to(device)

    # Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Training Loop
    train_losses, val_losses, test_losses = [], [], []
    train_accuracies, val_accuracies, test_accuracies = [], [], []

    best_val_acc = 0

    for epoch in range(cfg["training"]["n_epochs"]):

        model.train()
        train_loss, correct, total = 0, 0, 0
        y_true_train, y_pred_train = [], []

        train_steps = (
            len(train_val_dataset)
            // cfg["training"]["batch_size"]
            * cfg["training"]["augment_factor"]
        )

        train_steps = 200

        # train_generator = DataLoader(
        #     train_val_dataset,
        #     batch_size=cfg["training"]["batch_size"],
        #     shuffle=True,
        #     # num_workers=cfg["training"]["data_max_workers"],
        # )

        # train_steps = len(train_generator)

        train_generator = train_val_dataset.torch_generator(
            batch_size=cfg["training"]["batch_size"],
            n_batches=train_steps,
            batches_per_worker=cfg["dataset"]["batches_per_worker"],
            transforms=data_augmentations,
            max_workers=4,
        )
        for inputs, labels, _ in tqdm(
            # for inputs, labels in tqdm(
            train_generator,
            total=train_steps,
            desc=f"Epoch {epoch + 1}/{cfg['training']['n_epochs']}",
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

        train_loss /= train_steps
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation Loop
        model.eval()
        val_loss, correct, total = 0, 0, 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():

            val_steps = cfg["training"]["val_steps"]

            val_generator = train_val_dataset.torch_generator(
                batch_size=cfg["training"]["batch_size"],
                n_batches=val_steps,
                batches_per_worker=cfg["dataset"]["batches_per_worker"],
                transforms=data_augmentations,
                max_workers=4,
            )

            # val_generator = DataLoader(
            #     train_val_dataset,
            #     batch_size=cfg["training"]["batch_size"],
            #     shuffle=True,
            #     # num_workers=cfg["training"]["data_max_workers"],
            # )

            # steps = 0
            for inputs, labels, _ in val_generator:
                # for inputs, labels in val_generator:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

                # steps += 1
                # if steps == val_steps:
                #     break

        val_loss /= val_steps
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                Path(cfg["training"]["out_dir"]) / "best_model.pth",
            )

        # Test Loop

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
        )

        test_loss, correct, total = 0, 0, 0
        y_true_test, y_pred_test = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Accumulate test metrics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

        # Compute test metrics
        test_loss /= len(test_dataloader)
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Test Loss: {test_loss:.4f}, " f"Test Acc: {test_acc:.4f}")

        # Plot Loss and Metrics
        save_plot(
            train_losses,
            val_losses,
            test_losses,
            title="Loss",
            filename="loss.jpg",
        )
        save_plot(
            train_accuracies,
            val_accuracies,
            test_accuracies,
            title="Acc",
            filename="acc.jpg",
        )


if __name__ == "__main__":

    cfg = utils.load_config(Path("./models/patch_cls_simple/config.yaml"))

    # img_anno_paths_test = get_img_ano_paths(
    #     ds_folder=Path(cfg["dataset"]["folder"]), sample="train"
    # )

    # extract_and_save_subset(
    #     img_anno_paths=img_anno_paths_test,
    #     out_folder=Path("./data"),
    #     patch_size=cfg["dataset"]["patch_size"],
    #     layer=cfg["dataset"]["layer"],
    #     patches_per_class=3000,
    # )

    # prepare_test_patches(cfg)
    train(cfg)
