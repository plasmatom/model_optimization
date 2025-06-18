import numpy as np
import matplotlib.pyplot as plt
import PIL
import tqdm
import torch
from torchvision import datasets, transforms

image_size = 128

mean = [0.48825347423553467, 0.45504486560821533, 0.4168395400047302]
std = [0.2225690633058548, 0.21782387793064117, 0.218031108379364]

transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_val = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


def load_image(image_file):
    image = PIL.Image.open(image_file)
    image = np.array(image) / 255
    image = image.transpose((2, 0, 1))
    for i in range(3):
        image[i] /= utils.mean[i]
        image[i] -= utils.std[i]
    image = torch.from_numpy(image)
    image = transforms.functional.resize(image,(utils.image_size, utils.image_size))
    image = image.to(torch.float32)
    image = image.unsqueeze(0)
    return image


def test_network(model, test_loader, score_funcs=None, device="cpu"):
    model.to(device)
    model.eval()
    pred_prob = []
    ground_truth = []
    with torch.no_grad():
        for test_inputs, test_labels in tqdm.tqdm(test_loader,desc=f"Evaluating"):
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device).float()
            outputs = model(test_inputs).squeeze(1)
            pred_prob.append(outputs.cpu())
            ground_truth.append(test_labels.cpu())

    pred_prob = torch.cat(pred_prob)
    ground_truth = torch.cat(ground_truth)
    return pred_prob, ground_truth



def train_network(
    model,
    loss_func,
    train_loader,
    val_loader=None,
    score_funcs=None,
    epochs=50,
    checkpoint_file=None,
    optimizer=None,
    lr_scheduler=None,
    patience=10,
    device="cpu"
):


    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters())

    model.to(device)
    print(f"Model operating on: {device}")

    best_val_loss = float("inf")
    num_bad_epochs = 0
    if score_funcs is not None:
        metrics = {name:[] for name in score_funcs}
    metrics["train_loss"] = []
    metrics["val_loss"] = []
    

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] - Training")

        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        metrics["train_loss"].append(avg_train_loss)
        print(f"Epoch {epoch+1} | Training Loss: {avg_train_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                preds_list = []
                labels_list = []
                for val_inputs, val_labels in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device).float()
                    outputs = model(val_inputs).squeeze(1)
                    loss = loss_func(outputs, val_labels)
                    val_loss += loss.item()

                    preds = (outputs > 0.5).float()
                    correct += (preds == val_labels).sum().item()
                    total += val_labels.size(0)

                    preds_list.append(preds.cpu())
                    labels_list.append(val_labels.cpu())

            avg_val_loss = val_loss / len(val_loader)
            metrics["val_loss"].append(avg_val_loss)
            print(f"→ Validation Loss: {avg_val_loss:.4f}")

            # Score functions (optional)
            if score_funcs:
                preds_all = torch.cat(preds_list)
                labels_all = torch.cat(labels_list)
                for name, func in score_funcs.items():
                    metric = func(preds_all, labels_all)
                    metrics[name].append(metric)
                    print(f"→ {name}: {metric:.4f}")
                del preds_all
                del labels_all

            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                num_bad_epochs = 0
                if checkpoint_file:
                    checkpoint = {'model': model,
                                  'state_dict': model.state_dict(),
                                  'optimizer' : optimizer.state_dict()}

                    torch.save(checkpoint, checkpoint_file)
                    print("Model checkpoint saved.")
            else:
                num_bad_epochs += 1
                print(f"Validation loss did not improve for {num_bad_epochs} epoch(s).")

            if num_bad_epochs >= patience:
                print("Early stopping triggered.")
                break

        # Step LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

    print("Training complete.")
    return metrics


def imshow(inp, ax, title=None):
    # Unnormalize
    if isinstance(mean, torch.Tensor):
        mean_tensor = mean.view(3, 1, 1)
        std_tensor = std.view(3, 1, 1)
    else:
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)

    inp = inp.cpu() * std_tensor + mean_tensor  # unnormalize
    inp = inp.numpy().transpose((1, 2, 0))  # C x H x W → H x W x C
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')