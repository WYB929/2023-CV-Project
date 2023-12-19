import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
from torch import nn
import cv2
from torchvision import transforms
from datetime import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    jaccard_score,
)


# reference from CLAHE paper and cv2 documentation
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)
        # Check if image is not grayscale, convert it
        if len(img_np.shape) == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img_clahe = clahe.apply(img_gray)
        # If original image was RGB, replace the luminance component with the processed one
        if len(img_np.shape) == 3:
            img_clahe = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
            img_clahe[:, :, 0] = clahe.apply(img_clahe[:, :, 0])
            img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YCrCb2RGB)
        # Convert numpy array back to PIL image
        img = Image.fromarray(img_clahe)
        return img


class CloudDataset(Dataset):
    def __init__(
        self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True, transform=None
    ):
        super().__init__()

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [
            self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir)
            for f in r_dir.iterdir()
            if not f.is_dir()
        ]
        self.pytorch = pytorch
        self.transform = transform

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        files = {
            "red": r_file,
            "green": g_dir / r_file.name.replace("red", "green"),
            "blue": b_dir / r_file.name.replace("red", "blue"),
            "nir": nir_dir / r_file.name.replace("red", "nir"),
            "gt": gt_dir / r_file.name.replace("red", "gt"),
        }

        return files

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx, invert=False, include_nir=False):
        raw_rgb = np.stack(
            [
                np.array(Image.open(self.files[idx]["red"])),
                np.array(Image.open(self.files[idx]["green"])),
                np.array(Image.open(self.files[idx]["blue"])),
            ],
            axis=2,
        )

        if include_nir:
            img_nir = np.expand_dims(np.array(Image.open(self.files[idx]["nir"])), 2)
            raw_rgb = np.concatenate([raw_rgb, img_nir], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        # normalize
        return raw_rgb / np.iinfo(raw_rgb.dtype).max

    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]["gt"]))
        raw_mask = np.where(raw_mask == 255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        x = torch.tensor(
            self.open_as_array(idx, invert=self.pytorch, include_nir=False),
            dtype=torch.float32,
        )
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

        if self.transform:
            x = self.transform(x)

        return x, y

    def open_as_pil(self, idx):
        arr = 256 * self.open_as_array(idx)

        return Image.fromarray(arr.astype(np.uint8), "RGB")

    def __repr__(self):
        s = "Dataset class with {} files".format(self.__len__())

        return s


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # Decoding + concat path
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


import time
from IPython.display import clear_output


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
    start = time.time()
    model.cuda()

    train_loss, valid_loss = [], []

    best_acc = 0.0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()  # Set trainind mode = true
                dataloader = train_dl
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for x, y in dataloader:
                x = x.cuda()
                y = y.cuda()
                step += 1

                # forward pass
                if phase == "train":
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss.detach() * dataloader.batch_size
                if step % 100 == 0:
                    # clear_output(wait=True)
                    print(
                        "Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}".format(
                            step, loss, acc, torch.cuda.memory_allocated() / 1024 / 1024
                        )
                    )
                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 10)
            print("{} Loss: {:.4f} Acc: {}".format(phase, epoch_loss, epoch_acc))
            print("-" * 10)

            train_loss.append(epoch_loss) if phase == "train" else valid_loss.append(
                epoch_loss
            )

    time_elapsed = time.time() - start
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return train_loss, valid_loss


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()


def batch_to_img(xb, idx):
    img = np.array(xb[idx, 0:3])
    return img.transpose((1, 2, 0))


def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()


if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            CLAHETransform(),
            transforms.ToTensor(),
        ]
    )
    base_path = Path(
        "95-cloud/95-cloud_training_only_additional_to38-cloud/"
    )
    dataset = CloudDataset(
        base_path / "train_red_additional_to38cloud",
        base_path / "train_green_additional_to38cloud",
        base_path / "train_blue_additional_to38cloud",
        base_path / "train_nir_additional_to38cloud",
        base_path / "train_gt_additional_to38cloud",
        # transform=data_transform,
    )
    train_valid_size = int(0.8 * len(dataset))
    train_size = int(0.8 * train_valid_size)
    valid_size = train_valid_size - train_size
    test_size = len(dataset) - train_valid_size
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        dataset, (train_size, valid_size, test_size)
    )
    print(len(train_ds), len(valid_ds), len(test_ds))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=True)
    unet = AttU_Net(3, 2)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        unet = nn.DataParallel(unet)
        unet.to(device)
    else:
        device = torch.device("cpu")
    print(device)
    count_parameters(unet)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(unet.parameters(), lr=0.01)
    train_loss, valid_loss = train(
        unet, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50
    )
    # convert tensor list to list
    for i in range(len(train_loss)):
        train_loss[i] = float(train_loss[i].item())
        valid_loss[i] = float(valid_loss[i].item())
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss, label="Train loss")
    plt.plot(valid_loss, label="Valid loss")
    plt.legend()
    plt.savefig("att_unet_loss.png")

    xb, yb = next(iter(test_dl))

    with torch.no_grad():
        predb = unet(xb.cuda())

    bs = 16
    fig, ax = plt.subplots(bs, 3, figsize=(15, bs * 5))
    for i in range(bs):
        ax[i, 0].imshow(batch_to_img(xb, i))
        ax[i, 1].imshow(yb[i])
        ax[i, 2].imshow(predb_to_mask(predb, i))
    plt.savefig("att_unet_result.png")

    test_acc = 0.0
    for xt, yt in test_dl:
        with torch.no_grad():
            predt = unet(xt.cuda())
        acc = acc_metric(predt[: len(yt)], yt)

        test_acc += acc * test_dl.batch_size
    test_acc = test_acc / len(test_dl.dataset)
    print("Accuracy: {:.4f}".format(test_acc))

    test_acc, test_precision, test_recall, test_f1, test_auc, test_jaccard = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    all_probs, all_targets = [], []

    for xt, yt in test_dl:
        with torch.no_grad():
            predt = unet(xt.cuda())
        # Compute accuracy
        acc = acc_metric(predt, yt)
        test_acc += acc.item() * xt.size(0)
        # Reshape and process predictions and labels
        preds = (
            predt.argmax(dim=1).view(-1).cpu().numpy()
        )  # Convert to binary predictions and flatten
        labels = yt.view(-1).cpu().numpy()  # Flatten the ground truth labels
        # Accumulate all probabilities (for class 1) and true labels for AUC calculation
        probs = predt[:, 1, :, :].sigmoid().view(-1).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels)
        # Calculate and accumulate precision, recall, F1, and Jaccard for each batch
        test_precision += precision_score(labels, preds) * xt.size(0)
        test_recall += recall_score(labels, preds) * xt.size(0)
        test_f1 += f1_score(labels, preds) * xt.size(0)
        test_jaccard += jaccard_score(labels, preds) * xt.size(0)
    # Calculate averages over the entire test set
    num_samples = len(test_dl.dataset)
    test_acc = test_acc / num_samples
    test_precision = test_precision / num_samples
    test_recall = test_recall / num_samples
    test_f1 = test_f1 / num_samples
    test_jaccard = test_jaccard / num_samples
    test_auc = roc_auc_score(all_targets, all_probs)
    # Print the metrics
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Average Jaccard (IoU): {test_jaccard:.4f}")
    print(f"AUC: {test_auc:.4f}")
    now = datetime.now()
    formatted_time = now.strftime("%m_%d_%Y_%H%M%S")
    path = "models/residual_unet_" + formatted_time + "_acc_" + str(round(test_acc, 4)) +".pth"
    torch.save(unet, path)
