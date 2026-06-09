from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DAGM2007MultiClassDataset(Dataset):
    IMG_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.split = str(split).lower()
        self.transform = transform

        class_dirs = sorted(
            path for path in self.root.iterdir()
            if path.is_dir() and path.name.lower().startswith("class")
        )
        self.classes = [path.name for path in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        split_candidates = ["Train", "train"] if self.split == "train" else ["Test", "test"]
        samples = []
        for class_dir in class_dirs:
            split_dir = None
            for candidate in split_candidates:
                maybe_dir = class_dir / candidate
                if maybe_dir.is_dir():
                    split_dir = maybe_dir
                    break
            if split_dir is None:
                continue

            class_idx = self.class_to_idx[class_dir.name]
            for image_path in sorted(split_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in self.IMG_EXTS:
                    samples.append((str(image_path), class_idx))

        if not samples:
            raise RuntimeError("[DAGM] No samples found under {} for split={}.".format(self.root, split))

        self.samples = samples
        self.targets = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dagm_transforms(img_size=32):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
    ])


def stratified_iid_partition(labels, num_clients, seed):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    class_indices = [np.where(labels == cls)[0] for cls in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for cls_indices in class_indices:
        shuffled = cls_indices.copy()
        rng.shuffle(shuffled)
        splits = np.array_split(shuffled, num_clients)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for indices in client_indices:
        rng.shuffle(indices)
    return {cid: set(indices) for cid, indices in enumerate(client_indices)}


def dirichlet_noniid_partition(labels, num_clients, alpha=0.5, seed=9, min_size=10):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    num_samples = len(labels)
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    smallest_client = 0
    while smallest_client < int(min_size):
        client_indices = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]
            shuffled = cls_indices.copy()
            rng.shuffle(shuffled)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([
                p * (len(indices) < num_samples / num_clients)
                for p, indices in zip(proportions, client_indices)
            ])
            if proportions.sum() == 0:
                proportions = np.repeat(1.0 / num_clients, num_clients)
            else:
                proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) * len(shuffled)).astype(int)[:-1]
            for client_id, split in enumerate(np.split(shuffled, split_points)):
                client_indices[client_id].extend(split.tolist())

        smallest_client = min(len(indices) for indices in client_indices)

    for indices in client_indices:
        rng.shuffle(indices)
    return {cid: set(indices) for cid, indices in enumerate(client_indices)}
