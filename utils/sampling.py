import numpy as np


def cifar_iid(dataset, num_users):
    """Split CIFAR samples evenly and randomly across IID clients."""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_non_iid(dataset, num_classes, num_users, alpha=0.5):
    """Create a Dirichlet label-skew non-IID split for CIFAR-style datasets."""
    n_samples = len(dataset)
    min_size = 0
    print("Dataset size:", n_samples)

    dict_users = {}
    targets = np.asarray(dataset.targets)
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for cls in range(num_classes):
            idx_cls = np.where(targets == cls)[0]
            np.random.shuffle(idx_cls)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array([
                p * (len(idx_j) < n_samples / num_users)
                for p, idx_j in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_cls, split_points))
            ]
        min_size = min(len(idx_j) for idx_j in idx_batch)

    for user in range(num_users):
        np.random.shuffle(idx_batch[user])
        dict_users[user] = idx_batch[user]
    return dict_users
