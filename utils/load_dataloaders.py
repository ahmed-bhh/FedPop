from src.data.new_data_loader.data import get_dataloader_BFL
from utils.load_cifar10 import load_cifar10
from utils.load_cifar100 import load_cifar100
import copy

def load_dataloaders(
        dataset,
        n_clients,
        n_labels,
        relabel,
        device,
        batch_size,
        path_to_data,
        seed,
        max_data=None,
):
    # Utilisation de get_dataloader_BFL pour récupérer les DataLoaders
    train_loaders, test_loaders = get_dataloader_BFL(
        dataset=dataset,
        datadir=path_to_data,
        train_bs=batch_size,
        test_bs=batch_size,
        train_dataidxs=max_data,
        test_dataidxs=max_data
    )

    if dataset == "CIFAR10":
        client_labels = None  # Adapter selon ton besoin
        client_test_ind = copy.deepcopy(client_labels) if client_labels else None
        local_classes = n_labels if relabel else 10
        class_2_superclass = None

    elif dataset == "CIFAR100":
        if n_labels != 20 or not relabel:
            Warning("CIFAR100 datasets must be relabeled to 20 superclasses!")
        local_classes = 20
        relabel = True
        client_labels = None  # Adapter selon ton besoin
        class_2_superclass = None  # Adapter selon ton besoin
        client_test_ind = copy.deepcopy(client_labels) if client_labels else None

    else:
        raise NotImplementedError

    return train_loaders, test_loaders, local_classes, client_labels, class_2_superclass, client_test_ind, relabel
