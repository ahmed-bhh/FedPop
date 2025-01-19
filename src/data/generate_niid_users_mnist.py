import os
import json
from tqdm import trange
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms


def generate_mnist_json(train_path='data/train/mnist_train.json',
                        test_path='data/test/mnist_test.json',
                        num_users=10,
                        num_labels=5):
    """
    Génère des données MNIST réparties entre utilisateurs et les sauvegarde dans des fichiers JSON.

    Arguments :
        train_path (str) : Chemin du fichier JSON pour les données d'entraînement.
        test_path (str) : Chemin du fichier JSON pour les données de test.
        num_users (int) : Nombre d'utilisateurs.
        num_labels (int) : Nombre de labels par utilisateur.
    """
    # Vérifier si les fichiers JSON existent
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Both JSON files already exist. Skipping data loading.")
        return

    # Chargement des données MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    # Initialisation des paramètres
    random.seed(1)
    np.random.seed(1)

    # Création des dossiers si nécessaires
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    # Préparation des données
    fmnist_data_image = np.concatenate((trainset.data.cpu().numpy(), testset.data.cpu().numpy()), axis=0)
    fmnist_data_label = np.concatenate((trainset.targets.cpu().numpy(), testset.targets.cpu().numpy()), axis=0)

    fmnist_data = [fmnist_data_image[fmnist_data_label == i] for i in range(10)]

    print("\nNumb samples of each label:\n", [len(v) for v in fmnist_data])

    # Attribution des labels aux utilisateurs
    users_labels = [(user * num_labels + j) % 10 for user in range(num_users) for j in range(num_labels)]
    unique, counts = np.unique(users_labels, return_counts=True)
    print("Labels distribution per user:\n", dict(zip(unique, counts)))

    # Génération des échantillons aléatoires
    def random_sample_distribution(total, size):
        return [min(total // (size + 1), 1000)] * (size - 1) + [min(total // 2, 1000)]

    number_sample = [random_sample_distribution(len(fmnist_data[label]), count) for label, count in zip(unique, counts)]
    number_samples = [sample[i] for i in range(len(number_sample[0])) for sample in number_sample]

    # Répartition des données entre utilisateurs
    X = [[] for _ in range(num_users)]
    y = [[] for _ in range(num_users)]
    idx = np.zeros(10, dtype=np.int64)
    count = 0
    for user in range(num_users):
        for j in range(num_labels):
            label = (user * num_labels + j) % 10
            num_samples = number_samples[count]
            count += 1
            X[user] += fmnist_data[label][idx[label]:idx[label] + num_samples].tolist()
            y[user] += [label] * num_samples
            idx[label] += num_samples

    # Construction des structures JSON
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(num_users):
        uname = f'f_{i:05d}'
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i], y[i] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.2 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    # Sauvegarde des données dans les fichiers JSON
    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)

    print("Data generation completed and saved to JSON files.")
