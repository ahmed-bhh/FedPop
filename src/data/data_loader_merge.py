from src.data.generate_niid_users_mnist import generate_mnist_json


def get_dataloader_merge(algo):
    if algo == "FedPop":
        print("data loader of fedpop")
    elif algo == "FedBayes":
        generate_mnist_json()
        print("data loader of fedbayes")
    elif algo == "PFL":
        print("data loader of PFL")

    else:
        raise ValueError(f"Algorithme non reconnu : {algo}")
   # return dataloader
