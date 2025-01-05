from collections import defaultdict
import click
import numpy as np
import torch
import torch.nn as nn
from src.data import get_dataloaders
from src.fedsoul import FedSOUL
import argparse
from src.models import PersonalModel
from src.models.model_factory import set_model
from src.models.models import pBNN
from src.models.serverpFedbayes import pFedBayes
from src.samplers import optimization_factory
from src.utils import seed_everything, parse_args, get_config, dotdict, count_parameters, compute_metric, log_results
import yaml

from utils.plot_utils import average_data
from vem import train

def parse_yaml_to_flat_dict(config_data):
    """
    Convertit un fichier YAML structuré en un dictionnaire aplati avec des clés spécifiques.
    """
    flat_config = {
        "model": config_data["model_params"]["model_name"],
        "dataset": config_data["data_params"]["dataset_name"],
        "path_to_data": config_data["data_params"]["root_path"],
        "batch_size": config_data["data_params"]["train_batch_size"],
        "lr_head": config_data["optimization"]["learning_rate"],
        "lr_base": config_data["optimization"]["personal_learning_rate"],
        "base_epochs": config_data["optimization"]["local_epochs"],
        "momentum": 0.9,  # Momentum est manquant dans le YAML, on peut l'ajouter comme valeur par défaut
        "n_labels": config_data["data_params"]["specific_dataset_params"]["classes_per_user"],
        "n_rounds": config_data["optimization"]["global_iters"],
        "n_clients": config_data["data_params"]["specific_dataset_params"]["n_clients"],
        "sampling_rate": config_data["train_params"]["num_clients_per_round"] / config_data["data_params"]["specific_dataset_params"]["n_clients"],
        "seed": config_data["train_params"]["seeds"][0],
        "relabel": False,  # Relabel est manquant dans le YAML, on peut l'ajouter comme valeur par défaut
        "head_epochs": 10,  # Valeur par défaut ajoutée
        "scale": config_data["model_params"]["weight_scale"],
        "max_data": config_data["data_params"]["max_dataset_size_per_user"],
        "beta": config_data["model_params"]["beta"],
        "n_mc": 5,  # Valeur par défaut ajoutée
    }
    return flat_config
@click.command()
@click.option(
            "--config", help="Path to the configuration file (YAML).", default=None,
)
def PFL(**kwargs):
    if kwargs["config"]:
        with open(kwargs["config"], 'r') as f:
            # Charger le YAML
            config_data = yaml.safe_load(f)
            # Transformer en dictionnaire aplati
            flat_config = parse_yaml_to_flat_dict(config_data)
            kwargs.update(flat_config)
    # Supprimer la clé 'config' après chargement
    kwargs.pop("config", None)

    print(kwargs)  # Affichage pour vérification
    train(**kwargs)


def run(config, trial=None) -> dict:
    print(config)

    # Here we receive dotdicts, to access fields via dot operator.
    data_params = dotdict(config['data_params'])
    train_params = dotdict(config['train_params'])
    model_params = dotdict(config['model_params'])
    eval_params = dotdict(config['eval_params'])
    optimization_params = dotdict(config['optimization'])
    specific_dataset_params=dotdict(config['data_params']['specific_dataset_params'])


    train_params.n_personal_models = data_params.specific_dataset_params["n_clients"]

    OUTER_ITERS = train_params.outer_iters
    INNER_ITERS = train_params.inner_iters
    DEVICE = train_params.device
    #print(DEVICE)
    #exit()
    N_PERSONAL_MODELS = train_params.n_personal_models
    #print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",train_params.algorithm)
    if train_params.algorithm == 'FedPop':
        print("you are using",train_params.algorithm)
        # Here we define log prior function
        if train_params.prior:
            LOG_PRIOR_FN = torch.distributions.Normal(loc=torch.tensor(0., dtype=torch.float32, device=DEVICE),
                                                      scale=torch.tensor(1., dtype=torch.float32, device=DEVICE)).log_prob
        else:
            raise ValueError(f"No such prior available! '{train_params.prior}'")

        # Here we define log likelihood function
        if train_params.loss_fn_name == 'mse':
            LOG_LIKELIHOOD_FN = lambda x, y: -nn.MSELoss(reduction='sum')(x.squeeze(), y.squeeze())
        elif train_params.loss_fn_name == 'cross_entropy':
            LOG_LIKELIHOOD_FN = lambda x, y: -nn.CrossEntropyLoss(reduction='sum')(x, y)
        elif train_params.loss_fn_name == 'multivariate_gaussian_ll':
            def multivariate_gaussian_ll(y_pred, y):
                mu = y_pred[:2]
                lower_diag = y_pred[2:].view(2, 2)
                return torch.distributions.MultivariateNormal(loc=mu, scale_tril=lower_diag).log_prob(y).sum()

            LOG_LIKELIHOOD_FN = multivariate_gaussian_ll
        else:
            raise ValueError(f"Wrong loss name...{train_params.loss}")

        #######################################################################################
        ############################# Below, we define models #################################
        #######################################################################################

        # To define personal model (parameterized by \theta), we should decide which input size it takes
        personal_model_params = model_params.personal_model_params
        if model_params.composition_model_regime == "composition":
            personal_model_params['input_size'] = model_params.shared_model_params["shared_embedding_size"]
        else:
            personal_model_params['input_size'] = model_params.shared_model_params["shared_embedding_size"] + \
                                                  model_params.backbone_model_params["backbone_embedding_size"]

        metrics_dict = defaultdict(list)

        # Here we start cycle over different seeds
        for seed in train_params.seeds:
            seed_everything(seed)
            train_loaders, test_loaders = get_dataloaders(generate_data=data_params.generate_dataloaders,
                                                          dataset_name=data_params.dataset_name,
                                                          specific_dataset_params=data_params.specific_dataset_params,
                                                          root_path=data_params.root_path,
                                                          batch_size_train=data_params.train_batch_size,
                                                          batch_size_test=data_params.test_batch_size,
                                                          DEVICE=DEVICE,
                                                          regression=data_params.regression,
                                                          max_dataset_size_per_user=data_params.max_dataset_size_per_user,
                                                          min_dataset_size_per_user=data_params.min_dataset_size_per_user,
                                                          n_clients_with_min_datasets=data_params.n_clients_with_min_datasets,
                                                          )

            # We define one shared model
            shared_model = set_model(model_name=model_params.shared_model_name, device=DEVICE,
                                     model_params=model_params.shared_model_params, model_type='shared')
            shared_model.set_log_prior_fn(log_prior_fn=LOG_PRIOR_FN)
            shared_optim, shared_scheduler = optimization_factory(parameters=shared_model.parameters(),
                                                                  optimization_params={k[len('shared') + 1:]: v for k, v in
                                                                                       optimization_params.items() if
                                                                                       k.startswith('shared')})

            personal_models = []
            personal_optims = []
            personal_schedulers = []

            backbone_optims = []
            backbone_schedulers = []

            prior_models = []
            prior_optims = []
            prior_schedulers = []

            # And we define N_PERSONAL_MODELS personal models
            for i in range(N_PERSONAL_MODELS):
                #####################################
                #####################################
                # First -- backbone
                backbone_model = set_model(model_name=model_params.backbone_model_name, device=DEVICE,
                                           model_params=model_params.backbone_model_params,
                                           model_type='backbone')
                # Second -- high level personal model
                personal_model = set_model(model_name=model_params.personal_model_name, device=DEVICE,
                                           model_params=personal_model_params, model_type='personal')
                # Energy model takes as an input both vectors -- parameters of personal and personal backbone models
                model_params.prior_model_params.update({
                    "in_features": count_parameters(personal_model),
                    "log_prior_fn": LOG_PRIOR_FN,
                })
                # An instance of Personal Model consists of sets of parameters \theta, \theta_b
                personal_models.append(PersonalModel(
                    model=personal_model,
                    backbone_model=backbone_model))

                # And next, we define corresponding optims
                # For personal model
                personal_optim, personal_scheduler = optimization_factory(
                    parameters=list(personal_models[i].model.parameters()),
                    optimization_params={k[len("personal") + 1:]: v for k, v in
                                         optimization_params.items() if
                                         k.startswith('personal')})
                personal_optims.append(personal_optim)
                personal_schedulers.append(personal_scheduler)

                # And for the backbone
                if personal_models[i].backbone_n_parameters > 0:
                    backbone_optim, backbone_scheduler = optimization_factory(
                        parameters=list(personal_models[i].backbone_model.parameters()),
                        optimization_params={k[len("backbone") + 1:]: v for k, v in
                                             optimization_params.items() if
                                             k.startswith('backbone')})
                    backbone_optims.append(backbone_optim)
                    backbone_schedulers.append(backbone_scheduler)

                #####################################
                #####################################

                # We define \beta (prior model) as an instance of another special class
                if model_params.shared_prior_model and i == 0:  # if we share prior model, than we add it only once
                    prior_models.append(set_model(model_name=model_params.prior_model_name, device=DEVICE,
                                                  model_params=model_params.prior_model_params, model_type='prior').to(
                        DEVICE))
                    prior_optim, prior_scheduler = optimization_factory(
                        parameters=prior_models[0].parameters(),
                        optimization_params={k[len('prior_model') + 1:]: v for k, v in
                                             optimization_params.items() if
                                             k.startswith('prior_model')})
                    prior_optims.append(prior_optim)
                    prior_schedulers.append(prior_scheduler)
                else:  # else, we have an array of models
                    if model_params.shared_prior_model:
                        continue
                    prior_models.append(set_model(model_name=model_params.prior_model_name, device=DEVICE,
                                                  model_params=model_params.prior_model_params, model_type='prior').to(
                        DEVICE))
                    prior_optim, prior_scheduler = optimization_factory(
                        parameters=prior_models[i].parameters(),
                        optimization_params={k[len('prior_model') + 1:]: v for k, v in
                                             optimization_params.items() if
                                             k.startswith('prior_model')})
                    prior_optims.append(prior_optim)
                    prior_schedulers.append(prior_scheduler)

            ###########################################################################
            ###########################################################################
            initial_metrics = defaultdict(list)
            for metric_name in eval_params.metrics:
                print(f'Checking {metric_name} of initial models')
                m = compute_metric(metric=metric_name, personal_models=personal_models,
                                   loaders=test_loaders if isinstance(test_loaders, list) else train_loaders,
                                   shared_model=shared_model, composition_regime=model_params.composition_model_regime)
                initial_metrics[metric_name].append(m)

            print('Initial metrics:')
            for k, v in initial_metrics.items():
                print(f"{k} : {np.mean(v)} +/- {0 if len(v) == 0 else np.std(v)}")

            models = FedSOUL(
                outer_iters=OUTER_ITERS,
                inner_iters=INNER_ITERS,
                clients_sample_size=train_params.clients_sample_size,
                personal_models=personal_models,
                personal_optims=personal_optims,
                personal_schedulers=personal_schedulers,
                backbone_optims=backbone_optims,
                backbone_schedulers=backbone_schedulers,
                prior_models=prior_models,
                prior_optims=prior_optims,
                prior_schedulers=prior_schedulers,
                shared_model=shared_model,
                shared_optim=shared_optim,
                shared_scheduler=shared_scheduler,
                local_dataloaders=train_loaders,
                log_likelihood_fn=LOG_LIKELIHOOD_FN,
                burn_in=train_params.inner_burn_in,
                device=DEVICE,
                composition_regime=model_params.composition_model_regime,
                use_sgld=train_params.use_sgld,
                verbose=train_params.verbose,
                verbose_freq=train_params.verbose_freq,
                test_loaders=test_loaders,
                metrics=eval_params.metrics,
                trial=trial,
            )
            shared_model, personal_models, prior_models = models[0], models[1], models[2]

            for metric_name in eval_params.metrics:
                print(f'Checking {metric_name} of final models')
                m = compute_metric(metric=metric_name, personal_models=personal_models,
                                   loaders=test_loaders if isinstance(test_loaders, list) else train_loaders,
                                   shared_model=shared_model, composition_regime=model_params.composition_model_regime)
                metrics_dict[metric_name].append(m)

            print('Final metrics:')
            for k, v in metrics_dict.items():
                print(f"{k} : {np.mean(v)} +/- {0 if len(v) == 0 else np.std(v)}")

        return {
            "metrics": {k: [np.mean(v), 0 if len(v) == 0 else np.std(v)] for k, v in metrics_dict.items()},
            "shared_model": shared_model,
            "personal_models": personal_models,
            "prior_models": prior_models,
            "train_loaders": train_loaders,
            "test_loaders": test_loaders
        }
    if train_params.algorithm == 'FedBayes':
     runtime_params = dotdict(config['runtime_params'])

     print("you are using",train_params.algorithm)
     print("Configuration chargée avec succès depuis : {config_path}")
     print(config)

     print("Dataset Name:", data_params.dataset_name)
     print("Train Batch Size:", data_params.train_batch_size)
     print("Number of Clients:", specific_dataset_params.n_clients)
     print("Learning Rate:", optimization_params.learning_rate)
     print("Device:", train_params.device)
     print("Algorithm:", train_params.algorithm)
     print("times",runtime_params.num_runs)
     return main(
         dataset=data_params.dataset_name,
         algorithm=train_params.algorithm,
         model=model_params.model_name,
         batch_size=data_params.train_batch_size,
         learning_rate=optimization_params.learning_rate,
         beta=model_params.beta,
         lamda=model_params.lamda,
         num_glob_iters=optimization_params.global_iters,
         local_epochs=optimization_params.local_epochs,
         optimizer=optimization_params.optimizer_name,
         numusers=specific_dataset_params.n_clients,
         K=optimization_params.computation_steps,
         personal_learning_rate=optimization_params.personal_learning_rate,
         times=runtime_params.num_runs,
         device=train_params.device,
         weight_scale=model_params.weight_scale,
         rho_offset=model_params.rho_offset,
         zeta=model_params.zeta
     )
    if train_params.algorithm == 'PFL':

        print("PFL  IF")
        PFL()


def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, device,
         weight_scale, rho_offset, zeta):
    post_fix_str = 'plr_{}_lr_{}'.format(personal_learning_rate, learning_rate)
    model_path = []
    for i in range(times):
        print("---------------Running time:------------", i)
        if model == "pbnn":
            if dataset == "Mnist":
                model = pBNN(784, 100, 10, device, weight_scale, rho_offset, zeta).to(device), model
            else:
                model = pBNN(3072, 100, 10, device, weight_scale, rho_offset, zeta).to(device), model

        server = pFedBayes(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                           local_epochs, optimizer, numusers, i, device, personal_learning_rate,
                           post_fix_str=post_fix_str)

        model_path.append(server.train())
        _, nums_list, acc_list, _ = server.testpFedbayes()

    result_path = average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, beta=beta, algorithms=algorithm, batch_size=batch_size,
                               dataset=dataset, k=K, personal_learning_rate=personal_learning_rate, times=times,
                               post_fix_str=post_fix_str)
    return model_path, result_path


def runFedBayes():
    class DotDict(dict):
        """Un dictionnaire permettant l'accès par attribut."""

        def __getattr__(self, name):
            value = self.get(name)
            if isinstance(value, dict):
                return DotDict(value)
            return value

    # def load_config(config_path="data/config/mnist_config.yaml"):
    #     with open(config_path, "r") as file:
    #         return DotDict(yaml.safe_load(file))
    #
    # # Charger la configuration
    # config = load_config()
    def load_config(config_path):
        """Charge la configuration à partir d'un fichier YAML."""
        with open(config_path, "r") as file:
            return DotDict(yaml.safe_load(file))

    # Récupérer les arguments
    parser = argparse.ArgumentParser(description="Exécution du modèle avec un fichier de configuration.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin du fichier YAML de configuration."
    )
    args = parser.parse_args()

    # Vérifier et afficher le chemin de configuration
    print(f"Argument config_path reçu : {args.config}")
    if not args.config:
        raise ValueError("Aucun chemin de fichier de configuration spécifié.")

    # Charger la configuration
    config_path = args.config
    config = load_config(config_path)
    print(f"Configuration chargée avec succès depuis : {config_path}")
    print(config)

    print("Dataset Name:", config.data_params.dataset_name)
    print("Train Batch Size:", config.data_params.train_batch_size)
    print("Number of Clients:", config.data_params.specific_dataset_params.n_clients)
    print("Learning Rate:", config.optimization.learning_rate)
    print("Device:", config.train_params.device)
    print("Algorithm:", config.train_params.algorithm)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist"])
    #
    # parser.add_argument("--model", type=str, default="pbnn", choices=["pbnn"])
    # parser.add_argument("--batch_size", type=int, default=100)
    # parser.add_argument("--learning_rate", type=float, default=0.001,
    #                     help="Local learning rate")
    # parser.add_argument("--weight_scale", type=float, default=0.1)
    # parser.add_argument("--rho_offset", type=int, default=-3)
    # parser.add_argument("--zeta", type=int, default=10)
    # parser.add_argument("--beta", type=float, default=1.0,
    #                     help="Average moving parameter for pFedMe")
    # parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    # parser.add_argument("--num_global_iters", type=int, default=10)
    # parser.add_argument("--local_epochs", type=int, default=20)
    # parser.add_argument("--optimizer", type=str, default="SGD")
    # parser.add_argument("--algorithm", type=str, default="pFedBayes",
    #                     choices=["pFedMe", "FedAvg", "FedBayes"])
    # parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    # parser.add_argument("--K", type=int, default=5, help="Computation steps")
    # parser.add_argument("--personal_learning_rate", type=float, default=0.001,
    #                     help="Persionalized learning rate to caculate theta aproximately using K steps")
    # parser.add_argument("--times", type=int, default=1, help="running time")
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    # print("=" * 80)
    # print("Summary of training process:")
    # print("Algorithm: {}".format(args.algorithm))
    # print("Batch size: {}".format(args.batch_size))
    # print("Learing rate       : {}".format(args.learning_rate))
    # print("Average Moving       : {}".format(args.beta))
    # print("Subset of users      : {}".format(args.numusers))
    # print("Number of global rounds       : {}".format(args.num_global_iters))
    # print("Number of local rounds       : {}".format(args.local_epochs))
    # print("Dataset       : {}".format(args.dataset))
    # print("Local Model       : {}".format(args.model))
    # print("=" * 80)

    return main(
        dataset=config.data_params.dataset_name,
        algorithm=config.train_params.algorithm,
        model=config.model_params.model_name,
        batch_size=config.data_params.train_batch_size,
        learning_rate=config.optimization.learning_rate,
        beta=config.model_params.beta,
        lamda=config.model_params.lamda,
        num_glob_iters=config.optimization.global_iters,
        local_epochs=config.optimization.local_epochs,
        optimizer=config.optimization.optimizer_name,
        numusers=config.data_params.specific_dataset_params.n_clients,
        K=config.optimization.computation_steps,
        personal_learning_rate=config.optimization.personal_learning_rate,
        times=config.runtime_params.num_runs,
        device=config.train_params.device,
        weight_scale=config.model_params.weight_scale,
        rho_offset=config.model_params.rho_offset,
        zeta=config.model_params.zeta
    )



if __name__ == '__main__':
    args = parse_args()
    conf = get_config(args.config) #ici on utulise l argument --config passer comme argument est convertis en dictionnair python
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx",conf['train_params']['algorithm'])

   # if conf['train_params']['algorithm'] == 'FedPop':
    output = run(conf)

    if conf['train_params']['algorithm'] == 'FedPop':
     log_results(config_path=args.config, config=conf, output=output)
