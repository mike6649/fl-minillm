from collections import OrderedDict
from typing import List
import torch
from data_utils.prompt_datasets import PromptDataset
import flwr as fl
import os

from evaluate import _inner_evaluate

import numpy as np

from fl.fl_fine_tuning import setup_fine_tuning
from train_fl_minillm import fine_tune, get_student_model, setup_args
from utils import get_tokenizer, print_rank

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    print("loaded state dict")
    assert hasattr(net, "generate")


def client(args, ds_config, rank):

    tokenizer = get_tokenizer(args)
    device = torch.cuda.current_device()
    args.prompt_data_dir = os.path.join(args.prompt_data_dir, str(rank - 1), "")
    finetuning_args, fine_tune_dataset = setup_fine_tuning(args, tokenizer, rank)
    data_dir = "processed_data/dolly/prompt/gpt2"
    args.json_data = True
    eval_dataset = PromptDataset(args, tokenizer, "valid", data_dir, 10)
    model = get_student_model(args, device)

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, model):
            self.net = model
            assert hasattr(self.net, "generate")

        def get_parameters(self, config):
            return get_parameters(self.net)

        def fit(self, parameters, config):
            set_parameters(self.net, parameters)
            self.net = fine_tune(self.net, finetuning_args, tokenizer, fine_tune_dataset, ds_config)
            return get_parameters(self.net), len(fine_tune_dataset), {}

        def evaluate(self, parameters, config):
            set_parameters(self.net, parameters)
            results = _inner_evaluate(args, tokenizer, self.net, eval_dataset, device)
            # {'exact_match': 1.7, 'rougeL': 19.7316, 'loss': 3.5203}
            return float(results["loss"]), len(eval_dataset), results

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model))

def server(args):
    def weighted_average(metrics):
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["rougeL"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"rougeL": sum(accuracies) / sum(examples)}
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=args.num_clients,
    )

    # Start server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.fl_rounds),
        strategy=strategy,
    )


def main():
    ds_config, args = setup_args()
    
    rank = args.fl_rank
    print_rank(f"Launched process {rank}\n")

    if (rank == 0):
        server(args)
    else:
        client(args,ds_config, rank)


if __name__ == "__main__":
    main()
