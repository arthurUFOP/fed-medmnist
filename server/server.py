import flwr as fl
import time

dbs = [
    "bloodmnist",
    "breastmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
    "pathmnist",
    "pneumoniamnist",
    "retinamnist",
    "tissuemnist",
]

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

for db in dbs:
    print(f"Starting train for {db} DB!\n\n")
    history = fl.server.start_server(config=fl.server.ServerConfig(num_rounds=10),
                                    strategy=strategy)
    with open(f"./experiment-logs/{db}-10local-10global.log", "w+") as f:
        f.write(f"DB = {db}\n\n")
        f.write(str(history))
    print("---------------------------------------\n\n")
    time.sleep(2)