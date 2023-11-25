import json
from flwr.server import History
from datetime import datetime



def save_run_as_json(config, history: History, filename=None):
    if filename is None:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"results/{current_datetime}.json"
    config_dict = vars(config)
    history_dict = vars(history)
    results = {
        "config": config_dict,
        "results": history_dict,
    }
    json.dump(results, open(filename, "w"), indent=2)
    print(f"Saved run to {filename}")
