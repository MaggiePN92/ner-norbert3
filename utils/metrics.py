import os
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

def save_metrics(model_name:str, metrics:dict):
    
    """
    
    Writes out a .csv file with the name "model_name_metrics.csv". 
    
    - Supply metrics supplied in a dict.
    - Stick to the dict names below (or add new ones to it) to avoid empty cells.
    - model_name must match previously trained model to append, otherwise a new file will be created.
    - If a metric is not set, it will be -1.
    
    """

    # Get current time
    now = datetime.now()
    date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")
    
    # Dict structure
    metrics_dict = {
        'Date Time':date_and_time,
        'Epochs':-1,
        'Accuracy':-1, 
        'Precision':-1,
        'Recall':-1,
        'F1':-1,
        'Num Train Sentences':-1, # n_sentences used for training
        'Training Time':-1,
        'Learning Rate':-1,
        'Seed': -1
        }  
    
    # Handle unspecified columns.
    for key in metrics:
        if key in metrics_dict:
            metrics_dict[key] = metrics[key]
        else:
            assert key in metrics_dict, f"{key} is not a valid key. Check metrics.py for a list of valid keys."
    
    metrics_path = "./metrics_logs/" + model_name + "_metrics.csv"

    metrics_frame = pd.DataFrame(metrics_dict, index=[0])
    
    # Check if metrics csv exists before saving to avoid overwriting data.
    if (os.path.exists(metrics_path)):
        df = pd.read_csv(metrics_path, index_col=0)
        df = pd.concat([df, metrics_frame], ignore_index=True)
        df.to_csv(metrics_path)
    else:
        metrics_frame.to_csv(metrics_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path2data", type=str, 
                        default="/Users/magnusnytun/Documents/git/in5550/obligatory3/data/norne-nb-in5550-train.conllu.gz")
    parser.add_argument("--model_name", type=str, 
                        default="unnamed")
    parser.add_argument("--learning_rate", type=float, 
                        default=5e-5)
    parser.add_argument("--hidden_size", type=str, 
                        default="unnamed")
    parser.add_argument("--epochs", type=int, 
                        default=2)
    args = parser.parse_args()
    dictionary = {        
        'Accuracy': 123, 
        'Precision':123,
        'Recall':123,
        'F1':123,
    }

    dictionary2 = {        
        'Acuracy': 123, # Assertion
        'Precision':123,
        'Recall':123,
        'Epochs':123,
    }

    save_metrics("test_metrics", dictionary)
    save_metrics("test_metrics", dictionary2)