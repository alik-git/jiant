import wandb
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

# glue_tasks = [
#     "cola", "sst", "mrpc", "stsb", "qqp", 
#     "mnli", "mnli_mismatched", "qnli", "rte", "wnli", "glue_diagnostics"
# ]

glue_tasks = ["mrpc", "glue_diagnostics"]  # For testing

# Configuration dictionary
cfg = {
    "exp_dir": "/home/kuwajerw/repos/jiant/glue_exps",
    "data_dir": "/home/kuwajerw/repos/jiant/glue_exps/glue_tasks",
    "model_name_or_path": "roberta-base",
    "tasks": glue_tasks,
    "train_batch_size": 16,
    "num_train_epochs": 3,
    "write_test_preds": True,
    "run_name": "full_glue",
}

# Initialize wandb with cfg
wandb.init(project="glue_benchmark", config=cfg, entity="mbrl_ducky")

# Convert task list to a string for the API call
tasks_string = ",".join(cfg["tasks"])

# Download the Data
downloader.download_data(cfg["tasks"], cfg["data_dir"])

# Set up the arguments for the Simple API using the cfg dictionary
args = run.RunConfiguration(
   run_name=cfg["run_name"],
   exp_dir=cfg["exp_dir"],
   data_dir=cfg["data_dir"],
   hf_pretrained_model_name_or_path=cfg["model_name_or_path"],
   tasks=tasks_string,
   train_batch_size=cfg["train_batch_size"],
   num_train_epochs=cfg["num_train_epochs"],
   write_test_preds=cfg["write_test_preds"],
)

# Run the experiment
run.run_simple(args)

# Finish wandb session
wandb.finish()