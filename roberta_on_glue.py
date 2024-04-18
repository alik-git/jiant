import wandb
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader
import logging
# 89, 88, 86 

# Define the list of tasks
glue_tasks = [
    "cola", "sst", "mrpc", "stsb", "qqp", 
    "mnli", "mnli_mismatched", "qnli", "rte", "wnli", "glue_diagnostics"
]

# Base directories
base_exp_dir = "/home/kuwajerw/repos/jiant/glue_exps_tasks"
base_data_dir = "/home/kuwajerw/repos/jiant/glue_exps_tasks"

# Configuration dictionary
cfg = {
    "model_name_or_path": "prajjwal1/bert-tiny",
    "train_batch_size": 16,
    "num_train_epochs": 3,
    "write_test_preds": True,
}

# Iterate over each task and handle exceptions
for task in glue_tasks:
    try:
        # Set task-specific directories
        task_exp_dir = f"{base_exp_dir}/glue_{task}"
        task_data_dir = f"{task_exp_dir}/glue_{task}_tasks"

        # Update cfg for the current task
        cfg.update({
            "exp_dir": task_exp_dir,
            "data_dir": task_data_dir,
            "run_name": f"glue_{task}",
        })

        # Initialize wandb with cfg for the current task
        wandb.init(project="glue_benchmark", config=cfg, entity="mbrl_ducky", reinit=True)

        # Download the Data for the current task
        downloader.download_data([task], task_data_dir)

        # Set up the arguments for the Simple API using the cfg dictionary for the current task
        args = run.RunConfiguration(
            run_name=cfg["run_name"],
            exp_dir=cfg["exp_dir"],
            data_dir=cfg["data_dir"],
            hf_pretrained_model_name_or_path=cfg["model_name_or_path"],
            tasks=task,
            train_batch_size=cfg["train_batch_size"],
            num_train_epochs=cfg["num_train_epochs"],
            write_test_preds=cfg["write_test_preds"],
        )

        # Run the experiment for the current task
        run.run_simple(args)

    except Exception as e:
        # Log the error for the current task
        logging.error(f"Error processing task {task}: {e}")
        wandb.log({"error": f"Error processing task {task}: {str(e)}"})

    finally:
        # Finish wandb session
        wandb.finish()