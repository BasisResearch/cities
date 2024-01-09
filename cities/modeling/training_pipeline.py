import logging
import os
import time

from cities.modeling.model_interactions import InteractionsModel
from cities.utils.data_grabber import find_repo_root, list_interventions, list_outcomes

root = find_repo_root()
log_dir = os.path.join(root, "data", "model_guides")
log_file_path = os.path.join(log_dir, ".training.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s → %(name)s → %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# if you need to train from scratch
# clean data/model_guides folder manually
# automatic fresh start is not implemented
# for security reasons

num_iterations = 4000

interventions = list_interventions()
outcomes = list_outcomes()
shifts = [1, 2, 3]


N_combinations = len(interventions) * len(outcomes) * len(shifts)

files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
num_files = len(files)


logging.info(
    f"{(num_files-2)/2} guides already exist. "
    f"Starting to train {N_combinations - (num_files -2)/2} out of {N_combinations} guides needed."
)

remaining = N_combinations - (num_files - 2) / 2
for intervention_dataset in interventions:
    for outcome_dataset in outcomes:
        for forward_shift in shifts:
            # check if the corresponding guide already exists
            # existing_guides = 0  seems rendundant, remove if all works
            guide_name = f"{intervention_dataset}_{outcome_dataset}_{forward_shift}"
            guide_path = os.path.join(
                root, "data/model_guides", f"{guide_name}_guide.pkl"
            )
            if not os.path.exists(guide_path):
                # existing_guides += 1 seems redundat remove if all works

                logging.info(f"Training {guide_name} for {num_iterations} iterations.")

                start_time = time.time()
                model = InteractionsModel(
                    outcome_dataset=outcome_dataset,
                    intervention_dataset=intervention_dataset,
                    forward_shift=forward_shift,
                    num_iterations=num_iterations,
                    plot_loss=False,
                )

                model.train_interactions_model()
                model.save_guide()

                end_time = time.time()
                duration = end_time - start_time
                files = [
                    f
                    for f in os.listdir(log_dir)
                    if os.path.isfile(os.path.join(log_dir, f))
                ]
                num_files = len(files)
                remaining -= 1
                logging.info(
                    f"Training of {guide_name} completed in {duration:.2f} seconds. "
                    f"{remaining} out of {N_combinations} guides remain to be trained."
                )

logging.info("All guides are now available.")
