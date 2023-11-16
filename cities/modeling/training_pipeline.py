import os
import logging
import time

from cities.utils.data_grabber import list_interventions, list_outcomes, DataGrabber
from cities.modeling.model_interactions import InteractionsModel
from cities.utils.cleaning_utils import find_repo_root



root = find_repo_root()
log_dir = os.path.join(root, "data", "model_guides")
log_file_path = os.path.join(log_dir, ".training.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=log_file_path, filemode="w", 
                format="%(asctime)s → %(name)s → %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


# if you need to train from scratch
# clean data/model_guides folder manually
# automatic fresh start is not implemented
# for security reasons


num_iterations = 5000

interventions = list_interventions()
outcomes = list_outcomes()
shifts = [1,2,3]


N_combinations = len(interventions) * len(outcomes) * len(shifts)

files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
num_files = len(files)

logging.info(f"{num_files} guides already exist. Starting to train {N_combinations - num_files+1} out of {N_combinations} guides.")

dg = DataGrabber()
dg.get_features_std_long(list_interventions())

intervention_variables = []

for intervention_dataset in interventions:
    intervention_variables.append(dg.std_long[intervention_dataset].columns[-1])

for intervention_dataset, intervention_variable in zip(interventions, intervention_variables):
    for outcome_dataset in outcomes:
        for forward_shift in shifts:
            
            guide_name = (f"{intervention_dataset}_{outcome_dataset}_{forward_shift}")
            file_path = os.path.join(root, "data/model_guides", f"{guide_name}_guide.pkl")

            if not os.path.exists(file_path):
                logging.info(f"Training {guide_name} for {num_iterations} iterations.")
                
                start_time = time.time()
                model = InteractionsModel(outcome_dataset=outcome_dataset,
                        intervention_dataset=intervention_dataset,
                        intervention_variable=intervention_variable,
                        forward_shift = forward_shift,
                        num_iterations= num_iterations)
                
                model.train_interactions_model()
                model.save_guide()

                end_time = time.time()
                duration = end_time - start_time
                files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
                num_files = len(files)
                logging.info(f"Training of {guide_name} completed in {duration:.2f} seconds. "
                f"{N_combinations - num_files+1} out of {N_combinations} guides remain to be trained.")
                
