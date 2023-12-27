import logging
import os
import time

from cities.queries.causal_insight import CausalInsight
from cities.utils.cleaning_utils import find_repo_root
from cities.utils.data_grabber import DataGrabber, list_interventions, list_outcomes

root = find_repo_root()
log_dir = os.path.join(root, "data", "tau_samples")
log_file_path = os.path.join(log_dir, ".sampling.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=log_file_path,
    filemode="w",
    format="%(asctime)s → %(name)s → %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


session_start = time.time()


num_samples = 1000

data = DataGrabber()

interventions = list_interventions()
outcomes = list_outcomes()


N_combinations_samples = len(interventions) * len(outcomes)


files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
num_files = len(files)

logging.info(
    f"{(num_files-2)} sample dictionaries already exist. "
    f"Starting to obtain {N_combinations_samples - (num_files -2)}"
    f" out of {N_combinations_samples} sample dictionaries needed."
)
remaining = N_combinations_samples - (num_files - 2)
for intervention in interventions:
    for outcome in outcomes:
        tau_samples_path = os.path.join(
            root,
            "data/tau_samples",
            f"{intervention}_{outcome}_{num_samples}_tau.pkl",
        )

        if not os.path.exists(tau_samples_path):
            start_time = time.time()
            logging.info(f"Sampling {outcome}/{intervention} pair now.")
            ci = CausalInsight(
                outcome_dataset=outcome,
                intervention_dataset=intervention,
                num_samples=num_samples,
            )

            ci.generate_tensed_samples()
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
                f"Done sampling {outcome}/{intervention} pair, completed in {duration:.2f} seconds."
                f" {remaining} out of {N_combinations_samples}  samples remain."
            )


session_ends = time.time()

logging.info(
    f"All samples are now available."
    f"Sampling took {session_ends - session_start:.2f} seconds, or {(session_ends - session_start)/60:.2f} minutes."
)
