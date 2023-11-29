import glob
import os
import subprocess

from cities.utils.cleaning_utils import find_repo_root

root = find_repo_root()


def test_notebook_execution():
    os.environ["CI"] = "1"  # possibly redundant

    notebook_path = f"{root}/docs/guides/"
    notebooks = glob.glob(os.path.join(notebook_path, "*.ipynb"))

    # run this command from terminal if the test fails to identify the source of trouble, if any
    pytest_command = f"C1=1 python -m pytest --nbval-lax --dist loadscope -n auto {' '.join(notebooks)}"
    # print(pytest_command)

    subprocess.run(pytest_command, shell=True, check=True)
