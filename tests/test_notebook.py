import os
import sys

current_dir = os.getcwd()
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, grandparent_dir)


# import nbformat
# from nbconvert.preprocessors import ExecutePreprocessor
import os
import subprocess


def test_notebook_execution():
    notebook_path = "cities/docs/guides/notebook_test.ipynb"

    try:
        # Execute the notebook using Jupyter command-line interface
        subprocess.check_call(
            ["jupyter", "nbconvert", "--execute", "--inplace", notebook_path]
        )
    except subprocess.CalledProcessError as e:
        # If any cell execution fails, mark the test as failed
        raise AssertionError(f"Notebook execution failed: {e}")
