import os
import sys
import subprocess

from cities.utils.cleaning_utils import find_repo_root



sys.path.insert(0, os.path.dirname(os.getcwd()))
root = find_repo_root()



def test_notebook_execution():
    notebook_path = f"{root}/docs/guides/causal_insights_demo.ipynb"

    try:

        result = subprocess.check_output([
            'jupyter', 'nbconvert', '--to', 'script', '--execute', '--stdout', notebook_path
            ], stderr=subprocess.STDOUT, text=True)
        
        # check for error messages
        if 'Traceback (most recent call last):' in result:
            raise AssertionError("Notebook execution failed with an error.")
    except subprocess.CalledProcessError as e:
        # if the CalledProcessError occurs, test failed
        raise AssertionError(f"Notebook execution failed")
