<p align="center">
  <img width="665" alt="Screenshot 2023-12-07 at 6 12 27 PM" src="https://github.com/BasisResearch/cities/assets/31873755/e2d1f973-ba8c-4d19-9a1a-0b3e431ebd89">
</p>


## Evaluating Policy Transfer via Similarity Analysis and Causal Inference 
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cd tests && python -m pytest
```


Welcome to the repository for [polis](http://polis.basis.ai/), developed by the [Basis Research Institute](https://www.basis.ai/) for [The Opportunity Project (TOP)](https://opportunity.census.gov/) 2023 in collaboration with the U.S. Department of Commerce. The primary goal of this project is to enhance access to data for local policymakers, facilitating more informed decision-making.

This is the backend repository for more advanced users. For a more pleasant frontend experience and more information, please use the [app](http://polis.basis.ai/).


### The repository is structured as follows:

```
├── cities
│   ├── modeling
│   ├── queries
│   └── utils
├── data
│   ├── model_guides
│   ├── processed
│   └── raw
├── docs
│   ├── experimental_notebooks
│   └── guides
├── scripts
└── tests
```    


If you're interested in downloading the data or exploring advanced features beyond the frontend, check out the `guides` folder in the `docs` directory. There, you'll find:
- `data_sources.ipynb` for information on data sources,
- `similarity_demo.ipynb` demonstrating the use of the `DataGrabber` class for easy data acces, and of  our `FipsQuery` class, which is the key tool in the similarity-focused part of the project,
- `causal_insights_demo.ipynb` for an overview of how the `CausalInsight` class can be used to explore the influence of a range of intervention variables thanks to causal inference tools we employed. [WIP]

Feel free to dive into these resources to gain deeper insights into the capabilities of the Polis project, or to reach out if you have any comments or suggestions.

