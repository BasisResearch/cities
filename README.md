<p align="center">
  <img width="665" alt="Screenshot 2023-12-07 at 6 12 27 PM" src="https://github.com/BasisResearch/cities/assets/31873755/e2d1f973-ba8c-4d19-9a1a-0b3e431ebd89">
</p>


# Evaluating Policy Transfer via Similarity Analysis and Causal Inference


## Getting started


Welcome to the repository for [polis](http://polis.basis.ai/), developed by [Basis Research Institute](https://www.basis.ai/) for [The Opportunity Project (TOP)](https://opportunity.census.gov/) 2023 in collaboration with the U.S. Department of Commerce. The primary goal of this project is to enhance access to data for local policymakers, facilitating more informed decision-making.

This is the backend repository for more advanced users. For a more pleasant frontend experience and more information, please use the [app](http://polis.basis.ai/).


Installation
------------

**Basic Setup:**

```sh

    git clone git@github.com:BasisResearch/cities.git
    cd cities
    git checkout main
    pip install .
```

The above will install the minimal version that's ported to [polis.basis.ai](http://polis.basis.ai)

**Dev Setup:**

To install dev dependencies, needed to run models, train models and run all the tests, run the following command:

```sh
pip install -e .'[dev]'
```

Details of which packages are available in which see `setup.py`.


** Contributing: **

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally

```sh
make lint              # linting
make format            # runs black and isort, including on notebooks in the docs/ folder
make tests             # linting, unit and notebook tests
```


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

**WARNING: during the beta testing, the most recent version lives on the `staging-county-data` git branch, and so do the most recent versions of the notebooks. Please switch to this branch before inspecting the notebooks.

If you're interested in downloading the data or exploring advanced features beyond the frontend, check out the `guides` folder in the `docs` directory. There, you'll find:
- `data_sources.ipynb` for information on data sources,
- `similarity-conceptual.ipynb` for  a conceptual account of how similarity comparison works.
- `counterfactual-explained.ipynb` contains a rough explanation of how our causal model works. 
- `similarity_demo.ipynb` demonstrating the use of the `DataGrabber` class for easy data acces, and of  our `FipsQuery` class, which is the key tool in the similarity-focused part of the project,
- `causal_insights_demo.ipynb` for an overview of how the `CausalInsight` class can be used to explore the influence of a range of intervention variables thanks to causal inference tools we employed. [WIP]

## Interested? We'd love to hear from you.

[polis](http://polis.basis.ai/) is a research tool under very active development, and we are eager to hear feedback from users in the policymaking and public administration spaces to accelerate its benefit.

If you have feature requests, recommendations for new data sources, tips for how to resolve missing data issues, find bugs in the tool (they certainly exist!), or anything else, please do not hesitate to contact us at polis@basis.ai.

To stay up to date on our latest features, you can subscribe to our [mailing list](https://dashboard.mailerlite.com/forms/102625/110535550672308121/share). In the near-term, we will send out a notice about our upcoming batch of improvements (including performance speedups, support for mobile, and more comprehensive tutorials), as well as an interest form for users who would like to work closely with us on case studies to make the tool most useful in their work.

Lastly, we emphasize that this website is still in beta testing, and hence all predictions should be taken with a grain of salt.

Acknowledgments: polis was built by Basis, a non-profit AI research organization dedicated to creating automated reasoning technology that helps solve society's most intractable problems. To learn more about us, visit https://basis.ai.

