from setuptools import find_packages, setup

VERSION = "0.1.0"

TEST_REQUIRES = [
    "pytest == 7.4.3",
    "pytest-cov",
    "pytest-xdist",
    "mypy",
    "black",
    "flake8",
    "isort",
    "nbval",
    "nbqa",
    "autoflake",
]

DEV_REQUIRES = [
    "pyro-ppl==1.8.5",
    "torch",
    "plotly.express",
    "scipy",
    "chirho",
    "graphviz",
    "seaborn",
]

setup(
    name="cities",
    version=VERSION,
    description="Similarity and causal inference tools for policymakers.",
    packages=find_packages(include=["cities", "cities.*"]),
    author="Basis",
    url="https://www.basis.ai/",
    project_urls={
        #     "Documentation": "",
        "Source": "https://github.com/BasisResearch/cities",
    },
    install_requires=[
        "jupyter",
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "dill",
        "plotly",
        "matplotlib>=3.8.2",
    ],
    extras_require={"test": TEST_REQUIRES, "dev": DEV_REQUIRES + TEST_REQUIRES},
    python_requires=">=3.10",
    keywords="similarity, causal inference, policymaking, chirho",
    license="Apache 2.0",
)
