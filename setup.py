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
    "jupyter",
    "plotly.express",
    "graphviz",
    "python-dotenv",
    "google-cloud-storage",
    "dbt-core",
    "dbt-postgres",
    "seaborn",
    "pip-tools",
]

API_REQUIRES = [
    "psycopg2",
    "fastapi[standard]",
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
        "torch",
        "pyro-ppl==1.8.6",
        "chirho @ git+https://github.com/BasisResearch/chirho.git",
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "dill",
        "plotly",
        "matplotlib>=3.8.2",
    ],
    extras_require={
        "test": TEST_REQUIRES,
        "dev": API_REQUIRES + DEV_REQUIRES + TEST_REQUIRES,
        "api": API_REQUIRES,
    },
    python_requires=">=3.10",
    keywords="similarity, causal inference, policymaking, chirho",
    license="Apache 2.0",
)
