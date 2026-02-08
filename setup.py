from setuptools import setup, find_packages

setup(
    name="lob-contagion-regimes",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "polars>=0.19",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "arch>=6.1",
        "statsmodels>=0.14",
        "scikit-learn>=1.3",
        "hmmlearn>=0.3",
    ],
    python_requires=">=3.9",
)
