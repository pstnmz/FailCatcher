from setuptools import setup, find_packages

setup(
    name="UQ_Toolbox",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "torchvision>=0.11",
        "numpy>=1.20",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "shap>=0.40",
        "monai>=0.9"
    ],
    python_requires=">=3.8",
)