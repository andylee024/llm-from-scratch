from setuptools import setup, find_packages

setup(
    name="minigpt",    # Project name
    version="0.1",              # Version
    packages=find_packages(),   # Auto-find all packages
    install_requires=[          # Dependencies
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "tiktoken",
        "wandb>=0.15.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",    # Python version requirement
)