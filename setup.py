from setuptools import setup, find_packages

setup(
    name="gpt2-from-scratch",    # Project name
    version="0.1",              # Version
    packages=find_packages(),   # Auto-find all packages
    install_requires=[          # Dependencies
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "tiktoken"
    ],
    python_requires=">=3.8",    # Python version requirement
)