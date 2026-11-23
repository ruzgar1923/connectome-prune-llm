from setuptools import setup, find_packages

setup(
    name="connectome-prune-llm",
    version="1.0.0",
    author="Rüzgar Vatansever",
    author_email="rzvatansevertr@gmail.com",
    description="ConnectomePrune-LLM: Bio-Inspired Hub-Preserving Sparsification & Sharded Sculpting for Billion-Scale LLMs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RuzgarVatansever/connectome-prune-llm",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "connectome-prune=connectome_prune_llm.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors>=0.4.0",
        "numpy>=1.20",
        "scipy>=1.8",
        "matplotlib>=3.5",
    ],
)
