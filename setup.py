from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="resonance-neural-networks",
    version="0.1.0",
    author="Oluwatosin A. Afolabi",
    author_email="afolabi@genovotech.com",
    description="Frequency-Domain Information Processing with Holographic Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tafolabi009/NEURON_NEW",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "einops>=0.6.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
)
