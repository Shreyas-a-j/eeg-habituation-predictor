from setuptools import setup, find_packages

setup(
    name="eeg-habituation-predictor",
    version="0.1.0",
    description="ML-based prediction of EEG habituation to binaural beats in Parkinson's Disease",
    author="Your Name",
    author_email="your.email@college.edu",
    url="https://github.com/yourusername/eeg-habituation-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0",
        "mne>=0.24.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "jupyter>=1.0", "black>=21.0", "flake8>=3.9"],
    },
    entry_points={
        "console_scripts": [
            "ehp-pipeline=ehp.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="parkinson eeg machine-learning neuroscience binaural-beats",
)