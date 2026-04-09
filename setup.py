from setuptools import setup, find_packages

setup(
    name="medswarm",
    version="1.0.0",
    description="Multi-agent RL for urban disaster triage — CONVOKE 8.0",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "osmnx>=1.9.0",
        "networkx>=3.0",
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
        "plotly>=5.20.0",
        "streamlit>=1.35.0",
        "pyyaml>=6.0",
        "torch>=2.0.0",
        "tqdm>=4.60.0",
    ],
)