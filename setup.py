"""
MedSwarm Package Setup
=======================
Install with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="medswarm",
    version="1.0.0",
    author="MedSwarm Team",
    author_email="team@medswarm.ai",
    description="Multi-Agent Reinforcement Learning for Urban Disaster Triage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medswarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medswarm-data=scripts.prepare_data:main",
            "medswarm-train=scripts.train:main",
            "medswarm-dashboard=scripts.run_dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "medswarm": ["config/*.yaml"],
    },
)
