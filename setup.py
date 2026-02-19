from setuptools import setup, find_packages
from pathlib import Path

# version
here = Path(__file__).absolute().parent
version_data = {}
with open(here.joinpath("miguellib", "__init__.py"), "r") as f:
    exec(f.read(), version_data)
version = version_data.get("__version__", "0.0")

install_requires = [
    "numpy>=1.21,<2",
    "pandas>=2,<3",
    "scikit-learn>=1.2,<2",
    "scipy>=1.8,<2",
    "joblib>=1.2,<2",
]

extras_require = {
    "dev": [
        "pytest>=7.0,<9",
        "ruff>=0.6,<1",
        "black>=23.0,<25",
        "ipykernel>=6,<7",
        "jupyter>=1,<2",
    ]
}

setup(
    name="miguellib",
    version=version,
    install_requires=install_requires,
    extras_require=extras_require,
    package_dir={"miguellib": "miguellib"},
    python_requires=">=3.9",
    packages=find_packages(where=".", exclude=["docs", "examples", "tests"]),
)
