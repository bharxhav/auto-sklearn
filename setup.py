import os
import sys
from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

if sys.version_info < (3, 0):
    raise ValueError(
        "Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python "
        "3.0 or higher."
        % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

with open(os.path.join(HERE, "requirements.txt")) as fp:
    install_required = [
        r.rstrip()
        for r in fp.readlines()
        if not r.startswith("#") and not r.startswith("git+")
    ]

with open(os.path.join(HERE, "auto-sklearn", "__version__.py")) as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")


with open(os.path.join(HERE, "README.md")) as fh:
    long_description = fh.read()


setup(
    name="auto-sklearn",
    author="Bhargav Kantheti",
    author_email="bhargavkantheti@gmail.com",
    description="Automated Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    packages=find_packages(),
    install_requires=install_required,
    include_package_data=True,
    python_requires=">=3.0",
    url="https://github.com/bharxhav/auto-sklearn",
    license="MIT",
)
