import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name="birdgame",
    version="0.0.18",
    description="Utilities to help play the Bird Game at crunchdao.com",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/birdgame",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(include=["birdgame", "birdgame.*"]),
    package_data={
        "birdgame": [
            "config/certificates/ca.crt",
        ]
    },
    test_suite='pytest',
    tests_require=['pytest', 'scipy','statsmodels','sktime'],
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "birdgame=birdgame.__main__:main",
        ]
    },
)
