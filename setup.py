import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="birdgame",
    version="0.0.5",
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
    packages=["birdgame",
              "birdgame.stats",
              "birdgame.animation",
              "birdgame.datasources",
              "birdgame.examples",
              "birdgame.examples.selfcontained",
              "birdgame.examples.derived",
              "birdgame.config",
              "birdgame.trackers",
              "birdgame.model_benchmark",
              "birdgame.visualization"
              ],
    test_suite='pytest',
    tests_require=['pytest','scipy','density','densitypdf','redis','orjson'],
    include_package_data=True,
    install_requires=['pydantic'],
    entry_points={
        "console_scripts": [
            "birdgame=birdgame.__main__:main",
        ]
    },
)
