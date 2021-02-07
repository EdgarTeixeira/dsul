from setuptools import setup, find_packages
import eul

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setup(
    name="eul",
    version=eul.__version__,
    author="Edgar Teixeira",
    author_email="edgar.tx@outlook.com",
    description="Utility Library for Data Scientists",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[]
)
