import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opengen-tf",
    version="0.1.0",
    author="Andrei-Marius Avram",
    author_email="avram.andreimarius@gmail.com",
    description="OpenGen is a toolkit that offers an easy interface to generative models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avramandrei/OpenGen",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
