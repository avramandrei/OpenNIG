import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opengen-tf",
    version="0.1.5",
    author="Andrei-Marius Avram",
    author_email="avram.andreimarius@gmail.com",
    description="OpenGen is a toolkit that offers an easy interface to generative models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/avramandrei/OpenGen",
    keywords="generative-models tensorflow generative-adversarial-network gan variational-autoencoder vae",
    license="MIT",
    install_requires=[
	"tensorflow>=2.0.0",
	"Pillow"
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
	"Intended Audience :: Developers",
	"Intended Audience :: Information Technology",
	"Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
	"Topic :: Scientific/Engineering :: Visualization"
    ],
)
