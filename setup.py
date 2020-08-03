import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="C4.5_package_nikeros1974", # Replace with your own username
    version="0.0.1",
    author="Author",
    author_email="author@example.com",
    description="A basic implemenation of C4.5 algorithm",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/nikeros1974/C4.5",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)