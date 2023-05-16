import setuptools

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name='semantic_loss_pytorch',
    version='0.1',
    description='Semantic loss function for PyTorch based on PySDD for KC',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Jacopo Gobbi & Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    url='',
    python_requires='>=3.6',
    license=license,
    packages=setuptools.find_packages(),
    install_requires=["sympy", "torch", "pytest"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2.0",
        "Operating System :: OS Independent",
    ],
)
