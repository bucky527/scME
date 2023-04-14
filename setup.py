from setuptools import setup, find_packages

setup(
    name='scme',
    version='0.2.0',
    packages=find_packages(),
    author="zbOfZengBio",
    license='MIT',
    author_email='',
    install_requires=["numpy==1.21.6","scipy==1.7.3" ,"pandas==1.3.5" ,
    "scikit-learn==1.0.2" ,"pyro-ppl" ,"matplotlib" ,
    "scanpy==1.9.1" ,"anndata","scvi-tools==0.20.3"]
)