from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name='romapp',
    version='0.2.0',  
    description="My custom version of LeWagon project",
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url="https://github.com/Romaric1209/GAIDI", 
    author="Romaric",
    author_email="romaric.berger@gmail.com",
    packages=find_packages(), 
    install_requires=requirements,
    python_requires='>=3.7,<3.11',
    include_package_data=True,  
    zip_safe=False, 
)
