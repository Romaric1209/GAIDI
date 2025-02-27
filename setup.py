# setup.py
from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
		content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='romapp',
			description="packages for my own version of LeWagon project",
			packages=find_packages(),
			install_requires=requirements)
