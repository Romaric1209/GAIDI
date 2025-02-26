from setuptools import setup
from setuptools import find_packages

with open('requirement.txt') as f:
    content = f.readlines()

requirements = [x.strip() for x in content]

setup(name='romapp',
      description = 'Redo the whole project with my own work only, project and deployment',
      packages = find_packages(),
      install_requires=requirements)
