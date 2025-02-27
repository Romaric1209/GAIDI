install:
  @pip install-e

clean:
  @rm -f*/version.txt
  @rm -f .coverage
  @rm -f*/.ipynb_checkpoints
  @rm RF build
  @rm -Rf */ pycache
  @rm -Rf /.pyc

all: install clean
