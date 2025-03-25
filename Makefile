reinstall_package:
	@pip uninstall -y romapp || :
	@pip install -e .

clean:
  @rm -f*/version.txt
  @rm -f .coverage
  @rm -f*/.ipynb_checkpoints
  @rm RF build
  @rm -Rf */ pycache
  @rm -Rf /.pyc

all: install clean


run_api:
	uvicorn romapi.fast:app --reload
