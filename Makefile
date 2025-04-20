# Makefile

# Install dependencies
install:
	@pip install -r requirements.txt

# Install in editable mode for local development
reinstall_package:
	@pip uninstall -y romapp || :
	@pip install -e .

# Clean up unnecessary files
clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -rf build/
	@rm -rf */pycache/
	@rm -rf .pytest_cache/

# Install and clean
all: install clean

# Run API with uvicorn
run_api:
	@uvicorn romapi.fast:app --reload

# Run tests with pytest
test:
	@pytest tests/

# Create a distribution package
dist:
	@python setup.py sdist bdist_wheel
