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

# reinstall_package:
# 	@pip uninstall -y taxifare || :
# 	@pip install -e .

# run_preprocess:
# 	python -c 'from taxifare.interface.main import preprocess; preprocess()'

# run_train:
# 	python -c 'from taxifare.interface.main import train; train()'

# run_pred:
# 	python -c 'from taxifare.interface.main import pred; pred()'

# run_evaluate:
# 	python -c 'from taxifare.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate

# run_workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m taxifare.interface.workflow

run_api:
	uvicorn romapp.api.fast:app --reload
