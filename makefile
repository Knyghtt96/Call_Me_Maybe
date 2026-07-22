# VARIABLES

VENV = callmemaybe_env
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
#REQUIREMENTS = requirements.txt
SCRIPT = main.py

# Exclude venv from checks, otherwise many errors will raise.
FLAKE8_EXCLUDE = --exclude=$(VENV),__pycache__,.mypy_cache
MYPY_EXCLUDE = --exclude '^(($(VENV))|__pycache__|\.mypy_cache)(/|$$)'

# install virtual environement
venv:
	python3 -m venv $(VENV)

# install dependencies in virtual environement.
install: venv
	$(PIP) install -r $(REQUIREMENTS) 

# run the script under venv.
run: install
	$(PYTHON) $(SCRIPT)

# Run PDB for python
# post mortem debug ( if program crash ) : python3 -m pdb program.py
# insert breakpoint() in your code where you need to mark breaks for pdb
# to know value of a var: p var 
# to quit debugger: q
debug: install
	$(PYTHON) -m pdb $(SCRIPT)
# 	$(PYTHON) -m pdb $(SCRIPT) $(CONFIG)

# Remove every artifact.
clean:
	rm -rf __pycache__
	rm -rf parser/__pycache__
	rm -rf simulation/__pycache__
	rm -rf .mypy_cache
	rm -rf $(VENV)

#check flake8 & mypy / VENV folder excluded.
lint: install
	$(VENV)/bin/flake8 . $(FLAKE8_EXCLUDE)
	$(VENV)/bin/mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs $(MYPY_EXCLUDE)

git:
	git remote remove vog || true
	git remote remove git || true
	git remote add vog git@vogsphere-v2.42belgium.be:vogsphere/intra-uuid-0df5a4d2-0dbd-460c-91fc-c3b3d2309984-7551192-mde-bruy
	git remote add git git@github.com:Knyghtt96/Call_Me_Maybe.git
	git add .
	git commit -m "$(MSG)" || true
	# usage : make git MSG="feat: add parser"	
	git push git main
	git push vog main

pydoc:
	pydocstyle simulation parser main.py

.PHONY: venv install run debug clean lint lint-strict git