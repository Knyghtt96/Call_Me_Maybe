# VARIABLES

SRC = src
NAME = CALL_ME_MAYBE

# install dependencies in virtual environement.
install:
	uv sync

# run the script under venv.
run: install
	uv run python -m $(SRC)

# Run PDB for python
# post mortem debug ( if program crash ) : python3 -m pdb program.py
# insert breakpoint() in your code where you need to mark breaks for pdb
# to know value of a var: p var 
# to quit debugger: q
debug: install
	uv run python -m pdb -m $(SRC)

# Remove every artifact.
clean:
	rm -rf __pycache__ $(SRC)/__pycache__ .mypy_cache .pytest_cache data/output

fclean: clean
	rm -rf .venv

# Offline structural test: no model weights needed.
test: install
	uv run python tests/test_offline.py

#check flake8 & mypy / VENV folder excluded.
lint: install
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run flake8 .
	uv run mypy . --strict

#Usefull trick to push on github / vogsphere in one command
# usage : make git MSG="feat: add parser"
git:
	git remote remove vog || true
	git remote remove git || true
	git remote add vog git@vogsphere-v2.42belgium.be:vogsphere/intra-uuid-0df5a4d2-0dbd-460c-91fc-c3b3d2309984-7551192-mde-bruy
	git remote add git git@github.com:Knyghtt96/Call_Me_Maybe.git
	git add .
	git commit -m "$(MSG)" || true	
	git push git main
	git push vog main

tar: clean
	tar --exclude='.venv' \
	    --exclude='.git' \
	    --exclude='__pycache__' \
	    --exclude='.mypy_cache' \
	    --exclude='.pytest_cache' \
	    --exclude='*.pyc' \
	    --exclude='en.subject.pdf' \
	    -czvf /tmp/$(NAME).tar.gz .
	mv /tmp/$(NAME).tar.gz .

zip: clean
	zip -r /tmp/$(NAME).zip . \
		-x ".venv/*" \
		-x ".git/*" \
		-x "*__pycache__/*" \
		-x ".mypy_cache/*" \
		-x ".pytest_cache/*" \
		-x "*.pyc" \
		-x "en.subject.pdf" \
		-x "$(NAME).tar.gz" \
		-x "$(NAME).zip"
	mv /tmp/$(NAME).zip .


.PHONY: install run debug clean fclean lint lint-strict git zip tar