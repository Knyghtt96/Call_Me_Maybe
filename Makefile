# VARIABLES

SRC = src
NAME = CALL_ME_MAYBE

USER = mde-bruy
SGOINFRE = /sgoinfre/students/$(USER)

export UV_PROJECT_ENVIRONMENT := $(SGOINFRE)/.venv
export UV_CACHE_DIR := $(SGOINFRE)/.cache/uv
export HF_HOME := $(SGOINFRE)/.cache/huggingface

# Installe les dépendances dans l'environnement virtuel.
install:
	uv sync

# Lance le programme dans l'environnement virtuel.
run: install
	uv run python -m $(SRC)

# Lance le programme sous le débogueur pdb.
debug: install
	uv run python -m pdb -m $(SRC)

# Supprime les caches et les fichiers générés.
clean:
	rm -rf __pycache__ $(SRC)/__pycache__ .mypy_cache data/output

fclean: clean
	rm -rf .venv

# Vérifie flake8 et mypy avec les options imposées par le sujet.
lint: install
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	uv run flake8 .
	uv run mypy . --strict

# Push vers GitHub et Vogsphere.
# Usage : make git MSG="feat: add parser"
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
		--exclude='*.pyc' \
		--exclude='en.subject.pdf' \
		--exclude='$(NAME).tar.gz' \
		--exclude='$(NAME).zip' \
		-czvf /tmp/$(NAME).tar.gz .
	mv /tmp/$(NAME).tar.gz .

zip: clean
	zip -r /tmp/$(NAME).zip . \
		-x ".venv/*" \
		-x ".git/*" \
		-x "*__pycache__/*" \
		-x ".mypy_cache/*" \
		-x "*.pyc" \
		-x "en.subject.pdf" \
		-x "$(NAME).tar.gz" \
		-x "$(NAME).zip"
	mv /tmp/$(NAME).zip .

.PHONY: install run debug clean fclean lint lint-strict git tar zip
