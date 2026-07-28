*This project has been created as part of the 42 curriculum by mde-bruy.*

 # Call_Me_Maybe
***

If UV is not installed : 
sudo snap install astral-uv

This is a tool for project management written in Rust that can handle on it's own the different versions of python, dependencies and the project execution. 

it doesn't depend on python itself, even if you could install it with pip but it create a circular dependency since you need python to install a tool that manage python.

to run the program, the expected cli is : 
	uv run python -m src

it require a pyproject.toml for the dependencies

to create this pyproject : 
	uv init --bare (create a almost empty pyproject.toml file with bare minimum)

then we add dependencies with: 

	uv add numpy pydantic
	uv add --dev flake8 mypy

After this we can use those following commands :
	uv sync
	uv run python -m src --help
	uv run python -m src
	uv run flake8 .
	uv run mypy .

before being able to run cli with custom arguments, we need to specify them trough the argparse module
	import argparse

	parser = argparse.ArgumentParser() 
	=> This is the minimum variable declaration to instantiate a argument parser.

	parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

	To add an "expected argument" we can do : 

	parser.add_argument('x', metavar='x', type=str, help='enter your x')

	those are different behaviours add_argument can handle :

	parser.add_argument('filename')           # positional argument
					parser.add_argument('-c', '--count')      # option that takes a value
					parser.add_argument('-v', '--verbose',
										action='store_true')  # on/off flag

doc for argparse : 
https://docs.python.org/fr/3/library/argparse.html


l'ia propose d'utiliser ceci : 
	from __future__ import annotations

Without this line, python will try to interpret all types at the start.
With this line, python wait and handles them later on.

This is why we see this one often in modern python code with much type annotations.