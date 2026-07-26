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