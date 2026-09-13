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


Reading and comprehension of the SDK LLM

	Used Model: Qwen3-0.6B
	Model handle 151k tokens which are word, letters, partial-words, numbers, symbols,..
	Link to Model: https://huggingface.co/Qwen/Qwen3-0.6B/tree/main


	There are 3 important libraries which are used : 
	torch: motor for the tensors
	transformers: come from Hugging Face furnish high level classes to load models and pretrained tokenizers, select automaticly the correct architecture from the name/path of the model.


	Transformers is used to slice the text into tokens, convert them id numeric ID's and eventualy convert them back depending on the tokenizer.

	AutoModelForCausalLM.from_pretrained() loads a model of causal language which predict next token from previous context.
	it is the motor that produces logits.
	takes input_ids and return scores for next tokens.
	PreTrainedTokenizer & Model are there to type objects and organize the proper loading of the model.

	huggingface_hub: The most important function in this SDK, the doc states that it download a precise file & store it in cache on the drive, returns the path to the file.
	
	there are 3 functions in the SDK that loads specific files, 
		def get_path_to_vocab_file(self) -> str:
		def get_path_to_merges_file(self) -> str:
		def get_path_to_tokenizer_file(self) -> str:

	The vocab file is stored here : 
		https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/vocab.json
		The model will work with those id's to produce the logits of the next token.

		A visual example of how the token id's works, convention for GPT/BPE tokenizers :
			Ġ: indicates that what preceed the token is a space.
			the: token without a space before.
			Ġthe: token with a space before.

			The start of a word is often a token with Ġ at the start or a token that appear after a separator.
			middle of a word: fragment without Ġ like tion, ing, self, unction,...
			End of Word: often last fragment of a word sliced in multiple tokens like
			function -> fun + ction or Ġfunction depending on cases. 

		example with call me maybe: 
		["call", "Ġme", "Ġmaybe"]
		Texte brut :
			call me maybe
			│    │  │
			│    │  └── "Ġmaybe" = token avec espace avant
			│    └───── "Ġme"    = token avec espace avant
			└─────────── "call"  = début de phrase / pas forcément marqué par Ġ

	When we put arguments in our call me maybe program, the program will convert this argument into tokens ( known in the vocab.json ) and convert them in the id's given to each token.

	if we use a known word:
		Text :
		function

		Tokenisation :
		["function"]

	if we use a partially known word(word do not exist in full token in the vocab.json): 
		Texte :
		calling

		Tokenisation :
		["call", "ing"]

	if we use a unknown word: 
		Text :
		iujmahzefmkjzah

		Tokenisation possible :
		["Ġi", "u", "j", "m", "a", "h", "z", "e", "f", "m", "k", "j", "z", "a", "h"]

	The pipeline execution is as follow: 
		Texte brut
			↓
		Tokenization
			↓
		Tokens
			↓
		IDs
			↓
		Modèle LLM
			↓
		Logits sur tout le vocabulaire
			↓
		Choix du token suivant
			↓
		Nouveau contexte
			↓
		Boucle

	Token id is the number given to a token in the vocabulary.
		token "ach" -> id 610

	The logits is the representation of the brut score that the model give at a possible token for the next position.
		[... 0.2, -1.4, 3.8, 0.7, ...]
	Logits are the probability after softmax, tels which token it to choose.

	| Token   | Token ID | Logit | Probabilité après softmax | Choisi ? |
	| ------- | -------- | ----- | ------------------------- | -------- |
	| bonjour | 1532     | 2.8   | 0.61                      | Oui      |
	| salut   | 421      | 1.9   | 0.24                      | Non      |
	| hello   | 987      | 0.7   | 0.09                      | Non      |
	| merci   | 2044     | -0.4  | 0.03                      | Non      |
	| .       | 12       | -1.1  | 0.01                      | Non      |

	With input : "Le chat est..."

	| Token possible | Token ID | Logit | Probabilité après softmax | Choisi ? |
	| -------------- | -------- | ----- | ------------------------- | -------- |
	| mignon         | 3012     | 3.4   | 0.58                      | Oui      |
	| assis          | 1187     | 2.1   | 0.19                      | Non      |
	| fatigué        | 4420     | 1.5   | 0.11                      | Non      |
	| noir           | 209      | 0.6   | 0.06                      | Non      |
	| .              | 12       | -0.8  | 0.03                      | Non      |

	The token " mignon " will be chosen because it focuses on argmax/greedy.
	it could be one of the 2next options but with lower probability via Sampling.
	GREEDY / SAMPLING