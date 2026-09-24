*This project has been created as part of the 42 curriculum by mde-bruy.*

# Call Me Maybe

## Description

Call Me Maybe turns a natural language request into a structured function
call. Given a list of callable functions and a list of prompts, it decides
which function answers each prompt and extracts the arguments, then writes
everything as JSON.

The whole project rests on one idea: a 0.6B parameter model cannot be trusted
to emit well-formed JSON, so it is never asked to. The JSON skeleton is
written by the program, and the model is only allowed to fill the holes, one
token at a time, through a mask applied to its raw output scores. Invalid
output is not detected and retried; it is made unreachable.

The model is `Qwen/Qwen3-0.6B`, loaded through the provided `llm_sdk`.

## Instructions

```bash
make install      # uv sync: create .venv and install every dependency
make run          # run on data/input, write data/output
make debug        # same, under pdb
make lint         # flake8 + mypy with the required flags
make lint-strict  # flake8 + mypy --strict
make clean        # remove caches and generated output
```

The program can also be called directly:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional and default to the paths above. The first
run downloads the model weights (about 1.5 GB) into the Hugging Face cache.
On the 42 computers, where the home quota is small, the Makefile detects the
`/sgoinfre/students/<login>` folder and redirects the `uv` and Hugging Face
caches there; elsewhere nothing changes.

## Algorithm explanation

### 1. Reading and validating the inputs

`functions_definition.json` and the prompt file are parsed and validated by
pydantic models (`src/models.py`). Parameter types are restricted to
`string`, `number`, `integer` and `boolean`, the only ones the decoder knows
how to constrain. Unknown keys are ignored so that a richer definition file
still loads. A missing or malformed file produces a clear message and exit
code 1, never a traceback.

### 2. Building the token table

`src/vocab.py` reads `vocab.json` through `get_path_to_vocab_file()` to
learn which identifiers are real tokens, then calls the SDK `decode()` on
each one to obtain its text. Special tokens decode to nothing and tokens
holding a fragment of a multi-byte character decode to U+FFFD; both are
discarded. 150 186 tokens out of the 151 936 logit slots are usable, and
every other slot is masked out from the start.

Every token is classified once: the tokens legal inside a JSON string, the
ones starting with `"`, `,` or `}` (which close a value), and a text-to-id
table used to force the function name. The classification takes about one
second and is never repeated.

### 3. Choosing the function

The prompt lists every available function with its signature and
description, then the program itself writes the opening of the answer,
`{"name": "`. From there the model generates, but a mask keeps only the
tokens that continue at least one name declared in the definition file. The
decision is the model's, the alphabet is the program's. A name that does not
exist cannot be produced.

Because all names in the provided file share the prefix `fn_`, the mask
leaves a single legal token during the first steps, and the model
effectively chooses at the first point where the candidates diverge.

### 4. Generating the arguments

Once the name is known, so is the schema. The program writes each key
itself, because a key is imposed by the schema and is not a decision. Only
values are generated, each under a mask derived from its declared type
(`src/state.py`):

* **number** — the ten digits, an optional leading minus sign and at most
  one decimal point. Leading zeros are refused, because `01` is not valid
  JSON, and the value cannot end on a lone decimal point. Qwen tokenises
  digits one by one, which keeps this set tiny. A value without a decimal
  point becomes a Python `int`, so `2` is written `2` and not `2.0`.
* **integer** — same mask without the decimal point.
* **string** — every token that keeps the JSON string valid: no control
  character, no unescaped quote, and no backslash except in an escaped
  quote `\"`. The produced text is JSON source, so a quote inside a value
  comes out right (`Say "hello" to {name}`). A leading space is refused on
  the first token, so a value comes out as `shrek` and not ` shrek`.
* **boolean** — only the continuations of `true` and `false`.

### 5. Masking

`src/mask.py` is where constrained decoding happens. The logits returned by
`get_logits_from_input_ids()` are copied into an array filled with negative
infinity, at the positions of the allowed tokens only. Selecting the argmax
of that array is therefore guaranteed to return a legal token. Nothing is
sampled: decoding is deterministic, so two runs on the same input give the
same output.

### 6. Writing the results

Each value is typed by its slot (`str`, `int`, `float` or `bool`) and stored
in a pydantic `FunctionCallResult`. Results are written to
`data/output/function_calling_results.json` as an array of objects holding
exactly `prompt`, `name` and `parameters`.

## Design decisions

**The structure never comes from the model.** Braces, quotes, colons, commas
and parameter keys are written by the program. This is what makes 100% valid
JSON a property of the design rather than a measurement.

**The model chooses, the mask restricts.** No heuristic inspects the prompt
to guess a function. The function name is generated token by token by the
model; the mask only removes the tokens that would make the answer invalid.

**Closing a value must be reasoned about in tokens, not in characters.**
This was the single most important correction of the project. A value was
first closed only on the standalone `"` token, and the model almost never
chose it: it prefers merged tokens such as `",` or `"}`. Values therefore
ran until their token budget was exhausted and produced long, drifting
regular expressions. Accepting any token that *starts* with the delimiter
fixed the accuracy and halved the runtime.

**Escapes are restricted on purpose.** JSON allows `\\`, `\n` and friends
inside a string. Allowing them was measured harmful: given the chance, the
model appends regex escapes such as `\s` that are not in the request, and
prompt 9 of the provided set went from right to wrong. Only the escaped
quote is accepted, because a quote cannot appear in a value any other way.

**The chat format of the model is reproduced explicitly.** Qwen3 is
instruction-tuned, and it follows a request far better inside the turn
layout it was trained on.

**One format example, on functions that do not exist in the input file.** It
teaches the shape of an answer without teaching any answer, so it stays
valid for any other definition file.

**Repetition is blocked at token level.** Small models fall into loops such
as `$1 $2 $3 $1 $2 $3`. Any sequence of three tokens already produced inside
the current value cannot be produced again, which stops the drift without
constraining the content.

**A slot always yields a value.** If the token budget runs out, a number
keeps the digits produced so far, a string keeps its text and a choice
completes to the first matching candidate. The output file is complete even
when the model behaves badly.

## Performance analysis

Measured on a 2 vCPU Linux machine with `float32`, on the eleven provided
prompts:

| Metric | Result |
| --- | --- |
| Valid JSON | 11/11 (100%) |
| Correct function name | 11/11 (100%) |
| Fully correct calls | 10/11 (91%) |
| Slowest prompt | 62 s |
| Total generation time | 4 min 23 s |

No prompt comes close to the five minute limit. Model loading adds about one
minute on that machine, and the vocabulary classification about one second.

Runtime is dominated by the model itself, not by the constraint machinery:
masking costs a few milliseconds per token, against a few hundred
milliseconds for one forward pass. Because the SDK exposes logits only from
a full token sequence, cost grows with the length of the sequence; long
string values are the expensive ones (prompt 9, three string arguments,
is the slowest).

The remaining failure is prompt 10, *Replace all vowels in 'Programming is
fun' with asterisks*: the source string and the regular expression are
right, but the replacement comes out as a run of asterisks instead of a
single one. The model reads the plural literally. This is a semantic
limitation of a 0.6B model, not a structural one.

On a second, harder set of eleven prompts (integers, a compound interest
computation, SQL queries, file paths, templates with quotes), nine calls
out of eleven are correct. One failure is a leading `/` dropped from a path
by the model itself (`home` scores higher than `/home` as first token). The
other is a Windows path, `C:\Users\john\config.ini`, written with forward
slashes because the escaped backslash is deliberately refused: the same
rule that costs this prompt is what keeps prompt 9 of the provided set
right.

## Challenges faced

**Loading the model.** With CUDA available, the SDK passes
`device_map="auto"` to transformers, which then requires the optional
`accelerate` package. Since the subject forbids relying on those libraries
directly, the model is instantiated with `device="cpu"`, which leaves
`device_map` unset and adds no dependency.

**Padding in the logit vector.** The model returns 151 936 logits while
`vocab.json` holds 151 643 entries. Any index without a usable token must be
masked, otherwise the argmax can land on a token that cannot be rendered.

**Knowing when to stop.** Described above: the closing delimiter has to be
treated as a family of tokens rather than a single character.

**Integers versus numbers.** A first version cast every numeric value to
`float`, which turned `4` into `4.0` and failed every check expecting an
integer. The type is now decided by the produced text: no decimal point,
no float.

**Quotes inside values.** A value such as `Say "hello" to {name}` cannot be
produced if every quote token is banned. Treating the produced text as JSON
source, and accepting the escaped quote, lets the model write it while the
string stays valid.

## Testing strategy

Validation was done by running the program and measuring:

* the eleven provided prompts, checked one by one against the expected
  call, after every change to the prompt or to the masks;
* a second set of eleven prompts written to cover integers, decimal
  numbers, strings with quotes and paths, to make sure the fixes were not
  tuned to the provided file;
* broken inputs: missing file, invalid JSON, a definition with an unknown
  type, an empty prompt list. Each one must end with a clear message and
  exit code 1, without a traceback;
* `make lint`, which runs flake8 and mypy with the required flags, and
  `make lint-strict`.

The pydantic models act as a runtime contract on both the input files and
the generated result.

## Example usage

```console
$ make run
uv run python -m src
Loaded 5 function definitions.
Loaded 11 prompts.
Vocabulary ready: 150186 usable tokens.
  [1] fn_add_numbers {'a': 2, 'b': 3} (14.8s)
  [3] fn_greet {'name': 'shrek'} (13.4s)
  [5] fn_reverse_string {'s': 'hello'} (11.2s)
  [7] fn_get_square_root {'a': 16} (13.9s)
  [11] fn_substitute_string_with_regex {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'} (39.8s)
Wrote 11 results to data/output/function_calling_results.json.
```

Resulting file:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  }
]
```

## Resources

* Provided subject and `llm_sdk` package (42 curriculum).
* [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) —
  vocabulary and configuration, in particular the declared vocabulary size
  of 151 936.
* [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)
  — subword tokenisation and why a space is glued to the following word.
* [RFC 8259, The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
  — grammar of numbers and strings, escape sequences.
* [uv project layout](https://docs.astral.sh/uv/concepts/projects/layout/) —
  how `uv sync` creates and manages `.venv`.
* [pydantic v2 documentation](https://docs.pydantic.dev/latest/) — model
  configuration and validators.
* [NumPy reference](https://numpy.org/doc/stable/reference/) — masking and
  argmax on the logit vector.

### How AI was used

An AI assistant was used as a working partner throughout the project, in
three ways. First, to explain the concepts the subject relies on:
tokenization, the difference between logits and probabilities, greedy versus
sampled decoding, and what constrained decoding actually does to a
distribution. Second, as a review and debugging partner: it is what led to
identifying the padding gap in the logit vector and, above all, the token
level nature of the value-closing problem, which was the main blocker on
accuracy. Third, for assistance while writing the code and this document.

Every design decision was validated by measurement rather than accepted on
trust: each prompt-level or mask-level change was checked by rerunning the
full set and comparing the results, and the escape restriction was kept
only because the comparison showed it was needed.
