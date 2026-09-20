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
make test         # offline structural test, no model weights needed
make lint         # flake8 + mypy with the required flags
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
run downloads the model weights (about 1.2 GB) into `~/.cache/huggingface`.

## Algorithm explanation

### 1. Reading and validating the inputs

`functions_definition.json` and the prompt file are parsed and validated by
pydantic models (`src/models.py`). Unknown keys are rejected and parameter
types are restricted to `string`, `number` and `boolean`. A malformed input
file produces a clear message and exit code 1, never a traceback.

### 2. Building the vocabulary

`src/vocab.py` loads the `vocab.json` of the model through
`get_path_to_vocab_file()`. That file stores each token in the byte-level
form used by GPT-2 style tokenizers, where a space is written `Ġ`, so every
symbol is decoded back to its real text with the inverse of the byte-to-
unicode table. Tokens that are only a fragment of a multi-byte character
cannot stand on their own and are discarded: 150 195 tokens out of the
151 936 logit slots are usable, and the remaining slots are masked out from
the start.

The correctness of this step was verified before anything else was written:
encoding `What is the sum of 2 and 3?` gives
`[3838, 374, 279, 2629, 315, 220, 17, 323, 220, 18, 30]`, and decoding those
identifiers through the table returns the sentence character for character.

### 3. Choosing the function

The prompt lists every available function with its signature and description,
then the program itself writes the opening of the answer, `{"name": "`. From
there the model generates, but a mask keeps only the tokens that continue at
least one name declared in the definition file. The decision is the model's,
the alphabet is the program's. A name that does not exist cannot be produced.

Because all names in the provided file share the prefix `fn_`, the mask leaves
a single legal token during the first steps, and the model effectively chooses
at the first point where the candidates diverge.

### 4. Generating the arguments

Once the name is known, so is the schema. The program writes each key itself,
because a key is imposed by the schema and is not a decision. Only values are
generated, each under a mask derived from its declared type (`src/state.py`):

* **number** — the ten digits, an optional leading minus sign and at most one
  decimal point. Leading zeros are refused, because `01` is not valid JSON,
  and the value cannot end on a lone decimal point. Qwen tokenises digits one
  by one, which keeps this set tiny.
* **string** — every token containing a quote, a backslash or a control
  character is masked out, so the produced text never needs escaping. A
  leading space is also refused, so a value comes out as `shrek` and not
  ` shrek`.
* **boolean** — only the continuations of `true` and `false`.

### 5. Masking

`src/mask.py` is where constrained decoding happens. The logits returned by
`get_logits_from_input_ids()` are copied into an array filled with negative
infinity, at the positions of the allowed tokens only. Selecting the argmax
of that array is therefore guaranteed to return a legal token. Nothing is
sampled: decoding is deterministic, so two runs on the same input give the
same output.

### 6. Writing the results

The generated text is parsed with `json.loads`, validated against the schema
of the selected function, and numeric values are cast to `float`. Results are
written to `data/output/function_calling_results.json` as an array of objects
holding exactly `prompt`, `name` and `parameters`.

## Design decisions

**The structure never comes from the model.** Braces, quotes, colons, commas
and parameter keys are written by the program. This is what makes 100% valid
JSON a property of the design rather than a measurement.

**The model chooses, the mask restricts.** No heuristic inspects the prompt to
guess a function. The function name is generated token by token by the model;
the mask only removes the tokens that would make the answer invalid.

**Closing a value must be reasoned about in tokens, not in characters.** This
was the single most important correction of the project. A value was first
closed only on the standalone `"` token, and the model almost never chose it:
it prefers merged tokens such as `",` or `"}`. Values therefore ran until
their token budget was exhausted and produced long, drifting regular
expressions. Accepting any token that *starts* with the delimiter fixed the
accuracy and halved the runtime.

**The chat format of the model is reproduced explicitly.** Qwen3 is
instruction-tuned, and it follows a request far better inside the turn layout
it was trained on. The end-of-turn token is also accepted as a signal that a
value is complete.

**One format example, on functions that do not exist in the input file.** It
teaches the shape of an answer without teaching any answer, so it stays valid
for any other definition file.

**Repetition is blocked at token level.** Small models fall into loops such as
`$1 $2 $3 $1 $2 $3`. Any n-gram already produced inside the current value is
forbidden, which stops the drift without constraining the content.

**Every failure has a fallback.** Each generation state can return a safe
value if its mask ends up empty or its token budget runs out: `0` for a
number, the text produced so far for a string, the shortest candidate for a
closed choice. The program cannot crash on a bad generation.

## Performance analysis

Measured on CPU with `float32` (Intel desktop CPU under WSL), on the eleven
provided prompts:

| Metric | Result |
| --- | --- |
| Valid JSON | 11/11 (100%) |
| Correct function name | 11/11 (100%) |
| Fully correct calls | 10/11 (91%) |
| Slowest prompt | 34 s |
| Total generation time | 2 min 28 s |

No prompt comes close to the five minute limit: the slowest one takes 34 s,
and the whole set runs in under two and a half minutes.

Runtime is dominated by the model itself, not by the constraint machinery:
masking costs roughly 12 ms per token, against a few hundred milliseconds for
one forward pass. Because the SDK exposes logits only from a full token
sequence, there is no key-value cache, so cost grows with the length of the
sequence; long values are the expensive ones.

The remaining failure is prompt 10, *Replace all vowels in 'Programming is
fun' with asterisks*: the source string and the regular expression are right,
but the replacement comes out as a run of asterisks instead of a single one.
The model reads the plural literally. This is a semantic limitation of a 0.6B
model, not a structural one.

Accuracy improved in three steps: 8/11 with a plain prompt, then correct
extraction of the source strings once the chat format and repetition blocking
were added, then 10/11 once value closing was fixed at token level.

## Challenges faced

**Loading the model.** With CUDA available, the SDK passes
`device_map="auto"` to transformers, which then requires the optional
`accelerate` package. Since the subject forbids relying on those libraries
directly, the model is instantiated with `device="cpu"`, which leaves
`device_map` unset and adds no dependency.

**Byte-level decoding.** Comparing token text to the target text is impossible
without inverting the byte-to-unicode table first. This was validated on a
known sequence before any masking logic was written, because every later step
depends on it.

**Padding in the logit vector.** The model returns 151 936 logits while
`vocab.json` holds 151 643 entries. Any index without a decodable token must
be masked, otherwise the argmax can land on a token that cannot be rendered.

**Knowing when to stop.** Described above: the closing delimiter has to be
treated as a family of tokens rather than a single character.

**Number termination.** A number slot may only end once at least one digit has
been produced, otherwise the model can close on an empty value.

## Testing strategy

`make test` runs `tests/test_offline.py`, which replaces the model with a mock
returning **pseudo-random logits**. Random scores are the strongest possible
adversary for a structural guarantee: if validity depended on the model, the
test would fail immediately. Over three seeds, all eleven prompts produce a
valid call, with a name that exists, exactly the keys of its schema, and a
Python type matching each declared parameter type. The test needs no model
weights and runs in seconds.

Beyond that, `make lint` enforces flake8 and mypy with the required flags, and
the pydantic models act as a runtime contract on both the input files and the
generated result.

## Example usage

```console
$ make run
uv run python -m src
Loaded 5 function definitions.
Loaded 11 prompts.
Vocabulary ready: 150195 decodable tokens.
  [1] fn_add_numbers {'a': 2.0, 'b': 3.0} (8.7s)
  [3] fn_greet {'name': 'shrek'} (7.5s)
  [5] fn_reverse_string {'s': 'hello'} (6.2s)
  [7] fn_get_square_root {'a': 16.0} (8.4s)
  [11] fn_substitute_string_with_regex {'source_string': 'The cat sat on the mat with another cat', 'regex': 'cat', 'replacement': 'dog'} (23.8s)
Wrote 11 results to data/output/function_calling_results.json.
```

Resulting file:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Resources

* Provided subject and `llm_sdk` package (42 curriculum).
* [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) —
  vocabulary and configuration, in particular the declared vocabulary size of
  151 936.
* [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary)
  — byte-level BPE and the byte-to-unicode table.
* [uv project layout](https://docs.astral.sh/uv/concepts/projects/layout/) —
  how `uv sync` creates and manages `.venv`.
* [pydantic v2 documentation](https://docs.pydantic.dev/latest/) — model
  configuration, validators and private attributes.
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
trust: the byte-level decoding was checked against a known token sequence,
the structural guarantee against random logits, and each prompt-level change
by rerunning the full set and comparing the results.
