# RAILChain demo

[Langchain](https://python.langchain.com/en/latest/getting_started/getting_started.html) 💟 [Guardrails](https://shreyar.github.io/guardrails/getting_started/#objective).

An attempt to make Guardrails play more nicely with Langchain.

## Todo

[] Actually utilize the validation.

## Installation

1. On a Python virtual environment, run

```bash
pip install -r requirements.txt
```

2. Copy `example.env` to `.env` and add your OpenAPI API key.

## Running it

```bash
python3 main.py
```

## Tracing

If you need tracing for some reason:

1. Run the following command

```bash
docker-compose up
```

2. Change the `LANGCHAIN_TRACING` key on `.env` to `true`
