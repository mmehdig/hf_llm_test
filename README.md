# hf_llm_test
Easy test huggingface LLM device performance.

# How to Install

```
poetry install
```

# How to run

First get environment:
```
poetry shell
```

Run either of these to test the device performance:

```
python test.py -d cpu
python test.py -d mps
python test.py -d cuda:0
```

