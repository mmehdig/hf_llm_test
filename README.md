# hf_llm_test
An easy test for huggingface LLM performance on a single device.

# How to Install

```
poetry install
```

# How to run

First, get to the right environment:
```
poetry shell
```

Run either of these to test the device's performance:

```
python test.py -d cpu
python test.py -d mps
python test.py -d cuda:0
```

