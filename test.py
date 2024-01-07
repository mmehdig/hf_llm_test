import argparse
import torch
from timeit import timeit
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description='Test HF generate')
parser.add_argument("-d", "--device", default="cpu")
parser.add_argument("-p", "--prompt", default="Today is")

args = parser.parse_args()

device = torch.device(args.device)
prompt = args.prompt

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


def gen(promt):
   input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
   generated_token_ids = model.generate(
      inputs=input_ids,
      max_new_tokens=32,
      do_sample=True,
      temperature=1.0,
      top_p=1,
   )[0]

   generated_text = tokenizer.decode(generated_token_ids)
   return generated_text


time_result = timeit(lambda: print(gen(prompt)), number=10)
print(time_result)

