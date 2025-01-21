import argparse
import random
import torch
import sys
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self,tokenizer, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer
        self.batch_stop_list = []

    def __call__(self, batch_input_ids: torch.LongTensor, scores: torch.FloatTensor):
        source_id = 0
        for input_ids in batch_input_ids:
            if source_id in self.batch_stop_list:
                source_id += 1
                continue
            last_token = input_ids[-1]
            for stop in self.stops:
                if self.tokenizer.decode(stop) in self.tokenizer.decode(last_token):
                    #print("stop:",self.tokenizer.decode(stop))
                    #print("last token:",self.tokenizer.decode(last_token))
                    self.batch_stop_list.append(source_id)
            source_id += 1
        if len(self.batch_stop_list) == len(batch_input_ids):
            return True
        else:
            return False

def initialize_model(model_name, revision):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_equals_prompt(source_line, parallel_sources, parallel_targets):
    prompt = ""
    for src, tgt in zip(parallel_sources, parallel_targets):
        prompt += f"{src} = {tgt}\n"
    prompt += f"{source_line} ="
    return prompt

def generate_prompt(source_line, parallel_sources, parallel_targets):
    prompt = "Translate from English to Finnish\n\n"
    for src, tgt in zip(parallel_sources, parallel_targets):
        prompt += f"<|user|>{src}\n<|assistant|>{tgt}\n\n"
    prompt += f"<|user|>{source_line}\n<|assistant|>"
    return prompt

def generate_text_batch(prompts, model, tokenizer, stopping_criteria):
    #print(prompts)
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    input_ids = input_ids.to('cuda') 
    # Generate text using the language model
    output = model.generate(input_ids, max_new_tokens=200, stopping_criteria=stopping_criteria) 

    # Decode and return the generated text
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=False)
    return generated_texts

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("source_file", help="Path to the source file in English.")
    parser.add_argument("parallel_source_file", help="Path to the file containing English lines.")
    parser.add_argument("parallel_target_file", help="Path to the file containing Finnish lines.")
    parser.add_argument("prompt_type", help="Whether to construct prompt with equals sign or the special symbols, values: equals, symbols.")
    parser.add_argument("model1_name", help="Name of model 1")
    parser.add_argument("model1_revision", help="Revision of model 1")
    parser.add_argument("model2_name", help="Name of model 2")
    parser.add_argument("model2_revision", help="Revision of model 2")
    args = parser.parse_args()

    # Initialize the language model
    #model_name = "LumiOpen/Poro-34B"
    model_name = args.model_name
    revision = args.revision
    model, tokenizer = initialize_model(model_name,revision)

    # Stop at line break
    stop_words = ["\n"]
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    
    # Read source, parallel source, and parallel target files
    with open(args.source_file, "r", encoding="utf-8") as source_file, \
            open(args.parallel_source_file, "r", encoding="utf-8") as parallel_source_file, \
            open(args.parallel_target_file, "r", encoding="utf-8") as parallel_target_file:
        source_lines = source_file.readlines()
        parallel_source_lines = parallel_source_file.readlines()
        parallel_target_lines = parallel_target_file.readlines()

    # Batch size for prompts
    batch_size = 10
    context_size = 8
    # Iterate over source file lines in batches
    for i in range(0, len(source_lines), batch_size):
        batch_source_lines = source_lines[i:i + batch_size]

        prompts = []
        for source_line in batch_source_lines:
            # Pick random parallel translation pairs
            random_indices = random.sample(range(len(parallel_source_lines)), context_size)
            line_parallel_sources = [parallel_source_lines[i].strip() for i in random_indices]
            line_parallel_targets = [parallel_target_lines[i].strip() for i in random_indices]

            # Generate prompt
            if args.prompt_type == "equals":
                prompt = generate_equals_prompt(source_line.strip(), line_parallel_sources, line_parallel_targets)
                prompts.append(prompt)
            elif args.prompt_type == "symbols":
                prompt = generate_prompt(source_line.strip(), line_parallel_sources, line_parallel_targets)
                prompts.append(prompt)
            sys.stderr.write(f"prompt for line {source_line}:\n{prompt}\n")

        # Generate text for the batch of prompts
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stop_words_ids)])
        generated_texts = generate_text_batch(prompts, model, tokenizer, stopping_criteria)

        # Print the generated text for each prompt
        for prompt,generated_text in zip(prompts,generated_texts):
            if args.prompt_type == "equals":
                #remove the padding to make generated text length match prompt length
                cleaned = generated_text.replace("<pad>","")
                sys.stderr.write(f"cleaned generated text:\n{cleaned}\n") 
                print(cleaned[len(prompt):].split("\n")[0])
            elif args.prompt_type == "symbols":
                translation = generated_text.split("<|assistant|>")[context_size+1].split("\n")[0]
                print(translation)

if __name__ == "__main__":
    main()

