import argparse
import random
from transformers import AutoModel, AutoTokenizer

def initialize_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_prompt(source_line, parallel_sources, parallel_targets):
    prompt = ""
    for src, tgt in zip(parallel_sources, parallel_targets):
        prompt += f"{src} = {tgt}\n"
    prompt += f"{source_line} ="
    return prompt

def generate_text(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using the language model
    output = model.generate(input_ids, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model.")
    parser.add_argument("source_file", help="Path to the source file in English.")
    parser.add_argument("parallel_source_file", help="Path to the file containing English lines.")
    parser.add_argument("parallel_target_file", help="Path to the file containing Finnish lines.")
    args = parser.parse_args()

    # Initialize the language model
    model_name = "LumiOpen/Poro-34B"
    model, tokenizer = initialize_model(model_name)

    # Read source, parallel source, and parallel target files
    with open(args.source_file, "r", encoding="utf-8") as source_file, \
            open(args.parallel_source_file, "r", encoding="utf-8") as parallel_source_file, \
            open(args.parallel_target_file, "r", encoding="utf-8") as parallel_target_file:
        source_lines = source_file.readlines()
        parallel_source_lines = parallel_source_file.readlines()
        parallel_target_lines = parallel_target_file.readlines()

    # Iterate over source file lines
    for source_line in source_lines:
        # Pick five random parallel translation pairs
        random_indices = random.sample(range(len(parallel_source_lines)), 5)
        parallel_sources = [parallel_source_lines[i].strip() for i in random_indices]
        parallel_targets = [parallel_target_lines[i].strip() for i in random_indices]

        # Generate prompt
        prompt = generate_prompt(source_line.strip(), parallel_sources, parallel_targets)

        # Generate text using the language model
        generated_text = generate_text(prompt, model, tokenizer)

        # Stop generation at the first line break
        generated_text = generated_text.split("\n")[0]

        print(generated_text)

if __name__ == "__main__":
    main()

