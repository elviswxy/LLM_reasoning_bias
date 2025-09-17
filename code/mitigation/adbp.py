import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import copy 
import statistics
import argparse
import re

parser = argparse.ArgumentParser(
    description="Script to mitigate social bias from LLM reasoning using ADBP"
)

# Path to the input JSON file containing data to process
parser.add_argument(
    "jsonfilename", 
    type=str, 
    help="Path to the input JSON file containing questions"
)

# Path where the output (e.g. model responses) will be saved
parser.add_argument(
    "outputfilename", 
    type=str, 
    help="Path to the output CSV file to store the new answers under ADBP"
)

# Identifier of the language model to use (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
parser.add_argument(
    "modelid", 
    type=str, 
    help="The model ID or path to the Reasoning-based LLM"
)

# GPU device ID to run the model on
parser.add_argument(
    "gpu", 
    type=int, 
    help="GPU device ID"
)

# Get input arguments
args = parser.parse_args()
device = torch.device('cuda:'+ str(args.gpu))
df = pd.read_json(args.jsonfilename, lines=True)

# Initialize LLM
model_name = args.modelid
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:'+ str(args.gpu))
tokenizer = AutoTokenizer.from_pretrained(model_name)

def complete_chat0(model, tokenizer, messages, max_new_tokens=2048):
    """
    Generates a model response to a prompt.

    Parameters:
    - model: The language model to use for generating responses.
    - tokenizer: The tokenizer associated with the model.
    - messages: The prompt
    - max_new_tokens: The maximum number of new tokens to generate in the response.

    Returns:
    - The decoded text response from the model.
    """

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False)
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_answer(text):
    """
    Extracts the last occurrence of text enclosed in <answer>...</answer> tags.

    Parameters:
    - text: A string potentially containing <answer>...</answer> tags.

    Returns:
    - The content inside the last <answer>...</answer> tag, or an empty string if none are found.
    """

    match = re.findall(r"<answer>(.*?)</answer>", text)
    if match:
        result = match[-1]
        return result
    else:
        return ""

# Start of ADBP Algorithm
example_ids = [] # Example ID from BBQ dataset
labels = [] # Ground-truth answers
answer_fix = [] # Answers under ADBP
logss = [] # Model's reasoning under various incremental steps
new_log = [] # Model's final reasoning between two choices

marker = "\nProvide explanation based on known facts or Stay unbiased if no facts are known.\n"

# Loops through each question from the input file
for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    # Example ID from BBQ dataset
    example_ids.append(row['example_id'])

    # Get reasoning chain from CoT repsonse
    thoughts = row['cot_response_v1'].split('\n\n')

    # Get ground-truth answer
    labels.append(row['label'])
    
    cands = []
    logs = []

    # Loops through each reasoning step
    for i in range (len(thoughts)):

        prompt = copy.deepcopy(row['prompt'])
        
        # Adds one incremental step to the prompt
        prompt[0]['content'] = prompt[0]['content'] + " ".join(thoughts[:i]) + marker

        # Get model response
        response = complete_chat0(model, tokenizer, prompt)

        # Obtain final answer
        answer = get_answer(response).strip()

        # Record final answer and reasoning at all steps
        cands.append(answer)
        logs.append(response)

    # Record reasoning at all steps
    logss.append(logs)

    # Remove empty final answers
    new_cands = [x for x in cands if x != ""]

    # If no answers available, reuse the original final answer
    if len(new_cands) == 0:
        finalanswer = row['final_answer']

    # If one answer available, return the answer
    elif len(new_cands) == 1:
        finalanswer = new_cands[0]

    # Otherwise, there are two or more answers
    else:

        changes = 0

        # Compute the number of shifts in final answer
        for i in range (len(new_cands)-1):
            if new_cands[i] != new_cands[i+1]:
                changes += 1

        # If there are no shifts, all answers are identical, return the answer
        if changes == 0:
            finalanswer = new_cands[0]

        # Otherwise, there is at least one shift
        else:
            
            # Get the last answer as Candidate 1
            answer1 = cands[-1]

            # Get the majority answer that is not Candidate 1 as Candidate 2
            new_cands = [x for x in new_cands if x != answer1]
            answer2 = statistics.mode(new_cands)

            # Get the reasoning step corresponding to the initial appearance of Candidate 1
            index1 = cands.index(answer1)
            firstreasoning = logs[0].split(marker)[1]
            if index1 == 0:
                reasoning1 = [firstreasoning]
            else:
                reasoning1 = [thoughts[index1-1]]

            # Get the reasoning step corresponding to the initial appearance of Candidate 2
            index2 = cands.index(answer2)
            if index2 == 0:
                reasoning2 = [firstreasoning]
            else:
                reasoning2 = [thoughts[index2-1]]

            # Prompt the model again to choose between the two candidates and their isolated reasoning steps
            prompt = copy.deepcopy(row['prompt'])
            prompt[0]['content'] = prompt[0]['content'] + "\nPreviously you are hesitant between these two choices: " + answer1 + " and " + answer2 + " You picked " + answer1 + " becasue of " + str(reasoning1) + " You picked " + answer2 + " because of " + str(reasoning2) + ". Verify them to see if there is any bias and output the answer."
            
            # Get model response
            response = complete_chat0(model, tokenizer, prompt)

            # Get final answer
            finalanswer = get_answer(response).strip()

            # Record model reasoning
            new_log.append(response)
    
    # Record the final answer for each question under ADBP
    answer_fix.append(finalanswer)

    # If the last example did not have a new reasoning, add an empty one
    if len(new_log) != len(example_ids):
        new_log.append('')

# Construct results as a dataframe
results = pd.DataFrame({
'example_id': example_ids,
'answer': answer_fix,
'label': labels,
'logs': logss,
'new_log': new_log})

# Save the dataframe as a CSV file
results.to_csv(args.outputfilename, index=False)