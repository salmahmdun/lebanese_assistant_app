# This script is responsible for loading the trained Qwen 2.5 7B model with the LoRA adapter 
# for the Lebanese Assistant project , It defines a function load_lebanese_model, that :
# - loads the base Qwen model, 
# - merges the LoRA adapter weights, 
# - and returns the combined model and tokenizer for use in inference and evaluation.


# ------------------------------------------Lists of imports-----------------------------------------------------

import sys # handles system-specific parameters and functions
import os  # for handling file paths and directories
import torch # for tensor operations and model handling
from transformers import AutoModelForCausalLM, AutoTokenizer #loads pre-trained language models and their tokenizers 
                                                             # from the Hugging Face Transformers library
from peft import PeftModel # handles Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA,

# ------------------------------------------Lists of imports-----------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

MODEL_DIR = os.path.join(project_root, "outputs", "lora_adapter") 


# Function loads the base Qwen 2.5 7B model, merges the LoRA adapter weights, 
# and returns the combined model and tokenizer for use in inference and evaluation.
def load_lebanese_model(): 
    print("⏳ Loading Qwen 2.5 7B and Tokenizer on CPU...")

# 1. Load the tokenizer for the Qwen model, which will be used to convert text into input IDs
# that the model can process during inference and evaluation.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
# 2. Load the base Qwen 2.5 7B model in 4-bit precision to save memory and speed up loading,
    base_model = AutoModelForCausalLM.from_pretrained( # load the base Qwen 2.5 7B model from the Hugging Face model hub,
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float32, # specify the data type for the model's weights to be 32-bit floating point,
        device_map="cpu", # load the model on the CPU, which is suitable for inference 
                          # and evaluation without requiring a GPU,
        trust_remote_code=True # allow loading of custom code from the model repository,
    )

# 3. Merge the LoRA adapter weights with the base model to create a combined model 
# that incorporates the fine-tuned parameters from the training process,
    print("⏳ Merging Lebanese Assistant LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR) # merge the LoRA adapter weights from the 
                                                            #specified directory with the base model,
    print("✅ Qwen 2.5 7B Model Ready on CPU!")
    return model, tokenizer # return the combined model and tokenizer for use in inference and evaluation
