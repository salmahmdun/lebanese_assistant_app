# This file contains utility functions for formatting prompts and preparing the dataset for training the Qwen model.
# The main function, formatting_prompts_func, takes in examples from the dataset and a tokenizer, 
# and applies the Qwen-specific chat template to format the input and target sequences for training.
# This function ensures that the data is in the correct format for the Qwen model to learn effectively
# during fine-tuning with LoRA.

def formatting_prompts_func(examples, tokenizer): # this function takes in examples from the dataset and a tokenizer, 
    #and applies the Qwen-specific chat template to format the input and target sequences for training the Qwen model,
    tokenized = [
        tokenizer.apply_chat_template( # apply_chat_template is a method of the tokenizer 
                                      #that formats the conversation history according to the Qwen model's requirements,
            m, # apply the Qwen-specific chat template to the message m, 
               # which formats the conversation history in a way that is suitable for training the Qwen model,
            tokenize=True, # tokenize the formatted message using the tokenizer, 
                        # which converts the text into input IDs that can be processed by the model during training,
            add_generation_prompt=False, # whether to add a generation prompt to the input, 
                                         # which can help guide the model's responses during training,
            truncation=True,   # whether to truncate the input if it exceeds the maximum sequence length,
            max_length=2048,# the maximum sequence length for the input,
                            # which ensures that the model can process the input without running into memory issues 
                            # during training,
        )
        for m in examples["messages"] 
    ]

    return {
        "input_ids": tokenized, # the tokenized input sequences that will be fed into the model during training,
        "labels": tokenized, # the target sequences that the model will learn to generate during training, 
                        # which are the same as the input sequences in this case since we are using 
                        # a language modeling objective for fine-tuning the Qwen model with LoRA.
    }
