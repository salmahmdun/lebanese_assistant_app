#-- This script is responsible for generating responses from the trained Qwen 2.5 7B model with the LoRA adapter for 
# the Lebanese Assistant project. It defines a function generate_response that takes in the model, tokenizer,
# user message, and conversation history, and produces a response in Lebanese Arabic while ensuring that the model 
# does not repeat itself or produce empty responses.
# The function applies a system prompt to guide the model's behavior, formats the conversation history according 
# to the Qwen model's requirements, and uses specific generation parameters to ensure coherent and contextually relevant 
# responses.  


# ------------------------------------------Lists of imports-----------------------------------------------------
import torch # for tensor operations and model handling
# ------------------------------------------Lists of imports-----------------------------------------------------

def generate_response(model, tokenizer, message, history):
    SYSTEM_PROMPT = ( # system prompt that guides model behaviors 
        "أنت مساعد ذكي تتكلم باللهجة اللبنانية فقط وتساعد المستخدم بكل احترام. "
        "أجب دائماً بالعامية اللبنانية ولا تستخدم الفصحى إلا للضرورة القصوى."
    )

    chat_context = [{"role": "system", "content": SYSTEM_PROMPT}] # initialize the chat context with the system prompt
                                                            # that sets the tone and behavior for the model's responses,
    for user_msg, bot_msg in history: # loop through the conversation history and append 
                                    #user and assistant messages to the chat context in the required format 
        chat_context.append({"role": "user", "content": str(user_msg)}) # append the user's message to the chat context
                                                                #with the role "user",
        chat_context.append({"role": "assistant", "content": str(bot_msg)}) # append the assistant's message to the chat
                                                                            #context with the role "assistant",
    chat_context.append({"role": "user", "content": str(message)}) # append the current user message to the chat context
                                                                    #with the role "user",
    
   # 1. Apply the Qwen-specific chat template to the conversation history, tokenize it,
   # and prepare it for input to the model,
    inputs = tokenizer.apply_chat_template(  # apply the Qwen-specific chat template to the conversation history,
        chat_context, # format the conversation history according to the Qwen model's requirements,
        tokenize=True, # tokenize the formatted conversation history using the tokenizer,
        add_generation_prompt=True, # add a generation prompt to the input to guide the model's responses,
        return_tensors="pt").to(model.device) # return the tokenized input as PyTorch tensors
                                            # and move them to the same device as the model,
    
    input_length = inputs.shape[1] # get the length of the input sequence to ensure that
                                    #we only decode the newly generated tokens,

    with torch.no_grad(): # disable gradient calculations for inference to save memory and speed up generation,
        outputs = model.generate( # generate a response from the model using the tokenized input 
                                 #and specific generation parameters,
            input_ids=inputs, # the tokenized input sequence that will be fed into the model for generation,
            max_new_tokens=150, # the maximum number of new tokens to generate in the response,
            use_cache=True,  
            temperature=0.3, # the temperature parameter that controls the randomness of the generated response,
            top_p=0.9, # the nucleus sampling parameter that controls the diversity of the generated response,
            repetition_penalty=1.2, # the repetition penalty parameter that discourages the model
                                    #from repeating the same tokens in the response,
            eos_token_id=tokenizer.eos_token_id, # force the model to stop generating when it produces 
                                                #the end-of-sequence token,
            pad_token_id=tokenizer.pad_token_id, # specify the padding token ID to ensure that the model 
                                                # can handle variable-length inputs and outputs,
        )
    
# 2. Decode the generated tokens into text, skip special tokens, and clean up the response to ensure it is 
# in Lebanese Arabic and does not contain any unwanted artifacts,
    generated_tokens = outputs[0][input_length:] # get only the newly generated tokens by slicing the output tensor
                                                 # from the position of the input length,
    # Decode the generated tokens into text, skip special tokens
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() 
    
# 3. Clean up the response text to ensure it is in Lebanese Arabic and does not contain any unwanted artifacts,
    if "assistant" in response_text: # if the response contains the word "assistant", it may indicate that 
                                    # the model is repeating itself or including
        response_text = response_text.split("assistant")[-1].strip() # take only the part of the response after 
                                                        #the last occurrence of "assistant" to avoid repetition,
    return response_text # return the cleaned response text that is in Lebanese Arabic and does not contain
                        #any unwanted artifacts,
