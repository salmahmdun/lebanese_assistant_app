# This script is responsible for training the Qwen model using the processed dataset , it :
# - loads the model
# - applies LoRA for efficient fine-tuning 
# - prepares the dataset
# - then trains the model using the Hugging Face Trainer API.
# Finally, it saves the trained adapter for future use.



# ------------------------------------------Lists of imports-----------------------------------------------------
import os  # to handle file paths and directories in a platform-independent way
import sys # to manipulate the Python runtime environment, such as modifying the system path to include the project root directory for importing custom modules
from datasets import load_dataset # to load and handle datasets 
from transformers import ( # to work with the Hugging Face Transformers library for model training and fine-tuning
    Trainer, # to handle the training loop and evaluation of the model
    TrainingArguments, # to specify the training configuration and hyperparameters
    DataCollatorForSeq2Seq, # to create batches of data for sequence-to-sequence tasks, ensuring that the input and target sequences are properly padded and formatted for training
)
from unsloth import FastLanguageModel # to load the Qwen model and apply LoRA for efficient fine-tuning, allowing us to train the model with fewer resources while still achieving good performance on our specific task

# Set up the project root directory in the system path to allow importing custom modules from the training and inference directories, ensuring that we can access the configuration and utility functions needed for training the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.config import * #import all the configuration settings defined in the config.py file, such as model name, training hyperparameters, and system prompt, which will be used throughout the training process to ensure consistency and easy management of the training setup
from training.utils import formatting_prompts_func # import the utility function that formats the prompts for training, which will be applied to the dataset to prepare the input and target sequences in the format required by the Qwen model for effective training and fine-tuning

# ------------------------------------------Lists of imports-----------------------------------------------------




# ----------------------------------------Core Training Logic----------------------------------------------------
# There's 5 main steps in the training process :
# ----------------------------------------------
# Step 1 - Load the Qwen model and tokenizer using the FastLanguageModel class, specifying : 
print("⏳ 1. Loading Qwen...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, #model name 
    max_seq_length=MAX_SEQ_LENGTH, #maximum sequence length
    load_in_4bit=LOAD_IN_4BIT, #load the model in 4-bit precision for efficient training.
)


# Step 2 - Apply LoRA (Low-Rank Adaptation) to the model for efficient fine-tuning:
# which allows us to train the model with fewer resources 
# while still achieving good performance on our specific task 
# by only updating a small number of parameters during training
print("⏳ 2. Applying LoRA...")  
model = FastLanguageModel.get_peft_model( # peft stands for Parameter-Efficient Fine-Tuning,
                            #which is a technique that allows us to fine-tune large language models
                            # like Qwen by only updating a small subset of the model's parameters,
                            # making the training process more efficient and faster 
                            # while still achieving good performance on our specific task
 
                model,
            # Qwen model that we want to fine-tune using LoRA 
 
                r=LORA_R,
            # rank of the low-rank adaptation determines how many parameters will be updated during fine-tuning,
            #with a smaller rank leading to more efficient training but potentially less expressive power,
            #while a larger rank allows for more flexibility but may require more resources and time for training.
 
                target_modules=TARGET_MODULES, 
            # list of module names in the Qwen model that we want to apply LoRA to,
            # which typically includes the projection layers (q_proj, k_proj, v_proj, o_proj) 
            # and other relevant layers (gate_proj, up_proj, down_proj) that are crucial for the model's performance
            # and can benefit from fine-tuning with LoRA
 
                lora_alpha=LORA_ALPHA,
            # scaling factor for the LoRA updates, 
            # which controls the magnitude of the parameter updates during fine-tuning,
 
                lora_dropout=LORA_DROPOUT,
            # dropout rate for the LoRA updates, 
            # can help prevent overfitting during fine-tuning by randomly dropping out some of parameter updates, 
            # adding regularization to the training process and improving the generalization of the model on unseen data
 
                bias="none", 
            # whether to include bias parameters in the LoRA updates,

                use_gradient_checkpointing="unsloth",
            # whether to use gradient checkpointing during training, 
            # which can help reduce memory usage by saving intermediate activations and recomputing them 
            # allowing us to train larger models or use larger batch sizes without running out of memory
)


# Step 3 - Load the training and validation datasets from the specified JSONL files,
# and apply the formatting function to prepare the input and target sequences for training:
print("⏳ 3. Loading Dataset...") 
TRAIN_PATH = os.path.join(BASE_DIR, "datasets", "processed", "train.jsonl")
VAL_PATH = os.path.join(BASE_DIR, "datasets", "processed", "eval.jsonl")

dataset = load_dataset( # load the dataset from the specified JSONL files for training and validation,
    "json",
    data_files={ # specify the training and validation datasets
        "train": TRAIN_PATH,
        "val": VAL_PATH,
    },
)


# Step 4 - Tokenize the dataset using the formatting function 
# to prepare input and target sequences in the format required by Qwen model for effective training and fine-tuning:
print("⏳ 4. Tokenizing...")
dataset = dataset.map( 
                    # map function applies the formatting function to the dataset,
                    #which will tokenize the input and target sequences for training the Qwen model
    lambda x: formatting_prompts_func(x, tokenizer), 
                    # apply the formatting function to each example in the dataset,
    batched=True,  
                    # process the dataset in batches for efficiency, 
                    # allowing us to tokenize multiple examples at once 
                    # and speed up the preparation of the dataset for training
)


# Step 5 - Set up the Trainer with the model, tokenizer, datasets, data collator, and training arguments :
trainer = Trainer( 
                # trainer is a high-level API provided by Hugging Face Transformers library 
                # that handles the training loop, evaluation, and saving of the model during training,
                # making it easier to fine-tune language models like Qwen with custom datasets and configurations
    model=model, 
                # the Qwen model that we want to fine-tune using the Trainer API
    tokenizer=tokenizer,    
                # the tokenizer associated with the Qwen model, 
                # which will be used to prepare the input and target sequences for training 
    train_dataset=dataset["train"],
                # the training dataset that we loaded and tokenized
    eval_dataset=dataset["val"],
                # the validation dataset that we loaded and tokenized 
    data_collator=DataCollatorForSeq2Seq( 
                # data collator will create batches of data for sequence-to-sequence tasks,
        tokenizer=tokenizer, 
                # the tokenizer used by the data collator to properly pad and format the input and target sequences for training
        pad_to_multiple_of=8, 
                # pad the sequences to a multiple of 8 for efficient training on hardware
                #that benefits from such padding (like GPUs),    
        return_tensors="pt", 
                # return the batches as PyTorch tensors, 
                #which is the format expected by the Trainer API for training the model
        padding=True, 
                # pad the sequences to the maximum length in the batch for efficient training,
    ),
    args=TrainingArguments( # specify the training configuration and hyperparameters for the Trainer API,
        output_dir=OUTPUT_DIR, # directory where the trained model and checkpoints will be saved during training
        per_device_train_batch_size=TRAIN_BATCH_SIZE, # batch size for training, 
                            #which determines how many examples will be processed together in each training step,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
                            # number of steps to accumulate gradients before performing an optimization step,
        learning_rate=LEARNING_RATE, # learning rate for the optimizer, 
        num_train_epochs=NUM_EPOCHS, # number of epochs to train the model,
        warmup_steps=10, # number of steps to perform learning rate warmup at the beginning of training,
        weight_decay=0.01, # weight decay for regularization during training,
        logging_steps=5, # number of steps between logging training metrics and progress,
        save_strategy="steps", # strategy for saving model checkpoints during training,
        save_steps=50, # number of steps between saving model checkpoints during training,
        eval_strategy="no", # strategy for evaluating the model during training 
        fp16=True, 
        bf16=False,
        optim="adamw_8bit", # optimizer to use during training,
        report_to="none", # whether to report training metrics to external services (like TensorBoard or Weights & Biases),
        max_steps=MAX_STEPS,# maximum number of training steps to perform,
    ),
)
# ----------------------------------------Core Training Logic----------------------------------------------------


# ----------------------------------------Training Phase---------------------------------------------------------
print("🚀 Training Started...")
trainer.train() 
        # start the training process using the Trainer API 


print("💾 Saving Adapter...") # saving the weights of training 
SAVE_PATH = os.path.join(OUTPUT_DIR, "lora_adapter") 
model.save_pretrained(SAVE_PATH) 
                # save the trained model(specifically the LoRA adapter)to the specified output directory
                #for future use in inference or further fine-tuning, 
                # allowing us to reuse the trained weights without having to retrain the entire model from scratch.
tokenizer.save_pretrained(SAVE_PATH)
                # save the tokenizer associated with the model to the same directory as the trained model,
                # ensuring that we can properly tokenize input data when using the trained model for inference or further fine-tuning in the future.

print("Training finished ✅!")