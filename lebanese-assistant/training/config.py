# This configuration file defines the hyperparameters and settings for training the Qwen model using LoRA fine-tuning.
# It includes model specifications, training parameters, dataset paths, and system prompts
# that will be used during the training process to ensure that the model learns effectively and efficiently .

import os # for handling file paths and directories

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"  
                # the base pre-trained model that we will fine-tune using LoRA, 
                #which is a powerful language model that can be adapted for various tasks
                # through fine-tuning techniques like LoRA.

MAX_SEQ_LENGTH = 2048 
                    # the maximum sequence length for input and output sequences during training,
                    # which determines how much context the model can consider when generating responses.
LOAD_IN_4BIT = True
                    # whether to load the model in 4-bit precision, 
                    # which can reduce memory usage and speed up training while still maintaining good performance,
                    # especially when using LoRA for fine-tuning.

LORA_R = 16         # the rank of the low-rank adaptation matrices in LoRA, 
                    # which controls the capacity of the adapter layers
                    # and how much they can modify the original model's weights during fine-tuning.
LORA_ALPHA = 32     # the scaling factor for the LoRA updates,
LORA_DROPOUT = 0    # the dropout rate for the LoRA adapter layers, 
                    # which can help prevent overfitting during training
                    # by randomly dropping out some of the adapter connections.

TARGET_MODULES = [ # the specific modules in the Qwen model that will be adapted using LoRA during fine-tuning,
    "q_proj",  # the query projection layer in the attention mechanism of the model,
    "k_proj",  # the key projection layer in the attention mechanism of the model,
    "v_proj",  # the value projection layer in the attention mechanism of the model,
    "o_proj",  # the output projection layer in the attention mechanism of the model,
    "gate_proj", # the gating projection layer that controls the flow of information in the model,
    "up_proj",  # the up projection layer that transforms the input features 
                #before they are processed by the attention mechanism,
    "down_proj", # the down projection layer that transforms the output features
                # after they are processed by the attention mechanism.
]

TRAIN_BATCH_SIZE = 2  # batch size for training, which determines how many samples will be processed together in each training step,
                        # and can affect the stability and speed of training, especially when using LoRA for fine
GRAD_ACCUMULATION = 4 # the number of gradient accumulation steps, 
                        # which allows us to effectively increase the batch size without increasing memory usage,

LEARNING_RATE = 2e-4 # the learning rate ,controls how much the model's weights are updated during each training step,

NUM_EPOCHS = 1 # the number of times the entire training dataset will be passed through the model during training,
MAX_STEPS = 200 # the maximum number of training steps to perform, which can be used to
                # limit the training time and prevent overfitting, 


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SYSTEM_PROMPT = ( # the system prompt that will be used during training to guide the model's behavior and responses,
    "أنت مساعد ذكي لبناني. "
    "بتفهم وبتتكلم بالعامية اللبنانية "
    "وباللغة الإنجليزية بطلاقة، "
    "وأسلوبك دايماً قريب للقلب ومساعد."
)
