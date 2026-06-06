# This script performs a comprehensive evaluation of the Lebanese Assistant model using multiple metrics,
# including BLEU, ROUGE, BERTScore, METEOR, and a custom semantic similarity score based on sentence embeddings.
# It loads the evaluation dataset, generates responses using the model, and computes the scores for each metric,
# while ensuring that the model's responses are in Lebanese Arabic and do not contain unwanted artifacts.


#------------------------------------------Lists of imports-----------------------------------------------------
import sys 
import os
import json
import pandas as pd
import torch # for tensor operations and model handling
import nltk # for natural language processing tasks, specifically for computing the METEOR score
from evaluate import load # for loading evaluation metrics from the Hugging Face Evaluate library
from sentence_transformers import SentenceTransformer, util # for computing semantic similarity using 
                                        # sentence embeddings from the Sentence Transformers library
from inference.load_model import load_lebanese_model # for loading the trained Qwen 2.5 7B model with the LoRA adapter
                                                    # for the Lebanese Assistant 
from inference.generate import generate_response # for generating responses from the model based on 
                                                # user messages and conversation history
# ----------------------------------------Lists of imports------------------------------------------------------

# download necessary NLTK resources for METEOR score computation,
# ensuring that the evaluation can be performed without errors related to missing resources
nltk.download('wordnet', quiet=True) # for METEOR score, which relies on WordNet for synonym matching 
                                    # and other linguistic features
nltk.download('punkt', quiet=True) # for tokenizing sentences and words, which is necessary for computing 
                                    # various evaluation metrics that require tokenized input
nltk.download('omw-1.4', quiet=True) # for accessing the Open Multilingual WordNet,
                                    # which can enhance the METEOR score by providing


# additional linguistic resources for languages other than English, such as Arabic in this case.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



print("⏳ Loading All Evaluation Metrics (BLEU, ROUGE, BERTScore, METEOR)...")
bleu = load("bleu") # for computing the BLEU score, which measures the n-gram overlap between the generated responses 
                    # and reference responses,
rouge = load("rouge") # for computing the ROUGE score, which measures the longest common subsequence and n-gram overlap
                    # between the generated responses and reference responses, particularly useful 
                    # for summarization tasks, but also applicable for dialogue evaluation 
bertscore = load("bertscore") # for computing the BERTScore, which uses contextual embeddings from BERT to evaluate 
                            # the semantic similarity between the generated responses and reference responses, 
                            # providing a more nuanced evaluation
meteor = load("meteor") # for computing the METEOR score, which evaluates the generated responses based on 
                        #unigram matching, synonym matching, and other linguistic features, providing 
                        # a more comprehensive evaluation

# 1. Load the evaluation dataset, which contains user messages and reference responses,
# that will be used to benchmark the model's performance across multiple metrics,
EVAL_FILE = os.path.join(project_root, "datasets", "processed", "eval.jsonl")

evaluation_data = [] # empty list to store the evaluation samples loaded from the JSONL file, 
                    # where each sample contains a conversation history
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    for line in f: 
        if line.strip(): # ensure that we only process non-empty lines in the JSONL file to avoid errors during loading,
            evaluation_data.append(json.loads(line)) # load each line as a JSON object and append it to 
                                                # the evaluation_data list, which will be used for generating responses
                                                # and computing evaluation metrics

# 2. Initialize the model and tokenizer for evaluation, ensuring that we have the necessary components to generate
# responses from the model based on the user messages and conversation history in the evaluation dataset, 
print("⏳ Initializing Model and Tokenizer for Evaluation...")
model, tokenizer = load_lebanese_model() 

predictions = [] # empty list to store the generated responses from the model for each sample in the evaluation dataset,
references = [] # empty list to store the reference responses from the evaluation dataset, which will 
                #be used for computing the evaluation metrics by comparing them to the generated responses from the model 
rows = [] # empty list to store the detailed results for each sample, including the question, reference response,
            # generated response, and semantic similarity score, which will be saved to a CSV file for further analysis
            # and reporting 


print(f"🚀 Starting Benchmarking Pipeline for {len(evaluation_data)} samples...")
# 3. Loop through each sample in the evaluation dataset, generate a response from the model based on the user message
# and conversation history, and store the generated response, reference response, and other relevant information 
# for computing evaluation metrics and generating a comprehensive report at the end of the evaluation process. 
# The loop also includes progress updates to keep track of the evaluation status. 

for i, sample in enumerate(evaluation_data, 1): # loop through each sample in the evaluation dataset with
                                                # an index starting from 1 for better progress tracking,
    messages_list = sample["messages"] # extract the list of messages from the sample, which contains 
                                    # the conversation history between the user and the assistant,
    
    question = "" #  to store the user message extracted from the conversation history,
    reference = "" # to store the reference response extracted from the conversation history,
tuples_history = [] # to store the conversation history as (user, assistant) tuples,
                    # which can be passed to the generation function if multi-turn
                    # context is needed. In this evaluation pipeline it remains empty
                    # because each sample is evaluated independently.



predictions.append(response_text) # store the generated response from the model
                                  # for later evaluation against the reference response

references.append(reference) # store the ground-truth reference response extracted
                             # from the evaluation dataset for metric computation


clean_predictions = [p if p.strip() != "" else "no response" for p in predictions]
# replace empty generated responses with a placeholder string to prevent
# evaluation metrics from failing or producing invalid results

clean_references = [r if r.strip() != "" else "no reference" for r in references]
# replace empty reference responses with a placeholder string to ensure
# consistency during metric computation


bleu_result = bleu.compute(predictions=clean_predictions, references=[[r] for r in clean_references])
# compute BLEU score by comparing generated responses against reference responses,
# using a nested list format because BLEU supports multiple references per prediction


rouge_result = rouge.compute(predictions=clean_predictions, references=clean_references)
# compute ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, etc.) to evaluate
# lexical overlap and longest common subsequence similarity


bert_result = bertscore.compute(predictions=clean_predictions, references=clean_references, lang="ar")
# compute BERTScore using contextual embeddings for Arabic text,
# providing a semantic similarity evaluation beyond exact word matching

bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
# calculate the average BERTScore F1 value across all evaluated samples



meteor_result = meteor.compute(predictions=clean_predictions, references=clean_references)
# compute METEOR score, which considers exact matches, stem matches,
# synonym matches, and word order information

meteor_score = meteor_result["meteor"]
# extract the final METEOR score from the evaluation output


similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
# load a lightweight sentence embedding model for measuring semantic similarity
# between generated and reference responses

embeddings1 = similarity_model.encode(clean_predictions, convert_to_tensor=True)
# convert generated responses into dense vector representations

embeddings2 = similarity_model.encode(clean_references, convert_to_tensor=True)
# convert reference responses into dense vector representations

cosine_scores = util.cos_sim(embeddings1, embeddings2)
# compute pairwise cosine similarity matrix between prediction embeddings
# and reference embeddings

semantic_similarity_score = torch.diag(cosine_scores).mean().item()
# extract matching prediction-reference similarities from the diagonal
# and compute their average as the overall semantic similarity score



metrics_data = {
    "Evaluation Metric": ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "Semantic Similarity"],
    "Score Type": ["Lexical (N-gram)", "Lexical (LCS)", "Semantic (Token)", "Semantic/Lexical", "Pure Semantic"],
    "Model Score": [...]
}
# organize all evaluation metrics into a structured dictionary
# for easy visualization and reporting



updated_rows = []
# create a new list containing detailed per-sample evaluation results,
# including the semantic similarity score for each prediction-reference pair

row_sim = cosine_scores[i][i].item()
# extract the cosine similarity between the current prediction
# and its corresponding reference response



df_report = pd.DataFrame(updated_rows)
# convert detailed evaluation results into a DataFrame for analysis and export

output_path = os.path.join(project_root, "comprehensive_evaluation_results.csv")
# define the destination path where the detailed evaluation report
# will be saved as a CSV file

df_report.to_csv(output_path, index=False, encoding="utf-8-sig")
# save the report with UTF-8 encoding to preserve Arabic text correctly

print(f"💾 Comprehensive Local CSV report saved at: {output_path}")
# display the final location of the generated evaluation report