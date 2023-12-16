import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Load the dataset
file_path = 'comments.csv'  # Update with the actual path
data = pd.read_csv(file_path)

# Function to get embeddings from the model
def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Initialize the model and tokenizer (example using GPT-2, replace with your chosen model)
model_name = "gpt2"  # Replace with the model you're using
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to calculate a simple risk score
def calculate_risk_score(embedding):
    # This is a placeholder function. Replace with your actual risk scoring logic.
    # For demonstration, we'll just take the mean of the embedding values.
    risk_score = np.mean(embedding)
    return risk_score

# Calculate Context Scores
context_scores = []
for comment in data['comments']:
    embedding = get_embeddings(comment, model, tokenizer)
    score = calculate_risk_score(embedding)
    # Normalize score to be between 0 and 1
    normalized_score = (score - np.min(score)) / (np.max(score) - np.min(score))
    context_scores.append(normalized_score)

# Add the context scores to the dataframe
data['Context Score'] = context_scores

# Save the updated dataframe
data.to_csv('scored_comments.csv', index=False)
