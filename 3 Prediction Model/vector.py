import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

with open('abbreviations.json', 'r') as f:
    abbreviations_list = json.load(f)
    
data = pd.read_csv('paintrainingdata.csv')

for abbr in abbreviations_list:
    string_match = r"\b" + abbr + r"\b"
    data['chiefcomplaint'] = data['chiefcomplaint'].str.lower().str.replace(string_match, abbreviations_list[abbr], regex=True)
    
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

data['chiefcomplaint_vector'] = data['chiefcomplaint'].apply(lambda x: model.encode(x))
data.to_csv('paintrainingdata_with_vectors.csv', index=False)

from sklearn.metrics.pairwise import cosine_similarity

inputs = ["abd pain, n/v", "abd pain, dysteria"]

for input in inputs:
    for abbr in abbreviations_list:
        if abbr in input:
            input = input.replace(abbr, abbreviations_list[abbr])
            
print(inputs)