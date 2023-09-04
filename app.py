import streamlit as st
from bertviz import head_view
from transformers import AutoTokenizer, AutoModel, utils

st.title('Explainable NLP Demo using BERTviz')


input_text = st.text_input("Text to analyse", key="input_text")
utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer.encode(input_text, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

head_view(attention, tokens)

st.write(head_view(attention, tokens))