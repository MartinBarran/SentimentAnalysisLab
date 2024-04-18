from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch

# Especifica la ruta al directorio del checkpoint deseado
model_path = './results_robertuito/checkpoint-245'  # Ajustar este path al checkpoint que se quiera cargar

# Cargar el modelo
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_path)


input_file = 'inputUse.csv'
output_file = 'output.csv'
df = pd.read_csv(input_file)

sentiments = []
probas_list = []

pos_count = 0
neg_count = 0
neu_count = 0
pos_high_prob_count = 0
neg_high_prob_count = 0

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():  # Importante para asegurar que no se calculen gradientes
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_val, prediction = torch.max(probs, dim=1)
    return prediction.item(), probs.squeeze().tolist()

for text in df['content']:
    sentiment, probas = predict_sentiment(text)
    # Convertir la predicción numérica a etiqueta textual
    label_mapping = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    output = label_mapping[sentiment]
    sentiments.append(output)
    probas_list.append({label: prob for label, prob in zip(label_mapping.values(), probas)})



df['sentiment'] = sentiments
df['probas'] = probas_list


df.to_csv(output_file, index=False)
