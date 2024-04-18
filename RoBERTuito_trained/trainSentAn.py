from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os
import shutil

# Configuración de registro
logging.basicConfig(level=logging.INFO)

# Limpiar el directorio de resultados antes de comenzar
output_dir = "./results_robertuito"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Cargar el dataset desde un archivo CSV
input_file = 'inputTrain.csv'
df = pd.read_csv(input_file)

# Mapeo de etiquetas de sentimiento
label_dict = {'NEG': 0, 'NEU': 1, 'POS': 2}
df['label'] = df['sentiment'].map(label_dict)

# Balance de clases mediante sobremuestreo
df_majority = df[df.label == df['label'].value_counts().idxmax()]
df_minority1 = df[df.label == 0]
df_minority2 = df[df.label == 2]

df_minority_upsampled1 = resample(df_minority1, replace=True, n_samples=len(df_majority), random_state=123)
df_minority_upsampled2 = resample(df_minority2, replace=True, n_samples=len(df_majority), random_state=123)

df_balanced = pd.concat([df_majority, df_minority_upsampled1, df_minority_upsampled2])

# Convertir el DataFrame en un Dataset de Hugging Face
dataset = Dataset.from_pandas(df_balanced[['content', 'label']])
split_datasets = dataset.train_test_split(test_size=0.1)

# Cargar robertuito modelo y tokenizador
checkpoint = "pysentimiento/robertuito-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Función para tokenizar ejemplos
def tokenize_function(example):
    return tokenizer(example['content'], truncation=True, padding="max_length", max_length=128)

# Tokenizar los conjuntos de datos
tokenized_datasets = split_datasets.map(tokenize_function, batched=True)

# Collator de datos con padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Función para computar métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=3
)

# Crear el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entrenamiento y evaluación
logging.info("Comenzando el entrenamiento del modelo...")
trainer.train()
logging.info("Evaluando el modelo después del entrenamiento...")
trainer.evaluate()
logging.info("Proceso completado.")
