from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

def compute_metrics(pred):
    labels = pred.label_ids          # Las etiquetas reales
    predictions = pred.predictions.argmax(axis=-1)  # Las predicciones del modelo

    # Calcular las métricas
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,  # Precisión global
        'precision': precision,  # Precisión
        'recall': recall,  # Recall
        'f1': f1  # F1 Score
    }

# 1. Cargar los datos
df = pd.read_csv('data/hv_data.csv')

# Validar que haya datos
print(f"Datos cargados: {len(df)} filas")

# 2. Preparar dataset
texts = df['Text'].tolist()
labels = df['Label'].tolist()

# 3. Cargar tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizar los textos
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

# Crear dataset compatible
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

# Convertir Dataset a DataFrame de pandas para usar train_test_split
df_dataset = dataset.to_pandas()

# Dividir el dataset en entrenamiento y evaluación (80%/20%)
train_df, eval_df = train_test_split(df_dataset, test_size=0.2)


train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))


training_args = TrainingArguments(
    output_dir='./results',          # Directorio donde se guardan los resultados
    per_device_train_batch_size=8,   # Tamaño del batch para entrenamiento
    per_device_eval_batch_size=8,    # Tamaño del batch para evaluación
    num_train_epochs=3,              # Número de épocas de entrenamiento
    weight_decay=0.01,               # Decaimiento del peso
    logging_dir='./logs',            # Directorio para los logs
)


trainer = Trainer(
    model=model,                          # El modelo a entrenar
    args=training_args,                   # Los parámetros de entrenamiento
    train_dataset=train_dataset,          # Dataset de entrenamiento
    eval_dataset=eval_dataset,            # Dataset de evaluación
    compute_metrics=compute_metrics       # Pasar la función de métricas
)

# 6. Evaluación previa al entrenamiento (opcional)
eval_results = trainer.evaluate()
print("Resultados de la evaluación antes de entrenar:", eval_results)

# 7. Entrenar el modelo
print("🔥 Comenzando entrenamiento...")
trainer.train()

# 8. Evaluación final después del entrenamiento
eval_results_after = trainer.evaluate()
print("Resultados de la evaluación después de entrenar:", eval_results_after)

# 9. Guardar modelo y tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

print("✅ Entrenamiento terminado y modelo guardado en './results'.")
