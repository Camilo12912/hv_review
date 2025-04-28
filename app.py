from flask import Flask, render_template, request
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import fitz  # PyMuPDF para procesar PDFs
import re

# Inicializamos la app Flask
app = Flask(__name__)

# Cargar el modelo y el tokenizer de BERT entrenado desde la carpeta 'results'
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')

# Función para hacer predicción con BERT
def predict_resume(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convertir los logits en la predicción
    prediction = torch.argmax(logits, dim=-1).item()  # Cambiar esto según la clasificación que quieras
    return prediction

# Función para extraer información de la hoja de vida usando expresiones regulares
def extract_info(text):
    # Extraer el nombre (simplemente toma la primera línea como ejemplo)
    name = text.split('\n')[0]
    
    # Extraer el email usando regex
    email = re.search(r'\S+@\S+', text)
    email = email.group() if email else "No disponible"
    
    # Extraer el teléfono usando regex (simplificado)
    phone = re.search(r'\+?\d[\d -]{8,12}\d', text)
    phone = phone.group() if phone else "No disponible"
    
    # Extraer habilidades (simplificado, asumir que están después de "Habilidades:")
    skills = re.findall(r'\b\w+\b', text)  # Ajustar según el formato del texto
    skills = skills[:5]  # Tomar solo las primeras 5 habilidades
    
    # Extraer educación (simplificado, asumir que está después de "Educación:")
    education = re.findall(r'([A-Za-z\s]+)\s*[\—\-]\s*([\w\s,]+)', text)
    education = [f"{edu[0]} ({edu[1]})" for edu in education]
    
    return {
        'name': name,
        'email': email,
        'phone': phone,
        'skills': skills,
        'education': education
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    resumes = []

    if request.method == 'POST':
        files = request.files.getlist('files')

        for file in files:
            # Guardar el PDF temporalmente
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            # Extraer texto del PDF usando fitz (PyMuPDF)
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Predicción con el modelo BERT
            prediction = predict_resume(text)

            # Extraer información de la hoja de vida
            resume_info = extract_info(text)

            # Estructurar los datos de la hoja de vida para mostrar
            resume_data = {
                'filename': file.filename,
                'name': resume_info['name'],
                'email': resume_info['email'],
                'phone': resume_info['phone'],
                'skills': resume_info['skills'],
                'education': resume_info['education'],
                'score': prediction  # Aquí mostramos el puntaje o la categoría predicha
            }
            resumes.append(resume_data)

            # Borrar el archivo después de procesarlo
            os.remove(filepath)

    return render_template('index.html', resumes=resumes)

if __name__ == '__main__':
    app.run(debug=True)
