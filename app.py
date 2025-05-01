from flask import Flask, render_template, request
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import fitz  
import re

app = Flask(__name__)


model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')


def predict_resume(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    

    prediction = torch.argmax(logits, dim=-1).item()  
    return prediction


def extract_info(text):

    name = text.split('\n')[0]
    

    email = re.search(r'\S+@\S+', text)
    email = email.group() if email else "No disponible"
    

    phone = re.search(r'\+?\d[\d -]{8,12}\d', text)
    phone = phone.group() if phone else "No disponible"
    
    skills = re.findall(r'\b\w+\b', text)  
    skills = skills[:5]  
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
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)


            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()


            prediction = predict_resume(text)


            resume_info = extract_info(text)

            resume_data = {
                'filename': file.filename,
                'name': resume_info['name'],
                'email': resume_info['email'],
                'phone': resume_info['phone'],
                'skills': resume_info['skills'],
                'education': resume_info['education'],
                'score': prediction  
            }
            resumes.append(resume_data)

            os.remove(filepath)

    return render_template('index.html', resumes=resumes)

if __name__ == '__main__':
    app.run(debug=True)
