import gradio as gr
import librosa
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

# Model configurations
models = {
    "Urdu Model": {
        "checkpoint": "aarishshahmohsin/final_urdu_t5_finetuned",
        "vocoder": "microsoft/speecht5_hifigan",
        "processor": "aarishshahmohsin/urdu_processor_t5",  
    },
    "Technical Model": {  
        "checkpoint": "aarishshahmohsin/final_technical_terms_t5_finetuned",  
        "vocoder": "microsoft/speecht5_hifigan",
        "processor": "microsoft/speecht5_tts",  # Using same checkpoint for processor
    }
}

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# Initialize all models at startup
print("Loading models...")
loaded_models = {}
for model_name, config in models.items():
    processor = SpeechT5Processor.from_pretrained(config["processor"])
    model = SpeechT5ForTextToSpeech.from_pretrained(config["checkpoint"])
    vocoder = SpeechT5HifiGan.from_pretrained(config["vocoder"])
    
    loaded_models[model_name] = {
        "processor": processor,
        "model": model,
        "vocoder": vocoder
    }
print("Models loaded successfully!")

def predict(text, model_name):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))
    
    model_components = loaded_models[model_name]
    processor = model_components["processor"]
    model = model_components["model"]
    vocoder = model_components["vocoder"]

    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    speech = (speech.numpy() * 32767).astype(np.int16)
    
    return (16000, speech)

# UI Configuration
title = "Multi-Model SpeechT5 Demo"

examples = [
    # Urdu Model Examples
    ["میں نے آج بہت کام کیا۔", "Urdu Model"],
    ["آپ کا دن کیسا گزرا؟", "Urdu Model"],
    
    # Technical Model Examples
    ["JSON response with HTTP status code 200.", "Technical Model"],
    ["Nginx is the best", "Technical Model"],
]

description = """
Select a model and enter text to generate speech. 

1. Regional Language(Urdu)
2. Technical Speech

"""

# Create and launch the interface
gr.Interface(
    fn=predict,
    inputs=[
        gr.Text(label="Input Text"),
        gr.Dropdown(
            choices=list(models.keys()),
            label="Select Model",
            value="Technical Model"
        )
    ],
    outputs=[
        gr.Audio(label="Generated Speech", type="numpy"),
    ],
    title=title,
    description=description,
    examples=examples,  # Add examples to the interface
    cache_examples=True, 
).launch()