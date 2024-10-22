import os
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datetime import datetime

def create_samples_directory():
    """Create a timestamped directory for samples"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"speech_samples_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def generate_samples():
    samples_dir = create_samples_directory()
    print(f"\nCreating samples in directory: {samples_dir}")

    print("\nLoading models and processor...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    original_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    sample_texts = [
        'CUDA is an upcoming technology',
        'API is to be used',
        'OAuth technologies',
    ] 

    print("\nGenerating speech samples...")
    for text in sample_texts:
        print(f"\nProcessing: {text[0]}")
        
        inputs = processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = original_model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=vocoder
            )
        
        output_path = os.path.join(samples_dir, f"{text[0]}.wav")
        sf.write(output_path, speech.numpy(), samplerate=16000)
        
        text_path = os.path.join(samples_dir, f"{text[0]}.txt")
        with open(text_path, 'w') as f:
            f.write(text)

    print(f"\nSamples generated successfully in '{samples_dir}'")
    print(f"Generated {len(sample_texts)} samples with corresponding text files")
    print("Check the README.md file for detailed information about each sample")

if __name__ == "__main__":
    generate_samples()