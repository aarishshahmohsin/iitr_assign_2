import random
import torch
import csv
import os
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from datetime import datetime

sentence_templates = [
    "In our latest project, we're implementing {word} for better performance.",
    "The development team needs to learn {word} before the next sprint.",
    "Could you explain how {word} works in this context?",
    "We're experiencing issues with {word} in the production environment.",
    "The documentation for {word} is available in our wiki.",
    "I've been studying {word} to improve our infrastructure.",
    "The new version of {word} includes significant improvements.",
    "Can someone help me configure {word} on my development machine?",
    "According to best practices, {word} should be implemented with careful consideration.",
    "The team lead suggested using {word} for this particular use case.",
    "During the code review, we noticed incorrect usage of {word}.",
    "The latest update to {word} broke our continuous integration pipeline.",
    "We need to upgrade our {word} implementation before the security audit.",
    "The architecture team recommended {word} for the new microservices.",
    "I'm having trouble understanding how {word} fits into our tech stack.",
    "The performance impact of {word} has been significant.",
    "Let's schedule a workshop on {word} for the junior developers.",
    "The client specifically requested {word} in their requirements.",
    "We're seeing increased adoption of {word} in the industry.",
    "The security team raised concerns about our {word} configuration.",
    "Have you checked the latest release notes for {word}?",
    "The debugging process revealed issues with our {word} setup.",
    "According to the metrics, {word} has improved our response times.",
    "We need to migrate from the legacy system to {word}.",
    "The documentation team needs examples of {word} usage.",
    "The DevOps team is working on automating {word} deployment.",
    "Could you review my pull request for the {word} integration?",
    "We're experiencing high latency with our {word} implementation.",
    "The architecture decision record explains why we chose {word}.",
    "The monitoring dashboard shows issues with {word} performance."
]

def generate_speech_and_csv(terms, output_dir="audio_output"):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_data = []
    csv_headers = ["term", "sentence", "audio_path", "embedding_id"]
    
    for term in terms:
        selected_templates = random.sample(sentence_templates, 10)
        
        sentences = [template.format(word=term[1]) for template in selected_templates]
        
        for idx, sentence in enumerate(sentences):
            random_embedding_idx = random.randint(0, len(embeddings_dataset) - 1)
            speaker_embedding = torch.tensor(embeddings_dataset[random_embedding_idx]["xvector"]).unsqueeze(0)
            
            speech = synthesiser(sentence, forward_params={"speaker_embeddings": speaker_embedding})
            
            audio_filename = f"{term[0].lower().replace('.', '_')}_{idx+1}_{timestamp}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            
            sf.write(audio_path, 
                    speech["audio"], 
                    samplerate=speech["sampling_rate"])
            
            csv_data.append({
                "term": term[0],
                "sentence": sentence,
                "audio_path": audio_path,
                "embedding_id": random_embedding_idx
            })
            
            print(f"Generated audio file for {term}: {audio_filename}")
    
    csv_filename = f"tts_sentences_{timestamp}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nGenerated CSV file: {csv_filename}")
    print(f"Total audio files generated: {len(csv_data)}")
    return csv_filename

def load_terms_from_csv(csv_path):
    terms = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            terms.append(list(row.values()))
    return terms

def main():
    input_csv = "technical_terms.csv"  
    terms = load_terms_from_csv(input_csv)
    
    output_csv = generate_speech_and_csv(terms)
    
    print(f"\nProcess completed. Check {output_csv} for the sentence-audio mapping.")

if __name__ == "__main__":
    main()