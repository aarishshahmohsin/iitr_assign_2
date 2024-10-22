
# Fine-tuning Text-to-Speech Models for English Technical Speech and Regional Languages (Urdu) (IIT Roorkee Assignment II)

## Project Overview

This project focuses on fine-tuning the SpeechT5 model to improve its performance on two tasks:

1. **English Technical Speech**: Pronunciation accuracy for technical terms such as "API," "CUDA," and "OAuth."
2. **Regional Language (Urdu)**: Improving speech synthesis for a regional language, ensuring the naturalness and intelligibility of synthesized speech.

The report details the model selection, dataset preparation, fine-tuning methodology, and results, including optimization techniques like quantization and pruning to enhance inference speed and memory usage.

Link to the report: [Link](./report/aarish_final_report.pdf)

## Contents
- [Samples](#samples)
- [Usage](#usage)

## Samples

Samples can be seen [here](https://github.com/aarishshahmohsin/iitr_assign_2/blob/main/SAMPLES.md)

The samples wav files for Technical Terms can be found under the 'technical_terms/original' and 'technical_terms/finetuned' folders in [technical_terms folder](https://github.com/aarishshahmohsin/iitr_assign_2/tree/main/technical_terms)
The samples wav files for Technical Terms can be found under the 'regional_language_urdu/original' and 'regional_language_urdu/finetuned' folders in [regional_language_urdu folder](https://github.com/aarishshahmohsin/iitr_assign_2/tree/main/regional_language_urdu)

## Usage

For technical Words:
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("aarishshahmohsin/final_technical_terms_t5_finetuned")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


text = "Your text is to be entered here"

inputs = processor(text=text, return_tensors="pt")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)

```


For Regional Language(urdu):
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("aarishshahmohsin/urdu_processor_t5")
model = SpeechT5ForTextToSpeech.from_pretrained("aarishshahmohsin/final_urdu_t5_finetuned")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

text = "Your text is to be entered here"

inputs = processor(text=text, return_tensors="pt")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)

```
