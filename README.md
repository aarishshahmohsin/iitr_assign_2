
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

### Technical Terms

1. #### ```"API is to be Used"```
Finetuned Model:


https://github.com/user-attachments/assets/34c503f1-d515-4e99-93d2-43d512aad678


<br>
Original Model:


https://github.com/user-attachments/assets/9cb247a8-59de-46cb-9bdf-6e81b8a6ec45



<br>

2. #### ```"CUDA is an upcoming technology" ```
<br>
Finetuned Model:


https://github.com/user-attachments/assets/b153c581-38f9-4cfb-a1ed-bb34a8343930


<br>
Original Model:



https://github.com/user-attachments/assets/363d90f8-fb20-475a-87f3-60001cd5ee3c


<br>

3. #### ```"OAuth technologies"```
<br>
Finetuned Model:


https://github.com/user-attachments/assets/9ffd77f1-6a56-4d16-8cba-4bc3e53cdb07


<br>
Original Model:


https://github.com/user-attachments/assets/090f97cf-ede3-4778-828a-705920460c85



<br>

### Regional Language (Urdu)

1. #### ```  Roman: "Mera naam kya hai"  "میرا نام کیا ہے" ```
<br>
Finetuned Model:


https://github.com/user-attachments/assets/6d0fc13f-94a1-4130-9614-9a3a1a25676c


<br>
Original Model:


https://github.com/user-attachments/assets/6535f9d7-30b0-4634-96f8-4b985ce35e3c


<br>

2. ####  ```  Roman: "Aaj ek acha din hai" "آج ایک اچھا دن ہے" ```
<br>
Finetuned Model:


https://github.com/user-attachments/assets/24e63847-28b8-4e0c-8f04-296d7556c69a


<br>
Original Model:




https://github.com/user-attachments/assets/73975cfc-0d00-4b17-8230-45a5141dd570





<br>

3. #### ```  Roman: "Chalo kuch baat karte hai"  "چلو کچھ بات کرتے ہیں"```
<br>
Finetuned Model:


https://github.com/user-attachments/assets/421be25d-b5e5-45d9-9d5f-f3b36f797380


<br>
Original Model:



https://github.com/user-attachments/assets/e388d66b-6c32-4bd3-8820-d037704b28f9


<br>

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
