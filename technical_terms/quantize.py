from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import torch.nn.utils.prune as prune
import soundfile as sf
from datasets import load_dataset

def prune_model(model, amount=0.15):
    """
    Apply L1 unstructured pruning only to specific layers
    amount: percentage of weights to prune (0.15 = 15%)
    """
    for name, module in model.named_modules():
        # Only prune specific layers to maintain quality
        if isinstance(module, torch.nn.Linear) and ('encoder.feed_forward' in name):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

def apply_selective_quantization(model):
    """
    Apply very light quantization only to specific layers
    that don't heavily impact audio quality
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear}, 
        dtype=torch.float16,  
        inplace=True
    )
    
    return quantized_model

model = SpeechT5ForTextToSpeech.from_pretrained("aarishshahmohsin/final_technical_terms_t5_finetuned")

print("Applying light pruning...")
prune_model(model)

print("Applying minimal quantization...")
model = apply_selective_quantization(model)

# torch.save(model, "quantized_technical.pt")