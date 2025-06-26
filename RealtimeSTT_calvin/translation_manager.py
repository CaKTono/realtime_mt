from transformers import pipeline
import torch

class TranslationManager:
    def __init__(self, realtime_model_name, full_model_name, device):
        print("Loading translation models...")
        self.device = 0 if device == 'cuda' and torch.cuda.is_available() else -1
        
        print(f"Loading real-time translation model: {realtime_model_name}")
        self.realtime_translator = pipeline(
            'translation',
            model=realtime_model_name,
            device=self.device
        )
        
        print(f"Loading full-sentence translation model: {full_model_name}")
        self.full_translator = pipeline(
            'translation',
            model=full_model_name,
            device=self.device
        )
        print("Translation models loaded.")

    def translate(self, text, src_lang, tgt_lang, model_type='realtime'):
        if model_type == 'realtime':
            translator = self.realtime_translator
        else: # 'full'
            translator = self.full_translator

        # NLLB pipeline requires src_lang and tgt_lang parameters
        result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
        return result[0]['translation_text']