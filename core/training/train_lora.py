import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, AutoencoderKL
from peft import get_peft_model, LoraConfig, TaskType
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize the text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=77,  # CLIP's max length
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class LoRATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()
        
    def setup_model(self):
        # Load base model
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )
        
        # Get the text encoder and unet for LoRA
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        
        # Configure LoRA for text encoder
        text_encoder_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        
        # Configure LoRA for UNet
        unet_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha'],
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        
        # Apply LoRA to models
        self.text_encoder = get_peft_model(self.text_encoder, text_encoder_config)
        self.unet = get_peft_model(self.unet, unet_config)
        
        # Move models to device
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        
        # Setup tokenizer
        self.tokenizer = self.pipeline.tokenizer
        
    def prepare_dataset(self):
        dataset = TextDataset(self.config['dataset_path'], self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
    
    def train(self):
        logger.info("Starting LoRA training...")
        dataloader = self.prepare_dataset()
        
        # Setup optimizers
        text_encoder_optimizer = torch.optim.AdamW(
            self.text_encoder.parameters(),
            lr=self.config['learning_rate']
        )
        unet_optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config['learning_rate']
        )
        
        self.text_encoder.train()
        self.unet.train()
        
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get text embeddings
                text_embeddings = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0]
                
                # Forward pass through UNet
                # Note: This is a simplified training loop
                # In practice, you'd need to add noise and compute loss properly
                outputs = self.unet(text_embeddings)
                
                # Compute loss (simplified)
                loss = outputs.mean()
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights
                text_encoder_optimizer.step()
                unet_optimizer.step()
                
                # Zero gradients
                text_encoder_optimizer.zero_grad()
                unet_optimizer.zero_grad()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}, Average Loss: {avg_loss:.4f}")
        
        # Save the trained models
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save text encoder LoRA weights
        self.text_encoder.save_pretrained(output_dir / "text_encoder_lora")
        # Save UNet LoRA weights
        self.unet.save_pretrained(output_dir / "unet_lora")
        
        logger.info(f"Models saved to {output_dir}")

def main():
    # Load configuration
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize and run training
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 