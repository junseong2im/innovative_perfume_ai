from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import wandb
import json
import os
import logging
from datetime import datetime
import numpy as np
from datasets import Dataset as HFDataset
from ..core.config import settings

logger = logging.getLogger(__name__)


class FragranceDataset(Dataset):
    """향수 레시피 데이터셋"""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format the training example
        if "input" in item and "output" in item:
            text = f"### 입력:\n{item['input']}\n\n### 출력:\n{item['output']}<|endoftext|>"
        elif "prompt" in item and "completion" in item:
            text = f"### 요청:\n{item['prompt']}\n\n### 레시피:\n{item['completion']}<|endoftext|>"
        else:
            text = str(item.get("text", ""))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": encoding["input_ids"].flatten()
        }


class PEFTTrainer:
    """PEFT (Parameter-Efficient Fine-Tuning) 트레이너"""
    
    def __init__(
        self,
        model_name: str,
        use_quantization: bool = True,
        use_wandb: bool = True
    ):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self._load_base_model()
        
        # Initialize wandb if enabled
        if self.use_wandb:
            self._init_wandb()
    
    def _load_base_model(self) -> None:
        """기본 모델과 토크나이저 로드"""
        try:
            # Configure quantization
            if self.use_quantization and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                bnb_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="right"  # Important for training
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Prepare model for k-bit training if using quantization
            if self.use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
            
            logger.info(f"Loaded base model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def _init_wandb(self) -> None:
        """Weights & Biases 초기화"""
        try:
            wandb.init(
                project=settings.wandb_project,
                name=f"fragrance-peft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model_name": self.model_name,
                    "lora_r": settings.lora_r,
                    "lora_alpha": settings.lora_alpha,
                    "lora_dropout": settings.lora_dropout,
                    "learning_rate": settings.learning_rate,
                    "batch_size": settings.batch_size,
                    "max_seq_length": settings.max_seq_length,
                    "use_quantization": self.use_quantization
                }
            )
            logger.info("Initialized wandb logging")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def setup_lora(
        self,
        target_modules: Optional[List[str]] = None,
        lora_r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None
    ) -> None:
        """LoRA 설정 및 적용"""
        try:
            # Use default values if not provided
            lora_r = lora_r or settings.lora_r
            lora_alpha = lora_alpha or settings.lora_alpha
            lora_dropout = lora_dropout or settings.lora_dropout
            
            # Default target modules for common architectures
            if target_modules is None:
                if "llama" in self.model_name.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif "mistral" in self.model_name.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                else:
                    target_modules = ["q_proj", "v_proj"]
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"LoRA applied successfully")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def prepare_datasets(
        self,
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
        test_split: float = 0.1
    ) -> Tuple[FragranceDataset, Optional[FragranceDataset]]:
        """데이터셋 준비"""
        try:
            # Split data if eval_data is not provided
            if eval_data is None and test_split > 0:
                split_idx = int(len(train_data) * (1 - test_split))
                eval_data = train_data[split_idx:]
                train_data = train_data[:split_idx]
            
            # Create datasets
            train_dataset = FragranceDataset(
                train_data, 
                self.tokenizer, 
                settings.max_seq_length
            )
            
            eval_dataset = None
            if eval_data:
                eval_dataset = FragranceDataset(
                    eval_data, 
                    self.tokenizer, 
                    settings.max_seq_length
                )
            
            logger.info(f"Prepared datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def train(
        self,
        train_dataset: FragranceDataset,
        eval_dataset: Optional[FragranceDataset] = None,
        output_dir: str = "./models/fragrance_peft",
        num_epochs: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_early_stopping: bool = True,
        patience: int = 3
    ) -> None:
        """모델 훈련"""
        try:
            # Use default values if not provided
            num_epochs = num_epochs or settings.num_epochs
            learning_rate = learning_rate or settings.learning_rate
            batch_size = batch_size or settings.batch_size
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=1,
                learning_rate=learning_rate,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                logging_dir=f"{output_dir}/logs",
                logging_steps=logging_steps,
                save_steps=save_steps,
                eval_steps=eval_steps if eval_dataset else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                save_total_limit=3,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                fp16=torch.cuda.is_available(),
                report_to="wandb" if self.use_wandb else None,
                run_name=f"fragrance-peft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Callbacks
            callbacks = []
            if use_early_stopping and eval_dataset:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=patience,
                        early_stopping_threshold=0.001
                    )
                )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model()
            trainer.save_state()
            
            # Save training metrics
            metrics = {
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics["train_samples_per_second"],
                "train_loss": train_result.metrics["train_loss"]
            }
            
            with open(f"{output_dir}/training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.metrics['train_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(
        self,
        eval_dataset: FragranceDataset,
        model_path: Optional[str] = None
    ) -> Dict[str, float]:
        """모델 평가"""
        try:
            if model_path:
                # Load model from path
                model = PeftModel.from_pretrained(self.model, model_path)
            else:
                model = self.model
            
            model.eval()
            
            # Create data loader
            dataloader = DataLoader(
                eval_dataset,
                batch_size=settings.batch_size,
                shuffle=False
            )
            
            total_loss = 0.0
            num_batches = 0
            perplexities = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Calculate perplexity
                    perplexity = torch.exp(loss)
                    perplexities.append(perplexity.item())
            
            # Calculate metrics
            avg_loss = total_loss / num_batches
            avg_perplexity = np.mean(perplexities)
            
            metrics = {
                "eval_loss": avg_loss,
                "perplexity": avg_perplexity,
                "num_samples": len(eval_dataset)
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def generate_sample(
        self,
        prompt: str,
        model_path: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """샘플 생성"""
        try:
            if model_path:
                model = PeftModel.from_pretrained(self.model, model_path)
            else:
                model = self.model
            
            model.eval()
            
            # Format prompt
            formatted_prompt = f"### 요청:\n{prompt}\n\n### 레시피:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            return ""
    
    def save_adapter(self, output_dir: str) -> bool:
        """LoRA 어댑터 저장"""
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training configuration
            config = {
                "base_model": self.model_name,
                "lora_r": settings.lora_r,
                "lora_alpha": settings.lora_alpha,
                "lora_dropout": settings.lora_dropout,
                "max_seq_length": settings.max_seq_length,
                "created_at": datetime.now().isoformat()
            }
            
            with open(f"{output_dir}/adapter_config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Adapter saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            return False
    
    def load_adapter(self, adapter_path: str) -> bool:
        """LoRA 어댑터 로드"""
        try:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"Adapter loaded from {adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False
    
    def merge_and_save(self, output_dir: str) -> bool:
        """어댑터를 기본 모델에 병합하고 저장"""
        try:
            # Merge adapter weights into base model
            merged_model = self.model.merge_and_unload()
            
            # Save merged model
            merged_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Merged model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}")
            return False