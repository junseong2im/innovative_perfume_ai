from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import logging
import random
import re
from datetime import datetime
from ..core.config import settings

logger = logging.getLogger(__name__)


class FragranceRecipeGenerator:
    """향수 레시피 생성을 위한 미세조정된 언어 모델"""
    
    def __init__(self, model_name: Optional[str] = None, use_quantization: bool = True):
        self.model_name = model_name or settings.generation_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_quantization = use_quantization and torch.cuda.is_available()
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize generation config
        self._setup_generation_config()
        
        # Fragrance recipe templates and patterns
        self.recipe_templates = self._load_recipe_templates()
        
    def _load_model(self) -> None:
        """모델과 토크나이저 로드"""
        try:
            # Configure quantization for efficiency
            if self.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"Loaded generation model: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load generation model: {e}")
            raise
    
    def _setup_generation_config(self) -> None:
        """생성 설정 구성"""
        self.generation_config = GenerationConfig(
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            do_sample=settings.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    def _load_recipe_templates(self) -> Dict[str, str]:
        """향수 레시피 템플릿 로드"""
        return {
            "basic_recipe": """
### 향수 레시피: {fragrance_name}

**컨셉**: {concept}
**타겟 무드**: {target_mood}
**계절**: {season}

**구성비**:
- 톱노트 ({top_percentage}%): {top_notes}
- 미들노트 ({heart_percentage}%): {heart_notes}  
- 베이스노트 ({base_percentage}%): {base_notes}

**조향 노트**:
{blending_notes}

**착용감**: {wearing_experience}
            """,
            
            "detailed_recipe": """
### 프리미엄 향수 레시피: {fragrance_name}

**브랜드 스토리**: {brand_story}
**향수 철학**: {philosophy}
**타겟 고객**: {target_customer}

**원료 구성**:

**톱노트 ({top_percentage}%)**:
{detailed_top_notes}

**미들노트 ({heart_percentage}%)**:
{detailed_heart_notes}

**베이스노트 ({base_percentage}%)**:
{detailed_base_notes}

**조향 기법**:
{blending_technique}

**숙성 과정**:
{aging_process}

**최종 평가**:
- 지속력: {longevity}
- 실라지: {sillage}
- 프로젝션: {projection}
- 복잡성: {complexity}
            """
        }
    
    def generate_recipe(
        self,
        prompt: str,
        recipe_type: str = "basic_recipe",
        include_story: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """향수 레시피 생성"""
        try:
            # Create enhanced prompt
            enhanced_prompt = self._create_recipe_prompt(prompt, recipe_type, **kwargs)
            
            # Generate recipe
            generated_text = self._generate_text(enhanced_prompt)
            
            # Parse and structure the recipe
            structured_recipe = self._parse_recipe(generated_text, recipe_type)
            
            # Add story and additional details if requested
            if include_story:
                structured_recipe["story"] = self._generate_fragrance_story(structured_recipe)
            
            return structured_recipe
            
        except Exception as e:
            logger.error(f"Failed to generate recipe: {e}")
            return {"error": str(e)}
    
    def _create_recipe_prompt(
        self, 
        user_prompt: str, 
        recipe_type: str,
        **kwargs
    ) -> str:
        """향수 레시피 생성을 위한 프롬프트 구성"""
        
        system_prompt = """당신은 세계적으로 유명한 조향사입니다. 
고객의 요구에 맞는 독창적이고 매력적인 향수 레시피를 만들어주세요.
모든 향료의 비율과 조향 기법을 구체적으로 설명해주세요."""
        
        context_prompt = f"""
사용자 요청: {user_prompt}

다음 가이드라인을 따라 향수 레시피를 작성해주세요:
1. 톱노트, 미들노트, 베이스노트의 균형있는 구성
2. 각 향료의 정확한 비율 명시
3. 조향 순서와 기법 설명
4. 예상되는 향의 전개와 지속시간
5. 타겟 고객과 착용 상황
"""
        
        if kwargs.get("mood"):
            context_prompt += f"\n목표 무드: {kwargs['mood']}"
        
        if kwargs.get("season"):
            context_prompt += f"\n적합한 계절: {kwargs['season']}"
        
        if kwargs.get("notes_preference"):
            context_prompt += f"\n선호하는 노트: {kwargs['notes_preference']}"
        
        return f"{system_prompt}\n\n{context_prompt}\n\n향수 레시피:"
    
    def _generate_text(self, prompt: str) -> str:
        """텍스트 생성"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            # Safely extract generated text
            input_length = len(inputs["input_ids"][0])
            output_length = len(outputs[0])

            if output_length > input_length:
                generated_text = self.tokenizer.decode(
                    outputs[0][input_length:],
                    skip_special_tokens=True
                )
            else:
                # If no new tokens generated, decode the full output
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise
    
    def _parse_recipe(self, generated_text: str, recipe_type: str) -> Dict[str, Any]:
        """생성된 텍스트에서 구조화된 레시피 추출"""
        try:
            recipe = {
                "raw_text": generated_text,
                "generated_at": datetime.now().isoformat(),
                "recipe_type": recipe_type
            }
            
            # Extract fragrance name
            name_match = re.search(r'(?:향수명|이름|제품명):\s*([^\n]+)', generated_text)
            if name_match:
                recipe["name"] = name_match.group(1).strip()
            
            # Extract notes with percentages
            recipe["composition"] = self._extract_composition(generated_text)
            
            # Extract additional details
            recipe["concept"] = self._extract_field(generated_text, ["컨셉", "개념", "테마"])
            recipe["mood"] = self._extract_field(generated_text, ["무드", "분위기", "느낌"])
            recipe["season"] = self._extract_field(generated_text, ["계절", "시즌"])
            recipe["longevity"] = self._extract_field(generated_text, ["지속력", "지속시간"])
            recipe["sillage"] = self._extract_field(generated_text, ["실라지", "확산력"])
            
            return recipe
            
        except Exception as e:
            logger.error(f"Failed to parse recipe: {e}")
            return {"raw_text": generated_text, "error": str(e)}
    
    def _extract_composition(self, text: str) -> Dict[str, Any]:
        """향료 구성비 추출"""
        composition = {
            "top_notes": {"percentage": 0, "ingredients": []},
            "heart_notes": {"percentage": 0, "ingredients": []},
            "base_notes": {"percentage": 0, "ingredients": []}
        }
        
        # Extract top notes
        top_match = re.search(r'톱노트.*?(\d+)%.*?:([^\n]+)', text, re.IGNORECASE)
        if top_match:
            composition["top_notes"]["percentage"] = int(top_match.group(1))
            composition["top_notes"]["ingredients"] = [
                ingredient.strip() for ingredient in top_match.group(2).split(',')
            ]
        
        # Extract heart notes
        heart_match = re.search(r'(?:미들노트|하트노트).*?(\d+)%.*?:([^\n]+)', text, re.IGNORECASE)
        if heart_match:
            composition["heart_notes"]["percentage"] = int(heart_match.group(1))
            composition["heart_notes"]["ingredients"] = [
                ingredient.strip() for ingredient in heart_match.group(2).split(',')
            ]
        
        # Extract base notes
        base_match = re.search(r'베이스노트.*?(\d+)%.*?:([^\n]+)', text, re.IGNORECASE)
        if base_match:
            composition["base_notes"]["percentage"] = int(base_match.group(1))
            composition["base_notes"]["ingredients"] = [
                ingredient.strip() for ingredient in base_match.group(2).split(',')
            ]
        
        return composition
    
    def _extract_field(self, text: str, field_names: List[str]) -> Optional[str]:
        """특정 필드 값 추출"""
        for field_name in field_names:
            pattern = f'{field_name}\\s*:?\\s*([^\n]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _generate_fragrance_story(self, recipe: Dict[str, Any]) -> str:
        """향수의 스토리 생성"""
        try:
            story_prompt = f"""
다음 향수 레시피를 바탕으로 매력적인 브랜드 스토리를 작성해주세요:

향수명: {recipe.get('name', '신비로운 향수')}
컨셉: {recipe.get('concept', '')}
무드: {recipe.get('mood', '')}

300자 내외의 감성적이고 시적인 스토리를 작성해주세요.
이 향수를 착용하는 순간의 경험과 감정을 묘사해주세요.
            """
            
            story = self._generate_text(story_prompt)
            return story
            
        except Exception as e:
            logger.error(f"Failed to generate fragrance story: {e}")
            return "이 향수는 당신만의 특별한 이야기를 만들어갈 것입니다."
    
    def batch_generate_recipes(
        self,
        prompts: List[str],
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """배치로 여러 레시피 생성"""
        try:
            results = []
            
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                for prompt in batch_prompts:
                    recipe = self.generate_recipe(prompt)
                    results.append(recipe)
                
                logger.info(f"Generated batch {i//batch_size + 1}: {len(batch_prompts)} recipes")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to batch generate recipes: {e}")
            return []
    
    def fine_tune_with_lora(
        self,
        training_data: List[Dict[str, str]],
        output_dir: str = "./models/fragrance_lora",
        num_epochs: int = 3
    ) -> bool:
        """LoRA를 사용한 모델 미세조정"""
        try:
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=settings.lora_r,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Prepare training data
            train_dataset = self._prepare_training_dataset(training_data)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=settings.batch_size,
                learning_rate=settings.learning_rate,
                logging_steps=10,
                save_steps=500,
                evaluation_strategy="no",
                warmup_steps=100,
                logging_dir=f"{output_dir}/logs"
            )
            
            logger.info("Starting LoRA fine-tuning...")
            
            # 실제 훈련 루프 구현
            from transformers import Trainer, DataCollatorForLanguageModeling
            from torch.utils.data import Dataset
            
            # 커스텀 데이터셋 클래스
            class FragranceDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(),
                        'attention_mask': encoding['attention_mask'].squeeze(),
                        'labels': encoding['input_ids'].squeeze()
                    }
            
            # 훈련 데이터셋 준비
            formatted_texts = self._prepare_training_dataset(training_data)
            train_dataset = FragranceDataset(formatted_texts, self.tokenizer)
            
            # 데이터 콜레이터 설정
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                return_tensors="pt"
            )
            
            # 트레이너 초기화 및 훈련 실행
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # 훈련 실행
            trainer.train()
            
            # Save the fine-tuned model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fine-tune model: {e}")
            return False
    
    def _prepare_training_dataset(self, training_data: List[Dict[str, str]]) -> List[str]:
        """훈련 데이터셋 준비"""
        formatted_data = []
        
        for item in training_data:
            if "prompt" in item and "completion" in item:
                formatted_text = f"### 입력:\n{item['prompt']}\n\n### 출력:\n{item['completion']}"
                formatted_data.append(formatted_text)
        
        return formatted_data
    
    def load_fine_tuned_model(self, model_path: str) -> bool:
        """미세조정된 모델 로드"""
        try:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            logger.info(f"Loaded fine-tuned model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return False
    
    def evaluate_recipe_quality(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """생성된 레시피의 품질 평가"""
        try:
            scores = {
                "completeness": 0.0,
                "coherence": 0.0,
                "creativity": 0.0,
                "technical_accuracy": 0.0,
                "overall": 0.0
            }
            
            # Check completeness
            required_fields = ["composition", "concept", "mood"]
            complete_fields = sum(1 for field in required_fields if recipe.get(field))
            scores["completeness"] = complete_fields / len(required_fields)
            
            # Check composition balance
            composition = recipe.get("composition", {})
            if composition:
                total_percentage = sum([
                    composition.get("top_notes", {}).get("percentage", 0),
                    composition.get("heart_notes", {}).get("percentage", 0),
                    composition.get("base_notes", {}).get("percentage", 0)
                ])
                # Ideal total should be around 100%
                balance_score = 1.0 - abs(100 - total_percentage) / 100
                scores["technical_accuracy"] = max(0.0, balance_score)
            
            # Calculate overall score
            scores["overall"] = sum(scores.values()) / len([v for v in scores.values() if v > 0])
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to evaluate recipe quality: {e}")
            return {"overall": 0.0}