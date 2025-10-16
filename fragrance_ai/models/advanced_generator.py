"""
Advanced Multi-Conditional Fragrance Generator

이 모듈은 다양한 조건들을 동시에 처리하여 완벽하게 맞춤화된 향수 레시피를 생성합니다.
- 날씨, 계절, 시간, 장소, 무드, 연령, 성별, 상황 등 다중 조건 처리
- 실시간 적응형 학습 시스템
- 극한 상황 대응 능력
- 성능 예측 및 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Model,
    GPT2Config
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class MultiConditionEmbedding(nn.Module):
    """다중 조건을 임베딩하는 신경망"""

    def __init__(self, condition_vocab_sizes: Dict[str, int], embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 각 조건별 임베딩 레이어
        self.condition_embeddings = nn.ModuleDict({
            condition: nn.Embedding(vocab_size, embedding_dim // len(condition_vocab_sizes))
            for condition, vocab_size in condition_vocab_sizes.items()
        })

        # 조건 간 상호작용을 학습하는 어텐션
        self.condition_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 조건 조합 최적화
        self.condition_mixer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """조건들을 통합된 임베딩으로 변환"""
        condition_embeds = []

        for condition_name, condition_ids in conditions.items():
            if condition_name in self.condition_embeddings:
                embed = self.condition_embeddings[condition_name](condition_ids)
                condition_embeds.append(embed)

        # 모든 조건 임베딩을 연결
        combined_embed = torch.cat(condition_embeds, dim=-1)

        # Self-attention으로 조건 간 상호작용 학습
        attended_embed, _ = self.condition_attention(
            combined_embed, combined_embed, combined_embed
        )

        # 최종 조건 임베딩
        final_embed = self.condition_mixer(attended_embed)
        return final_embed


class AdaptivePerformancePredictor(nn.Module):
    """향수 성능을 예측하는 신경망"""

    def __init__(self, input_dim: int = 512):
        super().__init__()

        self.performance_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # longevity, sillage, projection, weather_resistance
        )

        # 성능 점수를 0-10 범위로 정규화
        self.performance_activation = nn.Sigmoid()

    def forward(self, conditions_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """조건을 바탕으로 향수 성능 예측"""
        performance_raw = self.performance_predictor(conditions_embed)
        performance_scores = self.performance_activation(performance_raw) * 10

        return {
            'longevity': performance_scores[:, 0],
            'sillage': performance_scores[:, 1],
            'projection': performance_scores[:, 2],
            'weather_resistance': performance_scores[:, 3]
        }


class AdvancedFragranceGenerator(nn.Module):
    """완벽한 다중 조건 향수 생성 시스템"""

    def __init__(
        self,
        base_model_name: str = "microsoft/DialoGPT-medium",
        condition_vocab_sizes: Optional[Dict[str, int]] = None,
        use_quantization: bool = False
    ):
        super().__init__()

        # 기본 조건 어휘 크기 설정
        if condition_vocab_sizes is None:
            condition_vocab_sizes = {
                'weather': 20,      # sunny, rainy, snowy, etc.
                'season': 4,        # spring, summer, autumn, winter
                'time': 8,          # morning, afternoon, evening, night, etc.
                'location': 50,     # indoor, outdoor, specific places
                'mood': 30,         # happy, sad, energetic, calm, etc.
                'age_group': 10,    # teens, 20s, 30s, 40s, 50s+
                'gender': 3,        # male, female, unisex
                'occasion': 25,     # casual, formal, business, romantic, etc.
                'intensity': 5,     # very_light, light, moderate, strong, very_strong
                'budget': 5,        # budget, affordable, mid_range, premium, luxury
                'skin_type': 6,     # normal, dry, oily, sensitive, combination, mature
                'personality': 15   # confident, shy, adventurous, traditional, etc.
            }

        self.condition_vocab_sizes = condition_vocab_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 토크나이저와 기본 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 기본 언어 모델
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32
        )

        # 고급 LoRA 설정
        lora_config = LoraConfig(
            r=64,               # 더 높은 rank
            lora_alpha=128,     # 더 높은 alpha
            target_modules=['c_attn', 'c_proj', 'c_fc'],  # 더 많은 모듈
            lora_dropout=0.05,  # 더 낮은 dropout
            bias='none',
            task_type=TaskType.CAUSAL_LM
        )
        self.base_model = get_peft_model(self.base_model, lora_config)

        # 다중 조건 임베딩 시스템
        self.condition_embedder = MultiConditionEmbedding(
            condition_vocab_sizes,
            embedding_dim=768
        )

        # 성능 예측 시스템
        self.performance_predictor = AdaptivePerformancePredictor(768)

        # 조건을 텍스트 생성에 통합하는 어댑터
        self.condition_text_adapter = nn.Sequential(
            nn.Linear(768, self.base_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        )

        # 실시간 적응을 위한 메모리 뱅크
        self.adaptation_memory = {}
        self.success_feedback = {}

        logger.info("Advanced Fragrance Generator initialized")

    def encode_conditions(self, conditions: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """문자열 조건들을 텐서로 인코딩"""
        # 조건 값을 인덱스로 변환하는 매핑
        condition_mappings = {
            'weather': {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3, 'windy': 4,
                       'hot': 5, 'cold': 6, 'humid': 7, 'dry': 8, 'stormy': 9},
            'season': {'spring': 0, 'summer': 1, 'autumn': 2, 'winter': 3},
            'time': {'early_morning': 0, 'morning': 1, 'late_morning': 2, 'noon': 3,
                    'afternoon': 4, 'evening': 5, 'night': 6, 'late_night': 7},
            'mood': {'happy': 0, 'sad': 1, 'energetic': 2, 'calm': 3, 'romantic': 4,
                    'confident': 5, 'mysterious': 6, 'playful': 7, 'sophisticated': 8},
            'age_group': {'teens': 0, '20s': 1, '30s': 2, '40s': 3, '50s': 4, '60s_plus': 5},
            'gender': {'male': 0, 'female': 1, 'unisex': 2},
            'intensity': {'very_light': 0, 'light': 1, 'moderate': 2, 'strong': 3, 'very_strong': 4},
            'budget': {'budget': 0, 'affordable': 1, 'mid_range': 2, 'premium': 3, 'luxury': 4}
        }

        encoded_conditions = {}
        for condition_name, condition_value in conditions.items():
            if condition_name in condition_mappings:
                mapping = condition_mappings[condition_name]
                condition_id = mapping.get(condition_value, 0)  # 기본값 0
                encoded_conditions[condition_name] = torch.tensor([condition_id], dtype=torch.long)

        return encoded_conditions

    def predict_performance(self, conditions: Dict[str, str]) -> Dict[str, float]:
        """조건을 바탕으로 향수 성능 예측"""
        encoded_conditions = self.encode_conditions(conditions)

        # 조건 임베딩 생성
        with torch.no_grad():
            condition_embed = self.condition_embedder(encoded_conditions)
            performance_pred = self.performance_predictor(condition_embed)

        # 텐서를 파이썬 값으로 변환
        performance_dict = {}
        for key, value in performance_pred.items():
            performance_dict[key] = float(value.item())

        return performance_dict

    def generate_recipe(
        self,
        prompt: str,
        conditions: Dict[str, str],
        max_length: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """조건에 맞는 향수 레시피 생성"""

        # 조건 인코딩 및 임베딩
        encoded_conditions = self.encode_conditions(conditions)
        condition_embed = self.condition_embedder(encoded_conditions)

        # 성능 예측
        performance_pred = self.predict_performance(conditions)

        # 조건 정보를 프롬프트에 통합
        condition_text = self.format_conditions_for_prompt(conditions)
        enhanced_prompt = f"{condition_text}\\n{prompt}"

        # 텍스트 생성
        inputs = self.tokenizer.encode(enhanced_prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe_text = generated_text[len(enhanced_prompt):].strip()

        return {
            'generated_recipe': recipe_text,
            'predicted_performance': performance_pred,
            'conditions_used': conditions,
            'adaptation_score': self.calculate_adaptation_score(conditions)
        }

    def format_conditions_for_prompt(self, conditions: Dict[str, str]) -> str:
        """조건들을 프롬프트 형식으로 변환"""
        condition_text = "### 조건:\\n"
        for key, value in conditions.items():
            formatted_key = key.replace('_', ' ').title()
            condition_text += f"- {formatted_key}: {value}\\n"
        return condition_text

    def calculate_adaptation_score(self, conditions: Dict[str, str]) -> float:
        """현재 조건에 대한 모델의 적응 점수 계산"""
        condition_key = str(sorted(conditions.items()))

        if condition_key in self.success_feedback:
            return self.success_feedback[condition_key]
        else:
            # 새로운 조건 조합에 대한 기본 점수
            return 0.7

    def update_adaptation(self, conditions: Dict[str, str], success_score: float):
        """사용자 피드백을 바탕으로 적응 학습"""
        condition_key = str(sorted(conditions.items()))

        if condition_key in self.success_feedback:
            # 기존 점수와 새 점수의 이동 평균
            old_score = self.success_feedback[condition_key]
            self.success_feedback[condition_key] = 0.8 * old_score + 0.2 * success_score
        else:
            self.success_feedback[condition_key] = success_score

        logger.info(f"Updated adaptation for conditions {condition_key}: {self.success_feedback[condition_key]:.3f}")

    def save_adaptation_state(self, filepath: str):
        """적응 상태 저장"""
        adaptation_state = {
            'success_feedback': self.success_feedback,
            'adaptation_memory': self.adaptation_memory,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(adaptation_state, f, ensure_ascii=False, indent=2)

    def load_adaptation_state(self, filepath: str):
        """적응 상태 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                adaptation_state = json.load(f)

            self.success_feedback = adaptation_state.get('success_feedback', {})
            self.adaptation_memory = adaptation_state.get('adaptation_memory', {})

            logger.info(f"Loaded adaptation state from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Adaptation state file {filepath} not found, starting fresh")

    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 정보 반환"""
        return {
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'condition_types': len(self.condition_vocab_sizes),
            'adaptation_experiences': len(self.success_feedback),
            'average_adaptation_score': np.mean(list(self.success_feedback.values())) if self.success_feedback else 0.0
        }


def create_advanced_training_data(dataset_path: str) -> List[Dict[str, Any]]:
    """고급 훈련 데이터 생성"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    training_examples = []

    for item in data['advanced_conditional_training_data']:
        conditions = item['conditions']
        input_text = item['input']
        output_data = item['output']

        # 조건을 텍스트 형식으로 변환
        condition_text = "### 조건:\\n"
        for key, value in conditions.items():
            formatted_key = key.replace('_', ' ').title()
            condition_text += f"- {formatted_key}: {value}\\n"

        # 출력 레시피 생성
        recipe_text = f"{output_data['fragrance_name']} ({output_data['korean_name']}) - "
        recipe_text += f"컨셉: {output_data['concept']}"

        if 'notes_breakdown' in output_data:
            notes = output_data['notes_breakdown']
            recipe_text += f" - 톱노트: {', '.join([n['note'] for n in notes.get('top_notes', [])])}"
            recipe_text += f" - 미들노트: {', '.join([n['note'] for n in notes.get('middle_notes', [])])}"
            recipe_text += f" - 베이스노트: {', '.join([n['note'] for n in notes.get('base_notes', [])])}"

        training_text = f"{condition_text}\\n### 요청: {input_text}\\n### 레시피: {recipe_text}"

        training_examples.append({
            'text': training_text,
            'conditions': conditions,
            'performance_targets': output_data.get('performance_specs', {})
        })

    return training_examples


if __name__ == "__main__":
    # 고급 생성기 테스트
    generator = AdvancedFragranceGenerator()

    test_conditions = {
        'weather': 'rainy',
        'season': 'autumn',
        'time': 'evening',
        'mood': 'romantic',
        'age_group': '20s',
        'gender': 'female',
        'intensity': 'moderate',
        'budget': 'mid_range'
    }

    result = generator.generate_recipe(
        "카페에서의 로맨틱한 데이트에 완벽한 향수를 만들어주세요",
        test_conditions
    )

    print("Generated Recipe:", result['generated_recipe'])
    print("Predicted Performance:", result['predicted_performance'])
    print("Model Stats:", generator.get_model_stats())