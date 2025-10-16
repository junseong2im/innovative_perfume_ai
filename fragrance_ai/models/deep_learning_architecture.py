"""
딥러닝 기반 향수 생성 모델 아키텍처
Universal Fragrance Generation Deep Learning System

이 모듈은 어떤 조건이나 요구에도 대응 가능한 범용적 향수 생성 딥러닝 시스템을 제공합니다.
다중 입력 모달리티(텍스트, 이미지, 음성, 센서 데이터)를 융합하여 향수 레시피를 생성합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FragranceGenerationConfig:
    """향수 생성 모델 설정 클래스"""

    # 모델 차원 설정
    text_embedding_dim: int = 768  # BERT 기본 차원
    image_embedding_dim: int = 2048  # ResNet50 기본 차원
    audio_embedding_dim: int = 512  # 오디오 특성 차원
    sensor_embedding_dim: int = 256  # 센서 데이터 차원

    # 통합 임베딩 차원
    unified_embedding_dim: int = 1024

    # 향수 구성 요소 차원
    num_fragrance_notes: int = 500  # 최대 향료 노트 수
    num_concentration_levels: int = 10  # 농도 레벨 수
    max_recipe_length: int = 50  # 최대 레시피 길이

    # 트랜스포머 설정
    num_transformer_layers: int = 8
    num_attention_heads: int = 16
    transformer_dropout: float = 0.1

    # 학습 설정
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 1000

    # 생성 설정
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_generation_length: int = 100

class MultiModalEncoder(nn.Module):
    """다중 모달리티 입력을 처리하는 인코더"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config

        # 텍스트 인코더 (BERT 기반)
        self.text_encoder = AutoModel.from_pretrained('klue/bert-base')
        self.text_projection = nn.Linear(config.text_embedding_dim, config.unified_embedding_dim)

        # 이미지 인코더 (ResNet 기반)
        self.image_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()  # 마지막 분류층 제거
        self.image_projection = nn.Linear(config.image_embedding_dim, config.unified_embedding_dim)

        # 오디오 인코더
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, config.audio_embedding_dim)
        )
        self.audio_projection = nn.Linear(config.audio_embedding_dim, config.unified_embedding_dim)

        # 센서 데이터 인코더 (환경, 생체 신호 등)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(64, 128),  # 다양한 센서 입력 가정
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config.sensor_embedding_dim)
        )
        self.sensor_projection = nn.Linear(config.sensor_embedding_dim, config.unified_embedding_dim)

        # 모달리티 융합 레이어
        self.modality_fusion = nn.MultiheadAttention(
            embed_dim=config.unified_embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 최종 융합 임베딩
        self.fusion_norm = nn.LayerNorm(config.unified_embedding_dim)

    def forward(self,
                text_input: Optional[Dict] = None,
                image_input: Optional[torch.Tensor] = None,
                audio_input: Optional[torch.Tensor] = None,
                sensor_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        다중 모달리티 입력을 통합 임베딩으로 변환

        Args:
            text_input: 토큰화된 텍스트 입력
            image_input: 이미지 텐서 (B, C, H, W)
            audio_input: 오디오 텐서 (B, 1, T)
            sensor_input: 센서 데이터 텐서 (B, sensor_dim)

        Returns:
            융합된 임베딩 텐서 (B, unified_embedding_dim)
        """
        embeddings = []

        # 텍스트 임베딩
        if text_input is not None:
            text_output = self.text_encoder(**text_input)
            text_emb = text_output.last_hidden_state.mean(dim=1)  # 평균 풀링
            text_emb = self.text_projection(text_emb)
            embeddings.append(text_emb.unsqueeze(1))

        # 이미지 임베딩
        if image_input is not None:
            with torch.no_grad():
                image_features = self.image_encoder(image_input)
            image_emb = self.image_projection(image_features)
            embeddings.append(image_emb.unsqueeze(1))

        # 오디오 임베딩
        if audio_input is not None:
            audio_features = self.audio_encoder(audio_input)
            audio_emb = self.audio_projection(audio_features)
            embeddings.append(audio_emb.unsqueeze(1))

        # 센서 임베딩
        if sensor_input is not None:
            sensor_features = self.sensor_encoder(sensor_input)
            sensor_emb = self.sensor_projection(sensor_features)
            embeddings.append(sensor_emb.unsqueeze(1))

        if not embeddings:
            raise ValueError("적어도 하나의 입력 모달리티가 필요합니다")

        # 다중 모달리티 융합
        if len(embeddings) == 1:
            fused_embedding = embeddings[0].squeeze(1)
        else:
            # 모든 임베딩을 연결
            stacked_embeddings = torch.cat(embeddings, dim=1)  # (B, num_modalities, embed_dim)

            # 어텐션을 통한 융합
            fused_embedding, _ = self.modality_fusion(
                stacked_embeddings, stacked_embeddings, stacked_embeddings
            )
            fused_embedding = fused_embedding.mean(dim=1)  # 평균 풀링

        return self.fusion_norm(fused_embedding)

class FragranceTransformer(nn.Module):
    """향수 생성을 위한 트랜스포머 모델"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config

        # 향료 임베딩
        self.note_embedding = nn.Embedding(config.num_fragrance_notes, config.unified_embedding_dim)
        self.concentration_embedding = nn.Embedding(config.num_concentration_levels, config.unified_embedding_dim)
        self.position_embedding = nn.Embedding(config.max_recipe_length, config.unified_embedding_dim)

        # 트랜스포머 인코더
        encoder_layers = TransformerEncoderLayer(
            d_model=config.unified_embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.unified_embedding_dim * 4,
            dropout=config.transformer_dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, config.num_transformer_layers)

        # 조건부 임베딩 통합
        self.condition_projection = nn.Linear(config.unified_embedding_dim, config.unified_embedding_dim)

        # 출력 헤드들
        self.note_head = nn.Linear(config.unified_embedding_dim, config.num_fragrance_notes)
        self.concentration_head = nn.Linear(config.unified_embedding_dim, config.num_concentration_levels)
        self.volume_head = nn.Linear(config.unified_embedding_dim, 1)  # 연속값 예측

        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(config.unified_embedding_dim)

    def forward(self,
                condition_embedding: torch.Tensor,
                target_notes: Optional[torch.Tensor] = None,
                target_concentrations: Optional[torch.Tensor] = None,
                target_volumes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        조건부 향수 생성

        Args:
            condition_embedding: 조건 임베딩 (B, embed_dim)
            target_notes: 타겟 향료 시퀀스 (B, seq_len)
            target_concentrations: 타겟 농도 시퀀스 (B, seq_len)
            target_volumes: 타겟 부피 시퀀스 (B, seq_len)

        Returns:
            예측 결과 딕셔너리
        """
        batch_size = condition_embedding.size(0)

        if target_notes is not None:
            # 학습 모드: teacher forcing
            seq_len = target_notes.size(1)

            # 향료, 농도, 위치 임베딩
            note_emb = self.note_embedding(target_notes)
            conc_emb = self.concentration_embedding(target_concentrations)
            pos_ids = torch.arange(seq_len, device=target_notes.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)

            # 임베딩 합성
            sequence_emb = note_emb + conc_emb + pos_emb

            # 조건 정보 통합
            condition_proj = self.condition_projection(condition_embedding).unsqueeze(1)
            sequence_emb = sequence_emb + condition_proj

            # 트랜스포머 처리
            transformer_output = self.transformer(sequence_emb)
            transformer_output = self.layer_norm(transformer_output)

            # 예측 헤드들
            note_logits = self.note_head(transformer_output)
            concentration_logits = self.concentration_head(transformer_output)
            volume_pred = self.volume_head(transformer_output).squeeze(-1)

            return {
                'note_logits': note_logits,
                'concentration_logits': concentration_logits,
                'volume_predictions': volume_pred,
                'hidden_states': transformer_output
            }
        else:
            # 추론 모드: 자기회귀 생성
            return self.generate(condition_embedding)

    def generate(self, condition_embedding: torch.Tensor, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        자기회귀적 향수 레시피 생성

        Args:
            condition_embedding: 조건 임베딩 (B, embed_dim)
            max_length: 최대 생성 길이

        Returns:
            생성된 레시피 딕셔너리
        """
        if max_length is None:
            max_length = self.config.max_generation_length

        batch_size = condition_embedding.size(0)
        device = condition_embedding.device

        # 시작 토큰 (특별한 시작 노트 ID 사용)
        generated_notes = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        generated_concentrations = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        generated_volumes = torch.zeros(batch_size, 1, device=device)

        # 조건 투영
        condition_proj = self.condition_projection(condition_embedding).unsqueeze(1)

        for step in range(max_length - 1):
            current_length = generated_notes.size(1)

            # 현재까지의 시퀀스 임베딩
            note_emb = self.note_embedding(generated_notes)
            conc_emb = self.concentration_embedding(generated_concentrations)
            pos_ids = torch.arange(current_length, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)

            # 임베딩 합성
            sequence_emb = note_emb + conc_emb + pos_emb + condition_proj

            # 트랜스포머 처리
            transformer_output = self.transformer(sequence_emb)
            transformer_output = self.layer_norm(transformer_output)

            # 마지막 스텝의 예측
            last_hidden = transformer_output[:, -1:, :]

            # 다음 토큰 예측
            note_logits = self.note_head(last_hidden)
            concentration_logits = self.concentration_head(last_hidden)
            volume_pred = self.volume_head(last_hidden)

            # 샘플링
            next_note = self._sample_token(note_logits.squeeze(1))
            next_concentration = self._sample_token(concentration_logits.squeeze(1))
            next_volume = volume_pred.squeeze()

            # 시퀀스에 추가
            generated_notes = torch.cat([generated_notes, next_note.unsqueeze(1)], dim=1)
            generated_concentrations = torch.cat([generated_concentrations, next_concentration.unsqueeze(1)], dim=1)
            generated_volumes = torch.cat([generated_volumes, next_volume.unsqueeze(1)], dim=1)

            # 종료 조건 검사 (EOS 토큰이나 특정 조건)
            if self._should_stop_generation(next_note):
                break

        return {
            'generated_notes': generated_notes,
            'generated_concentrations': generated_concentrations,
            'generated_volumes': generated_volumes
        }

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """토큰 샘플링 (temperature, top-k, top-p 적용)"""
        # Temperature 스케일링
        logits = logits / self.config.temperature

        # Top-k 필터링
        if self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) 필터링
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # 소프트맥스 및 샘플링
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def _should_stop_generation(self, tokens: torch.Tensor) -> bool:
        """생성 중단 조건 검사"""
        # EOS 토큰이나 특정 조건으로 중단 결정
        # 여기서는 간단히 False 반환 (실제로는 더 복잡한 로직 필요)
        return False

class UniversalFragranceGenerator(nn.Module):
    """범용 향수 생성 모델"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config

        # 다중 모달리티 인코더
        self.multimodal_encoder = MultiModalEncoder(config)

        # 향수 생성 트랜스포머
        self.fragrance_transformer = FragranceTransformer(config)

        # 품질 평가 모듈
        self.quality_evaluator = QualityEvaluator(config)

        # 문화적 적응 모듈
        self.cultural_adapter = CulturalAdapter(config)

    def forward(self,
                inputs: Dict[str, Any],
                targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        범용 향수 생성 전체 파이프라인

        Args:
            inputs: 다양한 입력 모달리티들
            targets: 학습용 타겟 (옵션)

        Returns:
            생성 결과 및 평가 메트릭
        """
        # 1. 다중 모달리티 인코딩
        condition_embedding = self.multimodal_encoder(
            text_input=inputs.get('text'),
            image_input=inputs.get('image'),
            audio_input=inputs.get('audio'),
            sensor_input=inputs.get('sensor')
        )

        # 2. 문화적 적응
        if 'cultural_context' in inputs:
            condition_embedding = self.cultural_adapter(
                condition_embedding,
                inputs['cultural_context']
            )

        # 3. 향수 생성
        generation_output = self.fragrance_transformer(
            condition_embedding=condition_embedding,
            target_notes=targets.get('notes') if targets else None,
            target_concentrations=targets.get('concentrations') if targets else None,
            target_volumes=targets.get('volumes') if targets else None
        )

        # 4. 품질 평가
        if not self.training:  # 추론 시에만 품질 평가
            quality_scores = self.quality_evaluator(
                generation_output,
                condition_embedding
            )
            generation_output.update(quality_scores)

        return generation_output

class QualityEvaluator(nn.Module):
    """생성된 향수의 품질을 평가하는 모듈"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config

        # 조화도 평가기
        self.harmony_evaluator = nn.Sequential(
            nn.Linear(config.unified_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # 실현가능성 평가기
        self.feasibility_evaluator = nn.Sequential(
            nn.Linear(config.unified_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # 창의성 평가기
        self.creativity_evaluator = nn.Sequential(
            nn.Linear(config.unified_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, generation_output: Dict[str, torch.Tensor],
                condition_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """품질 메트릭 계산"""

        # 생성된 시퀀스의 표현 획득
        if 'hidden_states' in generation_output:
            sequence_repr = generation_output['hidden_states'].mean(dim=1)
        else:
            # 추론 모드에서는 조건 임베딩 사용
            sequence_repr = condition_embedding

        # 각 품질 메트릭 계산
        harmony_score = self.harmony_evaluator(sequence_repr)
        feasibility_score = self.feasibility_evaluator(sequence_repr)
        creativity_score = self.creativity_evaluator(sequence_repr)

        return {
            'harmony_score': harmony_score,
            'feasibility_score': feasibility_score,
            'creativity_score': creativity_score,
            'overall_quality': (harmony_score + feasibility_score + creativity_score) / 3
        }

class CulturalAdapter(nn.Module):
    """문화적 맥락에 따른 향수 적응 모듈"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config

        # 문화권별 임베딩
        self.cultural_embeddings = nn.Embedding(50, config.unified_embedding_dim)  # 50개 문화권 가정

        # 적응 네트워크
        self.adaptation_network = nn.Sequential(
            nn.Linear(config.unified_embedding_dim * 2, config.unified_embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.unified_embedding_dim, config.unified_embedding_dim)
        )

    def forward(self, condition_embedding: torch.Tensor, cultural_context: torch.Tensor) -> torch.Tensor:
        """문화적 맥락을 반영한 조건 임베딩 적응"""

        # 문화적 임베딩 획득
        cultural_emb = self.cultural_embeddings(cultural_context)

        # 조건 임베딩과 문화적 임베딩 결합
        combined = torch.cat([condition_embedding, cultural_emb], dim=-1)

        # 적응된 임베딩 생성
        adapted_embedding = self.adaptation_network(combined)

        # 잔차 연결
        return condition_embedding + adapted_embedding

class AdaptiveLearningSystem(nn.Module):
    """실시간 적응형 학습 시스템"""

    def __init__(self, config: FragranceGenerationConfig):
        super().__init__()
        self.config = config
        self.base_model = UniversalFragranceGenerator(config)

        # 메타 학습을 위한 추가 네트워크
        self.meta_learner = nn.Sequential(
            nn.Linear(config.unified_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.unified_embedding_dim)
        )

        # 사용자 피드백 통합 모듈
        self.feedback_integrator = nn.Sequential(
            nn.Linear(config.unified_embedding_dim + 10, 512),  # 피드백 차원 가정
            nn.ReLU(),
            nn.Linear(512, config.unified_embedding_dim)
        )

    def forward(self, inputs: Dict[str, Any],
                user_feedback: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """사용자 피드백을 반영한 적응형 생성"""

        # 기본 모델 실행
        output = self.base_model(inputs)

        # 사용자 피드백이 있는 경우 적응
        if user_feedback is not None:
            # 피드백 통합
            condition_emb = output.get('condition_embedding', inputs.get('condition_embedding'))
            feedback_input = torch.cat([condition_emb, user_feedback], dim=-1)
            adapted_condition = self.feedback_integrator(feedback_input)

            # 적응된 조건으로 재생성
            adapted_output = self.base_model({
                **inputs,
                'condition_embedding': adapted_condition
            })

            return adapted_output

        return output

def create_universal_fragrance_model(config_path: Optional[str] = None) -> UniversalFragranceGenerator:
    """범용 향수 생성 모델 팩토리 함수"""

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = FragranceGenerationConfig(**config_dict)
    else:
        config = FragranceGenerationConfig()

    model = UniversalFragranceGenerator(config)

    logger.info(f"범용 향수 생성 모델 생성 완료")
    logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    return model

if __name__ == "__main__":
    # 모델 테스트
    config = FragranceGenerationConfig()
    model = create_universal_fragrance_model()

    # 더미 입력으로 테스트
    batch_size = 2

    # 텍스트 입력 (토큰화된 형태 가정)
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 32)),
        'attention_mask': torch.ones(batch_size, 32)
    }

    # 이미지 입력
    image_input = torch.randn(batch_size, 3, 224, 224)

    # 오디오 입력
    audio_input = torch.randn(batch_size, 1, 16000)

    # 센서 입력
    sensor_input = torch.randn(batch_size, 64)

    inputs = {
        'text': text_input,
        'image': image_input,
        'audio': audio_input,
        'sensor': sensor_input
    }

    # 추론 테스트
    model.eval()
    with torch.no_grad():
        output = model(inputs)

    print("모델 출력:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

    print("\n범용 향수 생성 딥러닝 시스템 초기화 완료!")