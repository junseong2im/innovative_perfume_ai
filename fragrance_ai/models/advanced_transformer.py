"""
고급 향수 생성 트랜스포머 모델
Advanced Fragrance Generation Transformer

이 모듈은 최신 트랜스포머 아키텍처를 기반으로 한 향수 생성 모델을 제공합니다.
어떤 조건이나 요구에도 대응할 수 있는 고도화된 생성 시스템입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTransformerConfig:
    """고급 트랜스포머 설정 클래스"""

    # 모델 기본 설정
    model_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    ff_dim: int = 4096
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # 향수 특화 설정
    vocab_size: int = 1000  # 향료 어휘 크기
    max_sequence_length: int = 256
    num_note_types: int = 3  # top, middle, base
    concentration_levels: int = 20
    volume_precision: int = 100  # 0.01ml 단위

    # 위치 인코딩 설정
    max_position_embeddings: int = 512
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0

    # 어텐션 설정
    use_flash_attention: bool = True
    attention_dropout: float = 0.1
    use_relative_attention_bias: bool = True

    # 생성 설정
    use_nucleus_sampling: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.2

    # 학습 설정
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # 특수 토큰
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 주파수 계산
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 캐시된 임베딩
        self._cached_embeddings = None
        self._cached_seq_len = 0

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        회전 위치 임베딩 계산

        Returns:
            cos_emb, sin_emb: (seq_len, dim//2)
        """
        if seq_len <= self._cached_seq_len and self._cached_embeddings is not None:
            cos_emb, sin_emb = self._cached_embeddings
            return cos_emb[:seq_len], sin_emb[:seq_len]

        # 위치 시퀀스 생성
        seq = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # 외적으로 주파수와 위치의 곱 계산
        freqs = torch.outer(seq, self.inv_freq)

        # cos, sin 계산
        cos_emb = torch.cos(freqs)
        sin_emb = torch.sin(freqs)

        # 캐시 업데이트
        self._cached_embeddings = (cos_emb, sin_emb)
        self._cached_seq_len = seq_len

        return cos_emb, sin_emb

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotary Position Embedding 적용

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        cos, sin: (seq_len, head_dim//2)

    Returns:
        회전 임베딩이 적용된 텐서
    """
    # x를 복소수 형태로 변환
    x1, x2 = x[..., ::2], x[..., 1::2]

    # 회전 적용
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # 원래 형태로 복원
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

class MultiHeadAttentionWithRoPE(nn.Module):
    """RoPE가 적용된 멀티헤드 어텐션"""

    def __init__(self, config: AdvancedTransformerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.scale = self.head_dim ** -0.5

        # 쿼리, 키, 밸류 프로젝션
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim)

        # RoPE
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, config.max_position_embeddings, config.rope_theta
            )

        # 어텐션 드롭아웃
        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # 상대적 어텐션 바이어스
        if config.use_relative_attention_bias:
            self.relative_attention_bias = nn.Parameter(
                torch.zeros(config.num_heads, config.max_sequence_length, config.max_sequence_length)
            )

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_value_states: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        멀티헤드 어텐션 계산

        Args:
            hidden_states: (batch, seq_len, model_dim)
            attention_mask: (batch, seq_len)
            key_value_states: 크로스 어텐션용 (batch, kv_seq_len, model_dim)
            past_key_value: 캐시된 키-밸류
            output_attentions: 어텐션 가중치 출력 여부

        Returns:
            output, attention_weights, present_key_value
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 쿼리, 키, 밸류 계산
        if key_value_states is None:
            # 셀프 어텐션
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
        else:
            # 크로스 어텐션
            query = self.q_proj(hidden_states)
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)

        # 헤드 차원으로 재구성
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key.shape[2]

        # 과거 키-밸류 연결
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            kv_seq_len = key.shape[2]

        # RoPE 적용
        if self.config.use_rotary_embeddings:
            cos_emb, sin_emb = self.rotary_emb(max(seq_len, kv_seq_len), query.device)

            # 쿼리에 RoPE 적용
            query = apply_rotary_pos_emb(
                query.transpose(1, 2),
                cos_emb[:seq_len],
                sin_emb[:seq_len]
            ).transpose(1, 2)

            # 키에 RoPE 적용
            key = apply_rotary_pos_emb(
                key.transpose(1, 2),
                cos_emb[:kv_seq_len],
                sin_emb[:kv_seq_len]
            ).transpose(1, 2)

        # 어텐션 스코어 계산
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # 상대적 어텐션 바이어스 추가
        if self.config.use_relative_attention_bias and hasattr(self, 'relative_attention_bias'):
            rel_bias = self.relative_attention_bias[:, :seq_len, :kv_seq_len]
            attn_scores = attn_scores + rel_bias.unsqueeze(0)

        # 어텐션 마스크 적용
        if attention_mask is not None:
            # 마스크를 어텐션 스코어 형태로 확장
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if mask.shape[-1] != kv_seq_len:
                # 키-밸류 시퀀스 길이에 맞게 조정
                mask = mask.expand(-1, -1, seq_len, kv_seq_len)

            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 소프트맥스 적용
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 어텐션 출력 계산
        attn_output = torch.matmul(attn_weights, value)

        # 헤드 차원 병합
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.config.model_dim
        )

        # 출력 프로젝션
        output = self.out_proj(attn_output)

        # 현재 키-밸류 캐시
        present_key_value = (key, value) if past_key_value is not None or self.training is False else None

        return output, attn_weights if output_attentions else None, present_key_value

class FragranceTransformerBlock(nn.Module):
    """향수 생성용 트랜스포머 블록"""

    def __init__(self, config: AdvancedTransformerConfig):
        super().__init__()
        self.config = config

        # 셀프 어텐션
        self.self_attention = MultiHeadAttentionWithRoPE(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)

        # 크로스 어텐션 (조건부 생성용)
        self.cross_attention = MultiHeadAttentionWithRoPE(config)
        self.cross_attn_layer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)

        # 피드포워드 네트워크
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, config.model_dim),
            nn.Dropout(config.dropout)
        )
        self.ffn_layer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)

        # 향수 특화 게이팅 메커니즘
        self.fragrance_gate = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.Sigmoid()
        )

        # 노트 타입별 특화 프로젝션
        self.note_type_projections = nn.ModuleList([
            nn.Linear(config.model_dim, config.model_dim) for _ in range(config.num_note_types)
        ])

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                note_type_ids: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        트랜스포머 블록 전방 전파

        Args:
            hidden_states: (batch, seq_len, model_dim)
            attention_mask: 셀프 어텐션 마스크
            encoder_hidden_states: 인코더 출력 (조건부 생성용)
            encoder_attention_mask: 인코더 어텐션 마스크
            note_type_ids: 노트 타입 ID (0: top, 1: middle, 2: base)
            past_key_value: 캐시된 키-밸류
            output_attentions: 어텐션 출력 여부

        Returns:
            output, attention_weights, present_key_value
        """
        residual = hidden_states

        # 1. 셀프 어텐션
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, self_attn_weights, present_key_value = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions
        )

        hidden_states = residual + hidden_states
        residual = hidden_states

        # 2. 크로스 어텐션 (조건부 생성시)
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_present_key_value = self.cross_attention(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions
            )

            hidden_states = residual + hidden_states
            residual = hidden_states

            if present_key_value is not None:
                present_key_value = present_key_value + cross_present_key_value

        # 3. 노트 타입별 특화 처리
        if note_type_ids is not None:
            note_specialized = torch.zeros_like(hidden_states)
            for note_type in range(self.config.num_note_types):
                mask = (note_type_ids == note_type).unsqueeze(-1)
                specialized = self.note_type_projections[note_type](hidden_states)
                note_specialized = note_specialized + mask * specialized

            # 향수 게이팅 적용
            gate = self.fragrance_gate(hidden_states)
            hidden_states = gate * note_specialized + (1 - gate) * hidden_states

        # 4. 피드포워드 네트워크
        hidden_states = self.ffn_layer_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output

        output_attentions_result = {}
        if output_attentions:
            output_attentions_result['self_attention'] = self_attn_weights
            if cross_attn_weights is not None:
                output_attentions_result['cross_attention'] = cross_attn_weights

        return hidden_states, output_attentions_result if output_attentions else None, present_key_value

class FragranceEmbedding(nn.Module):
    """향수 특화 임베딩 레이어"""

    def __init__(self, config: AdvancedTransformerConfig):
        super().__init__()
        self.config = config

        # 기본 토큰 임베딩
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)

        # 노트 타입 임베딩 (top, middle, base)
        self.note_type_embedding = nn.Embedding(config.num_note_types, config.model_dim)

        # 농도 임베딩
        self.concentration_embedding = nn.Embedding(config.concentration_levels, config.model_dim)

        # 위치 임베딩 (절대 위치)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.model_dim)

        # 계절성 임베딩
        self.season_embedding = nn.Embedding(4, config.model_dim)  # 봄, 여름, 가을, 겨울

        # 감정 임베딩
        self.emotion_embedding = nn.Embedding(8, config.model_dim)  # 8가지 기본 감정

        # 시간대 임베딩
        self.time_embedding = nn.Embedding(4, config.model_dim)  # 아침, 오후, 저녁, 밤

        # 임베딩 드롭아웃
        self.dropout = nn.Dropout(config.dropout)

        # 임베딩 스케일링
        self.embed_scale = math.sqrt(config.model_dim)

    def forward(self,
                input_ids: torch.Tensor,
                note_type_ids: Optional[torch.Tensor] = None,
                concentration_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                season_ids: Optional[torch.Tensor] = None,
                emotion_ids: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        향수 임베딩 계산

        Args:
            input_ids: 향료 토큰 ID (batch, seq_len)
            note_type_ids: 노트 타입 ID (batch, seq_len)
            concentration_ids: 농도 ID (batch, seq_len)
            position_ids: 위치 ID (batch, seq_len)
            season_ids: 계절 ID (batch,)
            emotion_ids: 감정 ID (batch,)
            time_ids: 시간대 ID (batch,)

        Returns:
            임베딩 텐서 (batch, seq_len, model_dim)
        """
        batch_size, seq_len = input_ids.shape

        # 기본 토큰 임베딩
        embeddings = self.token_embedding(input_ids) * self.embed_scale

        # 위치 임베딩
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.position_embedding(position_ids)

        # 노트 타입 임베딩
        if note_type_ids is not None:
            embeddings = embeddings + self.note_type_embedding(note_type_ids)

        # 농도 임베딩
        if concentration_ids is not None:
            embeddings = embeddings + self.concentration_embedding(concentration_ids)

        # 계절성 임베딩 (시퀀스 전체에 적용)
        if season_ids is not None:
            season_emb = self.season_embedding(season_ids).unsqueeze(1).expand(-1, seq_len, -1)
            embeddings = embeddings + season_emb

        # 감정 임베딩 (시퀀스 전체에 적용)
        if emotion_ids is not None:
            emotion_emb = self.emotion_embedding(emotion_ids).unsqueeze(1).expand(-1, seq_len, -1)
            embeddings = embeddings + emotion_emb

        # 시간대 임베딩 (시퀀스 전체에 적용)
        if time_ids is not None:
            time_emb = self.time_embedding(time_ids).unsqueeze(1).expand(-1, seq_len, -1)
            embeddings = embeddings + time_emb

        return self.dropout(embeddings)

class FragranceOutputHead(nn.Module):
    """향수 생성 출력 헤드"""

    def __init__(self, config: AdvancedTransformerConfig):
        super().__init__()
        self.config = config

        # 향료 예측 헤드
        self.note_head = nn.Linear(config.model_dim, config.vocab_size)

        # 노트 타입 예측 헤드
        self.note_type_head = nn.Linear(config.model_dim, config.num_note_types)

        # 농도 예측 헤드
        self.concentration_head = nn.Linear(config.model_dim, config.concentration_levels)

        # 부피 예측 헤드 (연속값)
        self.volume_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1)
        )

        # 호환성 점수 헤드
        self.compatibility_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1),
            nn.Sigmoid()
        )

        # 품질 점수 헤드
        self.quality_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1),
            nn.Sigmoid()
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        출력 헤드 계산

        Args:
            hidden_states: (batch, seq_len, model_dim)

        Returns:
            출력 딕셔너리
        """
        outputs = {}

        # 향료 로짓
        outputs['note_logits'] = self.note_head(hidden_states)

        # 노트 타입 로짓
        outputs['note_type_logits'] = self.note_type_head(hidden_states)

        # 농도 로짓
        outputs['concentration_logits'] = self.concentration_head(hidden_states)

        # 부피 예측
        outputs['volume_predictions'] = self.volume_head(hidden_states).squeeze(-1)

        # 호환성 점수
        outputs['compatibility_scores'] = self.compatibility_head(hidden_states).squeeze(-1)

        # 품질 점수
        outputs['quality_scores'] = self.quality_head(hidden_states).squeeze(-1)

        return outputs

class AdvancedFragranceTransformer(nn.Module):
    """고급 향수 생성 트랜스포머 모델"""

    def __init__(self, config: AdvancedTransformerConfig):
        super().__init__()
        self.config = config

        # 임베딩 레이어
        self.embeddings = FragranceEmbedding(config)

        # 트랜스포머 블록들
        self.layers = nn.ModuleList([
            FragranceTransformerBlock(config) for _ in range(config.num_layers)
        ])

        # 최종 레이어 정규화
        self.final_layer_norm = nn.LayerNorm(config.model_dim, eps=config.layer_norm_eps)

        # 출력 헤드
        self.output_head = FragranceOutputHead(config)

        # 조건부 인코더 (멀티모달 입력용)
        self.condition_encoder = nn.Sequential(
            nn.Linear(1024, config.model_dim),  # 멀티모달 입력 차원
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, config.model_dim)
        )

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                note_type_ids: Optional[torch.Tensor] = None,
                concentration_ids: Optional[torch.Tensor] = None,
                condition_embeddings: Optional[torch.Tensor] = None,
                season_ids: Optional[torch.Tensor] = None,
                emotion_ids: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None,
                labels: Optional[Dict[str, torch.Tensor]] = None,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_dict: bool = True) -> Union[Dict[str, torch.Tensor], Tuple]:
        """
        트랜스포머 전방 전파

        Args:
            input_ids: 입력 토큰 ID (batch, seq_len)
            attention_mask: 어텐션 마스크 (batch, seq_len)
            note_type_ids: 노트 타입 ID
            concentration_ids: 농도 ID
            condition_embeddings: 조건부 임베딩 (batch, condition_seq_len, model_dim)
            season_ids: 계절 ID
            emotion_ids: 감정 ID
            time_ids: 시간대 ID
            labels: 학습용 라벨 딕셔너리
            output_attentions: 어텐션 출력 여부
            output_hidden_states: 히든 스테이트 출력 여부
            return_dict: 딕셔너리 형태 반환 여부

        Returns:
            모델 출력
        """
        batch_size, seq_len = input_ids.shape

        # 임베딩 계산
        hidden_states = self.embeddings(
            input_ids=input_ids,
            note_type_ids=note_type_ids,
            concentration_ids=concentration_ids,
            season_ids=season_ids,
            emotion_ids=emotion_ids,
            time_ids=time_ids
        )

        # 조건부 인코더 처리
        encoder_hidden_states = None
        encoder_attention_mask = None
        if condition_embeddings is not None:
            encoder_hidden_states = self.condition_encoder(condition_embeddings)
            encoder_attention_mask = torch.ones(
                batch_size, encoder_hidden_states.shape[1],
                device=encoder_hidden_states.device
            )

        # 어텐션 마스크 생성
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        # 인과적 마스크 생성 (디코더용)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0).unsqueeze(0)

        # 트랜스포머 레이어들
        all_hidden_states = []
        all_attentions = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, layer_attentions, _ = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                note_type_ids=note_type_ids,
                output_attentions=output_attentions
            )

            if output_attentions:
                all_attentions.append(layer_attentions)

        # 최종 레이어 정규화
        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # 출력 헤드
        outputs = self.output_head(hidden_states)

        # 손실 계산 (학습 시)
        loss = None
        if labels is not None:
            loss = self._calculate_loss(outputs, labels, attention_mask)

        # 반환값 구성
        if not return_dict:
            output_tuple = (outputs['note_logits'],)
            if loss is not None:
                output_tuple = (loss,) + output_tuple
            return output_tuple

        result = {
            'loss': loss,
            'logits': outputs['note_logits'],
            'note_type_logits': outputs['note_type_logits'],
            'concentration_logits': outputs['concentration_logits'],
            'volume_predictions': outputs['volume_predictions'],
            'compatibility_scores': outputs['compatibility_scores'],
            'quality_scores': outputs['quality_scores'],
            'last_hidden_state': hidden_states
        }

        if output_hidden_states:
            result['hidden_states'] = all_hidden_states

        if output_attentions:
            result['attentions'] = all_attentions

        return result

    def _calculate_loss(self,
                       outputs: Dict[str, torch.Tensor],
                       labels: Dict[str, torch.Tensor],
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """손실 함수 계산"""
        total_loss = 0.0
        num_losses = 0

        # 향료 예측 손실 (Cross Entropy)
        if 'note_labels' in labels:
            note_loss = F.cross_entropy(
                outputs['note_logits'].view(-1, self.config.vocab_size),
                labels['note_labels'].view(-1),
                ignore_index=self.config.pad_token_id,
                reduction='mean'
            )
            total_loss += note_loss
            num_losses += 1

        # 노트 타입 예측 손실
        if 'note_type_labels' in labels:
            note_type_loss = F.cross_entropy(
                outputs['note_type_logits'].view(-1, self.config.num_note_types),
                labels['note_type_labels'].view(-1),
                reduction='mean'
            )
            total_loss += note_type_loss
            num_losses += 1

        # 농도 예측 손실
        if 'concentration_labels' in labels:
            concentration_loss = F.cross_entropy(
                outputs['concentration_logits'].view(-1, self.config.concentration_levels),
                labels['concentration_labels'].view(-1),
                reduction='mean'
            )
            total_loss += concentration_loss
            num_losses += 1

        # 부피 예측 손실 (MSE)
        if 'volume_labels' in labels:
            volume_mask = attention_mask.view(-1)
            volume_loss = F.mse_loss(
                outputs['volume_predictions'].view(-1)[volume_mask == 1],
                labels['volume_labels'].view(-1)[volume_mask == 1],
                reduction='mean'
            )
            total_loss += volume_loss
            num_losses += 1

        # 호환성 점수 손실
        if 'compatibility_labels' in labels:
            compatibility_mask = attention_mask.view(-1)
            compatibility_loss = F.binary_cross_entropy(
                outputs['compatibility_scores'].view(-1)[compatibility_mask == 1],
                labels['compatibility_labels'].view(-1)[compatibility_mask == 1],
                reduction='mean'
            )
            total_loss += compatibility_loss
            num_losses += 1

        # 품질 점수 손실
        if 'quality_labels' in labels:
            quality_mask = attention_mask.view(-1)
            quality_loss = F.mse_loss(
                outputs['quality_scores'].view(-1)[quality_mask == 1],
                labels['quality_labels'].view(-1)[quality_mask == 1],
                reduction='mean'
            )
            total_loss += quality_loss
            num_losses += 1

        return total_loss / max(num_losses, 1)

    def generate(self,
                 input_ids: torch.Tensor,
                 condition_embeddings: Optional[torch.Tensor] = None,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 do_sample: bool = True,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        향수 레시피 생성

        Args:
            input_ids: 시작 토큰 시퀀스 (batch, start_seq_len)
            condition_embeddings: 조건부 임베딩
            max_length: 최대 생성 길이
            temperature: 샘플링 온도
            top_k: Top-k 샘플링
            top_p: Nucleus 샘플링
            repetition_penalty: 반복 페널티
            do_sample: 샘플링 여부
            pad_token_id: 패딩 토큰 ID
            eos_token_id: 종료 토큰 ID

        Returns:
            생성된 시퀀스 (batch, generated_seq_len)
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 생성 루프
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_length):
            # 모델 포워드
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated,
                    condition_embeddings=condition_embeddings,
                    return_dict=True
                )

            # 다음 토큰 로짓
            next_token_logits = outputs['logits'][:, -1, :]

            # 반복 페널티 적용
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in set(generated[batch_idx].tolist()):
                        next_token_logits[batch_idx, token_id] /= repetition_penalty

            # 샘플링
            if do_sample:
                # 온도 적용
                next_token_logits = next_token_logits / temperature

                # Top-k 필터링
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) 필터링
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # 누적 확률이 top_p를 초과하는 토큰들 제거
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 그리디 디코딩
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # 시퀀스에 추가
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            # EOS 토큰으로 종료
            if (next_tokens == eos_token_id).all():
                break

        return generated

def create_advanced_fragrance_transformer(config_path: Optional[str] = None) -> AdvancedFragranceTransformer:
    """고급 향수 생성 트랜스포머 모델 팩토리"""

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = AdvancedTransformerConfig(**config_dict)
    else:
        config = AdvancedTransformerConfig()

    model = AdvancedFragranceTransformer(config)

    logger.info(f"고급 향수 생성 트랜스포머 모델 생성 완료")
    logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"설정: {config}")

    return model

if __name__ == "__main__":
    # 모델 테스트
    config = AdvancedTransformerConfig()
    model = create_advanced_fragrance_transformer()

    # 테스트 입력
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    note_type_ids = torch.randint(0, config.num_note_types, (batch_size, seq_len))
    concentration_ids = torch.randint(0, config.concentration_levels, (batch_size, seq_len))
    condition_embeddings = torch.randn(batch_size, 16, 1024)

    # 전방 전파 테스트
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            note_type_ids=note_type_ids,
            concentration_ids=concentration_ids,
            condition_embeddings=condition_embeddings,
            return_dict=True
        )

    print("모델 출력:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

    # 생성 테스트
    start_tokens = torch.tensor([[config.bos_token_id]], dtype=torch.long).repeat(batch_size, 1)
    generated = model.generate(
        input_ids=start_tokens,
        condition_embeddings=condition_embeddings,
        max_length=50,
        do_sample=True
    )

    print(f"\n생성된 시퀀스 shape: {generated.shape}")
    print("고급 향수 생성 트랜스포머 모델 테스트 완료!")