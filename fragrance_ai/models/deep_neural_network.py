"""
Deep Neural Network for Advanced Fragrance Generation
더욱 깊고 복잡한 신경망 아키텍처 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """커스텀 멀티헤드 셀프 어텐션"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        # Query, Key, Value 계산
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.out_proj(context)
        return output


class PositionalEncoding(nn.Module):
    """위치 인코딩"""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """커스텀 트랜스포머 인코더 레이어"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)

        return x


class DeepFragranceEncoder(nn.Module):
    """깊은 향수 인코더 네트워크"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1024
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Token embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class ConditionalFragranceDecoder(nn.Module):
    """조건부 향수 디코더"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        condition_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # 조건 임베딩
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for _ in range(num_layers)
        ])

        # 출력 레이어
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoded_features: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        # 조건 정보 임베딩
        condition_embed = self.condition_embedding(conditions)

        # 인코더 출력과 조건 결합
        # encoded_features: [batch, seq_len, embed_dim]
        # condition_embed: [batch, embed_dim]
        condition_expanded = condition_embed.unsqueeze(1).expand_as(encoded_features)
        combined = encoded_features + condition_expanded

        # 디코더 레이어들 통과
        for layer in self.decoder_layers:
            residual = combined
            combined = layer(combined) + residual

        # 출력 확률 분포
        logits = self.output_projection(combined)
        return logits


class DeepFragranceNetwork(nn.Module):
    """전체 깊은 향수 생성 네트워크"""

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 512,
        encoder_layers: int = 8,
        decoder_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        condition_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = DeepFragranceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        self.decoder = ConditionalFragranceDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            condition_dim=condition_dim,
            num_layers=decoder_layers,
            dropout=dropout
        )

        # 파라미터 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        conditions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 인코더 통과
        encoded = self.encoder(input_ids, attention_mask)

        # 디코더로 조건부 생성
        logits = self.decoder(encoded, conditions)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        conditions: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """조건부 텍스트 생성"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_length):
                # 현재 시퀀스로 예측
                logits = self.forward(input_ids, conditions)

                # 마지막 토큰의 로짓
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k 필터링
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Top-p 필터링
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                # 다음 토큰 샘플링
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 시퀀스에 추가
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class AdvancedOptimizer:
    """고급 옵티마이저 클래스"""

    def __init__(self, model: nn.Module, lr: float = 1e-4):
        self.model = model
        self.lr = lr

        # 다양한 옵티마이저 설정
        self.optimizers = {
            'adamw': optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999)),
            'sgd_momentum': optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4),
            'rmsprop': optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-8),
            'adagrad': optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=0),
        }

        # 학습률 스케줄러
        self.schedulers = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR(self.optimizers['adamw'], T_max=100),
            'step': optim.lr_scheduler.StepLR(self.optimizers['adamw'], step_size=30, gamma=0.1),
            'exponential': optim.lr_scheduler.ExponentialLR(self.optimizers['adamw'], gamma=0.95),
            'plateau': optim.lr_scheduler.ReduceLROnPlateau(self.optimizers['adamw'], mode='min', patience=5)
        }

        self.current_optimizer = 'adamw'
        self.current_scheduler = 'cosine'

    def step(self, loss: torch.Tensor):
        """옵티마이저 스텝"""
        optimizer = self.optimizers[self.current_optimizer]
        optimizer.zero_grad()
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        # 스케줄러 업데이트
        if self.current_scheduler in ['cosine', 'step', 'exponential']:
            self.schedulers[self.current_scheduler].step()
        elif self.current_scheduler == 'plateau':
            self.schedulers[self.current_scheduler].step(loss)

    def switch_optimizer(self, optimizer_name: str):
        """옵티마이저 변경"""
        if optimizer_name in self.optimizers:
            self.current_optimizer = optimizer_name
            print(f"Switched to {optimizer_name} optimizer")

    def get_lr(self) -> float:
        """현재 학습률 반환"""
        return self.optimizers[self.current_optimizer].param_groups[0]['lr']


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 깊은 신경망 테스트
    model = DeepFragranceNetwork(
        vocab_size=10000,
        embed_dim=512,
        encoder_layers=8,
        decoder_layers=6,
        num_heads=8,
        ff_dim=2048,
        condition_dim=256
    )

    print(f"Deep Neural Network created with {count_parameters(model):,} parameters")

    # 테스트 입력
    batch_size = 2
    seq_len = 50
    condition_dim = 256

    test_input = torch.randint(0, 10000, (batch_size, seq_len))
    test_conditions = torch.randn(batch_size, condition_dim)

    # Forward pass 테스트
    with torch.no_grad():
        output = model(test_input, test_conditions)
        print(f"Output shape: {output.shape}")
        print("Deep neural network forward pass successful!")