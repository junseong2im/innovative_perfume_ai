"""
고급 옵티마이저 설정 및 관리 모듈
딥러닝 옵티마이저와 유전 알고리즘 통합
"""

from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
from transformers import TrainingArguments
from transformers.optimization import (
    AdamW,
    Adafactor,
    get_scheduler
)
import logging
import math

# MOGA 옵티마이저 임포트
try:
    from .moga_optimizer import MOGAOptimizer, create_fragrance_optimizer
    MOGA_AVAILABLE = True
except ImportError:
    MOGA_AVAILABLE = False
    logging.warning("MOGA optimizer not available")

logger = logging.getLogger(__name__)


class OptimizerConfig:
    """옵티마이저 설정 클래스"""

    def __init__(
        self,
        optimizer_type: str = "adamw_torch",
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        **kwargs
    ):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.kwargs = kwargs

    def get_optimizer_kwargs(self) -> Dict[str, Any]:
        """옵티마이저 키워드 인수 반환"""
        base_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay
        }

        if self.optimizer_type.startswith("adamw"):
            base_kwargs.update({
                "betas": (self.adam_beta1, self.adam_beta2),
                "eps": self.adam_epsilon
            })

        base_kwargs.update(self.kwargs)
        return base_kwargs


class SchedulerConfig:
    """스케줄러 설정 클래스"""

    def __init__(
        self,
        scheduler_type: str = "cosine",
        warmup_ratio: float = 0.1,
        warmup_steps: Optional[int] = None,
        num_training_steps: int = 1000,
        cosine_restarts: int = 1,
        polynomial_power: float = 1.0,
        **kwargs
    ):
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.cosine_restarts = cosine_restarts
        self.polynomial_power = polynomial_power
        self.kwargs = kwargs

    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        """스케줄러 키워드 인수 반환"""
        # warmup_steps 계산
        if self.warmup_steps is None:
            warmup_steps = int(self.num_training_steps * self.warmup_ratio)
        else:
            warmup_steps = self.warmup_steps

        base_kwargs = {
            "num_warmup_steps": warmup_steps,
            "num_training_steps": self.num_training_steps
        }

        # 스케줄러별 특별 설정
        if self.scheduler_type == "cosine_with_restarts":
            base_kwargs["num_cycles"] = self.cosine_restarts
        elif self.scheduler_type == "polynomial":
            base_kwargs["power"] = self.polynomial_power

        base_kwargs.update(self.kwargs)
        return base_kwargs


class AdvancedOptimizerManager:
    """고급 옵티마이저 관리자"""

    def __init__(self):
        self.supported_optimizers = {
            "adamw_torch": torch.optim.AdamW,
            "adamw_hf": AdamW,
            "adafactor": Adafactor,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop
        }

        self.supported_schedulers = {
            "linear": "linear",
            "cosine": "cosine",
            "cosine_with_restarts": "cosine_with_restarts",
            "polynomial": "polynomial",
            "constant": "constant",
            "constant_with_warmup": "constant_with_warmup"
        }

    def create_optimizer(
        self,
        model: nn.Module,
        optimizer_config: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        try:
            optimizer_class = self.supported_optimizers.get(optimizer_config.optimizer_type)
            if optimizer_class is None:
                raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_config.optimizer_type}")

            # 파라미터 그룹 설정 (가중치 감쇠 차별화)
            param_groups = self._create_param_groups(model, optimizer_config.weight_decay)

            # 옵티마이저 생성
            optimizer_kwargs = optimizer_config.get_optimizer_kwargs()
            optimizer = optimizer_class(param_groups, **optimizer_kwargs)

            logger.info(f"Created optimizer: {optimizer_config.optimizer_type}")
            logger.info(f"Optimizer kwargs: {optimizer_kwargs}")

            return optimizer

        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            raise

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_config: SchedulerConfig
    ):
        """스케줄러 생성"""
        try:
            scheduler_name = self.supported_schedulers.get(scheduler_config.scheduler_type)
            if scheduler_name is None:
                raise ValueError(f"지원하지 않는 스케줄러: {scheduler_config.scheduler_type}")

            scheduler_kwargs = scheduler_config.get_scheduler_kwargs()

            scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=optimizer,
                **scheduler_kwargs
            )

            logger.info(f"Created scheduler: {scheduler_config.scheduler_type}")
            logger.info(f"Scheduler kwargs: {scheduler_kwargs}")

            return scheduler

        except Exception as e:
            logger.error(f"Failed to create scheduler: {e}")
            raise

    def _create_param_groups(
        self,
        model: nn.Module,
        weight_decay: float
    ) -> List[Dict[str, Any]]:
        """파라미터 그룹 생성 (가중치 감쇠 차별화)"""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        param_groups = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": weight_decay,
                "name": "decay"
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay"
            }
        ]

        # 통계 출력
        total_params = sum(len(group["params"]) for group in param_groups)
        decay_params = len(param_groups[0]["params"])
        no_decay_params = len(param_groups[1]["params"])

        logger.info(f"Parameter groups created:")
        logger.info(f"  - Decay parameters: {decay_params}")
        logger.info(f"  - No-decay parameters: {no_decay_params}")
        logger.info(f"  - Total trainable parameters: {total_params}")

        return param_groups

    def get_training_arguments_updates(
        self,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig
    ) -> Dict[str, Any]:
        """TrainingArguments를 위한 업데이트 딕셔너리 반환"""
        updates = {
            # 옵티마이저 설정
            "optim": optimizer_config.optimizer_type,
            "learning_rate": optimizer_config.learning_rate,
            "weight_decay": optimizer_config.weight_decay,
            "max_grad_norm": optimizer_config.max_grad_norm,

            # 스케줄러 설정
            "lr_scheduler_type": scheduler_config.scheduler_type,
            "warmup_ratio": scheduler_config.warmup_ratio,
            "warmup_steps": scheduler_config.warmup_steps,
        }

        # Adam 관련 설정
        if optimizer_config.optimizer_type.startswith("adamw"):
            updates.update({
                "adam_beta1": optimizer_config.adam_beta1,
                "adam_beta2": optimizer_config.adam_beta2,
                "adam_epsilon": optimizer_config.adam_epsilon
            })

        return updates

    def calculate_num_training_steps(
        self,
        num_samples: int,
        batch_size: int,
        num_epochs: int,
        gradient_accumulation_steps: int = 1
    ) -> int:
        """총 훈련 스텝 수 계산"""
        steps_per_epoch = math.ceil(num_samples / batch_size / gradient_accumulation_steps)
        total_steps = steps_per_epoch * num_epochs
        return total_steps

    def log_optimizer_info(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler=None
    ):
        """옵티마이저 정보 로깅"""
        logger.info("=== Optimizer Configuration ===")
        logger.info(f"Optimizer type: {type(optimizer).__name__}")

        # 파라미터 그룹 정보
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"Parameter group {i}:")
            logger.info(f"  - Learning rate: {group['lr']}")
            logger.info(f"  - Weight decay: {group.get('weight_decay', 0)}")
            logger.info(f"  - Num parameters: {len(group['params'])}")

        if scheduler:
            logger.info(f"Scheduler type: {type(scheduler).__name__}")

        logger.info("================================")


class GeneticOptimizerManager:
    """유전 알고리즘 최적화 관리자"""

    def __init__(self):
        self.moga_available = MOGA_AVAILABLE

    def create_moga_optimizer(
        self,
        num_ingredients: int = 20,
        population_size: int = 100,
        max_generations: int = 50,
        objectives: Dict[str, Any] = None
    ) -> Optional[MOGAOptimizer]:
        """MOGA 옵티마이저 생성"""
        if not self.moga_available:
            logger.error("MOGA optimizer not available")
            return None

        try:
            optimizer = create_fragrance_optimizer(
                num_ingredients=num_ingredients
            )

            # 커스텀 설정 적용
            optimizer.population_size = population_size
            optimizer.max_generations = max_generations

            logger.info(f"Created MOGA optimizer with {num_ingredients} ingredients")
            logger.info(f"Population: {population_size}, Generations: {max_generations}")

            return optimizer

        except Exception as e:
            logger.error(f"Failed to create MOGA optimizer: {e}")
            return None

    def optimize_fragrance(
        self,
        optimizer: MOGAOptimizer,
        objective_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """향수 레시피 최적화 실행"""
        if not optimizer:
            return {}

        try:
            # 최적화 실행
            pareto_front = optimizer.optimize(verbose=False)

            if not pareto_front:
                logger.warning("Optimization failed - empty Pareto front")
                return {}

            # 최적 해 선택
            best_solution = optimizer.get_best_solution(objective_weights)

            result = {
                'genes': best_solution.genes.tolist(),
                'objectives': best_solution.objectives,
                'fitness': best_solution.fitness,
                'pareto_front_size': len(pareto_front),
                'generation': optimizer.generation
            }

            logger.info(f"Optimization complete: {len(pareto_front)} solutions found")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {}

    def get_optimization_stats(self, optimizer: MOGAOptimizer) -> Dict[str, Any]:
        """최적화 통계 반환"""
        if not optimizer:
            return {}

        return {
            'population_size': optimizer.population_size,
            'current_generation': optimizer.generation,
            'max_generations': optimizer.max_generations,
            'pareto_front_size': len(optimizer.pareto_front),
            'best_fitness': max([ind.fitness for ind in optimizer.population]) if optimizer.population else 0,
            'moga_enabled': True
        }

# 전역 매니저 인스턴스
optimizer_manager = AdvancedOptimizerManager()
genetic_optimizer_manager = GeneticOptimizerManager()