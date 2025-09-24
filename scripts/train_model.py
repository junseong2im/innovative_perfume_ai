#!/usr/bin/env python3
"""
향수 AI 모델 훈련 스크립트
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import wandb
from transformers import set_seed

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fragrance_ai.core.config import settings
from fragrance_ai.training.peft_trainer import PEFTTrainer
from fragrance_ai.models.generator import FragranceGenerator
from fragrance_ai.models.embedding import FragranceEmbedding
from fragrance_ai.utils.data_loader import DatasetLoader
from fragrance_ai.evaluation.metrics import EvaluationMetrics

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="향수 AI 모델 훈련")
    
    # 기본 설정
    parser.add_argument("--model-type", type=str, choices=["embedding", "generation"], 
                       required=True, help="훈련할 모델 타입")
    parser.add_argument("--config-file", type=str, help="훈련 설정 파일 경로")
    parser.add_argument("--data-path", type=str, required=True, help="훈련 데이터 경로")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="출력 디렉토리")
    
    # 모델 설정
    parser.add_argument("--base-model", type=str, help="기본 모델명")
    parser.add_argument("--max-length", type=int, default=512, help="최대 시퀀스 길이")
    
    # 훈련 설정
    parser.add_argument("--epochs", type=int, default=3, help="훈련 에폭 수")
    parser.add_argument("--batch-size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--warmup-steps", type=int, default=500, help="워밍업 스텝")
    parser.add_argument("--save-steps", type=int, default=500, help="저장 간격")
    parser.add_argument("--eval-steps", type=int, default=500, help="평가 간격")

    # 고급 옵티마이저 설정
    parser.add_argument("--optimizer", type=str, default="adamw_torch",
                       choices=["adamw_torch", "adamw_hf", "adafactor", "sgd"],
                       help="옵티마이저 타입")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="가중치 감쇠")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="그래디언트 클리핑")

    # 스케줄러 설정
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
                       help="학습률 스케줄러")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="워밍업 비율")
    parser.add_argument("--cosine-restarts", type=int, default=1, help="코사인 재시작 횟수")
    parser.add_argument("--polynomial-power", type=float, default=1.0, help="다항식 파워")
    
    # PEFT 설정
    parser.add_argument("--use-lora", action="store_true", help="LoRA 사용 여부")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 양자화 설정
    parser.add_argument("--use-4bit", action="store_true", help="4bit 양자화 사용")
    parser.add_argument("--use-8bit", action="store_true", help="8bit 양자화 사용")
    
    # 기타 설정
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--wandb-project", type=str, help="W&B 프로젝트명")
    parser.add_argument("--resume-from", type=str, help="체크포인트 재개 경로")
    parser.add_argument("--dry-run", action="store_true", help="드라이런 모드")
    
    return parser.parse_args()

def load_training_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """훈련 설정 로드"""
    default_config = {
        "embedding": {
            "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "max_length": 512,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 5,
            "warmup_ratio": 0.1
        },
        "generation": {
            "base_model": "microsoft/DialoGPT-medium",
            "max_length": 1024,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "epochs": 3,
            "warmup_steps": 500,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        }
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        
        # 기본 설정과 커스텀 설정 병합
        for model_type in default_config:
            if model_type in custom_config:
                default_config[model_type].update(custom_config[model_type])
    
    return default_config

def setup_wandb(args: argparse.Namespace, config: Dict[str, Any]):
    """W&B 초기화"""
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config={
                "model_type": args.model_type,
                "base_model": args.base_model or config[args.model_type].get("base_model"),
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "use_lora": args.use_lora,
                "use_4bit": args.use_4bit,
                "seed": args.seed,
                # 옵티마이저 설정 추가
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_epsilon": args.adam_epsilon,
                "max_grad_norm": args.max_grad_norm,
                # 스케줄러 설정 추가
                "lr_scheduler": args.lr_scheduler,
                "warmup_ratio": args.warmup_ratio,
                "warmup_steps": args.warmup_steps,
                "cosine_restarts": args.cosine_restarts,
                "polynomial_power": args.polynomial_power
            },
            name=f"{args.model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

def train_embedding_model(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """임베딩 모델 훈련"""
    logger.info("임베딩 모델 훈련 시작")
    
    # 모델 초기화
    embedding_model = FragranceEmbedding(
        model_name=args.base_model or config["embedding"]["base_model"],
        max_length=args.max_length or config["embedding"]["max_length"]
    )
    
    # 데이터 로더 초기화
    data_loader = DatasetLoader()
    train_dataset, val_dataset = data_loader.load_embedding_dataset(
        args.data_path,
        test_size=0.2,
        random_state=args.seed
    )
    
    logger.info(f"훈련 데이터: {len(train_dataset)}, 검증 데이터: {len(val_dataset)}")
    
    # 훈련 설정
    training_args = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed
    }
    
    if not args.dry_run:
        # 훈련 실행
        trainer = embedding_model.create_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_args=training_args
        )
        
        trainer.train()
        
        # 최종 모델 저장
        final_output_dir = os.path.join(args.output_dir, "final")
        embedding_model.save_model(final_output_dir)
        logger.info(f"최종 모델 저장: {final_output_dir}")
        
        return final_output_dir
    else:
        logger.info("드라이런 모드 - 실제 훈련은 수행되지 않습니다.")
        return args.output_dir

def train_generation_model(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """생성 모델 훈련"""
    logger.info("생성 모델 훈련 시작")
    
    # PEFT Trainer 초기화
    peft_trainer = PEFTTrainer(
        model_name=args.base_model or config["generation"]["base_model"],
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # LoRA 설정
    if args.use_lora:
        lora_config = {
            "r": args.lora_r or config["generation"]["lora_r"],
            "lora_alpha": args.lora_alpha or config["generation"]["lora_alpha"],
            "lora_dropout": args.lora_dropout or config["generation"]["lora_dropout"],
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        peft_trainer.setup_lora(lora_config)
    
    # 데이터 로더 초기화
    data_loader = DatasetLoader()
    train_dataset, val_dataset = data_loader.load_generation_dataset(
        args.data_path,
        max_length=args.max_length,
        test_size=0.2,
        random_state=args.seed
    )
    
    logger.info(f"훈련 데이터: {len(train_dataset)}, 검증 데이터: {len(val_dataset)}")
    
    # 훈련 설정
    training_config = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "max_steps": -1,
        "gradient_accumulation_steps": 1,
        "dataloader_drop_last": False,
        "eval_accumulation_steps": None,
        "seed": args.seed
    }
    
    if not args.dry_run:
        # 훈련 실행
        peft_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            training_config=training_config
        )
        
        # 최종 모델 저장
        final_output_dir = os.path.join(args.output_dir, "final")
        peft_trainer.save_model(final_output_dir)
        logger.info(f"최종 모델 저장: {final_output_dir}")
        
        return final_output_dir
    else:
        logger.info("드라이런 모드 - 실제 훈련은 수행되지 않습니다.")
        return args.output_dir

def evaluate_model(model_path: str, model_type: str, eval_data_path: str):
    """모델 평가"""
    logger.info(f"모델 평가 시작: {model_path}")
    
    try:
        if model_type == "embedding":
            # 임베딩 모델 평가
            embedding_model = FragranceEmbedding()
            embedding_model.load_model(model_path)
            
            # 평가 데이터 로드
            data_loader = DatasetLoader()
            eval_dataset = data_loader.load_embedding_eval_dataset(eval_data_path)
            
            # 평가 수행
            metrics = embedding_model.evaluate(eval_dataset)
            
        elif model_type == "generation":
            # 생성 모델 평가
            generator = FragranceGenerator()
            generator.load_model(model_path)
            
            # 평가 데이터 로드
            data_loader = DatasetLoader()
            eval_prompts = data_loader.load_generation_eval_dataset(eval_data_path)
            
            # 생성 및 평가
            generated_recipes = []
            for prompt in eval_prompts[:100]:  # 샘플링
                result = generator.generate(prompt, max_tokens=500)
                generated_recipes.append(result)
            
            # 메트릭 계산
            metrics = EvaluationMetrics.calculate_generation_metrics(generated_recipes)
        
        # 평가 결과 저장
        eval_results_path = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"평가 완료. 결과 저장: {eval_results_path}")
        
        # W&B에 로그
        if wandb.run:
            wandb.log({"evaluation": metrics})
        
        return metrics
        
    except Exception as e:
        logger.error(f"모델 평가 중 오류 발생: {e}")
        return None

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 설정 로드
    config = load_training_config(args.config_file)
    
    # 인수로부터 설정 업데이트
    model_config = config[args.model_type]
    if args.base_model:
        model_config["base_model"] = args.base_model
    if args.learning_rate:
        model_config["learning_rate"] = args.learning_rate
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    if args.epochs:
        model_config["epochs"] = args.epochs
    
    logger.info(f"훈련 시작 - 모델 타입: {args.model_type}")
    logger.info(f"설정: {json.dumps(model_config, ensure_ascii=False, indent=2)}")
    
    # W&B 설정
    setup_wandb(args, config)
    
    try:
        # 모델별 훈련 실행
        if args.model_type == "embedding":
            model_path = train_embedding_model(args, config)
        elif args.model_type == "generation":
            model_path = train_generation_model(args, config)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {args.model_type}")
        
        logger.info(f"훈련 완료: {model_path}")
        
        # 평가 수행
        if not args.dry_run and os.path.exists(args.data_path):
            eval_metrics = evaluate_model(model_path, args.model_type, args.data_path)
            if eval_metrics:
                logger.info("평가 메트릭:")
                for key, value in eval_metrics.items():
                    logger.info(f"  {key}: {value}")
        
        # 훈련 요약 저장
        summary = {
            "model_type": args.model_type,
            "model_path": model_path,
            "training_config": model_config,
            "training_time": datetime.now().isoformat(),
            "success": True
        }
        
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"훈련 완료! 요약: {summary_path}")
        
    except Exception as e:
        logger.error(f"훈련 중 오류 발생: {e}")
        
        # 실패 요약 저장
        summary = {
            "model_type": args.model_type,
            "error": str(e),
            "training_time": datetime.now().isoformat(),
            "success": False
        }
        
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        sys.exit(1)
    
    finally:
        # W&B 정리
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    main()