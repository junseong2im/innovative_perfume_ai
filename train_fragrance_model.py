"""
향수 AI 모델 학습 스크립트
기존 데이터를 사용하여 모델 학습
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
from tqdm import tqdm
import os
import glob

class FragranceDataset(Dataset):
    """향수 데이터셋"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 입력과 출력 텍스트 생성
        input_text = item['input']
        output = item['output']

        # 출력을 텍스트로 변환
        output_text = self.format_output(output)

        # 전체 텍스트
        full_text = f"질문: {input_text}\n답변: {output_text}"

        # 토크나이징
        encoding = self.tokenizer(
            full_text,
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

    def format_output(self, output):
        """출력 딕셔너리를 텍스트로 변환"""
        text = f"{output['korean_name']} ({output['fragrance_name']})\n"
        text += f"컨셉: {output['concept']}\n"

        if 'notes_breakdown' in output:
            notes = output['notes_breakdown']
            if 'top_notes' in notes:
                top = ', '.join([n['note'] for n in notes['top_notes']])
                text += f"탑노트: {top}\n"
            if 'middle_notes' in notes:
                middle = ', '.join([n['note'] for n in notes['middle_notes']])
                text += f"미들노트: {middle}\n"
            if 'base_notes' in notes:
                base = ', '.join([n['note'] for n in notes['base_notes']])
                text += f"베이스노트: {base}\n"

        if 'mood' in output:
            text += f"무드: {', '.join(output['mood'])}\n"

        return text

def load_all_data():
    """모든 향수 데이터 로드"""
    all_data = []

    # 모든 향수 관련 JSON 파일 찾기
    data_files = glob.glob('data/**/*.json', recursive=True)

    print("Searching for fragrance data files...")

    for file_path in data_files:
        if 'fragrance' in file_path.lower() or 'perfume' in file_path.lower():
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 데이터 구조에 따라 처리 - 모든 데이터 사용
                        if 'multi_domain_training_data' in data:
                            all_data.extend(data['multi_domain_training_data'])
                            print(f"  Added {len(data['multi_domain_training_data'])} samples")
                        elif 'training_data' in data:
                            all_data.extend(data['training_data'])
                            print(f"  Added {len(data['training_data'])} samples")
                        elif 'fragrance_notes' in data:
                            # fragrance_notes 데이터를 학습 데이터로 변환
                            for category, cat_data in data['fragrance_notes'].items():
                                if 'notes' in cat_data:
                                    for note in cat_data['notes']:
                                        sample = {
                                            'input': f'{category} 계열의 향료를 추천해주세요',
                                            'output': {
                                                'fragrance_name': note.get('name', 'Unknown'),
                                                'korean_name': note.get('korean_name', note.get('name', 'Unknown')),
                                                'concept': note.get('description', ''),
                                                'notes_breakdown': {
                                                    'top_notes': [{'note': note.get('name', ''), 'percentage': 100}],
                                                    'middle_notes': [],
                                                    'base_notes': []
                                                },
                                                'mood': [category.lower()]
                                            }
                                        }
                                        all_data.append(sample)
                            print(f"  Converted fragrance notes to {len(all_data)} samples")
                except json.JSONDecodeError as e:
                    print(f"  JSON error in {file_path}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
                    continue

    # 데이터가 없으면 기본 데이터 생성
    if not all_data:
        print("Creating default training data...")
        all_data = [
            {
                'input': '여름에 어울리는 상쾌한 향수를 만들어주세요',
                'output': {
                    'fragrance_name': 'Summer Breeze',
                    'korean_name': '여름 바람',
                    'concept': '상쾌한 시트러스와 해양의 조화',
                    'notes_breakdown': {
                        'top_notes': [{'note': 'bergamot', 'percentage': 30}],
                        'middle_notes': [{'note': 'sea_breeze', 'percentage': 40}],
                        'base_notes': [{'note': 'white_musk', 'percentage': 30}]
                    },
                    'mood': ['fresh', 'energetic']
                }
            }
        ]

    print(f"Total training samples: {len(all_data)}")
    return all_data

def train_model():
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 토크나이저와 모델 로드 - 영어 GPT2 직접 사용
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print("Loaded GPT2 model")

    # 패딩 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    # 데이터 로드
    print("Loading training data...")
    train_data = load_all_data()

    # 데이터셋과 데이터로더 생성
    dataset = FragranceDataset(train_data, tokenizer)
    batch_size = min(4, len(train_data))  # 데이터 크기에 따라 배치 조정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 학습
    print("Starting training...")
    model.train()

    num_epochs = 50  # 데이터가 적으므로 충분히 학습
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Average loss: {avg_loss:.4f}")

        # 베스트 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss! Saving model...")
            os.makedirs('models', exist_ok=True)
            model.save_pretrained('models/fragrance_gpt2_best')
            tokenizer.save_pretrained('models/fragrance_gpt2_best')

        # Early stopping
        if avg_loss < 0.5:  # 충분히 학습되면 중단
            print(f"Loss is low enough ({avg_loss:.4f}), stopping early")
            break

    # 모델 저장
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    model.save_pretrained('models/fragrance_gpt2')
    tokenizer.save_pretrained('models/fragrance_gpt2')
    print("Model saved to models/fragrance_gpt2")

def test_model():
    """학습된 모델 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 학습된 모델 로드
    print("Loading trained model...")
    tokenizer = GPT2Tokenizer.from_pretrained('models/fragrance_gpt2')
    model = GPT2LMHeadModel.from_pretrained('models/fragrance_gpt2')
    model.to(device)
    model.eval()

    # 테스트 질문들
    test_queries = [
        "여름에 어울리는 상쾌한 향수를 만들어주세요",
        "로맨틱한 데이트를 위한 향수를 추천해주세요",
        "프로페셔널한 비즈니스 미팅용 향수를 만들어주세요"
    ]

    print("\n=== Model Testing ===")
    for query in test_queries:
        print(f"\n질문: {query}")

        # 입력 토크나이징
        input_text = f"질문: {query}\n답변:"
        inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(input_text):]
        print(f"답변: {answer}")

if __name__ == "__main__":
    print("=== 향수 AI 모델 학습 ===")

    # 학습
    train_model()

    # 테스트
    print("\n=== 모델 테스트 ===")
    test_model()