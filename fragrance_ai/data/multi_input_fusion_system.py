"""
다중 입력 융합 시스템 (Multi-Input Fusion System)
Universal Input Processing for Fragrance Generation

이 모듈은 텍스트, 이미지, 음성, 센서 데이터, 생체 신호 등
모든 형태의 입력을 향수 생성에 활용할 수 있도록 처리하는 시스템입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchaudio
import numpy as np
import cv2
import librosa
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import re
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

@dataclass
class InputProcessingConfig:
    """입력 처리 설정 클래스"""

    # 텍스트 처리 설정
    max_text_length: int = 512
    language_models: List[str] = None

    # 이미지 처리 설정
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # 오디오 처리 설정
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512

    # 센서 데이터 설정
    sensor_window_size: int = 100
    sensor_sampling_rate: int = 50

    # 생체 신호 설정
    ecg_sampling_rate: int = 250
    eeg_channels: int = 64
    gsr_baseline_duration: int = 60

    def __post_init__(self):
        if self.language_models is None:
            self.language_models = ['klue/bert-base', 'klue/roberta-base']

class TextProcessor:
    """텍스트 입력 처리기"""

    def __init__(self, config: InputProcessingConfig):
        self.config = config

        # 다국어 처리를 위한 토크나이저들
        from transformers import AutoTokenizer
        self.tokenizers = {}
        for model_name in config.language_models:
            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"토크나이저 로드 실패 {model_name}: {e}")

        # 감정 분석을 위한 키워드 사전
        self.emotion_keywords = {
            'romantic': ['로맨틱', '사랑', '연인', '데이트', '로맨스', 'romantic', 'love', 'date'],
            'fresh': ['상쾌', '시원', '프레시', '깨끗', 'fresh', 'clean', 'cool', 'crisp'],
            'warm': ['따뜻', '포근', '아늑', '편안', 'warm', 'cozy', 'comfort', 'soft'],
            'energetic': ['활기', '에너지', '활력', '역동', 'energetic', 'vibrant', 'dynamic'],
            'calming': ['진정', '평온', '안정', '릴렉스', 'calm', 'peaceful', 'relax', 'serene'],
            'sophisticated': ['세련', '우아', '고급', '품격', 'sophisticated', 'elegant', 'refined'],
            'mysterious': ['신비', '미스터리', '어둠', '깊이', 'mysterious', 'dark', 'deep', 'enigmatic']
        }

        # 계절 키워드
        self.seasonal_keywords = {
            'spring': ['봄', '벚꽃', '새싹', '따뜻', 'spring', 'cherry', 'blossom', 'bloom'],
            'summer': ['여름', '바다', '휴가', '시원', 'summer', 'ocean', 'beach', 'vacation'],
            'autumn': ['가을', '단풍', '추수', '코스모스', 'autumn', 'fall', 'maple', 'harvest'],
            'winter': ['겨울', '눈', '크리스마스', '따뜻함', 'winter', 'snow', 'christmas', 'cozy']
        }

        # 시간대 키워드
        self.time_keywords = {
            'morning': ['아침', '새벽', '일출', 'morning', 'dawn', 'sunrise'],
            'afternoon': ['오후', '점심', '낮', 'afternoon', 'noon', 'daytime'],
            'evening': ['저녁', '황혼', '석양', 'evening', 'twilight', 'sunset'],
            'night': ['밤', '자정', '달빛', 'night', 'midnight', 'moonlight']
        }

    def process(self, text_input: Union[str, Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        텍스트 입력을 처리하여 향수 생성용 특성 추출

        Args:
            text_input: 문자열 또는 다국어 텍스트 딕셔너리

        Returns:
            처리된 텍스트 특성 딕셔너리
        """
        if isinstance(text_input, str):
            texts = {'main': text_input}
        else:
            texts = text_input

        results = {}

        for key, text in texts.items():
            # 기본 토크나이제이션
            tokenized = self._tokenize_text(text)
            results[f'{key}_tokens'] = tokenized

            # 감정 분석
            emotions = self._analyze_emotions(text)
            results[f'{key}_emotions'] = emotions

            # 계절성 분석
            seasonality = self._analyze_seasonality(text)
            results[f'{key}_seasonality'] = seasonality

            # 시간대 분석
            time_context = self._analyze_time_context(text)
            results[f'{key}_time'] = time_context

            # 향료 관련 키워드 추출
            fragrance_keywords = self._extract_fragrance_keywords(text)
            results[f'{key}_fragrance_hints'] = fragrance_keywords

            # 문화적 맥락 분석
            cultural_context = self._analyze_cultural_context(text)
            results[f'{key}_cultural'] = cultural_context

        return results

    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """텍스트 토크나이제이션"""
        # 주요 토크나이저 사용
        primary_tokenizer = list(self.tokenizers.values())[0]

        encoded = primary_tokenizer(
            text,
            max_length=self.config.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def _analyze_emotions(self, text: str) -> torch.Tensor:
        """감정 분석"""
        text_lower = text.lower()
        emotion_scores = torch.zeros(len(self.emotion_keywords))

        for i, (emotion, keywords) in enumerate(self.emotion_keywords.items()):
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[i] = score

        # 정규화
        if emotion_scores.sum() > 0:
            emotion_scores = emotion_scores / emotion_scores.sum()

        return emotion_scores

    def _analyze_seasonality(self, text: str) -> torch.Tensor:
        """계절성 분석"""
        text_lower = text.lower()
        seasonal_scores = torch.zeros(len(self.seasonal_keywords))

        for i, (season, keywords) in enumerate(self.seasonal_keywords.items()):
            score = sum(1 for keyword in keywords if keyword in text_lower)
            seasonal_scores[i] = score

        # 정규화
        if seasonal_scores.sum() > 0:
            seasonal_scores = seasonal_scores / seasonal_scores.sum()

        return seasonal_scores

    def _analyze_time_context(self, text: str) -> torch.Tensor:
        """시간대 분석"""
        text_lower = text.lower()
        time_scores = torch.zeros(len(self.time_keywords))

        for i, (time_period, keywords) in enumerate(self.time_keywords.items()):
            score = sum(1 for keyword in keywords if keyword in text_lower)
            time_scores[i] = score

        # 정규화
        if time_scores.sum() > 0:
            time_scores = time_scores / time_scores.sum()

        return time_scores

    def _extract_fragrance_keywords(self, text: str) -> torch.Tensor:
        """향료 관련 키워드 추출"""
        # 향료명 사전 (실제로는 더 많은 향료들이 포함되어야 함)
        fragrance_notes = [
            'rose', '장미', 'jasmine', '자스민', 'vanilla', '바닐라',
            'sandalwood', '샌달우드', 'bergamot', '베르가못', 'lavender', '라벤더',
            'citrus', '시트러스', 'musk', '머스크', 'amber', '앰버'
        ]

        text_lower = text.lower()
        presence_vector = torch.zeros(len(fragrance_notes))

        for i, note in enumerate(fragrance_notes):
            if note in text_lower:
                presence_vector[i] = 1.0

        return presence_vector

    def _analyze_cultural_context(self, text: str) -> torch.Tensor:
        """문화적 맥락 분석"""
        cultural_markers = {
            'korean': ['한국', '전통', '한옥', '한복', '김치', 'korea', 'korean', 'hanbok'],
            'japanese': ['일본', '사쿠라', '와사비', 'japan', 'japanese', 'sakura'],
            'french': ['프랑스', '파리', '샹젤리제', 'france', 'french', 'paris'],
            'american': ['미국', '뉴욕', '할리우드', 'america', 'american', 'new york'],
            'chinese': ['중국', '베이징', '용', 'china', 'chinese', 'beijing']
        }

        text_lower = text.lower()
        cultural_scores = torch.zeros(len(cultural_markers))

        for i, (culture, markers) in enumerate(cultural_markers.items()):
            score = sum(1 for marker in markers if marker in text_lower)
            cultural_scores[i] = score

        return cultural_scores

class ImageProcessor:
    """이미지 입력 처리기"""

    def __init__(self, config: InputProcessingConfig):
        self.config = config

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])

        # 색상 분석을 위한 설정
        self.dominant_colors_count = 10

    def process(self, image_input: Union[str, np.ndarray, Image.Image, bytes]) -> Dict[str, torch.Tensor]:
        """
        이미지 입력을 처리하여 향수 생성용 특성 추출

        Args:
            image_input: 이미지 경로, numpy 배열, PIL 이미지, 또는 바이트 데이터

        Returns:
            처리된 이미지 특성 딕셔너리
        """
        # 이미지 로드 및 변환
        image = self._load_image(image_input)

        results = {}

        # 기본 이미지 텐서
        image_tensor = self.transform(image).unsqueeze(0)
        results['image_tensor'] = image_tensor.squeeze(0)

        # 색상 분석
        color_features = self._analyze_colors(image)
        results['color_palette'] = color_features['dominant_colors']
        results['color_harmony'] = color_features['harmony_score']
        results['brightness'] = color_features['brightness']
        results['saturation'] = color_features['saturation']

        # 질감 분석
        texture_features = self._analyze_texture(image)
        results['texture_features'] = texture_features

        # 구성 분석
        composition_features = self._analyze_composition(image)
        results['composition_features'] = composition_features

        # 감정적 톤 분석
        emotional_tone = self._analyze_emotional_tone(image)
        results['emotional_tone'] = emotional_tone

        # 계절성 추정
        seasonality = self._estimate_seasonality(image)
        results['visual_seasonality'] = seasonality

        return results

    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image, bytes]) -> Image.Image:
        """다양한 형태의 이미지 입력을 PIL 이미지로 변환"""
        if isinstance(image_input, str):
            # 파일 경로 또는 base64 문자열
            if image_input.startswith('data:image'):
                # base64 인코딩된 이미지
                header, data = image_input.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(BytesIO(image_data)).convert('RGB')
            else:
                # 파일 경로
                return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # numpy 배열
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                return Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
            else:
                raise ValueError("Numpy 배열은 (H, W, 3) 형태여야 합니다")
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        elif isinstance(image_input, bytes):
            return Image.open(BytesIO(image_input)).convert('RGB')
        else:
            raise ValueError(f"지원하지 않는 이미지 입력 타입: {type(image_input)}")

    def _analyze_colors(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """색상 분석"""
        # PIL 이미지를 numpy 배열로 변환
        img_array = np.array(image)

        # 주요 색상 추출
        pixels = img_array.reshape(-1, 3)

        # K-means를 사용한 주요 색상 추출 (간단한 버전)
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.dominant_colors_count, random_state=42)
        kmeans.fit(pixels)

        dominant_colors = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32) / 255.0

        # 색상 조화도 계산 (색상환에서의 거리 기반)
        harmony_score = self._calculate_color_harmony(dominant_colors)

        # 전체 밝기와 채도 계산
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image)

        brightness = torch.tensor(hsv_array[:, :, 2].mean() / 255.0)
        saturation = torch.tensor(hsv_array[:, :, 1].mean() / 255.0)

        return {
            'dominant_colors': dominant_colors,
            'harmony_score': harmony_score,
            'brightness': brightness,
            'saturation': saturation
        }

    def _calculate_color_harmony(self, colors: torch.Tensor) -> torch.Tensor:
        """색상 조화도 계산"""
        # RGB를 HSV로 변환하여 색상환에서의 각도 계산
        # 간단한 근사치 사용
        color_angles = torch.atan2(colors[:, 1] - colors[:, 0], colors[:, 2] - colors[:, 1])

        # 색상 간 각도 차이의 표준편차로 조화도 측정
        angle_std = torch.std(color_angles)
        harmony_score = torch.exp(-angle_std)  # 표준편차가 낮을수록 조화롭다고 가정

        return harmony_score

    def _analyze_texture(self, image: Image.Image) -> torch.Tensor:
        """질감 분석"""
        # 그레이스케일로 변환
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)

        # 텍스처 특성 계산
        # 1. 에지 강도 (Sobel 필터)
        from scipy import ndimage
        sobel_x = ndimage.sobel(gray_array, axis=1)
        sobel_y = ndimage.sobel(gray_array, axis=0)
        edge_strength = np.sqrt(sobel_x**2 + sobel_y**2).mean()

        # 2. 로컬 표준편차 (텍스처의 복잡성)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray_array.astype(float), size=5)
        local_variance = uniform_filter(gray_array.astype(float)**2, size=5) - local_mean**2
        texture_complexity = np.sqrt(local_variance).mean()

        # 3. 그라디언트 방향 히스토그램
        gradient_direction = np.arctan2(sobel_y, sobel_x)
        direction_hist, _ = np.histogram(gradient_direction.flatten(), bins=8, range=(-np.pi, np.pi))
        direction_features = direction_hist / direction_hist.sum()

        # 특성 벡터 구성
        texture_features = torch.tensor([
            edge_strength / 255.0,  # 정규화
            texture_complexity / 255.0,
            *direction_features
        ], dtype=torch.float32)

        return texture_features

    def _analyze_composition(self, image: Image.Image) -> torch.Tensor:
        """구성 분석"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # 1. 대칭성 분석
        left_half = img_array[:, :w//2]
        right_half = np.fliplr(img_array[:, w//2:])

        # 크기를 맞춤
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0

        # 2. 중심 집중도
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_distance = np.sqrt(center_y**2 + center_x**2)

        # 중심에서 멀어질수록 가중치 감소
        weights = 1 - (distances / max_distance)
        weighted_brightness = np.average(np.mean(img_array, axis=2), weights=weights)
        center_focus = weighted_brightness / 255.0

        # 3. 수직/수평 균형
        top_half = img_array[:h//2, :]
        bottom_half = img_array[h//2:, :]
        vertical_balance = 1.0 - abs(np.mean(top_half) - np.mean(bottom_half)) / 255.0

        composition_features = torch.tensor([
            symmetry_score,
            center_focus,
            vertical_balance
        ], dtype=torch.float32)

        return composition_features

    def _analyze_emotional_tone(self, image: Image.Image) -> torch.Tensor:
        """감정적 톤 분석"""
        img_array = np.array(image)

        # 색온도 분석
        r_mean = np.mean(img_array[:, :, 0])
        b_mean = np.mean(img_array[:, :, 2])
        warmth = (r_mean - b_mean) / 255.0  # 빨강이 많으면 따뜻함

        # 전체적인 밝기
        brightness = np.mean(img_array) / 255.0

        # 대비도
        contrast = np.std(img_array) / 255.0

        # 채도
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image)
        saturation = np.mean(hsv_array[:, :, 1]) / 255.0

        # 감정 차원으로 매핑
        # 발랜스-활력 차원
        valence = (brightness + saturation) / 2.0

        # 각성-평온 차원
        arousal = (contrast + saturation) / 2.0

        # 따뜻함-차가움 차원
        temperature = (warmth + 1.0) / 2.0  # -1~1을 0~1로 변환

        emotional_tone = torch.tensor([valence, arousal, temperature], dtype=torch.float32)

        return emotional_tone

    def _estimate_seasonality(self, image: Image.Image) -> torch.Tensor:
        """시각적 계절성 추정"""
        img_array = np.array(image)

        # 색상 기반 계절 추정
        # 봄: 연분홍, 연녹색
        spring_colors = np.array([[255, 182, 193], [144, 238, 144], [255, 255, 224]])  # 핑크, 라이트그린, 연노랑

        # 여름: 밝은 청록, 생생한 녹색
        summer_colors = np.array([[0, 206, 209], [34, 139, 34], [255, 215, 0]])  # 터키옥, 포레스트그린, 골드

        # 가을: 주황, 갈색, 빨강
        autumn_colors = np.array([[255, 140, 0], [165, 42, 42], [220, 20, 60]])  # 오렌지, 브라운, 크림슨

        # 겨울: 흰색, 파랑, 회색
        winter_colors = np.array([[255, 255, 255], [70, 130, 180], [128, 128, 128]])  # 화이트, 스틸블루, 그레이

        seasonal_colors = [spring_colors, summer_colors, autumn_colors, winter_colors]
        seasonal_scores = torch.zeros(4)

        # 각 계절별 색상과의 유사도 계산
        img_flat = img_array.reshape(-1, 3)

        for season_idx, season_palette in enumerate(seasonal_colors):
            total_similarity = 0
            for color in season_palette:
                # 각 픽셀과 계절 색상의 거리 계산
                distances = np.linalg.norm(img_flat - color, axis=1)
                similarity = np.exp(-distances / 100.0)  # 거리를 유사도로 변환
                total_similarity += np.mean(similarity)

            seasonal_scores[season_idx] = total_similarity / len(season_palette)

        # 정규화
        if seasonal_scores.sum() > 0:
            seasonal_scores = seasonal_scores / seasonal_scores.sum()

        return seasonal_scores

class AudioProcessor:
    """오디오 입력 처리기"""

    def __init__(self, config: InputProcessingConfig):
        self.config = config

    def process(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        오디오 입력을 처리하여 향수 생성용 특성 추출

        Args:
            audio_input: 오디오 파일 경로, numpy 배열, 또는 torch 텐서

        Returns:
            처리된 오디오 특성 딕셔너리
        """
        # 오디오 로드 및 전처리
        audio_data, sample_rate = self._load_audio(audio_input)

        results = {}

        # 기본 오디오 텐서
        results['audio_waveform'] = torch.tensor(audio_data, dtype=torch.float32)

        # 멜 스펙트로그램
        mel_spectrogram = self._compute_mel_spectrogram(audio_data, sample_rate)
        results['mel_spectrogram'] = mel_spectrogram

        # 음향 특성 분석
        acoustic_features = self._analyze_acoustic_features(audio_data, sample_rate)
        results.update(acoustic_features)

        # 감정 분석
        emotional_features = self._analyze_audio_emotions(audio_data, sample_rate)
        results['audio_emotions'] = emotional_features

        # 리듬 분석
        rhythm_features = self._analyze_rhythm(audio_data, sample_rate)
        results['rhythm_features'] = rhythm_features

        # 환경음 분석
        environmental_features = self._analyze_environmental_sounds(audio_data, sample_rate)
        results['environmental_features'] = environmental_features

        return results

    def _load_audio(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, int]:
        """오디오 로드"""
        if isinstance(audio_input, str):
            # 파일 경로
            audio_data, sample_rate = librosa.load(audio_input, sr=self.config.sample_rate)
        elif isinstance(audio_input, np.ndarray):
            audio_data = audio_input
            sample_rate = self.config.sample_rate
        elif isinstance(audio_input, torch.Tensor):
            audio_data = audio_input.numpy()
            sample_rate = self.config.sample_rate
        else:
            raise ValueError(f"지원하지 않는 오디오 입력 타입: {type(audio_input)}")

        return audio_data, sample_rate

    def _compute_mel_spectrogram(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """멜 스펙트로그램 계산"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )

        # 로그 스케일로 변환
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return torch.tensor(log_mel_spec, dtype=torch.float32)

    def _analyze_acoustic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, torch.Tensor]:
        """음향 특성 분석"""
        features = {}

        # 1. 스펙트럴 특성
        # 스펙트럴 중심
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid'] = torch.tensor(np.mean(spectral_centroids))

        # 스펙트럴 롤오프
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
        features['spectral_rolloff'] = torch.tensor(np.mean(spectral_rolloff))

        # 스펙트럴 대역폭
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)[0]
        features['spectral_bandwidth'] = torch.tensor(np.mean(spectral_bandwidth))

        # 2. 시간 영역 특성
        # RMS 에너지
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        features['rms_energy'] = torch.tensor(np.mean(rms_energy))

        # 제로 크로싱 비율
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zero_crossing_rate'] = torch.tensor(np.mean(zero_crossing_rate))

        # 3. MFCC 특성
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = torch.tensor(np.mean(mfccs, axis=1))
        features['mfcc_std'] = torch.tensor(np.std(mfccs, axis=1))

        return features

    def _analyze_audio_emotions(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """오디오 감정 분석"""
        # 감정 차원: 발랜스(긍정-부정), 활성화(활발-조용), 지배(강함-약함)

        # 발랜스: 주파수 분포와 조화도 기반
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        valence = min(1.0, spectral_centroid / 4000.0)  # 4kHz를 기준으로 정규화

        # 활성화: 에너지와 템포 기반
        rms_energy = np.mean(librosa.feature.rms(y=audio_data))
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        arousal = min(1.0, (rms_energy * 10 + tempo / 200.0) / 2.0)

        # 지배: 다이나믹 레인지와 스펙트럴 롤오프 기반
        dynamic_range = np.max(audio_data) - np.min(audio_data)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate))
        dominance = min(1.0, (dynamic_range * 2 + spectral_rolloff / 8000.0) / 2.0)

        return torch.tensor([valence, arousal, dominance], dtype=torch.float32)

    def _analyze_rhythm(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """리듬 분석"""
        # 템포 추출
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)

        # 비트 강도 변화
        beat_frames = librosa.util.fix_frames(beats, x_min=0, x_max=len(audio_data))
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)

        if len(beat_times) > 1:
            # 비트 간격의 변화량 (리듬의 규칙성)
            beat_intervals = np.diff(beat_times)
            rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals))
        else:
            rhythm_regularity = 0.0

        # 리듬적 복잡성
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sample_rate)
        rhythm_complexity = min(1.0, len(onset_frames) / (len(audio_data) / sample_rate * 10))  # 초당 10개를 최대로 가정

        rhythm_features = torch.tensor([
            tempo / 200.0,  # 200 BPM을 1.0으로 정규화
            rhythm_regularity,
            rhythm_complexity
        ], dtype=torch.float32)

        return rhythm_features

    def _analyze_environmental_sounds(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """환경음 분석"""
        # 환경음 카테고리별 특성 추출

        # 자연음 특성 (새소리, 바람소리, 물소리 등)
        # 고주파 성분이 많고 불규칙한 패턴
        high_freq_energy = np.mean(librosa.stft(audio_data, n_fft=2048)[-512:, :])  # 상위 1/4 주파수 대역
        irregularity = np.std(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
        nature_score = min(1.0, (abs(high_freq_energy) / 100.0 + irregularity / 1000.0) / 2.0)

        # 도시음 특성 (교통소음, 기계음 등)
        # 저주파 성분과 규칙적 패턴
        low_freq_energy = np.mean(librosa.stft(audio_data, n_fft=2048)[:512, :])  # 하위 1/4 주파수 대역
        regularity = 1.0 / (1.0 + irregularity / 1000.0)
        urban_score = min(1.0, (abs(low_freq_energy) / 100.0 + regularity) / 2.0)

        # 실내음 특성 (에어컨, 전자기기 등)
        # 중간 주파수 대역의 일정한 톤
        mid_freq_energy = np.mean(librosa.stft(audio_data, n_fft=2048)[512:1536, :])
        consistency = 1.0 - np.std(librosa.feature.rms(y=audio_data)) / np.mean(librosa.feature.rms(y=audio_data))
        indoor_score = min(1.0, (abs(mid_freq_energy) / 100.0 + consistency) / 2.0)

        environmental_features = torch.tensor([
            nature_score,
            urban_score,
            indoor_score
        ], dtype=torch.float32)

        return environmental_features

class SensorProcessor:
    """센서 데이터 처리기"""

    def __init__(self, config: InputProcessingConfig):
        self.config = config

    def process(self, sensor_data: Dict[str, Union[np.ndarray, List, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        센서 데이터를 처리하여 향수 생성용 특성 추출

        Args:
            sensor_data: 센서 데이터 딕셔너리
                - temperature: 온도 데이터
                - humidity: 습도 데이터
                - pressure: 기압 데이터
                - light: 조도 데이터
                - motion: 움직임 데이터
                - air_quality: 공기질 데이터

        Returns:
            처리된 센서 특성 딕셔너리
        """
        results = {}

        # 환경 센서 처리
        if 'temperature' in sensor_data:
            temp_features = self._process_temperature(sensor_data['temperature'])
            results['temperature_features'] = temp_features

        if 'humidity' in sensor_data:
            humidity_features = self._process_humidity(sensor_data['humidity'])
            results['humidity_features'] = humidity_features

        if 'pressure' in sensor_data:
            pressure_features = self._process_pressure(sensor_data['pressure'])
            results['pressure_features'] = pressure_features

        if 'light' in sensor_data:
            light_features = self._process_light(sensor_data['light'])
            results['light_features'] = light_features

        # 활동 센서 처리
        if 'motion' in sensor_data:
            motion_features = self._process_motion(sensor_data['motion'])
            results['motion_features'] = motion_features

        # 공기질 센서 처리
        if 'air_quality' in sensor_data:
            air_features = self._process_air_quality(sensor_data['air_quality'])
            results['air_quality_features'] = air_features

        # 통합 환경 컨텍스트 생성
        environmental_context = self._create_environmental_context(results)
        results['environmental_context'] = environmental_context

        return results

    def _process_temperature(self, temp_data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """온도 데이터 처리"""
        if not isinstance(temp_data, torch.Tensor):
            temp_data = torch.tensor(temp_data, dtype=torch.float32)

        # 통계적 특성
        mean_temp = torch.mean(temp_data)
        std_temp = torch.std(temp_data)
        min_temp = torch.min(temp_data)
        max_temp = torch.max(temp_data)

        # 정규화 (0-40도 범위 가정)
        normalized_mean = torch.clamp(mean_temp / 40.0, 0, 1)
        normalized_variation = torch.clamp(std_temp / 10.0, 0, 1)

        # 쾌적성 점수 (20-26도가 최적)
        comfort_score = 1.0 - torch.abs(mean_temp - 23.0) / 23.0
        comfort_score = torch.clamp(comfort_score, 0, 1)

        return torch.tensor([normalized_mean, normalized_variation, comfort_score])

    def _process_humidity(self, humidity_data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """습도 데이터 처리"""
        if not isinstance(humidity_data, torch.Tensor):
            humidity_data = torch.tensor(humidity_data, dtype=torch.float32)

        mean_humidity = torch.mean(humidity_data)
        std_humidity = torch.std(humidity_data)

        # 정규화 (0-100% 범위)
        normalized_mean = mean_humidity / 100.0
        normalized_variation = std_humidity / 50.0  # 50%를 최대 변화량으로 가정

        # 쾌적성 점수 (40-60%가 최적)
        comfort_score = 1.0 - torch.abs(mean_humidity - 50.0) / 50.0
        comfort_score = torch.clamp(comfort_score, 0, 1)

        return torch.tensor([normalized_mean, normalized_variation, comfort_score])

    def _process_pressure(self, pressure_data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """기압 데이터 처리"""
        if not isinstance(pressure_data, torch.Tensor):
            pressure_data = torch.tensor(pressure_data, dtype=torch.float32)

        mean_pressure = torch.mean(pressure_data)
        pressure_trend = torch.mean(torch.diff(pressure_data))  # 기압 변화 추세

        # 정규화 (980-1040 hPa 범위 가정)
        normalized_pressure = (mean_pressure - 980.0) / 60.0
        normalized_trend = torch.clamp(pressure_trend / 10.0, -1, 1)  # ±10 hPa 변화를 ±1로

        # 안정성 점수
        stability = 1.0 / (1.0 + torch.abs(pressure_trend))

        return torch.tensor([normalized_pressure, normalized_trend, stability])

    def _process_light(self, light_data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """조도 데이터 처리"""
        if not isinstance(light_data, torch.Tensor):
            light_data = torch.tensor(light_data, dtype=torch.float32)

        mean_light = torch.mean(light_data)
        light_variation = torch.std(light_data)

        # 정규화 (0-1000 lux 범위 가정)
        normalized_light = torch.clamp(mean_light / 1000.0, 0, 1)
        normalized_variation = torch.clamp(light_variation / 500.0, 0, 1)

        # 시간대 추정 (조도 기반)
        if mean_light > 500:
            time_context = torch.tensor([1.0, 0.0, 0.0])  # 낮
        elif mean_light > 100:
            time_context = torch.tensor([0.0, 1.0, 0.0])  # 저녁/새벽
        else:
            time_context = torch.tensor([0.0, 0.0, 1.0])  # 밤

        return torch.cat([torch.tensor([normalized_light, normalized_variation]), time_context])

    def _process_motion(self, motion_data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """움직임 데이터 처리"""
        if not isinstance(motion_data, torch.Tensor):
            motion_data = torch.tensor(motion_data, dtype=torch.float32)

        # 움직임이 3축 데이터인 경우 크기 계산
        if len(motion_data.shape) > 1 and motion_data.shape[-1] == 3:
            motion_magnitude = torch.norm(motion_data, dim=-1)
        else:
            motion_magnitude = motion_data

        # 활동 수준 특성
        mean_activity = torch.mean(motion_magnitude)
        activity_peaks = torch.sum(motion_magnitude > torch.mean(motion_magnitude) + torch.std(motion_magnitude))
        activity_consistency = 1.0 / (1.0 + torch.std(motion_magnitude))

        # 정규화
        normalized_activity = torch.clamp(mean_activity / 10.0, 0, 1)  # 10을 최대 활동으로 가정
        peak_frequency = activity_peaks / len(motion_magnitude)

        return torch.tensor([normalized_activity, peak_frequency, activity_consistency])

    def _process_air_quality(self, air_data: Dict[str, Union[np.ndarray, List, torch.Tensor]]) -> torch.Tensor:
        """공기질 데이터 처리"""
        features = []

        # PM2.5 처리
        if 'pm25' in air_data:
            pm25 = torch.tensor(air_data['pm25'], dtype=torch.float32)
            pm25_normalized = torch.clamp(torch.mean(pm25) / 100.0, 0, 1)  # 100을 최대로 가정
            features.append(pm25_normalized)

        # PM10 처리
        if 'pm10' in air_data:
            pm10 = torch.tensor(air_data['pm10'], dtype=torch.float32)
            pm10_normalized = torch.clamp(torch.mean(pm10) / 200.0, 0, 1)  # 200을 최대로 가정
            features.append(pm10_normalized)

        # CO2 처리
        if 'co2' in air_data:
            co2 = torch.tensor(air_data['co2'], dtype=torch.float32)
            co2_normalized = torch.clamp((torch.mean(co2) - 400.0) / 1600.0, 0, 1)  # 400-2000 ppm 범위
            features.append(co2_normalized)

        # VOC 처리
        if 'voc' in air_data:
            voc = torch.tensor(air_data['voc'], dtype=torch.float32)
            voc_normalized = torch.clamp(torch.mean(voc) / 1000.0, 0, 1)  # 1000을 최대로 가정
            features.append(voc_normalized)

        if not features:
            return torch.zeros(4)  # 기본값

        # 부족한 특성은 0으로 패딩
        while len(features) < 4:
            features.append(torch.tensor(0.0))

        return torch.stack(features)

    def _create_environmental_context(self, processed_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """처리된 센서 특성들을 통합하여 환경 컨텍스트 생성"""
        context_features = []

        # 각 센서 특성에서 대표값 추출
        for key, features in processed_features.items():
            if key != 'environmental_context':  # 재귀 방지
                if isinstance(features, torch.Tensor):
                    # 특성의 평균값 사용
                    context_features.append(torch.mean(features))

        if not context_features:
            return torch.zeros(8)  # 기본 컨텍스트 크기

        # 8차원으로 패딩 또는 축소
        context_tensor = torch.stack(context_features)
        if len(context_tensor) > 8:
            context_tensor = context_tensor[:8]
        elif len(context_tensor) < 8:
            padding = torch.zeros(8 - len(context_tensor))
            context_tensor = torch.cat([context_tensor, padding])

        return context_tensor

class UniversalInputFusion:
    """범용 입력 융합 시스템"""

    def __init__(self, config: InputProcessingConfig):
        self.config = config

        # 각 모달리티 처리기 초기화
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.sensor_processor = SensorProcessor(config)

    def process_all_inputs(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        모든 입력 모달리티를 처리하고 융합

        Args:
            inputs: 다양한 입력들의 딕셔너리
                - text: 텍스트 입력
                - image: 이미지 입력
                - audio: 오디오 입력
                - sensor: 센서 데이터
                - metadata: 메타데이터 (시간, 위치 등)

        Returns:
            융합된 특성 딕셔너리
        """
        results = {}

        # 텍스트 처리
        if 'text' in inputs and inputs['text'] is not None:
            try:
                text_features = self.text_processor.process(inputs['text'])
                results.update({f'text_{k}': v for k, v in text_features.items()})
                logger.info("텍스트 처리 완료")
            except Exception as e:
                logger.error(f"텍스트 처리 오류: {e}")

        # 이미지 처리
        if 'image' in inputs and inputs['image'] is not None:
            try:
                image_features = self.image_processor.process(inputs['image'])
                results.update({f'image_{k}': v for k, v in image_features.items()})
                logger.info("이미지 처리 완료")
            except Exception as e:
                logger.error(f"이미지 처리 오류: {e}")

        # 오디오 처리
        if 'audio' in inputs and inputs['audio'] is not None:
            try:
                audio_features = self.audio_processor.process(inputs['audio'])
                results.update({f'audio_{k}': v for k, v in audio_features.items()})
                logger.info("오디오 처리 완료")
            except Exception as e:
                logger.error(f"오디오 처리 오류: {e}")

        # 센서 데이터 처리
        if 'sensor' in inputs and inputs['sensor'] is not None:
            try:
                sensor_features = self.sensor_processor.process(inputs['sensor'])
                results.update({f'sensor_{k}': v for k, v in sensor_features.items()})
                logger.info("센서 데이터 처리 완료")
            except Exception as e:
                logger.error(f"센서 데이터 처리 오류: {e}")

        # 메타데이터 처리
        if 'metadata' in inputs and inputs['metadata'] is not None:
            try:
                metadata_features = self._process_metadata(inputs['metadata'])
                results.update({f'metadata_{k}': v for k, v in metadata_features.items()})
                logger.info("메타데이터 처리 완료")
            except Exception as e:
                logger.error(f"메타데이터 처리 오류: {e}")

        # 교차 모달리티 특성 추출
        cross_modal_features = self._extract_cross_modal_features(results)
        results.update(cross_modal_features)

        return results

    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """메타데이터 처리"""
        features = {}

        # 시간 정보 처리
        if 'timestamp' in metadata:
            time_features = self._process_timestamp(metadata['timestamp'])
            features.update(time_features)

        # 위치 정보 처리
        if 'location' in metadata:
            location_features = self._process_location(metadata['location'])
            features.update(location_features)

        # 사용자 프로필 처리
        if 'user_profile' in metadata:
            profile_features = self._process_user_profile(metadata['user_profile'])
            features.update(profile_features)

        # 컨텍스트 정보 처리
        if 'context' in metadata:
            context_features = self._process_context(metadata['context'])
            features.update(context_features)

        return features

    def _process_timestamp(self, timestamp: Union[str, datetime, int]) -> Dict[str, torch.Tensor]:
        """타임스탬프 처리"""
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, int):
            dt = datetime.fromtimestamp(timestamp)
        else:
            dt = timestamp

        # 시간 특성 추출
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)

        day_sin = np.sin(2 * np.pi * dt.timetuple().tm_yday / 365)
        day_cos = np.cos(2 * np.pi * dt.timetuple().tm_yday / 365)

        weekday = dt.weekday() / 6.0  # 0-1 범위로 정규화

        return {
            'time_cyclical': torch.tensor([hour_sin, hour_cos, day_sin, day_cos]),
            'weekday': torch.tensor(weekday)
        }

    def _process_location(self, location: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """위치 정보 처리"""
        lat = location.get('latitude', 0.0)
        lon = location.get('longitude', 0.0)

        # 위도 정규화 (-90 ~ 90 -> -1 ~ 1)
        lat_normalized = lat / 90.0

        # 경도 순환 인코딩 (-180 ~ 180)
        lon_sin = np.sin(2 * np.pi * lon / 360.0)
        lon_cos = np.cos(2 * np.pi * lon / 360.0)

        # 기후대 추정 (위도 기반)
        if abs(lat) < 23.5:
            climate_zone = torch.tensor([1.0, 0.0, 0.0])  # 열대
        elif abs(lat) < 66.5:
            climate_zone = torch.tensor([0.0, 1.0, 0.0])  # 온대
        else:
            climate_zone = torch.tensor([0.0, 0.0, 1.0])  # 극지

        return {
            'location_coords': torch.tensor([lat_normalized, lon_sin, lon_cos]),
            'climate_zone': climate_zone
        }

    def _process_user_profile(self, profile: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """사용자 프로필 처리"""
        features = {}

        # 나이 정규화
        if 'age' in profile:
            age_normalized = min(1.0, profile['age'] / 100.0)
            features['age'] = torch.tensor(age_normalized)

        # 성별 인코딩
        if 'gender' in profile:
            gender_encoding = {
                'male': torch.tensor([1.0, 0.0, 0.0]),
                'female': torch.tensor([0.0, 1.0, 0.0]),
                'other': torch.tensor([0.0, 0.0, 1.0])
            }
            features['gender'] = gender_encoding.get(profile['gender'], torch.tensor([0.0, 0.0, 1.0]))

        # 선호도
        if 'preferences' in profile:
            pref_vector = torch.zeros(10)  # 10차원 선호도 벡터
            for i, pref in enumerate(profile['preferences'][:10]):
                pref_vector[i] = float(pref)
            features['preferences'] = pref_vector

        return features

    def _process_context(self, context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """컨텍스트 정보 처리"""
        features = {}

        # 상황 컨텍스트
        if 'occasion' in context:
            occasion_encoding = {
                'work': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                'date': torch.tensor([0.0, 1.0, 0.0, 0.0]),
                'casual': torch.tensor([0.0, 0.0, 1.0, 0.0]),
                'formal': torch.tensor([0.0, 0.0, 0.0, 1.0])
            }
            features['occasion'] = occasion_encoding.get(context['occasion'], torch.tensor([0.0, 0.0, 1.0, 0.0]))

        # 계절 정보
        if 'season' in context:
            season_encoding = {
                'spring': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                'summer': torch.tensor([0.0, 1.0, 0.0, 0.0]),
                'autumn': torch.tensor([0.0, 0.0, 1.0, 0.0]),
                'winter': torch.tensor([0.0, 0.0, 0.0, 1.0])
            }
            features['season'] = season_encoding.get(context['season'], torch.tensor([0.25, 0.25, 0.25, 0.25]))

        return features

    def _extract_cross_modal_features(self, all_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """교차 모달리티 특성 추출"""
        cross_features = {}

        # 텍스트-이미지 일치도
        if any(k.startswith('text_') for k in all_features) and any(k.startswith('image_') for k in all_features):
            text_emotions = all_features.get('text_main_emotions')
            image_emotions = all_features.get('image_emotional_tone')

            if text_emotions is not None and image_emotions is not None:
                # 감정 일치도 계산
                emotion_similarity = F.cosine_similarity(
                    text_emotions[:min(len(text_emotions), len(image_emotions))].unsqueeze(0),
                    image_emotions[:min(len(text_emotions), len(image_emotions))].unsqueeze(0)
                )
                cross_features['text_image_emotion_alignment'] = emotion_similarity.squeeze()

        # 오디오-환경 일치도
        if any(k.startswith('audio_') for k in all_features) and any(k.startswith('sensor_') for k in all_features):
            audio_env = all_features.get('audio_environmental_features')
            sensor_env = all_features.get('sensor_environmental_context')

            if audio_env is not None and sensor_env is not None:
                # 환경 일치도 계산
                min_len = min(len(audio_env), len(sensor_env))
                env_similarity = F.cosine_similarity(
                    audio_env[:min_len].unsqueeze(0),
                    sensor_env[:min_len].unsqueeze(0)
                )
                cross_features['audio_sensor_environment_alignment'] = env_similarity.squeeze()

        # 전체 모달리티 일관성
        modal_count = sum(1 for k in all_features.keys() if any(k.startswith(prefix) for prefix in ['text_', 'image_', 'audio_', 'sensor_']))
        cross_features['modality_richness'] = torch.tensor(modal_count / 4.0)  # 4개 모달리티 대비 비율

        return cross_features

# 팩토리 함수
def create_universal_input_fusion(config_path: Optional[str] = None) -> UniversalInputFusion:
    """범용 입력 융합 시스템 생성"""

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = InputProcessingConfig(**config_dict)
    else:
        config = InputProcessingConfig()

    fusion_system = UniversalInputFusion(config)

    logger.info("범용 입력 융합 시스템 초기화 완료")

    return fusion_system

if __name__ == "__main__":
    # 시스템 테스트
    fusion_system = create_universal_input_fusion()

    # 테스트 입력
    test_inputs = {
        'text': "따뜻하고 로맨틱한 봄날의 향수를 만들어주세요",
        'metadata': {
            'timestamp': datetime.now(),
            'location': {'latitude': 37.5665, 'longitude': 126.9780},  # 서울
            'context': {'occasion': 'date', 'season': 'spring'},
            'user_profile': {'age': 25, 'gender': 'female', 'preferences': [0.8, 0.6, 0.9, 0.3, 0.7]}
        }
    }

    # 처리 실행
    results = fusion_system.process_all_inputs(test_inputs)

    print("처리 결과:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape} - {value[:3] if len(value) > 3 else value}")
        else:
            print(f"{key}: {value}")

    print(f"\n총 {len(results)}개의 특성이 추출되었습니다.")