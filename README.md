# Denoising and Deconvolution Project README

## 1. 과제 목표 (Goal)

이 프로젝트의 최종 목표는 **`dataset/test_y`** 폴더의 이미지들에 적용된 미지의 열화(degradation)를 복원하여, 원본 이미지인 **`dataset/label`** 과 가장 유사하게 만드는 것입니다.

## 2. 데이터셋 역할 분석 (Dataset Roles)

-   **`train/`**, **`val/`**: 모델 학습 및 검증에 사용되는 깨끗한 원본 이미지들.
-   **`test_y/`**: **[문제지]** 최종적으로 복원해야 할 대상. 알 수 없는 파라미터로 열화가 적용되어 있습니다.
-   **`label/`**: **[정답지]** `test_y` 이미지들의 원본. 최종 성능 평가에 사용됩니다.
-   **`test_y_v2/`**: **[힌트]** 열화 방식을 분석할 수 있도록 친절하게 파라미터 정보(컨볼루션 방향, 노이즈 레벨)를 포함한 샘플입니다. **학습이나 평가에는 직접 사용되지 않습니다.**
-   **`test_1/`**, **`test_1_noise/`**, **`test_1_conv/`**: **[자체 제작 학습 데이터]** `train` 원본 이미지에 `test_y_v2`에서 분석한 열화 방식을 적용하여 우리가 직접 생성한 학습용 데이터셋입니다.

## 3. 핵심 열화 분석 (Degradation Analysis)

`test_y_v2` 샘플을 분석한 결과, 이 프로젝트의 핵심 열화는 다음 두 가지로 구성됩니다.
1.  **컨볼루션 (Convolution):** `dataset/forward_simulator.py`의 `dipole_kernel`을 사용한 흐림 효과.
2.  **노이즈 (Noise):** 가우시안 노이즈. 분석된 노이즈 레벨은 표준편차(σ) 기준으로 **Level 1: σ≈0.070**, **Level 2: σ≈0.132** 입니다.

## 4. 최종 복원 워크플로우 (Final Restoration Workflow)

이 과제의 전체 프로세스는 '모델 학습' -> '이미지 복원' -> '성능 평가'의 3단계로 구성됩니다.

![Workflow Diagram](https://mermaid.ink/svg/eyJjb2RlIjoiZ3JhcGggVERcbiAgICBzdWJncmFwaCBcIjEuIG1vZGVsIHRyYWluaW5nKFxuICAgICAgICBBW3RyYWluIFx1YzY5MFxucmJvbl0gLS0-IEJ7XHU1MmY5XHUxZGE1IFx1Yzg1Y1xucm91YVxuaWNcbnRyYWluLnB5KX07XG4gICAgICAgIENbdGVzdF8xX25vaXNlXG5cdTUzZjlfXHUxZGE1XHU3NWFjXG5kYXRhXSAtLT4gQjtcbiAgICAgICAgQiAtLT4gRCoqXHU1MmY5XHUxZGE1XHU4YmY0XG5EZW5vaXNpbmcgXHU4YmY0XHU1ZGVmXG5iZXN0LmNrcHQqKjtcbiAgICBlbmRcblxuICAgIHN1YmdyYXBoIFwiMi4gcmVzdG9yYXRpb24oKFxuICAgICAgICBFW3Rlc3RfeVxuXHU4YmY0XHU1ZGVmXG5cdWQxYWNfYl0gLS0-IEZ7XHU4YmY0XHU1ZGVmIFx1YzgyY1xucm91YVxuaWNcbnRlc3QucHkpfTtcbiAgICAgICAgRCAtLT4gRjtcbiAgICAgICAgRiAtLT4gR1tyZXN1bHRzIFx1YzllNFxucmRlclxuKFx1ODJmNFx1NWRlZiBcdTY2Y2RcbnVjNjcwKV07XG4gICAgZW5kXG5cbiAgICBzdWJncmFwaCBcIjMuIGV2YWx1YXRpb24oKVxuICAgICAgICBHIC0tPiBIe1x1YzgxNVxuc2VuZ2RlXG5wcmVkaWN0ZVxuaWNcbmV2YWx1YXRlLmlweW5iKX07XG4gICAgICAgIElbbGFiZWxcblx1YzgxNVxuc2VuZ2RlXG5cdWM4MTZfYl0gLS0-IEg7XG4gICAgICAgIEggLS0-IEooKFx1Y2Y1ZVxuc2VvZ1xuXHU1MTRkXHUxcmFkXG5QU05SL1NTSU0pKTtcbiAgICBlbmRcblxuICAgIHN0eWxlIEIgZmlsbDojZTNmMmZkLHN0cm9rZTojMzMzLHN0cm9rZS13aWR0aDoycHhcbiAgICBzdHlsZSBGIGZpbGw6I2U4ZWNmMixzdHJva2U6IzMzMyxzdHJva2Utd2lkdGg6MnB4XG4gICAgc3R5bGUgSCBmaWxsOiNmZmVlYjMsc3Ryb2tlOiMzMzMsc3Ryb2tlLXdpZHRoOjJweFxuXG4gICAgICIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)

1.  **1단계: 모델 학습**
    -   `train` 데이터와 우리가 생성한 `test_1_*` 데이터셋을 사용하여 `code_denoising/train.py` 스크립트로 모델을 학습시킵니다.
    -   학습이 완료되면 가장 성능이 좋은 모델 가중치가 `best.ckpt` 파일로 저장됩니다.

2.  **2단계: 이미지 복원**
    -   `code_denoising/test.py` 스크립트를 사용합니다.
    -   학습된 모델(`best.ckpt`)을 불러온 뒤, **`test_y`** 폴더의 이미지들을 입력으로 넣어 복원을 수행합니다.
    -   복원된 결과 이미지들을 별도의 폴더(예: `results`)에 저장합니다.

3.  **3단계: 성능 평가**
    -   `dataset/evaluate.ipynb` 노트북을 사용합니다.
    -   2단계에서 생성된 `results` 폴더와 정답지인 `label` 폴더를 비교하여 최종 성능(PSNR, SSIM)을 채점합니다.

## 5. 실험 로드맵 (Experiment Roadmap)

최종 목표는 다양한 모델과 방법론을 점진적으로 테스트하여 최고의 성능을 내는 조합을 찾는 것입니다.

### 5.1. 핵심 전략

1.  **통제된 비교 실험:** 모든 모델(DnCNN, U-Net 등)은 **동일한 '통제된' 데이터 증강 방식** 하에서 학습하고 평가하여 공정한 성능을 비교합니다.
2.  **점진적 최적화:** 초기에는 빠른 가능성 타진을 위해 **'에폭별 순환'** 데이터 증강 방식을 사용하고, 가장 성능이 좋은 모델 아키텍처가 선정되면 최종적으로 **'모든 조합 활용'** 방식을 적용하여 성능을 극한으로 끌어올립니다.

> ***(주의) 과거 실수 기록:***
> -   ***초기 On-the-fly Augmentation 구현 시, 사용자님의 '통제된 환경' 요구사항을 무시하고 '완전 무작위' 방식으로 임의 구현하여 실험의 재현성을 해치는 심각한 오류를 범했음.***
> -   ***초기 계획 수립 시, 제안된 'Diffusion' 모델에 대해 구현 방안을 설계하지 않고 임의로 계획에서 생략하는 오류를 범했음.***

### 5.2. 세부 실행 계획

#### Phase 1: End-to-End 모델 아키텍처 비교 (현재 진행 단계)

-   **목표:** '통제된(에폭별 순환)' 환경에서 DnCNN과 U-Net의 End-to-End 성능을 비교하여 더 유망한 아키텍처를 선별합니다.
-   **공통 적용 사항:** `ReduceLROnPlateau` 학습률 스케줄러를 적용하여 과적합을 방지합니다.
-   **(1-1) End-to-End: DnCNN (통제된 ver.)**
-   **(1-2) End-to-End: U-Net (통제된 ver.)**

#### Phase 2: Step-by-Step 접근법 탐색

-   **목표:** Phase 1에서 더 우수했던 아키텍처를 기반으로, Denoising과 Deconvolution을 분리하는 Step-by-Step 방식의 성능을 확인합니다.
-   **(2-1) Step-by-Step: Denoising(선정 모델) + Deconvolution(선정 모델)**
-   **(2-2) Step-by-Step: Denoising(선정 모델) + Deconvolution(Least Square)**

#### Phase 3: 최종 모델 최적화 및 Diffusion 도입 (보류)

-   **목표:** 가장 성능이 좋았던 모델과 접근법에 대해 '모든 조합 활용' 데이터 증강을 적용하여 최종 성능을 극대화하고, Diffusion 모델 구현 및 실험을 진행합니다.

---

## 6. 실험 관리 전략 (Experiment Management Strategy)

다양한 모델, 데이터셋, 학습 방식을 체계적으로 실험하고 재현성을 확보하기 위해 다음 전략을 따릅니다.

1.  **파일 버전 관리 (File Versioning):**
    *   **원본 파일 유지:** `train.py`, `datawrapper.py`와 같은 핵심 원본 스크립트는 수정하지 않고 보존합니다.
    *   **파생 스크립트 생성:** 특정 실험을 위한 코드는 `train_{모델명}_{방식}.py` (예: `train_dncnn_controlled.py`)와 같이 명확한 이름의 파일을 새로 생성하여 관리합니다. 이를 통해 각 파일의 역할을 명확히 구분합니다.
    *   **파라미터 외부 주입:** 노이즈 레벨, 학습률 등 주요 하이퍼파라미터는 `params.py`를 직접 수정하는 대신, Colab 노트북에서 스크립트 실행 시 커맨드 라인 인자(argument)로 전달하여 유연성을 확보합니다.

2.  **Colab 노트북 관리 (Colab Notebook Management):**
    *   **실험 단위로 노트북 생성:** `'모델 + 데이터 + 방식'` 조합처럼, 특정 실험 하나당 하나의 Colab 노트북을 생성합니다. (예: `colab_train_dncnn_controlled_noise_v1.ipynb`)
    *   **(중요) 데이터 로딩 최적화:** Colab에서 학습 시, 전체 `dataset` 폴더를 복사하는 대신 **`train`과 `val` 폴더와 같이 학습에 필수적인 데이터만** 로컬 런타임으로 복사하여 I/O 병목 현상을 최소화하고 준비 시간을 단축합니다.

3.  **로그 및 결과 관리 (Log & Result Management):**
    *   실행 스크립트가 `logs/{실험명}/{날짜}` 구조로 로그와 체크포인트 파일을 자동으로 생성하므로, 각 실험 결과를 명확하게 추적하고 비교합니다.

---

### Advanced Course (여력 확보 시): 품질 우선 실험

-   **목표:** 강한 노이즈와 링잉(Ringing) 현상을 억제하는 데 초점을 맞춘 고급 하이브리드 파이프라인을 테스트합니다.
-   **Denoise:** Diffusion (5–10 steps)
-   **Deconvolution:** PnP (Plug-and-Play) (3–5 iterations)
-   **Fusion:** 대역 결합 (Frequency-based merging) 및 가중 평균
