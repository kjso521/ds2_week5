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

## 7. 개발 이력 및 교훈 (Development History & Lessons Learned)

본 프로젝트는 다수의 기술적 문제와 디버깅 과정을 거쳤습니다. 핵심적인 문제 해결 과정을 기록하여 향후 유사한 실수를 방지하고자 합니다.

- **초기 환경 설정 오류**: 로컬 Windows 환경과 Colab 환경의 차이(`pathlib` 경로 문제, `num_workers` 설정 등)로 인해 초기 학습에 어려움을 겪었으며, `num_workers=0` 설정 및 `pathlib`을 통한 경로 처리로 해결했습니다.
- **Git-Colab 동기화 문제**: `.gitignore`에 의해 `dataset` 폴더가 제외된 상태에서, Colab의 `git clone` 로직이 데이터셋 경로를 찾지 못하는 문제가 발생했습니다. 최종적으로 Colab 실행 시 Google Drive의 고정된 프로젝트 폴더로 이동 후 `git pull`로 코드만 업데이트하는 안정적인 'Upversioning' 방식으로 전환하여 해결했습니다.
- **산재된 설정 파일 문제**: 프로젝트 구조 리팩토링 과정에서 `code_denoising` 폴더 내에 구버전 `params.py`가 남아있어, 최상위 폴더의 신버전 `params.py`를 가리는(shadowing) 현상이 발생했습니다. 이로 인해 `ImportError`가 지속적으로 발생했으며, 불필요한 설정 파일을 삭제하여 해결했습니다.
- **반복적인 Import 경로 오류**: 코드 리팩토링 과정에서 `core_funcs.py`, `train_controlled.py` 등의 파일에서 다른 모듈을 불러오는 `import` 구문의 경로 설정(절대 경로 vs 상대 경로)을 일관성 있게 처리하지 못해 수많은 `ImportError`와 `ModuleNotFoundError`가 발생했습니다. 최종적으로 모든 스크립트가 프로젝트 최상위 폴더를 기준으로 모듈을 찾도록 `sys.path` 설정 및 절대/상대 경로를 명확히 구분하여 해결했습니다. 이 과정에서 **"하나의 파일을 수정하면, 그 파일을 참조하는 모든 다른 파일에 미치는 영향을 반드시 함께 점검해야 한다"**는 중요한 교훈을 얻었습니다.
- **노트북/스크립트 역할 분리 실패**: 초기에는 `run_evaluation.ipynb` 하나의 노트북에서 End-to-End와 Step-by-Step 평가를 모두 처리하려다 코드가 복잡해지고 새로운 버그가 발생하는 악순환을 겪었습니다. 사용자님의 지적에 따라, **"잘 되던 코드는 그대로 두고, 새로운 기능은 새로운 파일로 분리한다"**는 원칙을 적용하여 `run_evaluation_sbs.ipynb`를 새로 생성함으로써 문제를 해결했습니다.

## 8. 현재 진행 상황 (Current Status)

- **안정적인 학습/평가 파이프라인 구축 완료:**
    - **End-to-End 모델:** `colab_train_dncnn_e2e_controlled.ipynb`, `colab_train_unet_e2e_controlled.ipynb`를 통해 안정적인 학습이 가능합니다.
    - **Step-by-Step 모델:** `colab_train_step_by_step.ipynb`를 통해 Denoising, Deconvolution 모델의 개별 학습이 가능합니다.
    - **평가:** `run_evaluation.ipynb` (E2E용)와 `run_evaluation_sbs.ipynb` (SBS용)를 통해 어떤 모델이든 최종 성능(PSNR/SSIM)을 안정적으로 측정할 수 있습니다.
- **베이스라인 성능 측정 진행 중:**
    - **U-Net E2E (38 epochs): PSNR 20.902, SSIM 0.610**
    - 다른 모델들의 학습이 완료되는 대로 베이스라인 성능 확보 예정입니다.

## 9. 다음 목표: 고급 모델 설계 및 구현 (In Progress)

안정된 파이프라인을 기반으로, 성능을 극대화하기 위한 고급 모델링을 진행합니다.

1.  **`Step-by-Step: Denoising=Diffusion + Deconvolution=Least Square` 설계 및 구현**:
    *   **Denoising (Diffusion):** 현존 최고의 Denoising 성능을 보이는 DDPM(Denoising Diffusion Probabilistic Model)을 U-Net 아키텍처 기반으로 구현합니다. 이를 위한 별도의 학습 스크립트(`train_diffusion.py`) 및 Colab 노트북을 설계합니다.
    *   **Deconvolution (Least Square):** 딥러닝 없이, 주파수 영역에서 Convolution의 역연산을 수행하는 고전적인 복원 기법인 Least Squares (Wiener Filter)를 구현합니다. 이는 별도의 학습 과정 없이, Denoising된 이미지에 후처리 방식으로 적용됩니다.
2.  **`Plug-and-Play (PnP)` 알고리즘 설계 및 구현**:
    *   **Data-Fidelity와 Denoising의 결합:** Deconvolution 문제를 반복적으로 풀어가는 최적화 과정(Data-Fidelity) 중간에, 강력한 Denoising 모델을 '부품'처럼 끼워넣어(Plug-and-Play) 복원 성능을 극대화하는 고급 알고리즘입니다.
    *   **구현:** Least Square 방식의 Data-Fidelity 단계와, 위에서 학습시킨 **Diffusion Denoiser**를 결합하여 PnP 파이프라인을 구현합니다.
3.  **최종 성능 비교 및 최적 모델 선정**: 모든 실험 결과를 종합하여 가장 성능이 뛰어난 파이프라인을 선정하고 보고합니다.
