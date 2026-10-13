# Steel Plates Fault Classification

UCI의 **Steel Plates Faults** 데이터를 탐색하고, Dummy Classifier와 Random Forest의 성능을 비교한 다중 클래스 분류 프로젝트입니다.

단순히 높은 정확도를 얻는 데서 끝나지 않고 다음 질문을 직접 실험으로 확인했습니다.

- 데이터와 클래스 분포는 어떤 구조인가?
- Random Forest가 단순 기준 모델보다 얼마나 나은가?
- 훈련·테스트 성능 차이는 과적합을 의미하는가?
- 트리 깊이와 최소 리프 샘플 수를 제한하면 어떻게 달라지는가?
- 트리 기반 모델에 스케일링이 얼마나 영향을 주는가?

## 데이터

- 샘플: 1,941개
- 입력 특성: 27개
- 목표 클래스: 7개
- 결측치: 없음
- 가장 많은 클래스: `Other_Faults` 673개(34.7%)
- 가장 적은 클래스: `Dirtiness` 55개(2.8%)

클래스별 원-핫 타깃 7개를 각 행에서 값이 가장 큰 클래스명으로 변환했습니다. 이후 클래스 비율을 유지하도록 `stratify=y`를 적용해 훈련·테스트 데이터를 8:2로 분할했습니다.

## 모델 및 결과

동일한 데이터 분할(`random_state=42`)에서 얻은 결과입니다.

| 모델/설정 | 훈련 정확도 | 테스트 정확도 |
|---|---:|---:|
| Dummy Classifier (`most_frequent`) | - | 34.70% |
| Random Forest 기본 | 97.94% | 77.89% |
| Random Forest + StandardScaler | 97.94% | 78.15% |
| Random Forest (`max_depth=15`) | 96.84% | **79.18%** |

기본 Random Forest는 Dummy Classifier보다 테스트 정확도를 약 43%p 높였습니다. 한편 훈련 정확도와 테스트 정확도의 차이가 약 20%p여서 과적합 가능성을 확인했습니다.

`max_depth=15`에서는 기본 모델보다 테스트 정확도가 높아졌습니다. `min_samples_leaf`를 늘리면 훈련·테스트 정확도의 차이는 줄었지만 테스트 정확도도 함께 낮아졌습니다. 또한 Random Forest는 특성값의 거리보다 분할 순서를 이용하는 트리 기반 모델이므로 스케일링 전후 차이가 크지 않았습니다.

클래스 불균형을 완화하기 위해 Random Forest에 `class_weight="balanced"`를 적용했습니다. 정확도만으로 소수 클래스 성능을 충분히 판단하기는 어려우므로, 후속 작업에서는 macro F1-score와 혼동행렬을 함께 평가할 수 있습니다.

## 실행 방법

Python 3.10 이상을 권장합니다.

```bash
pip install -r requirements.txt
python steel_faults_analysis1.py --csv Faults.NNA --output-dir outputs
```

실행하면 데이터 구조와 모델별 정확도를 출력하고 다음 시각화 파일을 생성합니다.

- `outputs/eda_result.png`: 클래스 분포와 주요 특성 분포
- `outputs/analysis_result.png`: 모델 정확도, 특성 중요도, 하이퍼파라미터 실험 결과

## 프로젝트 구조

```text
.
├── steel_faults_analysis1.py  # 데이터 확인, 모델 학습 및 시각화
├── Faults.NNA                # 원본 데이터
├── requirements.txt          # Python 의존성
└── README.md
```

## 배운 점

- 모델 선택 전에 결측치, 변수 범위, 클래스 분포 등 데이터 구조를 먼저 확인해야 합니다.
- 높은 훈련 정확도가 좋은 일반화 성능을 보장하지 않으므로 테스트 성능과의 차이를 함께 봐야 합니다.
- 하이퍼파라미터가 과적합을 줄이더라도 테스트 정확도까지 항상 개선하는 것은 아닙니다.
- AI가 제안한 설명을 그대로 받아들이기보다 동일 조건에서 직접 실행하고 결과를 검증하는 과정이 중요합니다.

## 데이터 출처 및 라이선스

Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C5J88N

데이터는 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스로 제공됩니다.
