# Uncertainty-aware ML Valuation

국내 상장기업의 상대가치 지표를 추정하면서, 전통적 중앙배수와 ML의 비교 조건 및 결과의 안정성을 검토한 탐색 연구입니다.

## 연구 질문

동일 시점의 동일 비교기업 정보로 기업의 시장/장부 지표를 추정할 때 ML이 중앙배수보다 오차를 줄일 수 있는가? 표본·목표 정의·설정 선택이 결론에 어떤 영향을 주는가?

목표는 **KRX 보통주 상장시가총액 / 전년도 말 지배기업 장부지분**이라는 관측 지표입니다. 권리 범위를 완전히 맞춘 보통주 PBR 또는 전체 지분 내재가치로 해석하지 않습니다.

## 현재 결론

- 같은 날짜·같은 참조기업으로 비교하고, 같은 기업은 모든 연도에서 같은 외부 fold에 배정했습니다.
- 전체 161건에서 HGB가 업종 중앙보다 MAE가 낮은 경우는 6개 분할 중 1개였습니다.
- 공시 표상 다른 종류주식 발행수량이 없는 선별 134건에서는 HGB가 두 중앙값 기준보다 6개 분할 모두 낮았습니다. 이 표본은 모든 지분 권리를 인증한 표본이 아닙니다.
- 고정 alpha=100 Ridge의 개선이 학습 내부 설정 선택에서 안정적으로 재현되지는 않았습니다. 극단적인 재무비율의 외삽과 지수 복원이 일부 큰 오차를 만들었습니다.
- 따라서 ML의 보편적 우위를 결론 내리지 않았습니다.

## 최종 개발 비교

아래는 6개 외부 분할 MAE의 중앙값입니다. 같은 관측치를 반복 사용하는 분할 민감도 결과이며 신뢰구간이나 6회 독립 검증이 아닙니다.

| 모형 | 전체 161건 | 종류주식 단순성 선별 134건 |
|---|---:|---:|
| 업종 중앙 | 1.4083 | 1.0157 |
| 전체 중앙 | 1.4579 | 1.0127 |
| HGB 고정 | 1.4634 | 0.9673 |
| 학습 내부 선택 Ridge | 1.5850 | 17.7498 |

Ridge는 각 외부 학습 표본의 3-fold 내부 검증에서 평균 절대 로그비 오차로 설정을 선택했습니다. 선택 지표와 원배수 단위 MAE가 다르며, HGB는 고정 설정입니다. 두 모델의 튜닝 예산이 동일하다는 주장은 하지 않습니다.

## 실행과 결과

Python 3.9에서 실행한 코드입니다. 로컬 입력 스냅샷은 아래 형식을 사용합니다. 원자료는 포함하지 않았으므로 저장소만으로 원 실험의 재학습을 즉시 재현할 수는 없습니다. 공개 범위는 실험 코드·테스트·위 결과 요약입니다.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_matched_cross_section.py
python3 scripts/run_split_stability.py
python3 scripts/run_ridge_robustness.py
python3 scripts/run_nested_ridge.py
python3 -m unittest discover -s tests -v
```

실험은 위 순서로 실행하며 중간 예측은 scripts/ 아래 날짜별 폴더에 저장됩니다. 기존 결과가 있으면 중단해 보존합니다.

### 로컬 입력 형식

- `data/input/panel.csv`: `firm_id`, `valuation_date`, `stock_code`, `market`, `industry_2019_unverified`, `assets_krw`, `liabilities_krw`, `parent_equity_krw`, `cash_krw`, `sales_krw`, `parent_profit_krw`, `market_cap_krw`
- `data/input/filing_vintage.csv`: `firm_id`, `valuation_date`, `status`. 공시 시점 검토가 `PASS`인 관측치를 사용합니다.
- `data/input/share_tables.json`: `firm_id`, `valuation_date`, `classes`를 가진 배열. 각 `classes` 항목은 `label`, `issued`, `treasury`, `outstanding`을 포함하고 마지막에 합계 행을 둡니다. 원문 표시와 추론값의 근거는 별도 보존합니다.

현재 코드는 개발 스냅샷의 161건·선별 134건을 확인하도록 구성돼 있습니다. 입력자료는 OpenDART와 KRX에서 별도로 확보하고 단위·공개일·기업 연결을 검증해야 합니다.

## 후속 계획

이번 결과는 2019–2022년 개발 자료를 활용한 탐색 실험입니다. 후속 연구에서는 재무자료의 권리 범위·자기주식 기준일과 표본 적격성을 추가 검증하고, 설계를 고정한 뒤 2023–2025년 자료로 최종 평가를 진행할 예정입니다.

현재 동시점 분석에 맞춘 예측구간 검증도 확장할 예정입니다.

## 참고문헌

### 상대가치평가와 모형 비교

- Geertsema, P. & Lu, H. (2023). [Relative Valuation with Machine Learning](https://doi.org/10.1111/1475-679X.12464). *Journal of Accounting Research, 61*(1), 329–376. 재무특성을 활용한 상대가치 추정과 전통적 기준모형 비교의 출발점으로 참고했습니다.
- Staňková, V. (2024). [(Out)smart the Peer Group in Market Comparison: Building Business Valuation Multiples by Machine Learning](https://doi.org/10.11118/ejobsat.2024.011). *European Journal of Business Science and Technology, 10*(2), 156–172. 배수 추정에서 트리 기반 ML과 비교기업 방식의 성능을 비교하는 연구로 참고했습니다.
- Liu, J., Nissim, D. & Thomas, J. (2002). [Equity Valuation Using Multiples](https://doi.org/10.1111/1475-679X.00042). *Journal of Accounting Research, 40*(1), 135–172. 평가배수와 기준모형의 선택, 표본 조건에 따른 결과 해석을 검토할 때 참고했습니다.

### 예측구간 후속 연구

- Romano, Y., Patterson, E. & Candès, E. J. (2019). [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222). 현재 동시점 설계에 예측구간을 확장하기 위한 방법론으로 검토했습니다.
- Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2023). [Conformal Prediction Beyond Exchangeability](https://doi.org/10.1214/23-AOS2276). 시간 변화와 교환가능성 가정에 따른 예측구간의 해석을 검토할 때 참고했습니다.
