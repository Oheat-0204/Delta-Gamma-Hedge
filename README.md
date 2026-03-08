<<<<<<< HEAD
# Jump 상황에서 델타 헷지와 감마 헷지의 동적 헷지 성과 비교

이 프로젝트는 주가 급변(Jump) 상황에서 옵션 포트폴리오의 리스크를 관리하기 위한 두 가지 주요 전략인 **동적 델타 헷지(Dynamic Delta Hedging)**와 **동적 감마 헷지(Dynamic Gamma Hedging)**의 성과를 비교 분석합니다.

## 📌 프로젝트 개요
주식 시장에서 발생하는 비정상적인 수익률 변동(Jump)은 전통적인 Black-Scholes 모델의 가정을 위반하며 헷지 오차를 유발합니다. 본 프로젝트는 **Merton Jump Diffusion Model**을 활용하여 점프 위험을 반영한 옵션 가격 및 민감도를 산출하고, 실제 시장 데이터(SPY)를 바탕으로 각 전략의 헷지 성과(PnL)를 실증적으로 비교합니다.

## 🛠 주요 기능
- **데이터 수집**: `yfinance`를 이용한 SPY(S&P 500 ETF)의 최근 3년 데이터 분석
- **점프 탐지**: 로그 수익률의 분위수(Quantile)를 이용한 이상 변동 시점 식별
- **모델 구현**: 
    - Merton Jump Diffusion 모델 기반 Put Option 가격 결정
    - 모델 기반 Delta 및 Gamma 산출
- **시뮬레이션**:
    - 점프 발생 시점 전후 30일간의 동적 델타 헷지 시뮬레이션
    - 추가 옵션을 활용한 감마 중립(Gamma-Neutral) 포트폴리오 구성 및 시뮬레이션
- **성과 분석**: 
    - 전략별 PnL(손익) 계산 및 통계적 검정(Paired t-test, Sign test)
    - 왜도(Skewness), 첨도(Kurtosis), Jarque-Bera 검정 등을 통한 리스크 분포 분석

## 📂 파일 구조
- `13조_Jump 상황에서 델타헷지와 감마헷지의 동적 헷지 성과 비교.pdf`: 상세 연구 보고서
- `최종5월29일.py`: 전체 분석 및 시뮬레이션 수행 파이썬 스크립트
- `hedge_strategy_final_PnL_summary.xlsx`: 두 전략의 최종 PnL 기록 및 요약 통계
- `delta_hedge_daily_PL.xlsx`: 점프 시점별 델타 헷지의 일별 PnL 상세 기록
- `gamma_hedge.xlsx`: 특정 점프 시점의 감마 헷지 트래킹 상세 데이터
- `selected_jump_tracking.xlsx`: 특정 점프 시점의 델타 헷지 트래킹 상세 데이터
- `jump_day_prev_return.xlsx`: 점프 발생일 및 전일 수익률 데이터

## 🚀 실행 방법
### 요구 사항
이 프로젝트를 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다:
```bash
pip install yfinance numpy pandas scipy matplotlib koreanize_matplotlib xlsxwriter
```

### 실행
`최종5월29일.py` 파일을 실행하면 데이터를 다운로드하고 시뮬레이션을 수행한 후 결과 엑셀 파일들을 생성합니다.

## 📊 분석 결과 요약
- 본 시뮬레이션을 통해 점프 상황에서 감마 헷지가 델타 헷지에 비해 손익 변동성을 얼마나 효과적으로 관리하는지 확인할 수 있습니다.
- 통계적 검정을 통해 두 전략 간의 유의미한 성과 차이를 분석하였습니다. (상세 내용은 PDF 보고서 참조)
=======
# Jumping Bean
>>>>>>> 5a5f47f9a5cde25e54003156541daef53fe6a0fd
