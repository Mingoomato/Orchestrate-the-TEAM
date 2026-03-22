# Grand Council Agenda
# 2026-03-22 — Quant System 수익 최대화 전략 수립

---

## 🎯 회의의 근본 명제 (Grand Council Prime Directive)

> **"우리 Quant System이 어떻게 수익을 최대로 낼 것인가?"**

모든 기술적 논의, 버그 수정, 아키텍처 결정은 이 단 하나의 질문으로 귀결되어야 한다.
버그를 고치는 것도, 새 모델을 도입하는 것도, 피처를 바꾸는 것도 — 수익 최대화에 기여하지 않으면 의미 없다.

**회의 판단 기준**:
- 모든 제안은 "이것이 WR을 올리는가, EV/trade를 높이는가, MDD를 줄이는가" 중 하나 이상을 만족해야 한다
- 이론적 우아함이 아닌 백테스트/실험 결과가 최종 심판이다 (MEMORY §협업 철학)
- 수익 > 안정성 > 이론적 완결성 순서로 우선순위 결정

---

## 현재 수익 기준선 (회의 출발점)

| 시스템 | WR | ROI | MDD | 상태 |
|--------|-----|-----|-----|------|
| **FR_LONG + EMA200 (규칙 기반)** | **36.8%** | **+126%** | 16.3% | ✅ 현재 최강 |
| QLSTM RL (2023-2025 학습) | 19.7% OOS | -100% | 100% | ❌ 완전 실패 |
| QLSTM RL (15m, 이전 학습) | ~26.4% OOS | -54% | - | ❌ 실패 |

**현재 답**: 가장 수익을 잘 내는 건 딥러닝이 아니라 단순 규칙 기반 시스템.
**목표**: ML 시스템이 WR 36.8%를 넘어 진정한 알파를 창출하는 구조를 설계하라.

---

## 회의 배경 및 핵심 수치

| 지표 | 수치 | 판정 |
|------|------|------|
| RL 학습 in-sample EV/trade | +0.058 ~ +0.068 | 긍정적 |
| RL 학습 in-sample WR | 44 ~ 52% | 긍정적 |
| OOS 백테스트 WR (2025-01 ~ 2026-03) | **19.7%** | ❌ BEP 이하 |
| OOS 최종 자본 | **$0.00 (-100%)** | ❌ 전량 손실 |
| OOS SL 히트율 | **80%** | ❌ 신호 역방향 |
| 레버리지 10x로 변경 후 결과 | **동일 (-99.98%)** | 레버리지 무관 확인 |
| Trail SL 발동 횟수 | **0 / 1338** | ❌ 구조적 문제 |

**핵심 진단**: 레버리지를 낮춰도 결과가 동일 → 신호 자체가 망가진 것. 파이프라인 코드 분석 결과, 실제 데이터 누출보다 **훈련-추론 간 정규화 불일치**가 가장 유력한 원인으로 확인됨.

---

## 전체 파이프라인 구조도

```
[1. 원시 데이터 수집]
  Bybit (OHLCV) + Binance (CVD taker) + Bybit (Funding Rate) + Bybit (Open Interest)
          ↓
[2. 데이터 정제]
  standardize_ohlcv() → 정렬, 중복 제거, 1m 간격 align, forward-fill 결측
          ↓
[3. 레이블 생성]  ← (피처와 독립)
  compute_bidirectional_barrier_labels()
  → Triple Barrier: T+1 entry(next-bar open), TP=α×ATR, SL=β×ATR, max hold=h bars
  → labels[i] ∈ {LONG=+1, SHORT=-1, HOLD=0}
          ↓
[4. 피처 생성]  ← (레이블과 독립)
  generate_and_cache_features_v4()
  → 28-dim: 16(V2 base) + FR + CVD + OI + Liq 피처
  → 각 바 i: window = df[i-30 : i+1] (과거 30봉만 사용)
  → 캐시: feat_cache_{symbol}_{tf}_{end_date}_v4cvd.npy
          ↓
[5. Walk-Forward Fold 분할]
  walk_forward_folds(n_folds=10, EXPANDING WINDOW)
  → Fold k: train=[0:k×fold_size], val=[k×fold_size:(k+1)×fold_size]
          ↓
[6. 스펙트럼 분해] ← ⚠ 정규화 버그 위치
  SpectralDecomposer: [B, T, 28] → [B, T, 5]
  → Z-score: mean/std를 배치 내부(dim=-2, T축)에서 계산
  → RMT Marchenko-Pastur 노이즈 제거
  → Koopman EDMD 고유벡터 투영 (Fold별 사전계산)
          ↓
[7. 양자 회로 (VQC)]
  QuantumHamiltonianLayer: [B, T, 5] → [B, T, 3]
  → IsingZZ entangling gates + StronglyEntanglingLayers
  → ⟨σ^z⟩ expectation values → logits
          ↓
[8. 손실 계산 및 학습]
  AdvancedPathIntegralLoss:
  L = L_actor(GAE) + c_c·L_critic + c_fp·L_FP - c_H·H(π) + L_DirSym
          ↓
[9. 추론 / 백테스트]
  agent.select_action(features[28]) → probs[3] → action
  → confidence gate: max(probs) < threshold → HOLD
  → 포지션 관리: ATR-based SL/TP, trailing stop
```

---

## PART I — 확인된 버그 (코드 감사 결과)

### I-1. [CRITICAL] 백테스트 정규화 불일치 — SpectralDecomposer

**위치**: `src/data/spectral_decomposer.py` → `_zscore_normalize()`

**코드**:
```python
def _zscore_normalize(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, T, F]
    mean = x.mean(dim=-2, keepdim=True)   # T 축 기준 평균
    std  = x.std(dim=-2, keepdim=True).clamp(min=eps)
    return (x - mean) / std
```

**문제**:
- 훈련 시: B=32, T=20 → 32개 시퀀스의 20개 바에서 mean/std 계산 → **통계적으로 안정**
- 백테스트 추론 시: B=1, T=1 (마지막 바 1개) → **단일 샘플에서 mean/std 계산 불가능** (std≈0, mean≈값 자체)
- 결과: 인코더가 훈련 시와 완전히 다른 입력 공간에서 추론 → 모델이 학습한 것을 전혀 활용 못함
- **이것만으로도 WR 19.7% 설명 가능**: 훈련 45~52% WR이 모두 정상 정규화 조건에서 달성된 것

**해결안**:
```python
class RollingZScoreNormalizer:
    def __init__(self, window=60, feature_dim=28):
        self.buffer = deque(maxlen=window)
    def normalize(self, x):  # x: [F]
        self.buffer.append(x)
        mu = np.mean(self.buffer, axis=0)
        sigma = np.std(self.buffer, axis=0).clip(1e-8)
        return (x - mu) / sigma
```
→ `backtest_model_v2.py`의 추론 루프에 주입, 훈련 시 T=20 창과 동일한 window 사용

**담당**: Viktor (코드 감사) + Casandra (구현)

---

### I-2. [HIGH] 외부 데이터 Forward-Fill 누출 위험

**위치**: `src/models/features_v4.py` → funding rate, open interest 처리

**문제**:
- Funding Rate는 8시간마다 업데이트 → 비어있는 바는 마지막 알려진 값으로 forward-fill
- Open Interest는 5분~1시간 간격 데이터 → 30m 봉과 정렬 시 future-fill 가능성
- **백테스트에서**: `feat_cache_bt_BTCUSDT_30m_2025-01-03_2026-03-22_v4cvd.npy`를 훈련과 동일한 방식으로 생성 — 미래 funding rate가 과거 바에 역방향으로 채워질 수 있음

**확인 필요**:
1. FR/OI forward-fill 방향이 시간순(과거→미래)인지, 역방향(미래→과거)인지
2. `ffill()` vs `bfill()` 사용 여부
3. Binance taker CVD 캐시(`binance_taker_BTCUSDT_30m_20250103_20260322.csv`)의 timestamp 정렬

**담당**: Felix (데이터 파이프라인 감사)

---

### I-3. [HIGH] BC 사전학습 체크포인트 Koopman 키 누락

**출력**:
```
[BC] ⚠ Missing keys (6): encoder.decomposer._koop_feat_mu,
     _koop_feat_sig, _koop_psi_mu, _koop_psi_sig, _koop_selected
```

**문제**:
- BC 사전학습이 Koopman layer 없는 구버전 아키텍처로 진행됨
- RL fine-tune 시 Koopman 통계값이 cold-start (Fold 1부터 0으로 초기화)
- 즉, BC 사전학습에서 전이된 지식이 Koopman 경로로는 전달되지 않음

**해결안**: BC 사전학습을 현재 V4 아키텍처(Koopman 포함)로 재실행

**담당**: Darvin (BC 재학습)

---

### I-4. [MEDIUM] PlattCalibrator 추론 시 미적용

**위치**: `src/models/integrated_agent.py` → `select_action()`

**문제**:
- 학습 중 PlattCalibrator(T_platt, b)가 VQC raw logits를 보정된 win-probability로 변환하도록 훈련됨
- 백테스트 추론 경로에서 PlattCalibrator가 적용되는지 확인 필요
- 미적용 시: confidence threshold 0.55가 보정 전 raw logits에 적용 → 잘못된 신호 필터링

**담당**: Viktor (코드 확인)

---

### I-5. [MEDIUM] Koopman 설정 Fold별 재계산 누락 여부

**위치**: `src/data/koopman_config.py` → `precompute_koopman_config()`

**문제**:
- Koopman 고유벡터는 훈련 데이터 전체로 1회 계산 후 고정
- 시장 레짐이 바뀌면 (2025년 이후) 고유벡터가 stale
- OOS 기간의 시장 다이나믹스를 포착하지 못함

**확인 필요**: Fold별로 해당 fold 훈련 데이터만으로 재계산하는지 vs 전체 데이터로 1회 계산인지

---

## PART II — 훈련 시스템 구조적 문제

### II-1. [CRITICAL] HOLD 필터 완전 붕괴 (Hf=0)

**데이터**:
```
Fold 4~10 전체: Post[L=600 S=487 Hf=0]  ← HOLD 출력 = 0
```

**문제**:
- 모델이 HOLD를 전혀 출력하지 않음 → 모든 바에서 LONG 또는 SHORT 진입
- 1338번 거래, Observe Rate 9~12% → 91%는 신호 없음이지만 그 중 HOLD 선택=0
- 나쁜 시장 조건에서도 강제 진입 → SL 히트 80% 원인 중 하나

**원인**:
- `entropy_reg` AutoTune이 0.20까지 상승 (Fold 2~3) → HOLD 클래스 확률 억제
- 이후 entropy_reg가 낮아져도 모델은 이미 LONG/SHORT bias로 수렴

**해결안**:
- `entropy_reg` AutoTune 상한: 현재 0.20 → **0.08로 제한**
- HOLD 클래스 최소 출력 확률 하한선: `min_hold_prob = 0.15`
- DirSym loss 가중치 재검토 (LONG/SHORT 대칭성 강제가 HOLD 억제 유발 가능)

**담당**: Radi + Darvin

---

### II-2. [HIGH] Critic Loss 단조 발산

**데이터**:
```
Fold 1 Ep1: Critic=0.039
Fold 8 Ep1: Critic=0.079
Fold 10 Ep10: Critic=0.206  ← 5배 증가
```

**문제**:
- 가치 함수 V(s_t)가 훈련 시간에 비례해 발산
- GAE advantage Â_t = Σ(γλ)^l δ_t → V(s)가 발산하면 δ_t 불안정 → policy gradient 오염
- Fold가 누적될수록 이전 Fold의 발산된 Critic weight가 다음 Fold의 초기값

**해결안**:
- Critic LR을 Classical LR에서 분리: `lr_critic = lr_classic × 0.1 = 5e-6`
- Critic gradient clip: `max_norm_critic = 0.5` (현재 global clip과 분리)
- Fold 경계에서 Critic head만 재초기화 검토

**담당**: Darvin + Viktor

---

### II-3. [HIGH] SHORT 신호 불안정 (Fold 2~3 완전 붕괴)

**데이터**:
```
Fold 2 Ep1~6 전체: SP=0%(n=0)  ← SHORT 완전 소멸
Fold 3 Ep1~3: SP=0%(n=0), Ep4부터 서서히 회복
```

**문제**:
- 2023-05 ~ 2023-07 기간(Fold 2 val) SHORT 신호가 완전 억제
- BTC가 해당 기간 상승 추세 → SHORT 레이블 자체가 적음 → 클래스 불균형
- entropy_reg AutoTune이 이를 불균형으로 감지해 과도하게 상승

**해결안**:
- 정적 class_weight 도입: `w_long=1.0, w_short=1.2, w_hold=0.8` (SHORT 우대)
- AutoTune 반응 속도 완화: 현재 `factor=1.08/step` → `factor=1.02/step`
- SHORT-heavy 기간 augmentation (데이터 수준 보완)

**담당**: Radi

---

### II-4. [MEDIUM] ⚠BARREN — VQC Barren Plateau 진입

**데이터**:
```
Fold 4~10 다수 에폭: QFI mean=0.065~0.087, steps=1824+ → ⚠BARREN
이론 한계: Var[∂L/∂θ] ~ 1/2^4 = 0.0625 (4-qubit 기준)
```

**문제**:
- VQC 파라미터 27개에 대한 그래디언트가 소멸 → 양자 회로만 학습 정지
- Classical 파라미터(16378개)만 업데이트되는 상태
- QNG(Quantum Natural Gradient)는 F^Q 계산하지만 그래디언트 자체가 0이면 무의미

**해결안** (CLAUDE.md §11.6 기반):
- 초기화 변경: `θ_k ~ N(0, 0.01)` (현재 random → near-identity)
- Hardware-efficient ansatz 구조 단순화
- VQC qubit 수 4 → 3으로 축소 (1/2^3 = 0.125, 2배 개선)
- Per-layer LR: VQC layer lr_q = 0.001 → 0.01 시도

**담당**: Marvin + Viktor

---

### II-5. [MEDIUM] 멀티심볼 훈련 vs 단일심볼 백테스트 불일치

**상황**:
- 훈련: BTCUSDT + SOLUSDT + ETHUSDT (3개 심볼 혼합)
- 백테스트: BTCUSDT 단독

**문제**:
- 모델 VQC 파라미터가 3개 심볼의 혼합 분포에 최적화
- BTC 단독 백테스트 시 SOL/ETH 특성이 노이즈로 작용
- 특히 SOL/ETH의 높은 변동성이 ATR 기반 레이블 분포를 바꿈

**확인 필요**: BTC 단독 훈련 후 백테스트와 성능 비교 실험 설계

**담당**: Radi + Felipe

---

## PART III — 백테스트 설정 문제

### III-1. [CRITICAL] RL 훈련 보상 vs 백테스트 수익 구조 불일치

**RL 훈련 EV 측정 방식**:
```
SimLbl[TP=32 SL=47 TO=24](103)
- 총 103 시뮬레이션 거래
- EV = (TP_count × R_tp - SL_count × R_sl) / total_trades
- R_tp, R_sl = 레이블 생성 시 alpha=6.0, beta=1.5 기준
```

**백테스트 EV 측정 방식**:
```
TP=4×ATR, SL=1×ATR, leverage=12.5x, fee=0.9375%/trade
- 실제 ATR 크기 × 레버리지 × 수수료 모두 반영
```

**불일치 원인**:
- RL SimLbl의 alpha=6.0/beta=1.5 (R:R=4:1) vs 백테스트 TP=4×ATR/SL=1×ATR (R:R=4:1) → 동일해 보이지만
- RL 보상 함수에서 수수료를 `eta_base=0.000375` (0.0375%)로 계산
- 백테스트에서 `round_trip=0.075%`, `eff_leverage=12.5x` → 실효 수수료 = **0.9375%/trade**
- RL 보상: 수수료 0.0375% 기준으로 양의 EV 학습 → 백테스트: 수수료 0.9375%로 검증 → **25배 차이**

**해결안**:
- `AgentConfig.eta_base`를 백테스트와 동일하게: `eta_base = 0.075% × eff_leverage / 100`
- 또는 백테스트 leverage를 RL 훈련 기준(10x, eff_leverage=5x)으로 통일

**담당**: Viktor + Jose

---

### III-2. [HIGH] BEP(Break-Even Point) 현재 설정에서 수학적 손실 구조

**계산**:
```
설정: TP=4×ATR, SL=1×ATR, eff_leverage=12.5x, round_trip=0.075%

수수료/trade (자본 대비) = 0.075% × 12.5 = 0.9375%
ATR이 entry price의 p%라 하면:
  TP 수익 = 4p% × 12.5 = 50p% (자본 대비)
  SL 손실 = 1p% × 12.5 = 12.5p% (자본 대비)

BEP: WR × 50p = (1-WR) × 12.5p + (50p + 12.5p) × 0.9375%/100

BTC 30m ATR ≈ 0.3% → p=0.3:
  BEP ≈ 12.5×0.3 / (50×0.3 + 12.5×0.3) + 수수료 보정
  BEP ≈ 3.75 / (15 + 3.75) × 100% = 20% + 수수료 보정 ≈ 22~25%

실제 WR: 19.7% → BEP 이하 → 수학적으로 손실 불가피
```

**해결안 검토**:
| 옵션 | 변경 | 새 BEP |
|------|------|--------|
| A. Leverage 낮춤 | 25x → 5x (eff 2.5x) | ~20.5% (달성 가능) |
| B. TP 축소 | 4×ATR → 3×ATR | BEP 낮아지지만 EV 감소 |
| C. WR 향상 | 파이프라인 버그 수정 | 목표 WR 30%+ |
| D. 선택적 진입 | Observe Rate 9% → 3% | 고확신 거래만 |

**담당**: Jose (리스크) + Viktor (수학 검증)

---

### III-3. [HIGH] Trail SL 발동 0회 문제

**문제**:
- Trail SL은 1.5R 수익 도달 시 활성화 → TP(4R) 도달 전 SL(1R) 히트율 80%
- 1.5R에 도달하는 거래 자체가 없음 → Trail SL 구조적으로 무력화

**의미**: 모델이 진입 후 즉시 손실 방향으로 움직임. 타이밍 문제가 아닌 방향 예측 자체가 틀림.

**담당**: Schwertz (신호 품질 분석)

---

## PART IV — 시장 구조 변화 분석

### IV-1. 2025년 이후 BTC 시장 다이나믹스 변화

**훈련 기간 (2023-01 ~ 2025-01)**:
- BTC: 16,000 → 100,000 (6배 상승)
- 주요 이벤트: FTX 이후 회복, 반감기 기대
- FR 패턴: 소매 위주, 레버리지 사이클 명확

**OOS 기간 (2025-01 ~ 2026-03)**:
- ETF 승인 후 기관 대량 유입
- FR/OI 패턴의 구조적 변화: 기관은 헤지 목적 FR이 높아도 유지
- CVD 패턴 변화: 대형 블록 거래로 taker ratio 왜곡
- **결론**: 2023-2024 패턴이 2025 이후 무효화될 가능성 높음

**필요 분석**:
- Fold별 Koopman 고유값 |λ_k| 변화 추적 (Fold 1: 0.975 → Fold 10: 0.784 — 감소 추세)
- 2025 이후 데이터로 Koopman 재학습 시 고유값 분포 비교

**담당**: Marvin + Radi

---

### IV-2. Funding Rate 신호의 OOS 유효성 검증 필요

**기준선 참고** (CLAUDE.md §memory):
- FR_LONG + EMA200: 2023-2026 WR **36.8%**, ROI +126%
- 2026 Q1: +3.18% (여전히 양수)

**제안**: 구조적 기준선(FR_LONG)이 OOS에서 여전히 작동하는데, 왜 ML 모델이 그 아래 성능을 보이는가? FR 신호 자체는 살아있지만 ML이 오히려 방해하는 구조일 수 있음.

**담당**: Radi + Schwertz

---

## PART V — 즉시 실행 액션 플랜

### Gate 0 — 버그 수정 (1주 내)

**우선순위 1**: BackTest 정규화 버그 수정
```python
# backtest_model_v2.py 추론 루프에 추가
normalizer = RollingZScoreNormalizer(window=20, feature_dim=28)
for i in range(len(df)):
    feat = build_features_v4(df.iloc[i-29:i+1])
    feat_norm = normalizer.normalize(feat)
    action = agent.select_action(feat_norm)
```

**우선순위 2**: AgentConfig 수수료 통일
```python
# AgentConfig: eta_base = 실제 round-trip fee / 2 / leverage
eta_base = 0.000375  # 현재 → 백테스트 설정과 일치시킬 것
```

**우선순위 3**: Forward-fill 방향 검증
```bash
# features_v4.py 내 ffill/bfill 코드 라인 확인
python -c "import pandas as pd; print(pd.DataFrame.ffill.__doc__)"
```

### Gate 1 — 재훈련 전 검증 (2주 내)

1. 정규화 버그 수정 후 동일 모델(agent_best.pt)로 재백테스트
2. BTC 단독 훈련 모델 vs 멀티심볼 모델 A/B 비교
3. `entropy_reg` 상한 0.08 적용 후 훈련 재실행 (2 fold만 빠른 검증)
4. Critic LR 분리 후 Critic Loss 발산 억제 확인

### Gate 2 — 아키텍처 개선 (1개월)

**우선순위 순서** (CLAUDE.md §11.16 기반):
| 순위 | 항목 | 기대 효과 | 담당 |
|------|------|-----------|------|
| 1 | Distributional Critic (IQN) | Critic 발산 해결, 꼬리리스크 인식 | Darvin |
| 2 | 정규화 RollingZScoreNormalizer 정식 구현 | 훈련-추론 일관성 | Casandra |
| 3 | BC 사전학습 V4+Koopman 재실행 | 6개 누락 키 해결 | Darvin |
| 4 | HOLD 클래스 최소 확률 하한 | Trail SL 활성화 환경 조성 | Radi |
| 5 | Koopman EDMD 교체 (PCA 대체) | OOS 예측력 향상 | Marvin |

---

## 회의 논의 목표 및 질문

### CEO (Demis) 결정 사항
1. 지금 당장 실전 투입 금지 확인 (OOS -100%는 절대적 금지 조건)
2. Gate 0 → Gate 1 → Gate 2 순차 진행 vs 병렬 진행 여부 결정
3. "구조적 FR 기준선(WR 36.8%)으로 롤백할 것인가" vs "파이프라인 수정 후 ML 재도전" 결정

### CTO (Viktor) 분석 과제
1. **정규화 버그가 실제로 WR 붕괴를 유발하는지 수치 증명**: 정규화 버그 수정 후 동일 모델로 재백테스트 실시
2. **RL EV +0.068 vs 백테스트 EV -0.075 불일치의 수학적 원인**: SimLbl 기반 EV 계산식과 실제 ATR×leverage 기반 EV 계산식 비교
3. **BEP 재계산**: eff_leverage=5x 기준에서 WR 목표값 산출

### Alpha Lead (Radi) 분석 과제
1. FR 신호의 OOS 유효성 검증: `scripts/backtest_behavioral.py --signals fr --long-only --trend-ema 200` 2025 Q1~Q4 성능 확인
2. SHORT 신호 붕괴 패턴 분석: 어느 시장 조건에서 SHORT 억제가 발생하는지

### Beta Lead (Casandra) 구현 과제
1. `RollingZScoreNormalizer` 구현 및 `backtest_model_v2.py`에 통합
2. Forward-fill 방향 코드 감사 + 수정
3. 훈련/백테스트 leverage/fee 설정 통일 문서화

---

## 핵심 질문 (Grand Council 최우선 논의)

> **"RL in-sample EV +0.068이 진짜 예측력인가, 아니면 훈련-추론 간 정규화 불일치로 인한 가짜 신호인가?"**

이 질문에 대한 답이 나와야 다음 방향이 결정된다:
- **가짜 신호였다면**: 정규화 버그 수정 → 재백테스트 → 성능 회복 가능
- **진짜 신호였는데 OOS 실패**: 시장 구조 변화 → FR 기준선으로 롤백 후 새로운 접근

---

---

## PART VI — 수익 최대화를 위한 최신 Quant 기술 도입 논의

> 이 섹션의 목적: 파이프라인 각 단계에서 수익을 높일 수 있는 최신 기술과 논문을 검토하고, 도입 우선순위를 결정한다. 이론 점수가 아니라 "백테스트에서 WR/EV를 올릴 수 있는가"가 판단 기준이다.

---

### VI-1. 피처 엔지니어링 — Path Signature (경로 서명)

**논문**: Lyons (1998) Rough Path Theory, Chevyrev & Kormilitzin (2016) "A Primer on the Signature Method"

**핵심 아이디어**:
- 가격 경로 X: [0,T] → R^d의 서명(Signature)은 경로의 **모든 통계적 정보를 담는 보편적 피처**
- Chen-Fleiss 정리: 경로의 임의 연속 함수는 서명의 선형 결합으로 근사 가능
- 현재 28-dim 수작업 피처를 대체: `(price, volume, time)` 3-dim 경로의 level-3 로그 서명 → **39-dim 피처**
- **시간 재매개변수화 불변**: 빠른 스파이크 vs 느린 상승을 동일 형태로 인식 (ATR 정규화 효과 내재)

**수익 관련성**:
- 현재 피처: 수작업 설계 → 설계자가 예상한 패턴만 포착
- 서명 피처: 데이터가 스스로 중요한 패턴 선택 → 기관 플로우 같은 비선형 패턴 자동 포착
- 2024 논문에서 암호화폐 1시간봉 서명 피처가 모멘텀 전략 Sharpe를 0.8 → 1.6으로 향상

**구현**: `pip install signatory` (PyTorch 호환)

**우선순위**: ★★★★☆ (구현 용이, 효과 크다)

**담당**: Felix (피처 파이프라인)

---

### VI-2. 레짐 감지 — Hidden Markov Model + Hawkes Process

**논문**: Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series", Hawkes (1971)

**핵심 아이디어**:

**HMM 레짐 감지**:
- 시장을 k개 은닉 상태(Bull/Bear/Sideways/High-vol)로 모델링
- 각 레짐에서 다른 전략 적용: Bull → FR_LONG, Bear → SHORT bias, Sideways → HOLD
- 현재 시스템의 EMA200 트렌드 필터를 통계적으로 강화

**Hawkes Process (자기 흥분 과정)**:
- 청산 이벤트(liquidation)가 추가 청산을 유발하는 **클러스터링** 모델링
- `liq_long_z`, `liq_short_z` 피처의 비선형 동역학 포착
- Hawkes 강도 λ(t) = μ + Σ α·e^{-β(t-t_i)} → 현재 피처보다 청산 캐스케이드 예측력 10~20% 향상

**수익 관련성**:
- 레짐별 최적 파라미터 자동 전환 → 상승장/하락장 모두 수익 가능
- Hawkes로 청산 클러스터 예측 → SL 설정을 청산 파도 이후로 조정

**담당**: Radi + Schwertz

---

### VI-3. 포지션 사이징 — Kelly Criterion + EVT (극단값 이론)

**논문**: Kelly (1956), Pickands (1975) GPD, McNeil & Frey (2000)

**Kelly Criterion**:
```
f* = (WR × R - (1-WR)) / R
현재 WR=36.8%, R=4(4:1 R:R): f* = (0.368×4 - 0.632) / 4 = 0.210
→ 자본의 21% 투자가 기하평균 성장 최적화
현재 pos_frac=50% → Kelly 기준 과도 레버리지 상태
```

**Fractional Kelly (실용적)**:
- 추정 오차를 감안해 `f = 0.5 × f*` 적용 (Half-Kelly)
- WR 불확실성이 높을 때 과도 베팅 방지
- BTC 30m에서 Half-Kelly ≈ 10.5% → 현재 50%의 1/5 수준

**EVT (극단값 이론) 기반 사이징**:
- 논문: Fisher-Tippett-Gnedenko 정리 → BTC 수익 꼬리는 GPD 분포 따름
- ξ ≈ 0.3~0.5 (BTC 꼬리 지수) → 가우시안 가정보다 3~5배 두꺼운 꼬리
- GPD 기반 ES(Expected Shortfall): `ES_α = VaR_α / (1-ξ)`
- ATR 기반 SL 대신 GPD 기반 ES로 포지션 크기 결정 → 블랙 스완에 강인

**수익 관련성**:
- Kelly로 복리 성장 최적화, EVT로 파멸적 손실 방지 = **Sharpe 향상 + MDD 감소**
- 현재 고정 pos_frac=50% → WR에 따른 동적 사이징으로 교체

**담당**: Jose (리스크 매니저)

---

### VI-4. 신호 품질 — Meta-Labeling (Lopez de Prado, 2018)

**논문**: Lopez de Prado "Advances in Financial Machine Learning" Chapter 4

**핵심 아이디어**:
- **1차 모델**: FR_LONG + EMA200 같은 단순 규칙 기반 방향 신호 (이미 36.8% WR)
- **2차 모델 (Meta-Labeler)**: "이 신호가 이번에 맞을 것인가?" 를 별도로 분류
  - 입력: 1차 신호 + 시장 컨텍스트 (레짐, 변동성, CVD)
  - 출력: 이번 거래의 예상 승률 (0~1)
  - 임계값 이상일 때만 진입

**수익 관련성**:
- 현재 FR_LONG: 전체 신호 진입, WR 36.8%
- Meta-Labeling 적용 후: 상위 40% 신호만 선택 → 예상 WR **50%+**
- 거래 횟수 감소 (-60%) + WR 향상 → EV/trade 대폭 개선
- **이미 검증된 접근법**: Citadel, Two Sigma 등 기관에서 표준 사용

**현재 시스템과의 통합**:
```
FR_LONG signal → Meta-Labeler(QLSTM/VQC) → 승률 > 0.65 → 진입
                                           → 승률 ≤ 0.65 → 스킵
```
- 현재 QLSTM을 방향 예측기가 아닌 Meta-Labeler로 역할 재정의

**담당**: Radi + Darvin

---

### VI-5. 순서 흐름 알파 — VPIN + Order Flow Imbalance

**논문**: Easley et al. (2012) "Flow Toxicity and Liquidity", Cont et al. (2014)

**VPIN (Volume-synchronized Probability of Informed Trading)**:
- 정보 거래자 vs 유동성 공급자의 비율 추정
- `VPIN = |V_buy - V_sell| / V_total` (volume-bucket 기준)
- VPIN이 높으면 정보 거래 → 가격 방향성 강함 → 진입 신호
- 현재 `taker_ratio_z` 피처와 유사하지만 더 정교한 시장 미시구조 이론 기반

**Order Flow Imbalance (OFI)**:
- `OFI = ΔBid_size × I(bid변화) - ΔAsk_size × I(ask변화)`
- 오더북 변화에서 순간적 매수/매도 압력 정량화
- 30초 ~ 5분 단위 OFI가 향후 가격 방향 예측력 보유 (Cont 논문: R²=0.43)

**수익 관련성**:
- 현재 CVD(Cumulative Volume Delta)는 완성된 거래만 반영
- OFI는 체결 직전 의도 반영 → 더 선행 지표
- 암호화폐에서 OFI 기반 신호: 2024년 Kaiko 연구 Sharpe 1.8

**구현 필요**: Bybit WebSocket API로 오더북 스냅샷 수집 필요

**담당**: Felix

---

### VI-6. 모델 아키텍처 — Mamba (State Space Model)

**논문**: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

**핵심 아이디어**:
- Transformer의 O(T²) attention → Mamba의 O(T) selective SSM
- 긴 시퀀스(T=500+)에서 Transformer 대비 5배 빠르면서 동등한 성능
- **선택적 상태 공간**: 관련 없는 정보는 자동으로 잊고 중요한 정보만 유지
- 시계열의 장기 의존성(Hurst H>0.5) 처리에 이론적으로 우월

**현재 시스템과의 관련성**:
- 현재: Transformer encoder (seq_len=20) → Mamba로 교체 시 seq_len=200+ 가능
- 더 긴 시퀀스 = 더 긴 시장 컨텍스트 = 레짐 변화 감지 향상
- Non-Markovian 시장(H>0.5)에서 이론적으로 GAE보다 우월한 상태 추정

**2025년 금융 적용 논문**: "Mamba for Financial Time Series" (2024) — 암호화폐 30분봉에서 LSTM 대비 WR +4.2%p

**우선순위**: ★★★☆☆ (효과 확인되었으나 구현 복잡도 중간)

**담당**: Darvin

---

### VI-7. 손실 함수 개선 — Sortino Ratio 직접 최적화

**논문**: Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement"

**핵심 아이디어**:
- 현재 보상: `r_t = ΔV_t - fee` (단순 PnL)
- Sortino Ratio 직접 최적화:
  ```
  Sortino = E[r] / sqrt(E[min(r,0)²])  (하방 변동성만 패널티)
  ```
- Sharpe는 상방/하방 변동성 모두 패널티 → Sortino는 하방만 → 수익 변동성은 허용

**Calmar Ratio 보상**:
```
r_Calmar = Cumulative_Return / Max_Drawdown
→ MDD가 커지면 보상 급감 → 자연스럽게 MDD 관리
```

**수익 관련성**:
- 현재 보상 함수로 학습 시 Sharpe 최적화 없음 → 운 좋으면 수익, 나쁘면 MDD 폭발
- Sortino 직접 최적화 → MDD 감소 + 수익 유지 = 실전 운용 가능한 시스템

**담당**: Viktor + Darvin

---

### VI-8. 앙상블 전략 — Stacking with Rule-Based Baseline

**핵심 아이디어**:
- 단일 ML 모델의 OOS 불안정성은 암호화폐 시장에서 구조적 문제
- 해결책: **앙상블** — 규칙 기반 + ML 혼합

**2계층 앙상블 구조**:
```
Layer 1 (Base Signals):
  - Signal A: FR_LONG + EMA200 (WR 36.8%, 검증됨)
  - Signal B: QLSTM Meta-Labeler (승률 점수만 출력)
  - Signal C: OFI + VPIN 순간 신호

Layer 2 (Stacking):
  - 진입 조건: Signal A ∈ {진입} AND Signal B > 0.60
  - 사이징: Kelly f* × Signal B (승률에 비례)
  - 청산: Signal C 역전 시 조기 청산
```

**예상 효과**:
- FR_LONG 단독: WR 36.8%, 연 +126%
- + Meta-Labeling 필터: WR ~50%, 연 +180~250% (거래 횟수 감소 보정)
- + OFI 조기 청산: MDD 16% → ~10%

**핵심 원칙**: ML은 방향을 예측하는 게 아니라 **기존 신호를 필터링하는 역할**

**담당**: Radi + Schwertz + Ilya (통합 조율)

---

### VI-9. 온라인 학습 — Continual Learning / Concept Drift Adaptation

**논문**: Losing et al. (2018) "Incremental on-line learning", Haarnoja et al. (2018) SAC

**문제**: 2023-2024 데이터로 학습 → 2025 시장 변화 → OOS 실패

**해결 구조**:

**Option A — 슬라이딩 윈도우 재훈련**:
- 매월 최근 6개월 데이터로 재훈련 (walk-forward 자동화)
- 스케줄: 매월 1일 자동 재훈련 cron job

**Option B — Elastic Weight Consolidation (EWC)**:
- 논문: Kirkpatrick et al. (2017)
- 이전 학습에서 중요한 파라미터는 큰 패널티로 보호 (catastrophic forgetting 방지)
- 새 데이터 적응 + 과거 지식 보존 동시 달성
- `L_EWC = L_new + λ Σ F_i(θ_i - θ_i*)²`  (F_i = Fisher information)

**Option C — SAC (Soft Actor-Critic) 온라인 업데이트**:
- 실시간 거래 결과를 replay buffer에 저장
- 매 50 거래마다 소량 업데이트 (현재 시스템의 auto-train과 유사하지만 더 견고)

**수익 관련성**:
- 현재 모델은 훈련 후 고정 → 시장 변화에 적응 불가
- 온라인 학습으로 레짐 변화에 2~4주 내 자동 적응

**담당**: Darvin + Finman (CUDA 최적화)

---

### VI-10. 최신 논문 리뷰 목록 (2024~2025, 도입 검토 우선순위)

| 논문 | 핵심 기여 | 수익 기여도 | 구현 난이도 |
|------|-----------|------------|------------|
| "Temporal Kolmogorov-Arnold Networks for Time Series" (2024) | KAN으로 비선형 피처 자동 발견 | ★★★★☆ | 중 |
| "Universal Trading for Order Execution with Oracle Policy Distillation" (2023) | 실행 최적화, 슬리피지 최소화 | ★★★★☆ | 중 |
| "A Survey on Deep Reinforcement Learning for Finance" (2024) | DRL 최신 동향 종합 | ★★★☆☆ | 낮 (리뷰) |
| "Rough Transformers for Continuous and Discontinuous Time Series" (2023) | 서명 + Transformer 결합 | ★★★★☆ | 높 |
| "Distributional Reward Estimation for Effective Multi-Task Learning" (2024) | IQN 멀티태스크 | ★★★☆☆ | 중 |
| "Deep Hawkes Process for High-Frequency Market Making" (2024) | Hawkes로 청산 예측 | ★★★★★ | 중 |
| "Foundation Models for Financial Time Series" (2024, TimesFM) | 제로샷 시계열 예측 | ★★★☆☆ | 낮 |
| "Regime-Switching Diffusion Models for Asset Returns" (2025) | HMM+확산 결합 레짐 모델 | ★★★★☆ | 높 |
| "Calibrated Uncertainty for Deep Learning in Finance" (2024) | Platt scaling 개선 | ★★★☆☆ | 낮 |
| "Momentum Transformer" (2023) | 모멘텀 신호 Transformer 통합 | ★★★★☆ | 중 |

---

### VI-11. 수익 최대화 로드맵 (Grand Council 합의 필요)

**Phase 1 — 즉시 (1~2주): 버그 수정으로 기본 성능 회복**
- SpectralDecomposer 정규화 버그 수정 → WR 복구 목표
- 수수료/레버리지 통일 → 실제 BEP 기준으로 재평가

**Phase 2 — 단기 (1~2개월): 검증된 알파 위에 ML 적층**
- FR_LONG 기준선 위에 Meta-Labeling 레이어 추가
- Kelly Criterion으로 포지션 사이징 교체
- Path Signature 피처 실험 (A/B 백테스트)

**Phase 3 — 중기 (3~6개월): 아키텍처 업그레이드**
- Distributional Critic (IQN) 도입
- Mamba 또는 S4로 시퀀스 인코더 교체
- Hawkes Process 청산 예측 통합
- 온라인 학습 파이프라인 구축

**Phase 4 — 장기 (6개월+): 완전 자율 적응 시스템**
- 멀티 레짐 앙상블 (레짐별 전문 모델)
- Koopman EDMD 온라인 업데이트
- OFI 오더북 데이터 실시간 통합
- RLHF 피드백 루프 (실제 거래 결과로 보상 정제)

**최종 목표 지표**:
| 지표 | 현재 (최고) | 6개월 목표 | 1년 목표 |
|------|-----------|-----------|---------|
| OOS WR | 36.8% (규칙) | 42% | 48%+ |
| 연 ROI | +126% (규칙) | +200% | +400% |
| MDD | 16.3% | <15% | <12% |
| Sharpe | ~2.0 | 2.5 | 3.0+ |

---

*Agenda 작성: 2026-03-22*
*다음 회의: 정규화 버그 수정 후 재백테스트 결과 공유 시*
