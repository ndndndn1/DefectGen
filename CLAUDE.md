# CLAUDE.md — DefectGen 개발 가이드

## 프로젝트 컨텍스트
가상 불량 이미지 생성을 위한 연구 프레임워크.
Diffusion / Flow Matching 기반 모델을 PyTorch DDP로 학습하고,
다양한 산업 도메인에 PoC를 도출하며, 탑티어 컨퍼런스 페이퍼를 목표로 한다.

## 코드 작성 규칙

### 일반
- Python 3.10+, type hint 필수 (returns 포함)
- docstring은 Google style, 수식이 포함되면 LaTeX 표기
- 모든 config는 YAML + OmegaConf, 하드코딩 금지
- `src/` 하위는 순수 라이브러리, `scripts/`는 CLI 진입점

### 모델 구현
- 논문 구현 시 반드시 원 논문의 수식 번호를 주석에 기재
  ```python
  # Eq. (7) in Ho et al., DDPM (NeurIPS 2020)
  x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * eps
  ```
- forward()에 shape annotation 주석 필수
  ```python
  def forward(self, x: Tensor, t: Tensor) -> Tensor:
      # x: (B, C, H, W), t: (B,)
  ```
- 새 모델 추가 시 `src/models/base.py`의 `BaseGenerator` 상속

### DDP 관련
- 모든 학습 스크립트는 단일 GPU / DDP 양쪽에서 동작해야 함
- rank 0에서만 로깅/체크포인트 저장
- `torch.distributed.barrier()` 위치에 항상 주석 설명
- gradient accumulation steps는 config에서 제어

### 실험 관리
- 실험 config는 `configs/` 아래에 도메인_모델.yaml 네이밍
- wandb run name = `{model}_{domain}_{timestamp}`
- 체크포인트: `runs/{run_name}/epoch_{N}.pt`, `best.pt`
- 실험 리포트는 `experiments/reports/` 에 마크다운으로 작성

### 평가
- FID 계산 시 최소 5,000 샘플 (논문용은 50,000)
- 새로운 metric 추가 시 `src/evaluation/metrics/base.py` 상속
- 정성 평가용 시각화는 grid 형태, row=defect_type

### 테스트
- 모델의 forward/backward shape 테스트 필수
- DDP wrapper 테스트는 `torch.distributed.launch`로 2-GPU 시뮬레이션
- `pytest tests/ -x --tb=short`

## 상황별 처리

| 상황 | 처리 방법 |
|------|----------|
| 새 논문 모델 구현 | 1) `docs/paper_notes/`에 핵심 수식 정리 → 2) `src/models/`에 구현 → 3) shape test 작성 → 4) 단일 GPU 학습 검증 → 5) DDP 호환 확인 |
| 새 산업 도메인 PoC | 1) 데이터셋 분석 → 2) `src/data/loaders/`에 로더 추가 → 3) baseline config 생성 → 4) 실험 → 5) 리포트 |
| 학습 불안정 | gradient norm 로깅 확인 → lr/noise schedule 조정 → EMA 적용 여부 검토 → mixed precision 끄고 재현 |
| 메모리 부족 (OOM) | gradient checkpointing → batch size 축소 + accumulation → FSDP 전환 검토 |
| FID가 높을 때 | 생성 샘플 시각화 → training loss curve 확인 → data augmentation 점검 → 모델 capacity 확인 |
| 논문 작성 | `docs/paper/`에 LaTeX 소스 관리, 실험 결과는 자동 생성 스크립트 연결 |

## 디렉토리 소유권
- `src/models/` — 모델 연구자
- `src/training/ddp/` — 인프라 엔지니어
- `src/data/` — 데이터 엔지니어
- `experiments/` — 전원 공유
