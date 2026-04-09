# DefectGen — 가상 불량 생성 연구 프레임워크

> Industrial Defect Image Generation via Diffusion / Flow Matching Models  
> with PyTorch DDP Training Framework

## 프로젝트 개요

제조 산업에서 불량(defect) 이미지는 극도로 희소하며, 수집 비용이 높고 데이터 불균형 문제가 심각합니다.  
**DefectGen**은 최신 이미지 생성 모델(Diffusion, Flow Matching 등)을 활용하여 **고품질 가상 불량 이미지를 생성**하고,  
이를 통해 다운스트림 검사 모델의 성능을 향상시키는 연구 프레임워크입니다.

---

## 핵심 모듈

```
defectgen/
├── configs/            # YAML 실험 설정 (모델, 학습, 데이터, 평가)
├── src/
│   ├── models/         # 생성 모델 아키텍처
│   │   ├── diffusion/  # DDPM, DDIM, LDM, ControlNet-based
│   │   ├── flow_matching/ # Conditional Flow Matching (CFM), OT-CFM
│   │   └── gan/        # baseline 비교용 (StyleGAN2-ADA 등)
│   ├── data/           # 데이터 파이프라인
│   │   ├── loaders/    # 산업별 데이터셋 로더 (반도체, PCB, 금속 등)
│   │   ├── augmentation/ # defect-aware augmentation
│   │   └── synthesizers/ # mask/label conditioned 합성
│   ├── training/       # 학습 엔진
│   │   ├── ddp/        # PyTorch DDP 래퍼 및 유틸리티
│   │   ├── callbacks/  # 체크포인트, 로깅, EarlyStopping
│   │   └── schedulers/ # noise schedule, lr schedule
│   ├── evaluation/     # 정량/정성 평가
│   │   ├── metrics/    # FID, IS, LPIPS, defect-specific metrics
│   │   └── visualization/ # 생성 샘플 시각화, attention map
│   ├── inference/      # 추론 파이프라인, ONNX export
│   └── utils/          # 공통 유틸리티
├── experiments/        # 실험 설계 및 리포트 템플릿
├── scripts/            # 학습/평가/데이터 전처리 스크립트
├── tests/              # 단위 테스트
└── docs/               # 기술 문서, 논문 노트
```

## 지원 산업 도메인 (PoC)

| 도메인 | 불량 유형 예시 | 데이터셋 |
|--------|---------------|---------|
| 반도체 웨이퍼 | scratch, particle, pattern defect | WM-811K, 자체 |
| PCB 기판 | missing hole, short, spur, spurious copper | DeepPCB |
| 금속/철강 표면 | crack, inclusion, pitted surface, rolled-in scale | NEU-DET, Severstal |
| 직물/텍스타일 | hole, stain, thread error | AITEX, MVTec |
| 범용 이상탐지 | 다양한 anomaly 유형 | MVTec AD, VisA |

## 빠른 시작

```bash
# 환경 설정
pip install -e ".[dev]" --break-system-packages

# 단일 GPU 학습
python scripts/train.py --config configs/ddpm_wafer.yaml

# Multi-GPU DDP 학습
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/flow_matching_pcb.yaml \
    --ddp

# 평가
python scripts/evaluate.py --checkpoint runs/exp_001/best.pt --metrics fid,lpips

# 추론 (가상 불량 생성)
python scripts/generate.py --checkpoint runs/exp_001/best.pt \
    --num_samples 1000 --defect_type scratch --output generated/
```

## 기술 스택

- **PyTorch 2.x** + DDP / FSDP
- **Diffusers** (HuggingFace) — pretrained 활용 및 커스텀 파이프라인
- **wandb** — 실험 추적
- **Hydra / OmegaConf** — config 관리
- **einops** — 텐서 연산
- **torchmetrics** — 표준 메트릭

## 라이선스

연구 목적 내부 사용. 상업적 사용은 별도 협의 필요.
