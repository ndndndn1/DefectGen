# DefectGen 기술 아키텍처

## 1. 모듈 구성

```
src/
├── models/                # 생성 모델
│   ├── base.py            # BaseGenerator (추상)
│   ├── unet.py            # 공용 time-conditioned U-Net
│   ├── diffusion/ddpm.py  # DDPMGenerator (Ho+ 2020)
│   └── flow_matching/ot_cfm.py  # FlowMatchingGenerator (Lipman+/Tong+ 2023)
├── data/
│   ├── loaders/           # WM-811K · DeepPCB · NEU-DET · MVTec AD
│   ├── augmentation/      # CopyPaste · ElasticDeformation · Cutout
│   └── synthesizers/      # mask/label conditioned 합성 (확장 지점)
├── training/
│   ├── ddp/trainer.py     # DDPTrainer (AMP, accum, EMA, rank0 체크포인트)
│   ├── ddp/ema.py         # EMA shadow weights
│   ├── ddp/utils.py       # init/cleanup, rank helpers
│   └── schedulers/        # noise / lr 스케줄
├── evaluation/
│   ├── metrics/           # FID · DefectCoverage · IntraClassDiversity
│   └── visualization/     # grid · trajectory · real vs generated
├── inference/             # 배치 생성 + ONNX export
└── utils/                 # config · seed
```

## 2. 학습 파이프라인 (단일 / DDP)

1. `scripts/train.py`가 `setup_distributed()`로 rank/world_size를 결정.
2. `build_model(cfg.model)` → `BaseGenerator` 인스턴스.
3. `build_dataset(cfg.dataset)` → 도메인별 Dataset.
4. `DDPTrainer.fit()`:
   - DDP wrap (world_size > 1) / 단일 GPU 그대로
   - AMP autocast + GradScaler
   - gradient accumulation (`grad_accum_steps`)
   - EMA shadow weights (rank 0)
   - rank 0에서만 checkpoint (`last.pt`, `best.pt`) + wandb 로깅
   - epoch 종료 시 `dist.barrier()`로 동기화

## 3. 평가 파이프라인

`scripts/evaluate.py`는 체크포인트를 로드한 뒤 데이터셋의 실제 샘플과 모델 생성 샘플을 metrics에 update → compute. FID 5k 기본, 논문 보고용은 50k 권장 (CLAUDE.md).

## 4. 추론 / 배포

`GenerationPipeline.generate(num_samples, batch_size, defect_type)` 가 배치 생성. `export_onnx()`는 per-step denoiser (`model.net`)만 export — 샘플링 루프는 런타임 측에서 재구현.

## 5. 논문 로드맵

| 단계 | 주제 | 상태 |
|------|------|------|
| 1 | DDPM 베이스라인 (WM-811K) | 구현 완료 |
| 2 | OT-CFM 베이스라인 (DeepPCB) | 구현 완료 |
| 3 | Defect-conditioned guidance (mask/label) | TODO |
| 4 | Downstream classifier 효용 검증 (NEU-DET, MVTec) | TODO |
| 5 | Top-tier 컨퍼런스 paper draft (`docs/paper/`) | TODO |

## 6. 확장 지점

- **새 모델**: `src/models/`에 추가하고 `BaseGenerator` 상속, `models/__init__.py`의 `build_model` 레지스트리에 등록.
- **새 도메인 데이터셋**: `src/data/loaders/`에 추가하고 `_REGISTRY`에 등록.
- **새 metric**: `BaseMetric` 상속 후 `evaluation/__init__.py`에서 export.
