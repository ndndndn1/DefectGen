# DefectGen 개발 대화 이력

> README.md, CLAUDE.md, `defectgen_project_architecture.svg` 기반 초기 스캐폴드부터 MCP 서버 추가까지의 주요 요청·답변 요약. (보안 점검 / .gitignore 정비 단계는 제외)

---

## 1. 초기 프레임워크 구현

**요청**
README · CLAUDE.md · 아키텍처 SVG에 따라 다음을 구현:
- 모델 3개: `BaseGenerator` 추상 클래스 → `DDPMGenerator`(논문 수식 번호 주석) → `FlowMatchingGenerator`(OT-CFM, minibatch OT 커플링)
- DDP 학습 엔진: `DDPTrainer` — 단일/멀티 GPU, AMP, gradient accumulation, EMA, wandb, rank 0 전용 체크포인트
- 데이터 파이프라인: 4개 도메인 로더(WM-811K, DeepPCB, NEU-DET, MVTec AD) + defect-aware augmentation(CopyPaste, ElasticDeformation, Cutout)
- 평가: FID(Inception), DefectCoverage, IntraClassDiversity + 시각화(grid, trajectory, real vs generated)
- 추론: 배치 생성 + ONNX export
- 운영: configs/, experiments/templates/, docs/paper_notes/, docs/architecture.md

**답변 요약**
프로젝트 트리 일괄 생성. 핵심 산출물:

```
src/
├── models/
│   ├── base.py               # BaseGenerator (compute_loss / sample 추상)
│   ├── unet.py               # 공용 time-conditioned U-Net (sinusoidal temb)
│   ├── diffusion/ddpm.py     # Eq.(4)/(11)/(14) 주석, ε-prediction
│   └── flow_matching/ot_cfm.py  # Lipman+/Tong+ Eq.(22)/(23), Hungarian OT 커플링
├── data/loaders/             # _base + wm811k/deeppcb/neudet/mvtec
├── data/augmentation/        # copy_paste / elastic / cutout
├── training/ddp/             # trainer.py + ema.py + utils.py
│                             # AMP, accum, EMA, dist.barrier, rank0 ckpt
├── evaluation/metrics/       # base + fid + defect_coverage + intra_class_diversity
├── evaluation/visualization/ # save_grid / save_real_vs_generated / save_trajectory
├── inference/pipeline.py     # GenerationPipeline + export_onnx
└── utils/                    # config(OmegaConf) + seed
configs/{ddpm_wafer.yaml, flow_matching_pcb.yaml}
scripts/{train.py, evaluate.py, generate.py}
tests/{test_models_shape.py, test_ddp_wrapper.py}
docs/{architecture.md, paper_notes/_template.md}
experiments/templates/report_template.md
pyproject.toml
```

설계 원칙(CLAUDE.md 준수):
- 모든 모델은 `BaseGenerator` 상속 / forward shape 주석
- DDPM: 원 논문 수식 번호 주석 명시 (Eq. 4 forward, Eq. 11 reverse, Eq. 14 simple loss)
- OT-CFM: minibatch Hungarian assignment으로 source noise ↔ target image 매칭 후 직선 보간 vector field 회귀
- DDPTrainer: `RANK` 환경변수로 단일/DDP 자동 분기, `dist.barrier()`(epoch 종료 동기화) 위치에 주석, AMP·grad accum·EMA·wandb 모두 config 제어, rank 0만 체크포인트/로깅
- 모든 config는 YAML+OmegaConf, 하드코딩 금지

검증: `pytest tests/` → 3 passed (DDPM/FM forward·backward·sample 형상 + DDPTrainer single-process fit + checkpoint 생성).

---

## 2. 생성 테스트 결과 export

**요청**
생성 테스트 export 폴더에 내보내기.

**답변 요약**
학습 전(random init) 모델로 형상/파이프라인 검증용 산출물 생성, `export/` 저장:

| 파일 | 내용 |
|------|------|
| `ddpm_grid.png` | 16샘플 4×4 그리드 |
| `ddpm_trajectory.png` | DDPM 샘플링 trajectory (10 step → 8 frame) |
| `ddpm_real_vs_fake.png` | real vs generated 비교 |
| `ddpm_denoiser.onnx` (+ `.data`) | ε-prediction net ONNX export |
| `fm_grid.png`, `fm_trajectory.png`, `fm_real_vs_fake.png` | OT-CFM 시각화 |
| `fm_vector_field.onnx` (+ `.data`) | vector field net ONNX export |

학습 후 재실행 명령:
```bash
python scripts/generate.py --config configs/ddpm_wafer.yaml \
  --checkpoint runs/<run>/best.pt --num_samples 64 \
  --output export/ --export_onnx export/ddpm_denoiser.onnx
```

---

## 3. Model Context Protocol(MCP) 서버 추가

**요청**
client 측 요청에 대해 모델 지원, 데이터 생성, client가 지정한 위치에 데이터 반환 등의 기능을 MCP로 지원.

**답변 요약**
`src/mcp/server.py` 신규 — DefectGen을 stdio MCP 서버로 노출.

노출 도구:

| 도구 | 기능 |
|------|------|
| `list_models` | 지원/로드된 생성 모델 조회 |
| `list_datasets` | 지원 산업 데이터셋 조회 |
| `load_checkpoint` | YAML config + `.pt`로 모델 등록(`handle`) |
| `generate_defects` | **클라이언트 지정 `output_dir`** 에 N개 가상 불량 저장 후 절대경로 반환 |
| `evaluate` | FID / DefectCoverage / IntraClassDiversity 산출 |
| `export_onnx` | 클라이언트 지정 경로로 ONNX export |

핵심 설계:
- 세션 상태 `_Session.SESSION` — 호출 간 모델/파이프라인 캐싱
- 도구 함수가 트랜스포트와 분리(`TOOLS[name]["fn"]`) → 단위 테스트에서 직접 호출 가능
- 결과는 항상 JSON 직렬화 dict, MCP `TextContent`로 패킹
- 모든 path 인자는 `Path.expanduser().resolve()` 정규화

연계 변경:
- `pyproject.toml` — `[mcp]` extras(`mcp>=1.0`) + `defectgen-mcp` 콘솔 진입점
- `docs/mcp.md` — Claude Desktop 등록 예 + 호출 흐름(load → generate → evaluate → export)
- `tests/test_mcp_tools.py` — load_checkpoint + generate_defects 스모크 테스트

검증: `pytest tests/` → **5 passed**.

호출 예시:
```jsonc
// 1) 모델 로드
{ "name": "load_checkpoint", "arguments": {
    "handle": "ddpm_wafer",
    "config_path": "configs/ddpm_wafer.yaml",
    "checkpoint_path": "runs/ddpm_wafer_xxx/best.pt"
}}

// 2) 클라이언트 지정 위치로 생성 데이터 반환
{ "name": "generate_defects", "arguments": {
    "handle": "ddpm_wafer",
    "output_dir": "/path/from/client/inbox",
    "num_samples": 64,
    "defect_type": 6,
    "num_steps": 50,
    "seed": 0
}}
```

실행:
```bash
pip install -e ".[mcp]"
python -m src.mcp.server
```

---

## 부록: 디렉토리 소유권 (CLAUDE.md 규약)

- `src/models/` — 모델 연구자
- `src/training/ddp/` — 인프라 엔지니어
- `src/data/` — 데이터 엔지니어
- `experiments/` — 전원 공유

## 부록: 상태별 처리 가이드 (CLAUDE.md)

| 상황 | 처리 |
|------|------|
| 새 논문 모델 구현 | `docs/paper_notes/` 정리 → `src/models/` 구현 → shape test → 단일 GPU 검증 → DDP 호환 확인 |
| 새 도메인 PoC | 데이터셋 분석 → `src/data/loaders/` 추가 → baseline config → 실험 → 리포트 |
| 학습 불안정 | grad norm 로깅 → lr/noise schedule 조정 → EMA 검토 → AMP 끄고 재현 |
| OOM | gradient checkpointing → batch ↓ + accum ↑ → FSDP 검토 |
| FID 高 | 샘플 시각화 → loss curve → augmentation 점검 → capacity 확인 |
