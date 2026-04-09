# 실험 리포트: {run_name}

## 1. 개요
- **모델**: {model_name}
- **데이터셋 / 도메인**: {dataset_name}
- **목표**: (이 실험으로 검증하려는 가설 한 줄)
- **담당자**: 
- **날짜**: YYYY-MM-DD

## 2. 설정
- config 파일: `configs/{config_file}`
- GPU / world size: 
- 주요 하이퍼파라미터:
  - lr / batch / epochs / EMA decay
- 코드 커밋 해시: 

## 3. 결과 요약
| Metric | Value |
|--------|-------|
| FID (5k) | |
| DefectCoverage | |
| IntraClassDiversity | |

## 4. 정성 평가
- 생성 샘플 grid: `experiments/reports/{run_name}/grid.png`
- 실제 vs 생성 비교: `experiments/reports/{run_name}/real_vs_fake.png`
- 샘플링 trajectory: `experiments/reports/{run_name}/trajectory.png`

## 5. 분석
- 학습 곡선 (loss, grad norm)에서 관찰된 사항
- 성공/실패 케이스
- 다음 실험 제안

## 6. 재현 방법
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/{config_file} --ddp
python scripts/evaluate.py --config configs/{config_file} --checkpoint runs/{run_name}/best.pt
```
