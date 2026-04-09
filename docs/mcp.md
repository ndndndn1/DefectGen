# DefectGen MCP 서버

DefectGen 기능(모델 로딩, 가상 불량 생성, 클라이언트 지정 위치로 데이터 반환, 평가, ONNX export)을 **Model Context Protocol** 도구로 노출한다. Claude Desktop, Claude Code, 자체 MCP 클라이언트 등에서 그대로 호출 가능하다.

## 설치 / 실행

```bash
pip install -e ".[mcp]"

# stdio 서버로 실행
python -m src.mcp.server
# 또는
defectgen-mcp
```

## Claude Desktop 등록 예 (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "defectgen": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/workspaces/DefectGen"
    }
  }
}
```

## 노출 도구

| 도구 | 용도 |
|------|------|
| `list_models` | 지원 모델(DDPM/Flow Matching) + 현재 로드된 핸들 목록 |
| `list_datasets` | 지원 산업 데이터셋 (WM-811K, DeepPCB, NEU-DET, MVTec AD) |
| `load_checkpoint` | YAML config + 선택적 `.pt` → 메모리에 모델 등록 (`handle`) |
| `generate_defects` | N개 가상 불량 이미지를 **클라이언트가 지정한 `output_dir`** 에 저장 후 절대경로 반환 |
| `evaluate` | FID / DefectCoverage / IntraClassDiversity 산출 |
| `export_onnx` | per-step denoiser/vector field를 클라이언트 지정 경로로 ONNX export |

## 일반 호출 흐름

1. **모델 로드**
   ```json
   { "name": "load_checkpoint",
     "arguments": {
       "handle": "ddpm_wafer",
       "config_path": "configs/ddpm_wafer.yaml",
       "checkpoint_path": "runs/ddpm_wafer_xxx/best.pt"
     }}
   ```
2. **데이터 생성 + 클라이언트 위치로 반환**
   ```json
   { "name": "generate_defects",
     "arguments": {
       "handle": "ddpm_wafer",
       "output_dir": "/path/from/client/inbox",
       "num_samples": 64,
       "defect_type": 6,
       "num_steps": 50,
       "seed": 0
     }}
   ```
   응답에 저장된 파일들의 절대경로 + `grid.png`가 포함된다.
3. **평가**
   ```json
   { "name": "evaluate",
     "arguments": {
       "handle": "ddpm_wafer",
       "dataset_config_path": "configs/ddpm_wafer.yaml",
       "num_samples": 1024,
       "metrics": ["fid", "defect_coverage"]
     }}
   ```
4. **ONNX export**
   ```json
   { "name": "export_onnx",
     "arguments": { "handle": "ddpm_wafer", "output_path": "/path/from/client/ddpm.onnx" }}
   ```

## 구현 참고

- 세션 상태는 `src/mcp/server.py`의 `SESSION` (`_Session`)에 보관되어 동일 프로세스 내 도구 호출 간에 모델/파이프라인을 재사용한다.
- 도구 함수 본체는 MCP 트랜스포트와 분리되어 있어(`TOOLS[name]["fn"]`) 단위 테스트에서 직접 호출 가능 (`tests/test_mcp_tools.py`).
- 결과는 항상 JSON 직렬화 가능한 dict로 반환되어 MCP `TextContent`로 패킹된다. 예외는 `{"error": ..., "message": ...}`로 surface.
- `output_dir`/`output_path` 인자는 모두 `Path.expanduser().resolve()`를 거쳐 절대경로화된다 — 클라이언트 측에서 위치를 명시적으로 통제할 수 있다.
