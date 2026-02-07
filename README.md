# Portfolio Service

사용자가 보유 종목을 입력하면 실시간 시세를 조회해 원화 기준 보유금액을 계산하는 웹서비스입니다.

## 구성 파일

- `app.py`
  - FastAPI 서버 진입점
  - SQLite(`portfolio.db`)에 종목 정보 저장/수정/삭제
  - 가격 조회 로직
    - 한국 종목(6자리 숫자, 예: 005930, 0091P0 등): 네이버 조회
    - 해외/미국 종목: 야후 → 야후 HTML → Investing → Stooq 순서로 폴백
  - 환율은 야후 → exchangerate.host → open.er-api.com 순으로 폴백

- `index.html`
  - 사용자 입력 UI (종목유형/티커/수량)
  - CSV 텍스트 붙여넣기 가져오기
  - 실시간 가격 갱신 및 총합 표시

- `requirements.txt`
  - 실행에 필요한 파이썬 의존성 목록

- `portfolio.db`
  - 저장된 보유 종목 데이터베이스 (자동 생성)

## 주요 기능

- 종목 입력/수정/삭제 및 저장
- “구성 정보 저장” 시 화면에 보이는 항목만 DB에 저장
- “실시간 가격 갱신” 시 실시간 가격 재조회 및 보유금액 재계산
- CSV 텍스트 붙여넣기 가져오기 (기존 데이터 삭제 옵션 포함)
- 티커가 `na`인 경우
  - 종목명: `현금/해당없음`
  - 현재가: 1원

## 실행 방법

```bash
cd /Users/xxx/Desktop/portfolio_service (본인에 맞게 폴더로)
conda activate .tangoenv (본인에 맞게 conda 구성)
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
### uv가 설치되어 있다면 (설치: curl -LsSf https://astral.sh/uv/install.sh | sh) 일회성으로 수행하고 
```
uv run --with-requirements requirements.txt uvicorn app:app --host 127.0.0.1 --port 8000

```
브라우저에서 `http://127.0.0.1:8000` 접속


## CSV 텍스트 입력 포맷

```
ticker,quantity
064400,664
na,60000
463250,333
```

## 참고 사항

- 외부 데이터 소스가 차단되면 가격 조회가 실패할 수 있습니다.
- 해외 종목은 소스별 티커 규칙이 다를 수 있습니다.
