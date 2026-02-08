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

### 기본 실행
```bash
cd 소스폴더/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
APP_ENV=production uvicorn app:app --host 127.0.0.1 --port 8000
```

### uv가 설치되어 있다면 (설치: curl -LsSf https://astral.sh/uv/install.sh | sh) - one shot run
```bash
APP_ENV=production uv run --with-requirements requirements.txt uvicorn app:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속

## 백그라운드 실행 방법

터미널을 종료해도 서비스가 계속 실행되도록 하는 4가지 방법:

### 방법 1: nohup 사용 (가장 간단)
```bash
nohup env APP_ENV=production uvicorn app:app --host 127.0.0.1 --port 8000 > app.log 2>&1 &
```
- **장점**: 설정이 간단하고 즉시 사용 가능
- **단점**: 프로세스 관리가 제한적, 로그 관리가 수동
- **종료**: `pkill -f uvicorn` 또는 `ps aux | grep uvicorn`으로 PID 찾아 `kill PID`

### 방법 2: screen 사용
```bash
screen -S portfolio
# screen 세션에서 실행
uvicorn app:app --reload --host 127.0.0.1 --port 8000
# Ctrl+A, D로 detach
```
- **장점**: 세션 관리 가능, 언제든지 재접속 가능
- **단점**: screen 학습 필요, 시스템 재부팅 시 종료
- **재접속**: `screen -r portfolio`
- **종료**: screen 세션 접속 후 `Ctrl+C` 또는 `exit`

### 방법 3: tmux 사용
```bash
tmux new -s portfolio
# tmux 세션에서 실행
uvicorn app:app --reload --host 127.0.0.1 --port 8000
# Ctrl+B, D로 detach
```
- **장점**: screen보다 현대적, 세션 분할 가능
- **단점**: tmux 학습 필요, 시스템 재부팅 시 종료
- **재접속**: `tmux attach -t portfolio`
- **종료**: tmux 세션 접속 후 `Ctrl+C` 또는 `exit`

### 방법 4: systemd 서비스 등록 (영구적)
1. 서비스 파일 생성:
```bash
sudo nano /etc/systemd/system/portfolio.service
```

2. 아래 내용 추가:
```ini
[Unit]
Description=Portfolio Service
After=network.target

[Service]
Type=simple
User=kdm
WorkingDirectory=/home/kdm/www/portfolio_service
Environment=PATH=/home/kdm/.venv/bin
ExecStart=/home/kdm/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

3. 서비스 활성화:
```bash
sudo systemctl daemon-reload
sudo systemctl enable portfolio
sudo systemctl start portfolio
```

- **장점**: 시스템 재부팅 후 자동 시작, 안정적인 프로세스 관리
- **단점**: 초기 설정이 복잡, root 권한 필요
- **상태 확인**: `sudo systemctl status portfolio`
- **로그 확인**: `sudo journalctl -u portfolio -f`
- **종료**: `sudo systemctl stop portfolio`

## 추천 방법
- **개발 환경**: screen 또는 tmux
- **프로덕션 환경**: systemd 서비스 등록


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
