# Portfolio Service

실시간 시세를 기준으로 한국 주식, 미국 주식, ETF, 예수금을 KRW 기준 포트폴리오로 관리하는 FastAPI + Vanilla JS 서비스다.

## 개요

- Google OAuth 로그인 기반 세션 인증
- 사용자별 보유 종목 저장 및 실시간 평가금액 계산
- CSV 붙여넣기 기반 일괄 교체/가져오기
- 자산 비중 차트 및 종목별 서브합 집계
- 첫 가입자 관리자 자동 지정 및 사용자 관리 기능
- PWA 설치 지원, SEO 메타/사이트맵/robots 제공

## 구성

- `app.py`
  FastAPI 서버, OAuth, 세션, SQLite, 가격 조회 API, 관리자 API
- `index.html`
  메인 화면, SEO 메타, 관리자 모달, 사용 안내
- `static/index.js`
  인증 UI, 보유 종목 CRUD, CSV 가져오기, 관리자 화면, PWA 설치 흐름
- `static/index.css`
  서비스 전체 스타일
- `sitemap.xml`
  운영 사이트용 sitemap
- `portfolio.db`
  실행 중 자동 생성되는 SQLite 데이터베이스

## 환경 파일 규칙

앱은 아래 순서로 환경변수를 읽는다.

1. `.env`
2. `APP_ENV` 값 해석
3. `.env.dev` 또는 `.env.prod`를 override 로드

파일 역할은 아래와 같다.

- `.env`: 공통 비밀값과 공통 설정
- `.env.dev`: 개발환경 전용 설정
- `.env.prod`: 운영환경 전용 설정

`APP_ENV`는 `dev`, `development`, `prod`, `production`을 지원한다. 기본값은 `development`라서 로컬 실행 시 별도 지정이 없으면 `.env.dev`가 적용된다.

현재 권장 구성은 아래와 같다.

- `.env`: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `SESSION_SECRET`
- `.env.dev`: `GOOGLE_REDIRECT_URI`, `SESSION_HTTPS_ONLY`, `PROXY_HEADERS`
- `.env.prod`: `GOOGLE_REDIRECT_URI`, `SESSION_HTTPS_ONLY`, `PROXY_HEADERS`

`SESSION_SECRET`는 세션 쿠키 서명 키이므로 반드시 설정해야 한다. 운영환경에서는 최소 32자 이상의 강한 랜덤 문자열을 사용해야 한다.

생성 예시:

```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

## 로컬 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 7300 --reload
```

브라우저 접속 주소는 `http://127.0.0.1:7300`이다.

개발환경에서는 기본값이 `development`이므로 보통 `APP_ENV`를 따로 줄 필요가 없다.

## 운영 실행

운영환경 설정을 적용하려면 프로세스 시작 시 `APP_ENV=prod`를 명시해야 한다.

```bash
APP_ENV=prod uvicorn app:app --host 127.0.0.1 --port 7300
```

백그라운드 실행 예시:

```bash
nohup env APP_ENV=prod uvicorn app:app --host 127.0.0.1 --port 7300 > app.log 2>&1 &
```

운영에서는 `SESSION_SECRET`를 `.env` 파일에 넣지 말고 서버 환경변수로 주는 방식을 권장한다.

현재 README에는 이 방식이 부분적으로만 있었고, 아래처럼 실행하면 된다.

임시 셸 세션에서 실행:

```bash
export SESSION_SECRET='여기에-충분히-긴-랜덤-문자열'
export APP_ENV=prod
uvicorn app:app --host 127.0.0.1 --port 7300
```

한 줄 실행:

```bash
SESSION_SECRET='여기에-충분히-긴-랜덤-문자열' APP_ENV=prod uvicorn app:app --host 127.0.0.1 --port 7300
```

`nohup` 백그라운드 실행:

```bash
nohup env SESSION_SECRET='여기에-충분히-긴-랜덤-문자열' APP_ENV=prod uvicorn app:app --host 127.0.0.1 --port 7300 > app.log 2>&1 &
```

`systemd` 사용 시에는 서비스 파일에 직접 넣거나 `EnvironmentFile`로 분리한다.

예시:

```ini
[Service]
Environment=APP_ENV=prod
Environment=SESSION_SECRET=여기에-충분히-긴-랜덤-문자열
ExecStart=/경로/to/venv/bin/uvicorn app:app --host 127.0.0.1 --port 7300
```

## OAuth 및 세션 주의사항

- `GOOGLE_REDIRECT_URI`는 실제 로그인 시작 호스트와 정확히 같아야 한다.
- 로그인 시작 호스트와 콜백 호스트가 다르면 `mismatching_state` 오류가 발생한다.
- HTTPS 리버스 프록시 뒤의 운영환경에서는 `SESSION_HTTPS_ONLY=true`, `PROXY_HEADERS=true`를 사용해야 한다.
- `/auth/login`은 현재 요청 호스트 또는 `GOOGLE_REDIRECT_URI`를 기준으로 콜백 URL을 결정한다.
- 운영환경에서 약한 `SESSION_SECRET` 또는 누락된 `SESSION_SECRET`는 서버가 즉시 거부한다.

## 주요 기능

- `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/me`
  Google 로그인 및 세션 확인
- `/api/holdings`
  사용자 보유 종목 조회/추가/수정/삭제
- `/api/holdings_raw`
  원본 보유 종목 목록 조회
- `/api/holdings/bulk_replace`
  전체 종목 일괄 교체
- `/api/import_csv_text`
  CSV 텍스트 가져오기
- `/api/quote`
  원본 통화 기준 실시간 시세 조회
- `/api/quote_krw`
  KRW 환산 시세 조회
- `/admin/users`
  관리자용 사용자 목록/생성/수정/삭제

## UI 동작 규칙 (현행)

- 미로그인 상태에서 CSV 가져오기 버튼을 누르면 `/auth/login`으로 즉시 이동한다.
- 보유 자산 목록이 0건이면 `수정사항저장`, `실시간정보갱신` 버튼은 화면에서 숨긴다.
- 보유 자산 목록이 1건 이상이면 위 두 버튼을 표시한다.
- 보유 자산 차트 데이터가 없으면 차트 카드에는 `데이터 없음`만 표시한다.
- CSV 입력 placeholder는 보유 자산이 있을 때 `ticker,quantity`만 표시하고, 없을 때 샘플 데이터를 표시한다.
- CSV의 현금 티커는 `na`, `NA`, `현금`, `cash`, `예수금`을 모두 `NA`로 정규화한다.

## 가격 조회 규칙

- 한국 종목: 네이버 금융
- 해외 종목: Yahoo JSON -> Yahoo HTML -> Investing -> Stooq 순서로 fallback
- 환율: Yahoo -> exchangerate.host -> open.er-api.com 순서로 fallback
- `NA` 티커: 예수금으로 처리하며 1 KRW로 계산

## 관리자 기능

- 첫 로그인 사용자가 자동으로 관리자(`is_admin=1`)가 된다.
- 관리자는 사용자 목록 조회, 수동 사용자 추가, 이름/관리자 권한 수정, 사용자 삭제를 수행할 수 있다.
- 일반 사용자는 관리자 화면 버튼을 보지 못하며 `/admin/*` 요청 시 403을 받는다.

## 기타

- API 확인은 `http://127.0.0.1:7300/docs`에서 가능하다.
- 포트가 점유되어 있으면 `lsof -ti:7300 | xargs kill -9`로 종료할 수 있다.
- `portfolio.db`는 실행 시 자동 생성된다.

## one-shot 스크립트

- 개발 시작: `./one-shot-startup.sh`
- 개발 중지: `./one-shot-stop.sh`
- 운영 시작: `APP_ENV=prod ./one-shot-startup.sh`
- 운영 중지: `APP_ENV=prod ./one-shot-stop.sh`
