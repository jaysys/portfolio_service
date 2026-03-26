# 로컬 개발 가이드

이 프로젝트는 기본적으로 개발환경에서 `.env` + `.env.dev` 조합으로 실행된다.

## 1. 준비

- Python 3.9 이상
- Google OAuth 클라이언트 ID/Secret

## 2. 환경 파일

현재 파일 역할은 아래와 같다.

- `.env`
  공통 비밀값과 공통 설정
- `.env.dev`
  로컬 개발 전용 설정
- `.env.prod`
  운영 전용 설정

앱 로딩 순서는 아래와 같다.

1. `.env` 로드
2. `APP_ENV` 해석
3. 개발이면 `.env.dev`, 운영이면 `.env.prod` override 로드

개발환경 기본값은 `development`라서 로컬에서는 보통 `APP_ENV`를 지정하지 않는다.

현재 권장 값은 아래와 같다.

- `.env`
  `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `SESSION_SECRET`
- `.env.dev`
  `GOOGLE_REDIRECT_URI=http://127.0.0.1:8200/auth/callback`
  `SESSION_HTTPS_ONLY=false`
  `PROXY_HEADERS=false`

주의사항:

- `GOOGLE_REDIRECT_URI`는 브라우저가 실제로 접속하는 호스트와 같아야 한다.
- 로그인 시작 호스트와 콜백 호스트가 다르면 `mismatching_state`가 발생한다.
- 운영환경에서는 `APP_ENV=prod`로 실행해야 `.env.prod`가 적용된다.
- `SESSION_SECRET`는 필수이며, 운영환경에서는 최소 32자 이상의 강한 랜덤 문자열이어야 한다.

랜덤 값 생성 예시:

```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
```

## 3. 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8200 --reload
```

접속 주소:

- 앱: `http://127.0.0.1:8200`
- Swagger UI: `http://127.0.0.1:8200/docs`

## 4. 종료

터미널에서 `Ctrl + C`로 종료한다.

포트 점유로 강제 종료가 필요하면:

```bash
lsof -ti:8200 | xargs kill -9
```

## 5. 기능 확인 포인트

- 로그인 전에는 보유 종목 조작 버튼이 비활성화된다.
- 로그인 후 보유 종목 CRUD와 CSV 가져오기를 사용할 수 있다.
- 첫 가입자는 관리자 권한을 받는다.
- 관리자 계정으로 로그인하면 사용자 관리 대시보드를 사용할 수 있다.
