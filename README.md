# Stock Decision Helper MVP

한국/미국 주식을 함께 조회하고, 보유 종목 기준으로 간단한 매수/매도 판단을 보여주는 Streamlit 기반 MVP입니다.

## 기능

- 미국/한국 종목 가격 조회
- 이동평균, RSI, MACD 계산
- 보유 수량과 평균 단가 입력
- 평가손익 계산
- `홀드`, `일부매수`, `일부매도`, `관망` 추천
- 일부매수/일부매도 비중 힌트
- 추천 사유 표시
- 미국/한국 대표 종목군 대상 오늘 매수 후보 스캐너
- 보유 종목보다 강한 후보 비교
- 보유 종목 로컬 저장/불러오기
- CSV 보유 종목 업로드 및 샘플 다운로드
- 추천 후보군 직접 편집 및 저장
- 추천 점수 기준과 상위 표시 개수 조절
- 점수 기반 간단 백테스트 화면
- 추천 종목별 세부 점수와 급등 후보 보드
- 분봉 기반 실시간 급등주 스캐너
- 위험/성장/배당/집중도 기반 포트폴리오 분석
- 월간/주간/일간/내일 급등 후보 자동 후보군
- 후보군 스냅샷 누적 저장
- 자동 후보 추적 성과판
- 장기 복리형 후보 스크리너

## 빠른 시작

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

파이썬 경로를 직접 써야 하면:

```powershell
& "C:\Users\njwjs\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

## 한국 종목 입력 예시

- 삼성전자: `005930.KS`
- SK하이닉스: `000660.KS`
- NAVER: `035420.KS`

## 프로젝트 구조

```text
app.py
data/
src/
  backtest/
  data/
  indicators/
  portfolio/
  storage/
  strategy/
  ui/
```

## Git 업로드 전 참고

- `.venv`, `__pycache__`, `.tmp`는 Git에서 제외합니다.
- `data/*.json` 같은 로컬 누적 데이터는 개인 실행 기록이라 Git에서 제외합니다.
- 회사 서버에서 돌릴 때는 서버 쪽에서 새로 누적되도록 두는 편이 깔끔합니다.

## 회사 서버 실행 메모

```powershell
git clone <your-repo-url>
cd 주식
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 다음 확장 포인트

- PostgreSQL 연동
- 사용자별 포트폴리오 저장
- 전 종목 기반 실시간 급등주 스캐너
- 텔레그램/디스코드 알림
- 백테스트 엔진 고도화
