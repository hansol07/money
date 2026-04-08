# Stock Decision Helper

한국/미국 주식을 함께 보면서 보유 종목 관리, 추천 후보 확인, 단타 플랜, 배당주 분석, 자동 후보 추적까지 한 번에 보는 Streamlit 기반 개인용 투자 보조 앱입니다.

## 빠른 실행

가장 간단한 실행:

```powershell
& "C:\Users\njwjs\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

가상환경으로 실행:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501` 을 열면 됩니다.

## 주요 기능

- 종목 분석: 단일 종목 차트, 지표, 현재 판단 확인
- 보유 종목: 포트폴리오 입력, 저장, 분석, 리밸런싱 제안
- 오늘 추천: 오늘 강한 후보와 급등 후보 보드 확인
- 전략별 추천: 안정형, 배당형, 성장형 후보 확인
- 배당주: 미국/한국 배당 후보, 배당 성장, 배당락일, 모아가기 구간 확인
- 단타: 일반 단타와 고위험 단타 분리 확인
- 관심 추적: 직접 골라둔 종목만 따로 추적
- 자동 후보군: 월간, 주간, 일간, 내일 후보 자동 생성
- 추적: 추천 후보의 이후 성과, 목표 도달, 손절 도달 추적
- 실시간: 분봉 기반 장중 후보 확인
- 백테스트: 점수 로직 기반 간단 백테스트
- 후보군 관리: 미국/한국 관찰 후보 직접 편집

## 파일 안내

- [상세 한국어 매뉴얼](C:\Users\njwjs\Desktop\개발\주식\MANUAL_KO.md)
- 실행 파일: [app.py](C:\Users\njwjs\Desktop\개발\주식\app.py)
- 저장 폴더: [data](C:\Users\njwjs\Desktop\개발\주식\data)

## 저장 방식

현재는 로컬 JSON 저장입니다.

- 포트폴리오 저장
- 후보군 저장
- 추천 스냅샷 저장
- 학습용 피처 로그 저장
- 관심 추적 저장

즉 혼자 쓰는 환경에 맞춘 가벼운 구조입니다.

## Git 업로드 참고

다음 항목은 저장소에서 제외됩니다.

- `.venv`
- `__pycache__`
- `.tmp`
- `data/*.json`
- `data/*.db`

코드와 구조만 GitHub에 올리고, 누적 데이터는 실행 환경에서 새로 쌓는 방식입니다.

## 회사 서버 실행 예시

```powershell
git clone https://github.com/hansol07/money.git
cd money
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
