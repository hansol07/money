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
- 보유 종목: 포트폴리오 입력, 저장, 현재가/당일등락 반영 진단, 리밸런싱 제안
- 오늘 추천: 오늘 진입 시 이번 주 또는 근시일 안의 상승을 기대할 만한 최근 차트 후보 확인
- 전략별 추천: 안정형, 배당형, 성장형 후보 확인
- 배당주: 미국/한국 배당 후보, 배당 성장, 배당락일, 모아가기 구간 확인
- 단타: 지금 진입해서 짧게 수익을 먹고 나오는 일반 단타와 고위험 단타 분리 확인
- 장기 추천: 차트, 뉴스/이벤트, 장세, 상대강도를 종합해 계속 모아갈 후보 확인
- 관심 추적: 직접 골라둔 종목만 따로 추적하고 현재가, 현재수익률, 목표/손절 상태 요약 확인
- 자동 후보군: 월간, 주간, 일간, 내일 후보 자동 생성
- 추적: 추천 후보의 이후 성과, 목표 도달, 손절 도달 추적
- 실시간: 분봉 기반 장중 후보 확인
- 백테스트: 점수 로직 기반 간단 백테스트
- 후보군 관리: 미국/한국 관찰 후보 직접 편집

## 현재 상태 요약

- 추천/단타/실시간/장기 후보에는 기본적으로 `현재가`, `진입가`, `손절가`, `1~3차 목표`, `시세기준`, `데이터 최신성` 표시가 들어갑니다.
- `관심 추적` 탭은 현재가, 현재수익률, 목표/손절 상태 요약까지 바로 볼 수 있습니다.
- `실시간` 탭은 초단위 체결형이 아니라 `분봉 기반 준실시간 스캐너`입니다.
- 장이 닫혀 있거나 분봉 데이터가 비어 있으면 후보가 없을 수 있지만, 화면은 오류 없이 유지되는 방향으로 방어되어 있습니다.

## 파일 안내

- [상세 한국어 매뉴얼](C:/work/테스트/주식/money/MANUAL_KO.md)
- [다음 개발 TODO](C:/work/테스트/주식/money/TODO.md)
- 실행 파일: [app.py](C:/work/테스트/주식/money/app.py)
- 저장 폴더: [data](C:/work/테스트/주식/money/data)

## 저장 방식

현재는 `JSON + SQLite` 혼합 저장 구조입니다.

- JSON:
- 포트폴리오 저장
- 워치리스트 저장
- 관심 추적 저장
- 데일리 브리핑 / 결정 로그 저장

- SQLite:
- 시세 캐시 저장
- 추천 스냅샷 저장
- 학습용 피처 로그 저장
- 가격 바(일봉/분봉) 웨어하우스 저장

즉 가벼운 로컬 앱 구조는 유지하되, 추천/학습/캐시 쪽은 로컬 DB로 확장한 상태입니다.

## Git 업로드 참고

다음 항목은 저장소에서 제외됩니다.

- `.venv`
- `__pycache__`
- `.tmp`
- `data/cache.sqlite3-wal`
- `data/cache.sqlite3-shm`

운영 시에는 JSON 데이터와 `data/cache.sqlite3`를 같이 관리할 수 있습니다.

## 회사 서버 실행 예시

```powershell
git clone https://github.com/hansol07/money.git
cd money
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 문서 운영 원칙

앞으로 Git에 push 할 때는 아래 문서를 같이 갱신하는 것을 기본 원칙으로 둡니다.

- `README.md`: 빠른 실행, 핵심 기능, 진입 안내
- `MANUAL_KO.md`: 실제 사용 방법과 화면별 설명
- `TODO.md`: 다음 개발 항목과 우선순위
