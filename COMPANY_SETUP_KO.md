# 회사 PC 이전/운영 가이드

이 프로젝트는 코드뿐 아니라 학습용 로컬 DB(`data/cache.sqlite3`)를 같이 가져가야 같은 상태로 이어서 학습할 수 있습니다.

## 1. 회사 PC 최초 설치

```powershell
git clone https://github.com/hansol07/money.git
cd money
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

## 2. 실행

```powershell
python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false
```

또는:

```powershell
.\scripts\start_app.ps1
```

## 3. 집 PC에서 회사 PC로 데이터 갱신 전 준비

서버를 끈 뒤 아래 명령으로 SQLite WAL을 본 DB에 합칩니다.

```powershell
.\scripts\prepare_data_for_git.ps1
git add .gitattributes .gitignore app.py src data/cache.sqlite3 data/*.json data/universe_*.csv requirements.txt COMPANY_SETUP_KO.md scripts
git commit -m "Sync stock helper app and local learning database"
git push
```

## 4. 회사 PC에서 최신 데이터 받기

```powershell
git pull
git lfs pull
.\scripts\start_app.ps1
```

## 5. 중요한 운영 원칙

- `data/cache.sqlite3`는 Git LFS로 관리합니다. 일반 Git 파일로 올리면 GitHub 용량 제한에 걸립니다.
- `data/cache.sqlite3-wal`, `data/cache.sqlite3-shm`은 올리지 않습니다. `prepare_data_for_git.ps1`이 본 DB에 합칩니다.
- 회사 PC에서 계속 학습한 뒤 집 PC로 다시 가져오려면 회사 PC에서도 같은 방식으로 commit/push 해야 합니다.
- 동시에 집/회사 양쪽에서 DB를 수정하고 각각 push하면 SQLite DB 충돌이 날 수 있습니다. 한쪽을 주 운영 PC로 정하는 게 안전합니다.

