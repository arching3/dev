# Week 10 — 컴퓨터 비전 학습 환경

## 이 브랜치의 실행 방법

```bash
git clone --branch week10 git@github.com:arching3/dev.git
cd dev
python build.py --setup-only
```

`--setup-only`는 환경만 준비하므로 `.ipynb` 작업에도 사용할 수 있습니다. 학습 스크립트는 `team.py`이며, CIFAR-10 데이터셋은 최초 실행 시 `data/`로 다운로드합니다. `build.py`의 기본 진입 파일은 `train.py`이므로 아래와 같이 `--script team.py`를 지정하세요.

환경 구성과 두 실험을 한 번에 실행하려면:

```bash
python build.py --script team.py

# 두 실험의 에포크를 모두 10으로 변경하고 그래프를 화면에도 표시
python build.py --script team.py -- --epochs 10 --show
```

`team.py`는 다음 두 실험을 순서대로 수행합니다. 각 실험에 공통 `fit`을 적용하고 `evaluate_history`가 2행 × 2열 subplot으로 손실과 정확도를 비교합니다. tqdm으로 실험·에포크·배치 진행을 표시합니다.

| 실험 | 모델·데이터 | 기본 에포크 | Optimizer | 학습률 | 배치 크기 |
| --- | --- | --- | --- | --- | --- |
| 2 | CNN_v4, 정규화 | 50 | Adam | 0.001 | 100 |
| 3 | CNN_v4, 정규화 + RandomHorizontalFlip + RandomErasing | 100 | Adam | 0.001 | 100 |

각 실험의 `history.csv`와 `model_final.pth`는 `outputs/experiment2_cnn_v4/`, `outputs/experiment3_cnn_v4_aug/`에 저장됩니다. 비교 그래프는 `outputs/experiments.png`로 저장되며, `--show`로 화면에도 표시할 수 있습니다.

`--script` 경로와 학습 작업 디렉터리는 `build.py`가 있는 디렉터리를 기준으로 합니다. 다른 디렉터리에서 `python /path/to/dev/build.py --setup-only`로 호출해도 환경과 패키지 파일 위치는 동일합니다. 재실행 시 `.venv`를 재사용하고 패키지 설치를 다시 확인합니다. 기존 환경의 불필요한 패키지를 제거하거나 Python 버전을 바꾸지는 않습니다.

### 노트북 사용

VS Code에서는 노트북 커널 선택 메뉴에서 이 저장소의 `.venv` Python을 선택합니다. 또는 환경 구성 후 다음 명령으로 JupyterLab을 실행합니다.

```bash
# Linux
.venv/bin/python -m jupyterlab

# Windows PowerShell
.venv/Scripts/python.exe -m jupyterlab
```

이 JupyterLab에서 `Python 3 (ipykernel)` 커널을 사용합니다. `ipykernel`, JupyterLab은 requirements에 포함되어 있습니다. 사용자 전역 커널 등록은 자동으로 수행하지 않습니다.

### 호환 조건과 공통 개발 환경

`requirements.txt`는 학습과 노트북 작업에 필요한 주요 패키지 이름만 기록합니다. 버전, CUDA 전용 인덱스, NVIDIA 런타임과 Triton의 직접 지정을 제거하고, 하위 의존성은 pip가 현재 Python과 플랫폼에 맞춰 결정하도록 합니다.

`build.py`는 `pip install --upgrade --upgrade-strategy eager`로 실행할 때마다 현재 Python과 패키지 간 의존성 조건에서 허용되는 최신 버전을 설치합니다. cuDNN은 독립적으로 최신 버전을 강제하지 않고, 설치되는 PyTorch 배포판이 요구하는 호환 버전을 사용합니다. PyTorch가 특정 cuDNN 버전을 요구하면 그 조건이 우선합니다. CPU 또는 macOS용 배포판에는 cuDNN이 필요하지 않습니다.

패키지가 해당 운영체제·CPU 아키텍처·Python 버전을 지원해야 하며, 버전 미지정만으로 모든 플랫폼의 호환성이 보장되지는 않습니다. NVIDIA GPU 사용에는 호환 드라이버와 CUDA 지원 PyTorch 배포판이 필요합니다. 플랫폼에 따라 기본 PyPI 설치가 CPU용일 수 있으므로 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 환경에 맞는 설치 명령과 인덱스를 확인하세요. 사용자 지정 인덱스는 pip의 설정 또는 `PIP_EXTRA_INDEX_URL` 환경 변수로 전달할 수 있습니다. 드라이버는 이 스크립트에서 설치하지 않습니다. Python의 `venv`·`ensurepip` 지원과 설치용 네트워크가 필요합니다.

현재 개발 프로젝트의 공통 `../.venv`는 계속 사용할 수 있습니다. `build.py`는 저장소 내부의 별도 `.venv`만 생성·갱신합니다. 새로운 의존성은 패키지 이름으로 추가하며, 버전을 고정하지 않으므로 설치 시점에 따라 환경이 달라질 수 있습니다. 이번 변경에 대한 패키지 재설치와 GPU 학습 검증은 수행하지 않았습니다.

---

## 저장소 운영 방향

주차별 학습 코드를 관리하고, 더 높은 컴퓨팅 성능을 갖춘 환경에서 한 번의 명령으로 가상환경 구성, 패키지 설치, 학습 실행을 이어가는 저장소입니다.

## 브랜치 운영

- `main`: 저장소 목적과 공통 운영 방향을 기록합니다.
- `week10`: 10주차 학습 코드, `build.py`, `requirements.txt`, 실행 안내를 관리합니다.
- 이후 학습 코드는 해당 주차 브랜치에서 관리합니다.

## Windows에서 Git 설치

CMD 또는 PowerShell에서 다음 명령으로 Git CLI를 설치합니다. `winget`이 없다면 Microsoft Store에서 **앱 설치 관리자(App Installer)**를 설치하거나 업데이트하세요.

```powershell
winget install --id Git.Git -e --source winget
```

설치가 완료되면 터미널을 닫았다가 다시 열고 설치 여부를 확인합니다.

```powershell
git --version
```

SSH 키를 아직 설정하지 않았다면 HTTPS 주소로 원하는 브랜치를 복제할 수 있습니다. 비공개 저장소는 GitHub 인증이 필요합니다.

```powershell
git clone --branch week10 https://github.com/arching3/dev.git
cd dev
```

이후 아래 명령의 SSH 주소도 `https://github.com/arching3/dev.git`으로 바꿔 사용할 수 있습니다. 설치 명령은 [Git 공식 Windows 설치 안내](https://git-scm.com/install/windows)를 참고하세요.

## 저장소 복제와 브랜치 선택

Git이 설치되어 있어야 합니다. 아래 SSH 주소를 사용하려면 GitHub 계정의 SSH 키 인증이 설정되어 있어야 합니다. 두 복제 방법 중 하나를 선택하세요.

### 특정 브랜치만 복제

10주차 작업만 필요한 경우:

```bash
git clone --branch week10 --single-branch git@github.com:arching3/dev.git
cd dev
python build.py --setup-only
```

저장소 소개만 확인하려면 `--branch main`으로 변경합니다. `main`에는 실행 파일이 없으므로 `build.py`는 학습 브랜치에서 실행하세요.

`--single-branch`로 복제한 뒤 다른 브랜치도 필요해지면, 예를 들어 다음과 같이 가져올 수 있습니다.

```bash
git fetch origin main:refs/remotes/origin/main
git switch main
```

### 전체 브랜치를 가져온 뒤 전환

여러 주차를 오가며 작업하려면 `--single-branch` 없이 복제합니다. 원격 브랜치는 가져오지만, 각 브랜치의 파일은 선택한 브랜치로 전환할 때 작업 디렉터리에 나타납니다.

```bash
git clone git@github.com:arching3/dev.git
cd dev
git branch -a
git switch week10
python build.py --setup-only

# 저장소 소개로 돌아가기
git switch main
```

이미 복제한 저장소라면 해당 디렉터리에서 `git fetch origin`으로 브랜치 정보를 갱신하고 `git switch week10`으로 전환합니다. 로컬에 없는 브랜치가 `origin`에 있으면 `git switch`가 추적 브랜치를 자동 생성합니다. 전환 전 작업 내용은 커밋하거나 `git stash`로 보관하세요.

## 변경 사항을 최신 버전으로 업데이트

새로 복제할 필요 없이, 기존 저장소 디렉터리에서 사용할 브랜치로 전환한 뒤 업데이트합니다.

```bash
git switch week10
git pull --ff-only origin week10
```

`main`의 README를 업데이트하려면 다음 명령을 사용합니다.

```bash
git switch main
git pull --ff-only origin main
```

`git fetch`는 원격 변경 정보를 가져오고, `git pull`은 현재 브랜치의 파일에도 변경을 반영합니다. `--ff-only`는 자동 병합 커밋을 만들지 않으며, 로컬과 원격 이력이 갈라져 있으면 중단합니다.

로컬에서 수정한 파일이 있다면 먼저 커밋하거나 다음과 같이 임시 보관합니다. `-u`는 새로 만든 미추적 파일도 포함하며, `.venv`나 데이터처럼 Git에서 무시하는 파일은 제외합니다.

```bash
git stash push -u -m "local work before update"
git switch week10
git pull --ff-only origin week10
# 변경을 보관했던 원래 브랜치에서 복원
git stash pop
```

위 예시는 `week10`에서 작업하던 경우입니다. 다른 브랜치에서 보관했다면 그 브랜치로 돌아온 뒤 `git stash pop`을 실행하세요. 복원 중 충돌이 나면 해당 파일의 충돌을 해결해야 합니다.

패키지 목록이 바뀌었다면 `week10`에서 다음 명령으로 기존 가상환경의 의존성을 다시 설치합니다. 학습은 실행하지 않습니다.

```bash
python build.py --setup-only
```

## 목표 실행 흐름

실행 플랫폼에는 Python이 설치되어 있다고 가정합니다. 각 주차 브랜치에서 다음 명령으로 실행합니다.

```bash
python build.py
```

`build.py`는 자신의 디렉터리에 `.venv`를 생성하고, 해당 가상환경의 Python으로 `requirements.txt`를 설치한 뒤 학습 코드를 실행합니다. 기존 `.venv`는 재사용하며 수동 활성화는 필요하지 않습니다. 설치나 학습에 실패하면 종료 코드를 전달합니다.

노트북 작업을 위해 환경 구성만 수행할 수도 있습니다.

```bash
python build.py --setup-only
```

## 환경 관리

현재 개발 프로젝트에서는 루트의 `.venv`를 week1~10이 공유합니다. `week10/requirements.txt`에는 주요 패키지 이름을 버전 없이 기록합니다. 다른 환경에서는 저장소 내부에 별도의 `.venv`를 생성합니다. 가상환경, 데이터셋, 모델 가중치, 학습 산출물과 비밀정보는 Git에서 제외합니다.

패키지는 현재 플랫폼과 의존성 조건에서 허용하는 최신 버전을 사용하고, CUDA·cuDNN 관련 의존성은 PyTorch 배포판에 맡깁니다. GPU 드라이버와 Python의 venv/pip 지원은 실행 환경에서 준비해야 합니다. 설치 시점에 따라 버전이 달라지므로 동일한 환경의 재현을 보장하지 않습니다.

학습 진입 파일, 실행 인자, 데이터셋 준비 방법은 학습 코드를 추가하면서 구체화합니다. 실제 사용 방법과 구현 상태는 해당 주차 브랜치를 확인하세요.
