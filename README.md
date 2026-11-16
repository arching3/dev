# 컴퓨터 비전 학습 코드

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

현재 개발 프로젝트에서는 루트의 `.venv`를 week1~10이 공유합니다. `week10/requirements.txt`는 이 공통 환경의 설치 버전을 기록합니다. 다른 환경에서는 저장소 내부에 별도의 `.venv`를 생성합니다. 가상환경, 데이터셋, 모델 가중치, 학습 산출물과 비밀정보는 Git에서 제외합니다.

현재 기준 환경은 Python 3.14.4, Linux, CUDA 13.0용 PyTorch입니다. 패키지 버전 고정만으로 운영체제, GPU 드라이버, 데이터셋까지 재현되지는 않습니다. GPU 드라이버와 Python의 venv/pip 지원은 실행 환경에서 준비해야 하며, 플랫폼별 설치 조건은 주차 브랜치의 README에서 설명합니다.

학습 진입 파일, 실행 인자, 데이터셋 준비 방법은 학습 코드를 추가하면서 구체화합니다. 실제 사용 방법과 구현 상태는 해당 주차 브랜치를 확인하세요.
