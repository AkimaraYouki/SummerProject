# IsaacLab 최초 설치 (랩 PC, 한 번만)

`~/IsaacLab`는 클론만 돼 있고 확장 패키지가 설치 안 된 상태였다(2026-07-25 확인).
`convert_urdf.sh`/`train.sh`/`play.sh`를 처음 쓰기 전에 이 설치를 한 번 해야 한다.

## 1. conda 비활성화부터

인터랙티브 쉘에서 `.bashrc`의 conda init 블록 때문에 `base` 환경이 자동 활성화돼 있다.
`isaaclab.sh`는 `$CONDA_PREFIX`가 설정돼 있으면 **번들 파이썬 대신 conda 파이썬을 씀**
(이 랩 PC의 Isaac Sim은 conda 환경이 아니라 독립 번들 설치본이라 conda 쪽엔 `isaacsim`
모듈이 없음 → `ModuleNotFoundError: No module named 'isaacsim'`로 실패):

```bash
conda deactivate
```

## 2. 설치 실행

```bash
cd ~/IsaacLab
./isaaclab.sh --install
```

PyTorch(1.1GB) 등 큰 다운로드가 있어서 시간이 걸린다.

## ⚠️ 알려진 실패 지점: flatdict 빌드 에러

```
Getting requirements to build wheel: finished with status 'error'
ModuleNotFoundError: No module named 'pkg_resources'
```

`flatdict==4.0.1`(오래된 패키지, `setup.py`가 구식 `pkg_resources` 임포트에 의존)이 pip의
격리된 빌드 환경에서 실패한다. 격리된 빌드 환경은 최신 setuptools를 새로 받는데, 거기엔
`pkg_resources`가 없을 수 있음. **메인 환경의 setuptools는 이미 갖고 있으므로**, 격리 없이
직접 설치해서 우회:

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip install flatdict==4.0.1 --no-build-isolation
```

성공하면(`Successfully installed flatdict-4.0.1`) `./isaaclab.sh --install`을 다시 실행 —
`flatdict`는 "already satisfied"로 건너뛰고 나머지 확장(`isaaclab_mimic`, `isaaclab_rl` 등)을
이어서 설치한다.

## 정상적으로 오래 걸리는 구간 (멈춘 게 아님)

- `pytransform3d`/`eigenpy` 버전 호환성 때문에 pip이 여러 버전을 내려받으며 되짚는
  (backtracking) 구간이 있다 — 수 분 걸릴 수 있음, 로그가 안 바뀌어 보여도 정상 진행 중.
- `isaaclab_rl` 설치 과정에서 torch가 2.7.0 → 2.13.0으로 재설치될 수 있음(다른 확장의
  버전 요구사항 때문) — 정상 동작.

## 설치 확인

```bash
~/IsaacLab/_isaac_sim/python.sh -m pip list | grep isaaclab
```

`isaaclab`, `isaaclab_assets`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks` 5개가 다 떠야
한다.

## 검증된 결과 (2026-07-25)

설치 후 `ISAACLAB_PATH=~/IsaacLab ./scripts/convert_urdf.sh --headless`로 `robot/robot.urdf`를
`robot/usd/open_duck_mini_v2.usd`로 성공적으로 변환함. (출력 경로는 원래 최상위 `assets/usd/`였다가,
모든 임포트 결과물을 한곳에 모으기 위해 `robot/usd/`로 옮김 — `robot/`가 이제 URDF·메시·USD 전부의
캐노니컬 위치.)

**한 가지 주의(해결됨, 2026-07-25 같은 날 후속조치)**: 첫 변환 로그엔
```
The path xm430_어셈 is not a valid usd path, modifying to xm430_______
```
경고가 있었다 — USD는 경로 이름에 비-ASCII 문자(한글)를 허용 안 해서 자동으로 밑줄로
치환됐던 것. OnShape에서 `xm430_어셈` → `xm430_assem`으로 영문 리네임한 뒤 재임포트 +
재변환하니 이 경고가 완전히 사라짐(`grep "not a valid usd path"` 결과 0건). 확인된
교훈: **OnShape 서브어셈블리·파츠 이름은 반드시 영문으로 지을 것** — 한글 이름은 USD
변환 시 자동으로 밑줄 치환되어 원래 이름을 알아보기 어려워진다.
