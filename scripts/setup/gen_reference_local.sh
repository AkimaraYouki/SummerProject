#!/usr/bin/env bash
# 레퍼런스 보행을 **이 PC 에서** 생성한다. (랩PC 원격판은 gen_reference_remote.sh)
#
# 왜 로컬이 되는가. 2026-07-30 에 "이 PC 에는 placo 를 못 깐다" 고 결론냈던 건
# 원인 진단이 틀렸다. 진짜 원인은 설치하려던 환경(~/.odm-tools)의 **numpy 가 2.x**
# 라서였다 — placo 0.6.3 이 딸고 오는 cmeel 스택(eigenpy 3.5.1 / pin 2.7.0 /
# hpp-fcl 2.4.4)은 numpy 1.x ABI 로 빌드돼 있어서, numpy 2 환경에서는 pip 가
# 맞는 wheel 을 못 고르고 소스 빌드로 떨어진다. numpy==1.26.4 를 먼저 박은
# **전용 venv** 에 깔면 42개 패키지가 전부 wheel 로 들어온다 (2026-08-08 확인).
#
#   python3 -m venv ~/.placo-env
#   ~/.placo-env/bin/pip install --only-binary=:all: numpy==1.26.4 placo==0.6.3 matplotlib scipy
#
# ~/.odm-tools 를 건드리지 않고 따로 두는 이유: 거기엔 numpy 2 + libpinocchio
# 4.1 이 있고 teleop / 간섭검사가 그걸 쓴다. 둘을 한 환경에 못 섞는다.
#
# ⚠️ auto_waddle.py 는 서브프로세스를 PATH 의 `python3` 로 띄운다. 그래서 venv 를
#    PATH 맨 앞에 놓아야 한다 — 안 그러면 시스템 python3 가 placo 를 못 찾고
#    로그 파일만 0 바이트로 쌓인 채 "성공" 처럼 끝난다 (2026-07-26 에 겪었다).
#
#   ./scripts/setup/gen_reference_local.sh --height 0.1873 --out ref_g115m
#
# 결과: source/open_duck_mini_isaaclab/reference_motion/data/<out>.pkl
# 쓰려면 joystick_env_cfg 의 reference_motion_pkl 을 그 파일로 바꾸고,
# scripts/diag/calc_home.py 로 READY_JOINT_POS / READY_BASE_HEIGHT 를 다시 뽑아야
# 한다 — 기본 자세와 레퍼런스가 어긋나면 v1~v9 가 아홉 번 실패한 그 상황이 된다.
set -euo pipefail

PLACO_PY="${ODM_PLACO_PY:-$HOME/.placo-env/bin/python}"
HEIGHT=""
YAW_SWEEP=""
PRESET_SET=""
OUT="polynomial_coefficients"
JOBS=8

while [ $# -gt 0 ]; do
  case "$1" in
    --height) HEIGHT="$2"; shift 2 ;;
    # yaw 격자를 대칭으로. 기본값이 0 을 비껴가는 문제는 gen_reference_remote.sh
    # 의 주석에 자세히 적어 두었다.
    --yaw-sweep) YAW_SWEEP="$2"; shift 2 ;;
    # medium 프리셋의 임의 키를 덮어쓴다: --preset foot_zmp_target_y=0.0
    --preset) PRESET_SET="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    *) echo "모르는 인자: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN_DIR="$REPO_ROOT/reference_motion_generator"
DATA_DIR="$REPO_ROOT/source/open_duck_mini_isaaclab/reference_motion/data"
# 레포 밖에서 돌린다. 프리셋을 그 자리에서 고쳐야 하는데 레포를 더럽히면 안 되고,
# 녹화물(수백 MB)도 레포에 남기고 싶지 않다.
WORK="$HOME/odm_out/refgen_local/$OUT"

echo "== 1/5 placo 확인 ($PLACO_PY)"
[ -x "$PLACO_PY" ] || { echo "!! placo venv 가 없다: $PLACO_PY  (파일 상단 주석의 설치 명령 참고)" >&2; exit 1; }
"$PLACO_PY" -c "import placo" 2>/dev/null \
  || { echo "!! placo 를 못 찾습니다: $PLACO_PY" >&2; exit 1; }
export PATH="$(dirname "$PLACO_PY"):$PATH"     # 서브프로세스의 python3 도 이걸 쓰게

echo "== 1.5/5 생성기 URDF 를 robot/robot.urdf 와 맞춤"
"$PLACO_PY" "$REPO_ROOT/scripts/setup/patch_urdf_for_placo.py" | tail -3

echo "== 2/5 생성기 복사 -> $WORK"
mkdir -p "$WORK"
rsync -a --delete --exclude 'recordings/' --exclude '__pycache__/' \
  "$GEN_DIR/" "$WORK/reference_motion_generator/"

if [ -n "$HEIGHT" ]; then
  echo "== 3/5 walk_com_height -> $HEIGHT (medium 프리셋)"
  "$PLACO_PY" - "$WORK" "$HEIGHT" <<'PY'
import json, pathlib, sys
work, height = sys.argv[1], sys.argv[2]
p = pathlib.Path(work) / 'reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json'
d = json.loads(p.read_text())
old = d['walk_com_height']
d['walk_com_height'] = float(height)
p.write_text(json.dumps(d, indent=2))
print(f'  walk_com_height {old} -> {d["walk_com_height"]}')
PY
else
  echo "== 3/5 높이 그대로 (--height 없음)"
fi

if [ -n "$YAW_SWEEP" ]; then
  echo "== 3.5/5 yaw 스윕 -> +-$YAW_SWEEP (대칭 격자)"
  "$PLACO_PY" - "$WORK" "$YAW_SWEEP" <<'PY'
import json, pathlib, sys
import numpy as np
work, lim = sys.argv[1], float(sys.argv[2])
p = pathlib.Path(work) / 'reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/auto_gait.json'
d = json.loads(p.read_text())
old = (d['min_sweep_theta'], d['max_sweep_theta'])
d['min_sweep_theta'], d['max_sweep_theta'] = -lim, lim
g = d['sweep_theta_granularity']
grid = np.round(np.arange(-lim, lim + g, g), 6)
assert all(round(-x, 6) in set(grid) for x in grid), f'격자가 여전히 비대칭이다: {grid}'
p.write_text(json.dumps(d, indent=2))
print(f'  sweep_theta {old} -> ({-lim}, {lim}), 간격 {g}, {len(grid)}종, 대칭 확인')
PY
fi

if [ -n "$PRESET_SET" ]; then
  echo "== 3.6/5 medium 프리셋 덮어쓰기: $PRESET_SET"
  "$PLACO_PY" - "$WORK" "$PRESET_SET" <<'PY'
import json, pathlib, sys
work, sets = sys.argv[1], sys.argv[2]
p = pathlib.Path(work) / 'reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json'
d = json.loads(p.read_text())
for kv in sets.split(','):
    k, v = kv.split('=', 1)
    old = d.get(k, '(없음)')
    d[k] = json.loads(v)
    print(f'  {k}: {old} -> {d[k]}')
p.write_text(json.dumps(d, indent=2))
PY
fi

echo "== 4/5 보행 생성 + 다항식 피팅 (-j$JOBS)"
G="$WORK/reference_motion_generator"
REC="$WORK/recordings_$OUT"
mkdir -p "$REC"
# --no-speed-filter: 5단계 필터는 medium 대역(slow~fast) 밖 녹화를 지운다. 우리는
# 명령 범위 전체를 덮는 표 하나를 만드는 것이라 그러면 격자에 구멍이 난다
# (2026-08-11 실측: 충전율 53~55 %, 직진 슬라이스에서 dx=0/0.148/0.222 소실).
"$PLACO_PY" "$G/scripts/auto_waddle.py" -j"$JOBS" --duck open_duck_mini_v2 --sweep \
  --no-speed-filter --output_dir "$REC"
# 녹화가 하나도 안 나왔는데 조용히 넘어가면 옛 pkl 을 그대로 쓰게 된다.
N=$(find "$REC" -maxdepth 1 -name '*.json' | wc -l)
echo "  녹화 $N 개"
[ "$N" -gt 0 ] || { echo "!! 녹화가 0 개다 — $REC/log 의 로그를 볼 것" >&2; exit 1; }
# 적합 **전에** 직진 녹화를 대칭화한다. placo 원본은 거의 대칭인데(무릎 +3.3 %)
# 격자점마다 독립으로 15차를 맞추는 적합이 +11~41 % 로 벌려 놓는다. 대칭인 신호를
# 넣어야 적합 결과도 대칭이 나온다.
echo "== 4.5/5 직진 녹화 대칭화"
"$PLACO_PY" "$REPO_ROOT/scripts/setup/symmetrize_recordings.py" "$REC" | tail -5

(cd "$WORK" && "$PLACO_PY" "$G/scripts/fit_poly.py" --ref_motion "$REC")
mv "$WORK/polynomial_coefficients.pkl" "$WORK/$OUT.pkl"

echo "== 5/5 결과 배치"
mkdir -p "$DATA_DIR"
cp "$WORK/$OUT.pkl" "$DATA_DIR/$OUT.pkl"
ls -l "$DATA_DIR/$OUT.pkl"

cat <<EOF

완료: $DATA_DIR/$OUT.pkl
작업 디렉터리(녹화 포함): $WORK

다음 단계 (이것까지 해야 실제로 키가 바뀐다):
  1. joystick_env_cfg 의 reference_motion_pkl 을 이 파일로 가리키게 한다
  2. scripts/diag/calc_home.py 로 READY_JOINT_POS / READY_BASE_HEIGHT 재계산
  3. 새 cfg 클래스 + 태스크 등록 후 학습
EOF
