#!/usr/bin/env bash
# 레퍼런스 보행을 **랩PC에서** 생성한다.
#
# 왜 원격인가. 이 PC에는 placo 를 못 깐다 — 0.6.3 을 설치하면 의존성 하나가
# 소스 빌드(cythonize)로 빠져 실패한다(2026-07-30 확인). 반면 랩PC
# (do@192.168.137.111) 에는 placo 0.6.3 + pinocchio + scipy + matplotlib 이
# 이미 있고, **지금 쓰는 레퍼런스를 만든 게 바로 그 환경이다.** 로컬에 미묘하게
# 다른 환경을 새로 만들면 레퍼런스가 달라져도 알아채기 어렵다.
# 순수 CPU 작업이라 이 PC 의 GPU 학습과 충돌하지 않는다.
#
# 로봇 키를 바꾸는 손잡이는 medium 프리셋의 walk_com_height 다.
# auto_waddle.py 가 preset_speeds = ["medium"] 로 고정돼 있어 fast.json 은
# 쓰이지 않는다 (fast 의 0.21 을 고쳐봐야 아무 일도 안 일어난다).
# 현재 0.16 -> 로봇이 서는 높이 121 mm.
#
#   ./scripts/setup/gen_reference_remote.sh --height 0.175 --out tall175
#
# 결과: source/open_duck_mini_isaaclab/reference_motion/data/<out>.pkl
# 쓰려면 joystick_env_cfg 의 reference_motion_pkl 을 그 파일로 바꾸고,
# scripts/diag/calc_home.py 로 READY_JOINT_POS / READY_BASE_HEIGHT 를 다시 뽑아야
# 한다 — 기본 자세와 레퍼런스가 어긋나면 v1~v9 가 아홉 번 실패한 그 상황이 된다.
#
# 랩PC 는 공용 장비다. **절대 아무것도 지우지 않는다** — 쓰기는 REMOTE_DIR
# 안에서만 한다 (docs/handoff/feedback_labpc_no_deletion.md).
set -euo pipefail

HOST="${ODM_LABPC:-do@192.168.137.111}"
REMOTE_DIR="/home/do/Pictures/Claude/refgen"
HEIGHT=""
YAW_SWEEP=""
PRESET_SET=""
OUT="polynomial_coefficients"
JOBS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --height) HEIGHT="$2"; shift 2 ;;
    # yaw 격자를 대칭으로. 기본값(-0.3~0.3, 간격 0.07)은 0.6 이 0.07 로 나누어
    # 떨어지지 않아 격자가 -0.02 / +0.05 로 0 을 비껴가고, 어떤 명령도 부호를
    # 뒤집은 짝을 갖지 못한다 (2026-07-30 확인). 좌회전과 우회전의 모방 목표가
    # 서로 다른 위치의 격자에서 보간된다는 뜻이다. dx / dy 는 멀쩡하다.
    #   --yaw-sweep 0.28  ->  -0.28 .. 0.28, 간격 0.07 그대로, 9종 그대로,
    #                         0 포함, 모든 명령에 거울짝 존재
    --yaw-sweep) YAW_SWEEP="$2"; shift 2 ;;
    # medium 프리셋의 임의 키를 덮어쓴다: --preset foot_zmp_target_y=0.0
    # (쉼표로 여러 개). walk_com_height 는 --height 를 쓰는 편이 낫다.
    --preset) PRESET_SET="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --host)   HOST="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    *) echo "모르는 인자: $1" >&2; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN_DIR="$REPO_ROOT/reference_motion_generator"
DATA_DIR="$REPO_ROOT/source/open_duck_mini_isaaclab/reference_motion/data"

echo "== 1/5 랩PC 접속 확인 ($HOST)"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'python3 -c "import placo"' \
  || { echo "!! placo 를 못 찾습니다. 랩PC 환경을 확인하세요." >&2; exit 1; }

echo "== 2/5 생성기 전송"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$REMOTE_DIR'"
# recordings/ 는 산출물이라 보내지 않는다 (수 GB 가 될 수 있고 랩PC 디스크는
# 95% 차 있다). --delete 는 쓰지 않는다 — 랩PC 에서 지우지 않는다는 규칙.
rsync -az --exclude 'recordings/' --exclude '__pycache__/' \
  "$GEN_DIR/" "$HOST:$REMOTE_DIR/reference_motion_generator/"

if [ -n "$HEIGHT" ]; then
  echo "== 3/5 walk_com_height -> $HEIGHT (medium 프리셋)"
  ssh -o BatchMode=yes "$HOST" "python3 - <<'PY'
import json, pathlib
p = pathlib.Path('$REMOTE_DIR/reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json')
d = json.loads(p.read_text())
old = d['walk_com_height']
d['walk_com_height'] = float('$HEIGHT')
p.write_text(json.dumps(d, indent=2))
print(f'  walk_com_height {old} -> {d[\"walk_com_height\"]}')
PY"
else
  echo "== 3/5 높이 그대로 (--height 없음)"
fi

if [ -n "$YAW_SWEEP" ]; then
  echo "== 3.5/5 yaw 스윕 -> +-$YAW_SWEEP (대칭 격자)"
  ssh -o BatchMode=yes "$HOST" "python3 - <<'PY'
import json, pathlib, numpy as np
p = pathlib.Path('$REMOTE_DIR/reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/auto_gait.json')
d = json.loads(p.read_text())
old = (d['min_sweep_theta'], d['max_sweep_theta'])
lim = float('$YAW_SWEEP')
d['min_sweep_theta'], d['max_sweep_theta'] = -lim, lim
g = d['sweep_theta_granularity']
grid = np.round(np.arange(-lim, lim + g, g), 6)
assert all(round(-x, 6) in set(grid) for x in grid), f'격자가 여전히 비대칭이다: {grid}'
p.write_text(json.dumps(d, indent=2))
print(f'  sweep_theta {old} -> ({-lim}, {lim}), 간격 {g}, {len(grid)}종, 대칭 확인')
PY"
fi

if [ -n "$PRESET_SET" ]; then
  echo "== 3.6/5 medium 프리셋 덮어쓰기: $PRESET_SET"
  ssh -o BatchMode=yes "$HOST" "python3 - <<'PY'
import json, pathlib
p = pathlib.Path('$REMOTE_DIR/reference_motion_generator/open_duck_reference_motion_generator/robots/open_duck_mini_v2/placo_presets/medium.json')
d = json.loads(p.read_text())
for kv in '$PRESET_SET'.split(','):
    k, v = kv.split('=', 1)
    old = d.get(k, '(없음)')
    d[k] = json.loads(v)
    print(f'  {k}: {old} -> {d[k]}')
p.write_text(json.dumps(d, indent=2))
PY"
fi

echo "== 4/5 보행 생성 + 다항식 피팅 (수십 분)"
ssh -o BatchMode=yes "$HOST" "bash -lc '
set -e
cd \"$REMOTE_DIR\"
G=\"$REMOTE_DIR/reference_motion_generator\"
REC=\"$REMOTE_DIR/recordings_$OUT\"
mkdir -p \"\$REC\"
python3 \"\$G/scripts/auto_waddle.py\" -j$JOBS --duck open_duck_mini_v2 --sweep --output_dir \"\$REC\"
cd \"$REMOTE_DIR\"
python3 \"\$G/scripts/fit_poly.py\" --ref_motion \"\$REC\"
mv \"$REMOTE_DIR/polynomial_coefficients.pkl\" \"$REMOTE_DIR/$OUT.pkl\"
ls -l \"$REMOTE_DIR/$OUT.pkl\"
'"

echo "== 5/5 결과 회수"
mkdir -p "$DATA_DIR"
rsync -az "$HOST:$REMOTE_DIR/$OUT.pkl" "$DATA_DIR/$OUT.pkl"
ls -l "$DATA_DIR/$OUT.pkl"

cat <<EOF

완료: $DATA_DIR/$OUT.pkl

다음 단계 (이것까지 해야 실제로 키가 바뀐다):
  1. joystick_env_cfg 의 reference_motion_pkl 을 이 파일로 가리키게 한다
  2. scripts/diag/calc_home.py 로 READY_JOINT_POS / READY_BASE_HEIGHT 재계산
  3. 새 cfg 클래스 + 태스크 등록 후 학습

랩PC 의 $REMOTE_DIR 는 지우지 않았습니다. 정리는 사용자가 판단할 일입니다.
EOF
