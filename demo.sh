#!/usr/bin/env bash
# 시연용 한 곳. 인자 없이 실행하면 목록이 나온다.
#
#   ./demo.sh drive       Xbox 패드로 로봇 조종 (실기 후보: 정책 + 안전 필터)
#   ./demo.sh drive-raw   같은 정책, 안전 필터 없이 (차이 비교용)
#   ./demo.sh ref         레퍼런스 보행을 브라우저에서 (meshcat, Isaac Sim 불필요)
#   ./demo.sh measure     6방향 추종 오차 측정
#   ./demo.sh safety      다리-몸통 간격을 정확 메시로 검증 (5 mm 기준)
#   ./demo.sh graphs      그래프 4장 -> ~/Desktop/openduck_graphs/
#   ./demo.sh stop        떠 있는 Isaac Sim / meshcat 정리
#   ./demo.sh status      지금 뭐가 돌고 있는지
#   ./demo.sh setup       ref / safety 용 파이썬 도구 설치 (처음 한 번)
#   ./demo.sh colors      URDF 색을 USD 에 재주입 (USD 를 다시 변환했을 때)
set -euo pipefail
cd "$(dirname "$0")"

ODM=./scripts/odm
# pinocchio + meshcat + usd-core. 외장 SSD 는 심볼릭 링크를 못 만들어 venv 가
# 안 생기므로 홈에 둔다. 없으면 `./demo.sh setup` 이 만든다.
VENV="$HOME/.odm-tools/bin/python"
VIZ=scripts/viz_reference.py
BARRIER=source/open_duck_mini_isaaclab/barrier_h5d.pt

# 실기 후보 = v32 정책(안쪽 고관절 이탈 벌점) + CBF 안전 필터
SAFE_VER=v32s
RAW_VER=v32

banner() { printf '\n\033[1m%s\033[0m\n' "$1"; }

case "${1:-help}" in

drive)
  banner "조종 — $SAFE_VER (정책 + 안전 필터)"
  cat <<'TXT'
  왼쪽 스틱  세로 = 전후 (±0.15 m/s)   가로 = 회전 (±1.0 rad/s)
  오른쪽 스틱 가로 = 게걸음 (±0.2 m/s)
  A 버튼     비상정지

  다리가 몸통에 5 mm 이내로 접근하지 못하도록 매 스텝 강제된다.
  창이 뜨는 데 1~2분 걸린다.
TXT
  [ -e /dev/input/js0 ] || echo "  !! /dev/input/js0 없음 — 패드를 켜세요"
  $ODM play "$SAFE_VER" --joystick
  ;;

drive-raw)
  banner "조종 — $RAW_VER (안전 필터 없음)"
  echo "  같은 정책인데 필터만 뺐다. 5 mm 위반이 1.0% 나온다 (필터를 켜면 0%)."
  $ODM play "$RAW_VER" --joystick
  ;;

ref)
  banner "레퍼런스 보행 (meshcat)"
  cmd="${2:-forward}"
  echo "  명령: $cmd   (forward backward left right turn stop)"
  echo "  브라우저에서 http://127.0.0.1:7000/static/ 를 여세요."
  "$VENV" "$VIZ" "$cmd"
  ;;

measure)
  ver="${2:-$RAW_VER}"
  banner "6방향 추종 측정 — $ver  (약 7분)"
  $ODM measure "$ver"
  ;;

safety)
  ver="${2:-v32safe}"
  npz="$HOME/odm_out/gait_${ver}.npz"
  banner "안전 검증 — $npz (기준 5 mm, 정확 메시)"
  [ -f "$npz" ] || { echo "  없음. 먼저 ./demo.sh measure 를 돌리세요."; exit 1; }
  [ -x "$VENV" ] || { echo "  도구 없음. ./demo.sh setup 을 먼저 돌리세요."; exit 1; }
  "$VENV" scripts/diag/leg_trunk_clearance.py --npz "$npz" \
      --urdf robot/robot.urdf --mesh-dir robot --stride 8
  ;;

graphs)
  banner "그래프 4장"
  "$HOME/Desktop/IsaacLab/isaaclab.sh" -p scripts/diag/plot_best.py --ver v28
  echo "  -> $HOME/Desktop/openduck_graphs/"
  ;;

colors)
  banner "머티리얼 주입"
  echo "  IsaacLab URDF 변환기가 시각 머티리얼을 버려서 로봇이 회색으로 나온다."
  echo "  USD 는 생성물이라 git 에 없으므로, 재변환하면 이걸 다시 돌려야 한다."
  "$VENV" scripts/setup/inject_materials.py
  ;;

setup)
  banner "도구 설치 (~/.odm-tools)"
  echo "  pinocchio + meshcat + usd-core. Isaac Sim 파이썬과 별개다 --"
  echo "  pinocchio 는 Isaac 안에서 임포트가 실패하고, pxr 은 앱을 띄워야 잡힌다."
  python3 -m venv "$HOME/.odm-tools"
  "$HOME/.odm-tools/bin/pip" install -q pin meshcat usd-core scipy
  "$VENV" -c "import pinocchio, meshcat, pxr, scipy; print('  완료:', pinocchio.__version__)"
  ;;

stop)
  banner "정리"
  $ODM stop || true
  P="viz""_ref"; pkill -f "$P" 2>/dev/null || true
  echo "  완료"
  ;;

status)
  banner "상태"
  printf '  GPU        %s\n' "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
  P="viz""_ref"
  printf '  meshcat    %s개\n' "$(pgrep -cf "$P" 2>/dev/null || echo 0)"
  printf '  조이스틱   %s\n' "$([ -e /dev/input/js0 ] && echo 연결됨 || echo 없음)"
  printf '  장벽함수   %s\n' "$([ -f "$BARRIER" ] && echo 있음 || echo 없음)"
  printf '  도구 venv  %s\n' "$([ -x "$VENV" ] && echo 있음 || echo '없음 (./demo.sh colors)
  banner "머티리얼 주입"
  echo "  IsaacLab URDF 변환기가 시각 머티리얼을 버려서 로봇이 회색으로 나온다."
  echo "  USD 는 생성물이라 git 에 없으므로, 재변환하면 이걸 다시 돌려야 한다."
  "$VENV" scripts/setup/inject_materials.py
  ;;

setup)')"
  ;;

*)
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  ;;
esac
