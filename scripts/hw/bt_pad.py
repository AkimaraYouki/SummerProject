#!/usr/bin/env python3
"""블루투스 패드 붙이기 — 콘솔 UI 든 CLI 든.

    ssh -t … 'python3 ~/bt_pad.py'                  콘솔 UI (기본). 번호만 고른다
    ssh -t … 'python3 ~/bt_pad.py scan'             스캔 -> 번호 매긴 목록
    ssh -t … 'python3 ~/bt_pad.py connect 3'        그 번호에 붙는다 (MAC 도 됨)
    ssh -t … 'python3 ~/bt_pad.py remove 3'         등록 삭제
    ssh -t … 'python3 ~/bt_pad.py disconnect'       연결만 끊기
    ssh -t … 'python3 ~/bt_pad.py forget'           패드 등록 전부 삭제
    ssh    … 'python3 ~/bt_pad.py status'           상태만 (TTY 없어도 된다)
    ssh -t … 'python3 ~/bt_pad.py auto'             대화 없이 알아서

`scan` 이 매긴 번호를 `connect`/`remove` 에 그대로 쓴다 (`~/.bt_pad_last.json`
에 남긴다 — 서브커맨드는 매번 새 프로세스라 메모리로는 못 넘긴다).

목록의 **● 는 지금 전파가 잡히는 장치**, **· 는 예전에 등록만 된 것**이다.
· 에 붙으려 하면 `br-connection-create-socket` 만 받는다 — 패드 전원이 꺼져
있다는 뜻이다. 이 구분이 없어서 2026-08-13 에 한참 헤맸다.

콘솔 UI 키:

    번호   그 장치에 붙는다 (pair -> trust -> connect -> js0 확인)
    r      새로고침(3초)      d  연결 끊기
    x3     3번 등록 삭제      f  패드 등록 전부 삭제      q  종료

패드를 페어링 모드로: Xbox 버튼으로 켠 뒤, 위쪽 작은 **연결 버튼**을 로고가
빠르게 깜빡일 때까지 3 초 누른다.

## 이 파일이 있는 이유

붙일 때마다 걸리던 것들을 절차로 굳혔다.

  * **ERTM 을 꺼야 페어링이 붙는다.** 리눅스 블루투스 스택의 ERTM 과 Xbox
    패드가 안 맞는다.

    2026-08-13 정정: 이걸 `/etc/modprobe.d/99-xbox-ertm.conf` 에 넣어 뒀는데
    **처음부터 효과가 없었다.** 이 젯슨은 bluetooth 가 커널 내장이라
    (`lsmod | grep bluetooth` 가 비어 있다) modprobe.d 가 적용되지 않는다.
    그래서 재부팅할 때마다 ERTM 이 켜진 채였고, "지난번엔 붙었는데 오늘은
    안 붙는다" 를 반복했다. 지금은 `/etc/tmpfiles.d/xbox-ertm.conf` 가
    부팅 때 sysfs 에 직접 쓴다.

  * **joydev 모듈이 있어야 `/dev/input/js0` 이 생긴다.** 없으면 페어링은
    되는데 장치 노드가 안 나와서 "붙었는데 안 읽힌다" 가 된다.

  * **패드가 재연결마다 MAC 끝자리를 바꾼다** (…1A -> …1B). 주소를 적어 두고
    쓰면 다음에 안 붙는다 — 그래서 매번 스캔해서 고르는 방식으로 만들었다.

## 붙고 나서

    python3 ~/joy_monitor.py     # 축·버튼 번호 확인 (패드마다 다르다)
    python3 ~/joy_local.py       # 속도 명령으로 어떻게 변환되는지
    python3 ~/pad_ctl.py --onnx ~/policy_v46/policy.onnx    # 패드로 운전
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time

ERTM = "/sys/module/bluetooth/parameters/disable_ertm"
DEV = "/dev/input/js0"
#: 이름에 이게 들어가면 패드로 본다 (자동 모드에서 후보를 고를 때).
NAME_HINTS = ("xbox", "controller", "gamepad", "joystick", "pad")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
#: bluetoothctl 은 색을 readline 의 **비출력 구간 마커**로 감싸서 내보낸다
#: (\x01 = RL_PROMPT_START_IGNORE, \x02 = END). ANSI 만 벗기면 실제 줄이
#: `[\x01\x02NEW\x01\x02] Device …` 로 남아 `[NEW]` 매칭이 전부 빗나간다.
#: 2026-08-13 에 이것 때문에 "스캔은 도는데 살아 있는 장치가 하나도 없다" 로
#: 한참 헤맸다 — Discovering 은 yes 였고 RSSI 도 들어오고 있었다.
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MAC = r"([0-9A-F]{2}(?::[0-9A-F]{2}){5})"


def clean(s: str) -> str:
    return CTRL.sub("", ANSI.sub("", s))


def connected_macs() -> set[str]:
    """지금 연결된 장치의 MAC 집합.

    `bluetoothctl devices Connected` 를 쓰면 안 된다 — **그 필터 인자는 5.65
    부터**다. 이 젯슨은 5.64 라 "Too many arguments" 를 뱉는데, 그걸 빈 목록
    으로 읽으면 패드가 붙어 있는데도 "연결된 장치 없음" 이라고 보고한다
    (2026-08-13 에 실제로 그렇게 거짓말했다). 대신 아는 장치마다 `info` 를
    물어서 `Connected: yes` 를 본다.
    """
    known = re.findall(rf"Device {MAC}", clean(sh("bluetoothctl devices", 8.0)))
    out = set()
    for mac in known[:24]:          # 아는 장치가 많아도 여기서 끊는다
        if re.search(r"Connected:\s*yes", clean(sh(f"bluetoothctl info {mac}", 5.0))):
            out.add(mac)
    return out


def sh(cmd: str, timeout: float = 15.0) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


class Bt:
    """bluetoothctl 을 하나 띄워 놓고 계속 쓴다.

    매번 `bluetoothctl <명령>` 을 새로 부르면 그때마다 스캔이 끊긴다. 스캔
    결과는 휘발성이라 끊기면 방금 본 장치가 목록에서 사라진다 — 그래서 한
    프로세스를 붙들고 그 안에서 스캔과 연결을 다 한다.
    """

    def __init__(self) -> None:
        self.p = subprocess.Popen(
            ["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines: list[str] = []
        self.devices: dict[str, dict] = {}     # mac -> {name, rssi, connected}
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("power on", "agent on", "default-agent")
        time.sleep(0.5)

    def _reader(self) -> None:
        assert self.p.stdout is not None
        for raw in self.p.stdout:
            if self._stop:
                return
            line = clean(raw).rstrip()
            with self._lock:
                self.lines.append(line)
                if len(self.lines) > 400:
                    del self.lines[:200]
            self._parse(line)

    def _parse(self, line: str) -> None:
        m = re.search(rf"Device {MAC}\s*(.*)$", line)
        if not m:
            return
        mac, rest = m.group(1), m.group(2).strip()
        # **지금 전파가 오는 장치**와 **예전에 등록만 된 장치**를 구분한다.
        # bluetoothctl 은 스캔으로 본 것에만 [NEW]/[CHG] 를 붙이고 `devices`
        # 명령의 출력에는 안 붙인다. 이걸 안 나누면 꺼져 있는 패드가 목록에
        # 그대로 떠서 "보이는데 왜 연결이 안 되지" 가 된다 (2026-08-13 에
        # 그렇게 헤맸다 — 캐시 항목에 붙으려다 br-connection-create-socket 만
        # 받았다).
        #
        # 줄 **앞**만 보면 안 된다. 프롬프트가 살아 있으면
        # `[bluetooth]# [NEW] Device …` 처럼 앞에 붙어 나온다 — 처음에
        # startswith 로 짰다가 전부 캐시로 잘못 분류했다. RSSI 가 왔다는 것도
        # 이번 세션에 실제로 봤다는 뜻이므로 같이 live 로 친다.
        live = bool(re.search(r"\[(NEW|CHG)\]", line))
        with self._lock:
            d = self.devices.setdefault(mac, {"name": "", "rssi": None,
                                              "connected": False, "seen": 0.0,
                                              "live": False})
            if live:
                d["live"] = True
                d["seen"] = time.time()
            if rest.startswith("RSSI:"):
                try:
                    d["rssi"] = int(rest.split(":", 1)[1].strip())
                    d["live"] = True
                    d["seen"] = time.time()
                except ValueError:
                    pass
            elif rest.startswith("Connected:"):
                d["connected"] = rest.split(":", 1)[1].strip() == "yes"
            elif rest and ":" not in rest.split()[0]:
                # "[NEW] Device AA:.. Xbox Wireless Controller" 의 이름 부분
                if not rest.startswith(("Paired", "Trusted", "Bonded",
                                        "ServicesResolved", "UUIDs", "Modalias",
                                        "LegacyPairing", "Icon", "Class",
                                        "Appearance", "Blocked", "WakeAllowed",
                                        "TxPower", "ManufacturerData",
                                        "ServiceData", "AdvertisingFlags")):
                    d["name"] = rest

    def send(self, *cmds: str) -> None:
        assert self.p.stdin is not None
        for c in cmds:
            try:
                self.p.stdin.write(c + "\n")
                self.p.stdin.flush()
            except (BrokenPipeError, ValueError):
                return

    def run(self, cmd: str, wait: float = 6.0, until: str | None = None) -> str:
        """명령을 보내고 wait 초 동안(또는 until 이 보일 때까지) 나온 출력을 준다."""
        with self._lock:
            start = len(self.lines)
        self.send(cmd)
        t0 = time.time()
        while time.time() - t0 < wait:
            time.sleep(0.15)
            with self._lock:
                out = "\n".join(self.lines[start:])
            if until and until in out:
                break
        with self._lock:
            return "\n".join(self.lines[start:])

    def refresh_known(self) -> None:
        """이미 등록/연결된 장치도 목록에 넣는다.

        연결 여부는 **따로 띄운 bluetoothctl** 로 본다. 이 클래스의 공용
        스트림에서 `devices Connected` 출력을 잘라 쓰면, 직전 `devices` 의
        출력이 아직 흐르고 있을 때 그게 섞여 안 붙은 장치가 [연결됨] 으로
        찍힌다 (2026-08-13 에 실제로 두 대가 그렇게 나왔다).
        """
        self.run("devices", wait=1.5)
        conn = connected_macs()
        with self._lock:
            for mac in conn:
                self.devices.setdefault(mac, {"name": "", "rssi": None,
                                              "connected": False, "seen": 0.0,
                                              "live": False})
            for mac, d in self.devices.items():
                d["connected"] = mac in conn

    def listing(self) -> list[tuple[str, dict]]:
        """지금 보이는 것 먼저 -> 신호 센 순 -> 이름 있는 순."""
        with self._lock:
            items = list(self.devices.items())
        def key(kv):
            _mac, d = kv
            return (0 if d["connected"] else 1,
                    0 if d.get("live") else 1,
                    -(d["rssi"] if d["rssi"] is not None else -999),
                    0 if d["name"] else 1)
        return sorted(items, key=key)

    def live_only(self) -> list[tuple[str, dict]]:
        return [(m, d) for m, d in self.listing() if d.get("live")]

    def close(self) -> None:
        self._stop = True
        self.send("scan off", "quit")
        try:
            self.p.terminate()
            self.p.wait(timeout=3)
        except Exception:
            self.p.kill()


# ── 사전 조건 ──────────────────────────────────────────────────────────
def check_prereqs(verbose: bool = True) -> bool:
    ok = True
    def say(s):
        if verbose:
            print(s)
    if os.path.exists(ERTM):
        v = open(ERTM).read().strip()
        if v in ("Y", "1"):
            say(f"  ERTM 꺼짐 (disable_ertm={v})  OK")
        else:
            say(f"  !! ERTM 이 켜져 있다 (disable_ertm={v}) — 페어링이 실패한다.")
            say(f"     지금 끄기:  sudo sh -c 'echo Y > {ERTM}'")
            say(f"     영구 적용:  /etc/tmpfiles.d/xbox-ertm.conf 에")
            say(f"                 w {ERTM} - - - - Y")
            say(f"     (modprobe.d 는 안 먹는다 — 이 젯슨은 bluetooth 가 커널 내장이다)")
            ok = False
    else:
        say("  ERTM 노드 없음 — 블루투스 모듈이 안 올라왔을 수 있다")
    if re.search(r"^joydev", sh("lsmod"), re.M):
        say("  joydev 모듈 있음  OK")
    else:
        say("  !! joydev 모듈이 없다 — 페어링돼도 /dev/input/js0 이 안 생긴다.")
        say("     sudo modprobe joydev   (영구: /etc/modules-load.d/joydev.conf)")
        ok = False
    if sh("systemctl is-active bluetooth").strip() == "active":
        say("  bluetooth 서비스 active  OK")
    else:
        say("  !! bluetooth 서비스가 안 돌고 있다 — sudo systemctl start bluetooth")
        ok = False
    return ok


def wait_js0(secs: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < secs:
        if os.path.exists(DEV):
            return True
        time.sleep(0.4)
    return False


def _attempt(bt: Bt, mac: str) -> tuple[bool, str]:
    """pair -> trust -> connect 한 바퀴. (성공, 마지막 connect 메시지)."""
    last = ""
    for step, wait, until in (("pair", 12.0, "Pairing successful"),
                              ("trust", 3.0, "trust succeeded"),
                              ("connect", 12.0, "Connection successful")):
        out = bt.run(f"{step} {mac}", wait=wait, until=until)
        hit = [l.strip() for l in out.splitlines()
               if re.search(r"(Failed|successful|succeeded|already|not available)", l)]
        msg = hit[-1] if hit else "(응답 없음)"
        print(f"  {step:8s} {msg}")
        if step == "connect":
            last = msg
    # bluetoothctl 의 성공 문구는 못 믿는다. 장치 노드로 판단한다.
    return wait_js0(), last


def try_connect(bt: Bt, mac: str, name: str, live: bool = True) -> bool:
    print(f"\n--- {mac}  {name or '(이름없음)'} ---")
    if not live:
        print("  ! 이 장치는 지금 전파가 안 잡힌다 (예전에 등록된 항목이다).")
        print("    패드가 꺼져 있을 가능성이 높다. 그래도 시도해 본다.")
    ok, msg = _attempt(bt, mac)
    if ok:
        print(f"\n  ✓ 붙었다. {DEV} 생성 확인.")
        return True

    # 흔한 막다른 길: 예전 본딩이 남아 pair 는 AlreadyExists, connect 는
    # br-connection-create-socket. 본딩을 지우고 한 번 더 하면 대개 붙는다.
    # 패드가 재연결마다 MAC 을 바꾸므로 옛 기록이 실제와 안 맞아서 생긴다.
    if "create-socket" in msg or "AlreadyExists" in msg or "br-connection" in msg:
        print("\n  옛 페어링 기록이 걸린 것 같다 — 지우고 한 번 더 시도한다.")
        bt.run(f"remove {mac}", wait=3.0)
        time.sleep(2.0)
        ok, _ = _attempt(bt, mac)
        if ok:
            print(f"\n  ✓ 붙었다. {DEV} 생성 확인.")
            return True

    print(f"\n  ✗ {DEV} 이 안 생겼다.")
    if not live:
        print("    → 패드 전원을 켜고(Xbox 버튼), 위쪽 작은 연결 버튼을 로고가")
        print("      **빠르게 깜빡일 때까지** 3초 누른 뒤 r 로 새로고침할 것.")
    return False



# ── 목록 표시 ──────────────────────────────────────────────────────────
#: `scan` 이 매긴 번호를 다음 명령에서 쓰려면 어딘가 남겨야 한다.
#: 서브커맨드는 매번 새 프로세스라 메모리로는 못 넘긴다.
CACHE = os.path.expanduser("~/.bt_pad_last.json")


def show(items, title="") -> None:
    js = "있음" if os.path.exists(DEV) else "없음"
    print("\n" + "=" * 68)
    print(f"{title or '장치'} {len(items)}대   ·   {DEV}: {js}")
    print("=" * 68)
    if not items:
        print("  없다. 패드를 페어링 모드로 두고 다시 스캔할 것.")
        return
    for i, (mac, d) in enumerate(items, 1):
        rssi = f"{d['rssi']:>4}dBm" if d["rssi"] is not None else "   —   "
        live = "●" if d.get("live") else "·"
        hint = " ←패드?" if any(h in d["name"].lower() for h in NAME_HINTS) else ""
        conn = " [연결됨]" if d["connected"] else ""
        print(f"  {i:2d}) {live} {rssi}  {mac}  "
              f"{d['name'] or '(이름없음)'}{conn}{hint}")
    print("\n  ● = 지금 전파가 잡힌다   · = 예전에 등록만 됨(전원 꺼진 상태일 것)")


def save_cache(items) -> None:
    import json
    try:
        json.dump([{"mac": m, **d} for m, d in items], open(CACHE, "w"))
    except OSError:
        pass


def load_cache() -> list[tuple[str, dict]]:
    import json
    try:
        raw = json.load(open(CACHE))
    except (OSError, ValueError):
        return []
    return [(r.pop("mac"), r) for r in raw]


def resolve(target: str) -> tuple[str, dict] | None:
    """'3' 같은 번호나 MAC 을 (mac, info) 로. 번호는 마지막 scan 결과 기준."""
    if re.fullmatch(MAC, target.upper()):
        return (target.upper(), {"name": "", "rssi": None,
                                 "connected": False, "live": True})
    items = load_cache()
    if target.isdigit() and 1 <= int(target) <= len(items):
        return items[int(target) - 1]
    if not items:
        print("  번호를 쓰려면 먼저 `bt_pad.py scan` 을 돌려야 한다.")
    else:
        print(f"  1~{len(items)} 사이 번호나 MAC 주소를 줄 것.")
    return None


# ── 서브커맨드 ─────────────────────────────────────────────────────────
def cmd_scan(secs: float) -> int:
    bt = Bt()
    try:
        bt.send("scan on")
        bt.refresh_known()
        print(f"{secs:.0f}초 스캔… (패드를 페어링 모드로: Xbox 버튼으로 켜고,")
        print("위쪽 작은 연결 버튼을 로고가 빠르게 깜빡일 때까지 3초)")
        time.sleep(secs)
        items = bt.listing()
        save_cache(items)
        show(items, "발견")
        live = [x for x in items if x[1].get("live")]
        if live:
            print(f"\n  붙이려면:  python3 ~/bt_pad.py connect <번호>")
        else:
            print("\n  ● 이 하나도 없다 — 켜져 있는 장치가 안 잡혔다.")
            print("    패드 전원과 페어링 모드를 확인하고 다시 스캔할 것.")
        return 0
    finally:
        bt.close()


def cmd_connect(target: str, secs: float) -> int:
    got = resolve(target)
    if not got:
        return 1
    mac, info = got
    bt = Bt()
    try:
        # 연결 전에 잠깐 스캔해서 지금 실제로 있는지 확인한다. 캐시 항목에
        # 그냥 붙으려 들면 br-connection-create-socket 만 받는다.
        bt.send("scan on")
        bt.refresh_known()
        time.sleep(secs)
        cur = dict(bt.listing()).get(mac, {})
        live = bool(cur.get("live"))
        name = cur.get("name") or info.get("name", "")
        return 0 if try_connect(bt, mac, name, live=live) else 1
    finally:
        bt.close()


def cmd_remove(target: str) -> int:
    got = resolve(target)
    if not got:
        return 1
    mac, info = got
    bt = Bt()
    try:
        print(f"  {mac}  {info.get('name','')} 등록을 지운다")
        bt.run(f"disconnect {mac}", wait=3.0)
        bt.run(f"remove {mac}", wait=3.0)
        print("  완료. 다시 붙이려면 scan -> connect.")
        return 0
    finally:
        bt.close()


def cmd_disconnect() -> int:
    macs = sorted(connected_macs())
    if not macs:
        print("연결된 장치가 없다.")
        return 0
    bt = Bt()
    try:
        for mac in macs:
            print(f"  끊는다 {mac}")
            bt.run(f"disconnect {mac}", wait=3.0)
        return 0
    finally:
        bt.close()


def cmd_forget_all() -> int:
    bt = Bt()
    try:
        bt.send("scan on")
        bt.refresh_known()
        time.sleep(2.0)
        n = 0
        for mac, d in bt.listing():
            if any(h in d["name"].lower() for h in NAME_HINTS) or d["connected"]:
                print(f"  제거 {mac}  {d['name']}")
                bt.run(f"disconnect {mac}", wait=2.0)
                bt.run(f"remove {mac}", wait=2.0)
                n += 1
        print(f"{n}대 제거. 이제 scan -> connect 로 처음부터.")
        return 0
    finally:
        bt.close()


def cmd_status() -> int:
    print("=== 사전 조건 ===")
    check_prereqs()
    print("\n=== 연결된 장치 ===")
    names = dict(re.findall(rf"Device {MAC} (.+)",
                            clean(sh("bluetoothctl devices", 8.0))))
    conn = sorted(connected_macs())
    print("  " + ("\n  ".join(f"{a}  {names.get(a,'')}" for a in conn)
                  if conn else "없음"))
    print(f"\n=== {DEV} ===")
    print("  있음 — joy_monitor.py 로 축/버튼 확인" if os.path.exists(DEV)
          else "  없음 — 패드가 안 붙었거나 joydev 가 없다")
    return 0


def cmd_auto(tries: int) -> int:
    bt = Bt()
    try:
        bt.send("scan on")
        bt.refresh_known()
        for k in range(1, tries + 1):
            print(f"\n=== 시도 {k}/{tries} — 8초 스캔 ===")
            time.sleep(8.0)
            cand = [(m, d) for m, d in bt.live_only()
                    if any(h in d["name"].lower() for h in NAME_HINTS)]
            if not cand:
                print("  켜져 있는 패드가 안 잡힌다. 전원과 페어링 모드를 확인할 것.")
                continue
            for mac, d in cand:
                if try_connect(bt, mac, d["name"], live=True):
                    return 0
        print("\n실패. `bt_pad.py ui` 로 목록을 직접 보고 고를 것.")
        return 1
    finally:
        bt.close()


# ── 콘솔 UI ────────────────────────────────────────────────────────────
def cmd_ui() -> int:
    bt = Bt()
    try:
        print("\n스캔을 켠 채로 목록을 갱신한다. 패드를 페어링 모드로 두어라 —")
        print("Xbox 버튼으로 켠 뒤 위쪽 작은 연결 버튼을 로고가 **빠르게")
        print("깜빡일 때까지** 3초.")
        bt.send("scan on")
        bt.refresh_known()
        time.sleep(3.0)
        while True:
            items = bt.listing()
            save_cache(items)
            show(items)
            print("\n  번호=연결   r=새로고침(3초)   d=연결끊기   x=등록지우기(번호)")
            print("  f=패드 등록 전부 지우기        q=종료")
            try:
                s = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return 1
            if s == "q":
                return 0
            if s == "r":
                time.sleep(3.0)
                continue
            if s == "d":
                for mac, d in items:
                    if d["connected"]:
                        print(f"  끊는다 {mac}")
                        bt.run(f"disconnect {mac}", wait=3.0)
                continue
            if s == "f":
                n = 0
                for mac, d in items:
                    if any(h in d["name"].lower() for h in NAME_HINTS) or d["connected"]:
                        print(f"  제거 {mac}  {d['name']}")
                        bt.run(f"disconnect {mac}", wait=2.0)
                        bt.run(f"remove {mac}", wait=2.0)
                        n += 1
                print(f"  {n}대 제거.")
                continue
            if s.startswith("x"):
                rest = s[1:].strip()
                if rest.isdigit() and 1 <= int(rest) <= len(items):
                    mac, d = items[int(rest) - 1]
                    print(f"  제거 {mac}  {d['name']}")
                    bt.run(f"disconnect {mac}", wait=2.0)
                    bt.run(f"remove {mac}", wait=2.0)
                else:
                    print("  x3 처럼 번호를 붙여서 줄 것")
                continue
            if s.isdigit() and 1 <= int(s) <= len(items):
                mac, d = items[int(s) - 1]
                if try_connect(bt, mac, d["name"], live=bool(d.get("live"))):
                    print("\n다음:")
                    print("  python3 ~/joy_monitor.py    # 축·버튼 번호 확인")
                    print("  python3 ~/joy_local.py      # 속도 명령 변환 확인")
                    return 0
                continue
            print("  ?")
    finally:
        bt.close()


USAGE = """블루투스 패드 붙이기

  python3 ~/bt_pad.py                    콘솔 UI (기본) — 번호만 고르면 된다
  python3 ~/bt_pad.py scan [초]          스캔해서 번호 매긴 목록
  python3 ~/bt_pad.py connect <번호|MAC> 붙인다
  python3 ~/bt_pad.py remove  <번호|MAC> 등록 삭제
  python3 ~/bt_pad.py disconnect         연결만 끊기
  python3 ~/bt_pad.py forget             패드 등록 전부 삭제
  python3 ~/bt_pad.py status             상태만 (TTY 없어도 된다)
  python3 ~/bt_pad.py auto [횟수]        대화 없이 알아서

  scan 이 매긴 번호를 connect/remove 에 그대로 쓴다.
  목록의 ● 는 지금 전파가 잡히는 장치, · 는 예전에 등록만 된 것이다.
  · 에는 붙어도 실패한다 — 패드 전원이 꺼져 있다는 뜻이다."""


def main() -> int:
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "ui"
    arg = argv[1] if len(argv) > 1 else None

    if cmd in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if cmd in ("status", "--status"):
        return cmd_status()

    # 상태 외의 모든 명령은 사전 조건이 맞아야 의미가 있다.
    if not check_prereqs(verbose=False):
        print("=== 사전 조건 ===")
        check_prereqs()
        print("\n위 항목을 먼저 고치고 다시 실행할 것.")
        return 1

    if cmd == "scan":
        return cmd_scan(float(arg) if arg else 10.0)
    if cmd == "connect":
        if not arg:
            print("사용법: bt_pad.py connect <번호|MAC>")
            return 1
        return cmd_connect(arg, 4.0)
    if cmd == "remove":
        if not arg:
            print("사용법: bt_pad.py remove <번호|MAC>")
            return 1
        return cmd_remove(arg)
    if cmd == "disconnect":
        return cmd_disconnect()
    if cmd in ("forget", "--forget"):
        return cmd_forget_all()
    if cmd in ("auto", "--auto"):
        return cmd_auto(int(arg) if arg and arg.isdigit() else 3)
    if cmd == "ui":
        if not sys.stdin.isatty():
            print("콘솔 UI 는 TTY 가 필요하다 — ssh -t 로 붙을 것.")
            print("비대화형이면 `scan` / `connect` / `auto` 를 쓸 것.\n")
            print(USAGE)
            return 1
        return cmd_ui()

    print(f"모르는 명령: {cmd}\n")
    print(USAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main())
