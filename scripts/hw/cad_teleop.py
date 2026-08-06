#!/usr/bin/env python3
"""웹 슬라이더로 CAD 모델을 움직이고 실기가 같이 움직이는지 실시간으로 본다.

    # 1) Jetson 에서 브리지 (로봇을 매달아 놓고)
    ssh -t parksuho@192.168.137.7 'python3 ~/dxl_bridge.py --arm'

    # 2) 데스크탑에서
    ~/.odm-tools/bin/python scripts/hw/cad_teleop.py
    # -> http://localhost:8099  (3D 뷰는 meshcat, 같은 페이지에 끼워 넣는다)

    # 실기 없이 UI 만 보려면
    ~/.odm-tools/bin/python scripts/hw/cad_teleop.py --no-robot

3D 는 meshcat, 관절 표는 **명령값 / 실측값 / 오차**를 나란히 보여준다. 오차가
임계를 넘으면 빨갛게 된다 -- 부호가 뒤집혔거나 관절이 걸리면 바로 눈에 띈다.

외부 의존성 없이 stdlib `http.server` 만 쓴다 (이 환경에 flask/websockets 가 없다).
"""

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NAMES = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
GROUP = {"왼다리": NAMES[0:5], "머리·목": NAMES[5:9], "오른다리": NAMES[9:14]}
READY = {"left_hip_yaw": 0.0003, "left_hip_roll": 0.0213, "left_hip_pitch": 0.9910,
         "left_knee": -1.7852, "left_ankle": 0.8647,
         "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
         "right_hip_yaw": -0.0005, "right_hip_roll": -0.0092, "right_hip_pitch": 1.0114,
         "right_knee": 1.8163, "right_ankle": -0.8754}

S = {"cmd": [0.0] * 14, "meas": [0.0] * 14, "tick": [0] * 14,
     "clamped": [False] * 14, "link": False, "err": ""}
LOCK = threading.Lock()


# ── Jetson 링크 ────────────────────────────────────────────────────────
def link_thread(host, port, hz):
    period = 1.0 / hz
    while True:
        sk = None
        try:
            sk = socket.create_connection((host, port), timeout=3.0)
            sk.settimeout(3.0)
            with LOCK:
                S["link"], S["err"] = True, ""
            f = sk.makefile("rwb")
            while True:
                with LOCK:
                    q = list(S["cmd"])
                f.write((json.dumps({"q": q}) + "\n").encode())
                f.flush()
                line = f.readline()
                if not line:
                    raise OSError("연결이 닫혔다")
                r = json.loads(line)
                with LOCK:
                    S["meas"] = r.get("q", S["meas"])
                    S["tick"] = r.get("tick", S["tick"])
                    S["clamped"] = r.get("clamped", S["clamped"])
                time.sleep(period)
        except Exception as e:                      # 끊기면 재접속
            with LOCK:
                S["link"], S["err"] = False, str(e)[:80]
            if sk:
                try:
                    sk.close()
                except OSError:
                    pass
            time.sleep(1.5)


# ── meshcat ───────────────────────────────────────────────────────────
class Viz:
    def __init__(self, urdf, mesh_dir):
        m, c, v = pin.buildModelsFromUrdf(urdf, mesh_dir)
        self.model = m
        self.viz = MeshcatVisualizer(m, c, v)
        self.viz.initViewer(open=False)
        self.viz.loadViewerModel()
        self.slot = [m.idx_qs[m.getJointId(n)] for n in NAMES]
        self.url = self.viz.viewer.url()

    def show(self, q14):
        q = pin.neutral(self.model)
        for k, s in enumerate(self.slot):
            q[s] = q14[k]
        self.viz.display(q)


def viz_thread(viz, hz=20.0):
    last = None
    while True:
        with LOCK:
            q = list(S["cmd"])
        if q != last:
            try:
                viz.show(q)
            except Exception:
                pass
            last = q
        time.sleep(1.0 / hz)


# ── 웹 ────────────────────────────────────────────────────────────────
PAGE = """<!doctype html><meta charset=utf-8><title>CAD ↔ 실기 텔레옵</title>
<style>
:root{--bg:#14161a;--fg:#e8e6e1;--dim:#8b9099;--line:#2a2f37;--ok:#5fb37a;--bad:#d4674f;--acc:#6ba3d6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;
     display:grid;grid-template-columns:minmax(430px,1fr) 1.3fr;height:100vh}
@media(max-width:900px){body{grid-template-columns:1fr;height:auto}}
#panel{overflow-y:auto;padding:14px 16px;border-right:1px solid var(--line)}
h1{font-size:15px;margin:0 0 4px;letter-spacing:.04em}
#status{font-size:12px;color:var(--dim);margin-bottom:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:1px}
.on{background:var(--ok)}.off{background:var(--bad)}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);
   margin:16px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--line)}
.row{display:grid;grid-template-columns:112px 1fr 62px 62px 54px;gap:8px;align-items:center;padding:3px 0}
.row label{font-size:12px;color:var(--dim);overflow:hidden;text-overflow:ellipsis}
input[type=range]{width:100%;accent-color:var(--acc)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.err{text-align:right;font-variant-numeric:tabular-nums;font-size:12px;color:var(--dim)}
.err.hot{color:var(--bad);font-weight:700}
.meas{color:var(--acc)}
.clamped{outline:1px solid var(--bad);border-radius:3px}
#bar{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap}
button{background:#1e232b;color:var(--fg);border:1px solid var(--line);border-radius:5px;
       padding:7px 13px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.warn{border-color:var(--bad);color:var(--bad)}
iframe{border:0;width:100%;height:100%;min-height:420px;background:#000}
#note{font-size:11px;color:var(--dim);margin-top:14px;line-height:1.6}
</style>
<div id=panel>
  <h1>CAD ↔ 실기 텔레옵</h1>
  <div id=status><span class="dot off"></span><span id=stxt>연결 확인 중…</span></div>
  <div id=rows></div>
  <div id=bar>
    <button onclick="preset('ready')">READY 자세</button>
    <button onclick="preset('zero')">전부 0 (2048)</button>
    <button class=warn onclick="preset('hold')">현재 실측으로 맞추기</button>
  </div>
  <div id=note>
    슬라이더 = <b>명령</b>, 파란 숫자 = <b>실측</b>, 오른쪽 = 오차(°).
    오차가 3° 넘으면 빨갛게 된다 — 부호가 뒤집혔거나 관절이 걸린 것.<br>
    잿슨 브리지가 슬루 60 °/s 로 제한하므로 슬라이더를 확 당겨도 천천히 따라간다.
  </div>
</div>
<iframe id=v src="__MESHCAT__"></iframe>
<script>
const N=__NAMES__, LIM=__LIMS__, D=180/Math.PI;
let cmd=new Array(14).fill(0), dirty=false;
const rows=document.getElementById('rows');
let html='';
for(const [g,list] of Object.entries(__GROUP__)){
  html+=`<h2>${g}</h2>`;
  for(const n of list){const i=N.indexOf(n);
    html+=`<div class=row><label title="${n}">${n}</label>`+
      `<input type=range id=s${i} min=${(LIM[i][0]*D).toFixed(1)} max=${(LIM[i][1]*D).toFixed(1)} step=0.5 value=0 oninput="mv(${i},this.value)">`+
      `<span class=num id=c${i}>0.0</span><span class="num meas" id=m${i}>–</span>`+
      `<span class=err id=e${i}>–</span></div>`;}
}
rows.innerHTML=html;
function mv(i,v){cmd[i]=v*Math.PI/180;document.getElementById('c'+i).textContent=(+v).toFixed(1);dirty=true;}
function setAll(q){cmd=q.slice();for(let i=0;i<14;i++){
  document.getElementById('s'+i).value=(q[i]*D).toFixed(1);
  document.getElementById('c'+i).textContent=(q[i]*D).toFixed(1);}dirty=true;}
async function preset(k){const r=await fetch('/preset?k='+k,{method:'POST'});setAll(await r.json());}
setInterval(async()=>{
  if(dirty){dirty=false;fetch('/set',{method:'POST',body:JSON.stringify(cmd)});}
  const s=await (await fetch('/state')).json();
  document.querySelector('.dot').className='dot '+(s.link?'on':'off');
  document.getElementById('stxt').textContent=s.link
    ? '잿슨 연결됨 — 실측 갱신 중' : ('잿슨 미연결 ' + (s.err?('· '+s.err):''));
  for(let i=0;i<14;i++){
    const m=s.meas[i]*D, c=cmd[i]*D, e=m-c;
    document.getElementById('m'+i).textContent=s.link?m.toFixed(1):'–';
    const el=document.getElementById('e'+i);
    el.textContent=s.link?(e>=0?'+':'')+e.toFixed(1):'–';
    el.className='err'+(s.link&&Math.abs(e)>3?' hot':'');
    document.getElementById('s'+i).className=s.clamped[i]?'clamped':'';
  }
},200);
</script>"""


class H(BaseHTTPRequestHandler):
    page = ""

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/state"):
            with LOCK:
                self._send(json.dumps({"meas": S["meas"], "tick": S["tick"],
                                       "clamped": S["clamped"], "link": S["link"],
                                       "err": S["err"]}))
        else:
            self._send(H.page, "text/html")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path.startswith("/set"):
            try:
                q = json.loads(raw)
                if isinstance(q, list) and len(q) == 14:
                    with LOCK:
                        S["cmd"] = [float(v) for v in q]
            except ValueError:
                pass
            self._send("{}")
        elif self.path.startswith("/preset"):
            k = self.path.split("k=")[-1]
            with LOCK:
                if k == "ready":
                    q = [READY[n_] for n_ in NAMES]
                elif k == "zero":
                    q = [0.0] * 14
                else:
                    q = list(S["meas"])
                S["cmd"] = q
            self._send(json.dumps(q))
        else:
            self._send("{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=os.path.join(ROOT, "robot", "robot.urdf"))
    ap.add_argument("--mesh-dir", default=os.path.join(ROOT, "robot"))
    ap.add_argument("--jetson", default="192.168.137.7")
    ap.add_argument("--jetson-port", type=int, default=5599)
    ap.add_argument("--web-port", type=int, default=8099)
    ap.add_argument("--hz", type=float, default=20.0, help="잿슨으로 보내는 주기")
    ap.add_argument("--no-robot", action="store_true", help="실기 없이 UI 만")
    args = ap.parse_args()

    viz = Viz(args.urdf, args.mesh_dir)
    lims = []
    for n in NAMES:
        j = viz.model.getJointId(n)
        lims.append([float(viz.model.lowerPositionLimit[viz.model.idx_qs[j]]),
                     float(viz.model.upperPositionLimit[viz.model.idx_qs[j]])])

    H.page = (PAGE.replace("__MESHCAT__", viz.url)
                  .replace("__NAMES__", json.dumps(NAMES))
                  .replace("__LIMS__", json.dumps(lims))
                  .replace("__GROUP__", json.dumps(GROUP, ensure_ascii=False)))

    threading.Thread(target=viz_thread, args=(viz,), daemon=True).start()
    if not args.no_robot:
        threading.Thread(target=link_thread,
                         args=(args.jetson, args.jetson_port, args.hz),
                         daemon=True).start()
    else:
        print("[--no-robot] 실기에 접속하지 않는다. 3D 와 UI 만 돈다.")

    srv = ThreadingHTTPServer(("0.0.0.0", args.web_port), H)
    print(f"meshcat  {viz.url}")
    print(f"조작 UI  http://localhost:{args.web_port}")
    print("Ctrl+C 로 종료 (잿슨 브리지는 연결이 끊기면 그 자리에서 홀드한다)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
