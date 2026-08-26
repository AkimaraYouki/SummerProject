#!/usr/bin/env python3
"""반복 측정 결과를 **한 장의 HTML**로 만든다.

    python3 scripts/diag/build_dashboard.py --out ~/odm_out/dashboard.html

`~/odm_out/rep/<ver>_<i>.npz` 를 읽는다 (`odm measure <ver> --repeat=N` 산출물).

## 왜 표가 아니라 이 그림인가

2026-08-24 에 잡음을 실측하니 같은 체크포인트에서 회전 오차가 11 배 흔들렸다.
**값만 그리면 거짓말이 된다.** 그래서 모든 막대를 최소~최대 범위로 그리고
평균을 표식으로 얹는다 — 두 정책의 범위가 겹치면 그 차이는 읽지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source"))
from open_duck_mini_isaaclab.joint_order import ACT_LEG_JOINT_IDX   # noqa: E402

TAU_SAT, SKIP, TRACK_FRAC = 3.16, 100, 0.5
PRIO = {"forward": 3.0, "backward": 3.0, "turn": 2.0, "left": 1.0, "right": 1.0}


def metrics(path):
    z = np.load(path, allow_pickle=True)
    conds = [str(c) for c in z["conds"]]
    dt = float(z["ctrl_dt"]) if "ctrl_dt" in z else 0.02
    e, rr, fa, st, wt, rt, bad = {}, [], [], [], [], [], []
    for c in conds:
        v = z[f"{c}__v_base"][SKIP:]; w = z[f"{c}__w_base"][SKIP:]; cmd = z[f"{c}__cmd"]
        e[c] = (float(abs(w.mean() - cmd[2])) if c == "turn"
                else float(np.linalg.norm(v.mean(axis=(0, 1)) - cmd[:2])))
        vb = v.mean(axis=(0, 1)); wz = float(w.mean())
        for tgt, act, nm in ((cmd[0], vb[0], "vx"), (cmd[1], vb[1], "vy"), (cmd[2], wz, "wz")):
            if abs(tgt) < 1e-6:
                continue
            if np.sign(tgt) != np.sign(act) or abs(act) < TRACK_FRAC * abs(tgt):
                bad.append(f"{c}.{nm}")
        g = z[f"{c}__grav"][SKIP:]
        r = np.degrees(np.arctan2(-g[..., 1], -g[..., 2]))
        fa.append(float(100 * (np.abs(r) > 40).mean()))
        if c == "stop":
            continue
        rr.append(float(r.std()))
        t = z[f"{c}__tau"][SKIP:][..., ACT_LEG_JOINT_IDX]; a = np.abs(t)
        st.append(float(100 * (a > TAU_SAT * 0.995).mean()))
        d = z[f"{c}__dq"][SKIP:]; n = min(len(t), len(d))
        wt.append(float(np.abs(t[:n] * d[:n]).sum(-1).mean()))
        rt.append(float(np.diff(r, axis=0).std() / dt))
    return {
        "sat": float(np.mean(st)), "watt": float(np.mean(wt)),
        "roll": float(np.mean(rr)), "rrate": float(np.mean(rt)),
        "fall": float(max(fa)),
        "fb": (e["forward"] + e["backward"]) / 2, "turn": e["turn"],
        "lr": (e["left"] + e["right"]) / 2,
        "score": sum(PRIO[k] * e[k] for k in PRIO) / sum(PRIO.values()),
        "obey": len(bad) == 0,
    }, os.path.basename(str(z["checkpoint"]))


def collect(repdir):
    out = {}
    for p in sorted(glob.glob(os.path.join(repdir, "*_*.npz"))):
        m = re.match(r"(v[\w]+)_(\d+)\.npz$", os.path.basename(p))
        if not m:
            continue
        out.setdefault(m.group(1), []).append(p)
    rows = []
    for ver, paths in out.items():
        runs, ck = [], ""
        for p in sorted(paths):
            try:
                d, ck = metrics(p)
                runs.append(d)
            except Exception:                                    # noqa: BLE001
                continue
        if not runs:
            continue
        agg = {"ver": ver, "n": len(runs), "ck": ck,
               "obey": all(r["obey"] for r in runs)}
        for k in ("sat", "watt", "roll", "rrate", "fall", "fb", "turn", "lr", "score"):
            v = np.array([r[k] for r in runs], float)
            agg[k] = {"m": float(v.mean()), "lo": float(v.min()), "hi": float(v.max())}
        rows.append(agg)
    rows.sort(key=lambda r: int(re.sub(r"\D", "", r["ver"]) or 0))
    return rows


#: 표시할 지표. (키, 이름, 단위, 잡음 신뢰도, 설명)
METRICS = [
    ("watt", "일률", "W", "높음", "|τ·ω| 합의 평균. 반복측정 폭 3 % 로 가장 믿을 만하다"),
    ("sat", "토크 포화", "%", "보통", "|τ| 가 실효 한계 3.16 N·m 에 붙어 있는 시간 비율"),
    ("roll", "몸통 흔들림", "°", "보통", "roll RMS. 걸을 때 좌우로 휘청이는 정도"),
    ("fb", "앞뒤 추종", "", "낮음", "전진·후진 명령 오차. 반복측정 폭이 146 % 라 순위를 믿지 말 것"),
    ("turn", "회전 추종", "", "낮음", "회전 명령 오차. 반복측정 폭 223 % — 같은 정책이 11 배 흔들린다"),
    ("lr", "옆걸음 추종", "", "보통", "게걸음 명령 오차"),
]

NOTES = {
    "v47": "액션 지연 2~3 스텝. 회전이 여기서 무너졌다",
    "v55": "토르소 각속도 벌점 −0.7 을 처음 켰다",
    "v57": "순수 단일축 명령 도입",
    "v59": "질량 무작위화 0.90~1.10 으로 되돌림",
    "v61": "실기에서 가장 잘 걷던 정책",
    "v65": "imitation_w_ang_vel_xy 0.1 → 1.0. 흔들림 돌파구",
    "v68": "토크 −0.2 · 낙상종료 0.85 · 탐색 σ1.3",
    "v70": "몸통 CoM 뒤로 10 mm",
    "v73": "발바닥 큰 것(big_foot)",
    "v74": "v73 + v68 묶음",
    "v75": "v74 + 토르소 벌점. 11,800 iter 장기 학습",
    "v77": "마찰 0.4~1.3 · 강성 0.8~1.2 로 넓힘",
    "v78": "path frame 제거 + heading 0.5",
    "v79": "path frame 제거만 (대조군)",
    "v81": "v75 + heading 0.2",
    "v82": "path frame 의 횡방향만 제거",
}


def build(rows, out):
    data = json.dumps(rows, ensure_ascii=False)
    mets = json.dumps([{"k": k, "n": n, "u": u, "c": c, "d": d} for k, n, u, c, d in METRICS],
                      ensure_ascii=False)
    notes = json.dumps(NOTES, ensure_ascii=False)
    total_runs = sum(r["n"] for r in rows)
    tpl = TEMPLATE.replace("__DATA__", data).replace("__METRICS__", mets)
    tpl = tpl.replace("__NOTES__", notes)
    tpl = tpl.replace("__NVER__", str(len(rows))).replace("__NRUN__", str(total_runs))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tpl)
    print(f"[ok] {out}  ({len(rows)} 버전 · 측정 {total_runs} 회)")


TEMPLATE = r"""<title>Duck Mini 정책 계측판</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+KR:wght@300;400;500;600;700&display=swap">
<style>
:root{
  --bg:#eef0ef; --panel:#f7f8f7; --line:#d3d8d6; --line-soft:#e4e8e6;
  --ink:#171b21; --ink-2:#3d454f; --muted:#6d7883;
  --pen:#2b6d8f; --pen-soft:#a8ccdc; --hot:#c2542a; --hot-soft:#eec6b4;
  --good:#2f7d5c; --grid:#dfe4e2;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans KR","IBM Plex Sans",system-ui,-apple-system,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#171b21; --panel:#1e242c; --line:#333c47; --line-soft:#272f38;
  --ink:#e7ecf1; --ink-2:#bcc6d1; --muted:#8b96a3;
  --pen:#7fb8d8; --pen-soft:#33505f; --hot:#e0703f; --hot-soft:#5c3524;
  --good:#7fc4a0; --grid:#28313a;
}}
:root[data-theme="dark"]{
  --bg:#171b21; --panel:#1e242c; --line:#333c47; --line-soft:#272f38;
  --ink:#e7ecf1; --ink-2:#bcc6d1; --muted:#8b96a3;
  --pen:#7fb8d8; --pen-soft:#33505f; --hot:#e0703f; --hot-soft:#5c3524;
  --good:#7fc4a0; --grid:#28313a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-weight:300;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px) 96px}
header{display:flex;flex-direction:column;gap:10px;padding-bottom:26px;
  border-bottom:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--muted)}
h1{font-size:clamp(28px,4.2vw,44px);font-weight:600;letter-spacing:-.02em;
  margin:0;text-wrap:balance}
.sub{color:var(--ink-2);max-width:64ch;font-size:15px}
.verdict{margin-top:30px;background:var(--panel);border:1px solid var(--line);
  border-radius:3px;padding:22px 24px;display:flex;flex-direction:column;gap:12px}
.verdict h2{margin:0;font-size:13px;font-family:var(--mono);letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);font-weight:500}
.verdict p{margin:0;font-size:16px;color:var(--ink-2)}
.verdict strong{color:var(--ink);font-weight:600}
section{margin-top:52px}
h3{font-size:19px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}
.note{color:var(--muted);font-size:14px;margin:0 0 20px;max-width:70ch}
.mgrid{display:grid;gap:26px;grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;
  padding:18px 20px 20px;display:flex;flex-direction:column;gap:4px}
.chead{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.cname{font-size:15px;font-weight:600}
.conf{font-family:var(--mono);font-size:10px;letter-spacing:.1em;padding:2px 7px;
  border-radius:2px;border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.conf.hi{color:var(--good);border-color:var(--good)}
.conf.lo{color:var(--hot);border-color:var(--hot)}
.cdesc{font-size:12.5px;color:var(--muted);margin:0 0 12px;min-height:2.6em}
.rows{display:flex;flex-direction:column;gap:5px}
.row{display:grid;grid-template-columns:46px 1fr 62px;align-items:center;gap:9px}
.vname{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);text-align:right}
.row.best .vname{color:var(--pen);font-weight:600}
.track{position:relative;height:15px;background:var(--grid);border-radius:2px;overflow:hidden}
.rng{position:absolute;top:0;bottom:0;background:var(--pen-soft)}
.mean{position:absolute;top:-1px;bottom:-1px;width:2px;background:var(--pen)}
.row.best .mean{background:var(--hot);width:3px}
.val{font-family:var(--mono);font-size:11.5px;text-align:right;
  font-variant-numeric:tabular-nums;color:var(--ink-2)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:3px;background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:820px;font-family:var(--mono);font-size:12px}
th,td{padding:8px 11px;text-align:right;white-space:nowrap;
  border-bottom:1px solid var(--line-soft);font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:var(--panel);font-weight:500;color:var(--muted);
  font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;cursor:pointer;
  border-bottom:1px solid var(--line)}
th:hover{color:var(--ink)}
th:first-child,td:first-child{text-align:left}
td.ver{font-weight:600;color:var(--ink)}
tr:hover td{background:var(--line-soft)}
.spread{color:var(--muted);font-size:10.5px}
.pill{display:inline-block;padding:1px 6px;border-radius:2px;font-size:10px;
  border:1px solid currentColor}
.pill.ok{color:var(--good)} .pill.no{color:var(--hot)}
.desc{font-family:var(--sans);font-weight:300;font-size:11.5px;color:var(--muted);
  text-align:left;white-space:normal;max-width:270px}
footer{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:12.5px;display:flex;flex-direction:column;gap:6px}
code{font-family:var(--mono);font-size:.92em;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Open Duck Mini V2 · Isaac Lab</div>
  <h1>정책 계측판</h1>
  <p class="sub">버전 __NVER__ 개를 같은 조건에서 다시 쟀다. 각 버전 3 회, 총 __NRUN__ 회.
     막대는 <strong>최소~최대 범위</strong>이고 세로선이 평균이다 —
     두 정책의 범위가 겹치면 그 차이는 읽지 않는다.</p>
</header>

<div class="verdict">
  <h2>읽는 법</h2>
  <p id="verdict-body"></p>
</div>

<section>
  <h3>지표별 순위</h3>
  <p class="note">각 지표에서 좋은 순으로 위쪽 10 개. 신뢰도 딱지는 같은 체크포인트를
     반복해 쟀을 때의 흔들림 폭에서 나온 것이다 — <em>낮음</em>이면 순위 자체가 잡음이다.</p>
  <div class="mgrid" id="cards"></div>
</section>

<section>
  <h3>전체 표</h3>
  <p class="note">머리글을 눌러 정렬한다. 괄호 안은 3 회 측정의 최소~최대.
     <strong>방향</strong>은 6 방향 명령을 부호와 크기(명령의 50 % 이상)로 따라갔는지 —
     이것만은 잡음에 강해서 문턱으로 쓴다.</p>
  <div class="tablewrap"><table id="tbl"></table></div>
</section>

<footer>
  <div>측정: <code>gait_compare.py</code> · 6 방향 각 500 스텝 · 과도구간 100 스텝 제외 · env 4 개</div>
  <div>토크 실효 한계 3.16 N·m 는 effort_limit 4.1 이 아니라 토크-속도 모델이 먼저 자른 값이다.</div>
</footer>
</div>

<script>
const DATA = __DATA__, METRICS = __METRICS__, NOTES = __NOTES__;
const fmt = (k,v) => (k==="watt") ? v.toFixed(2)
                   : (k==="sat"||k==="roll"||k==="rrate"||k==="fall") ? v.toFixed(2)
                   : v.toFixed(4);

/* 판정 문구 — 일률(가장 믿을 만한 지표)과 방향 준수로 고른다 */
(function(){
  const ok = DATA.filter(d=>d.obey);
  const pool = ok.length ? ok : DATA;
  const best = pool.slice().sort((a,b)=>a.watt.m-b.watt.m)[0];
  const turnSpread = DATA.map(d=>d.turn.hi-d.turn.lo).sort((a,b)=>b-a)[0];
  document.getElementById("verdict-body").innerHTML =
    `일률이 가장 낮으면서 6 방향을 모두 따라간 것은 <strong>${best.ver}</strong> `
    + `(${best.watt.m.toFixed(2)} W, 범위 ${best.watt.lo.toFixed(2)}~${best.watt.hi.toFixed(2)}). `
    + `반면 회전 추종은 한 정책 안에서 최대 <strong>${turnSpread.toFixed(4)}</strong> 까지 흔들렸다 — `
    + `추종 순위는 이 폭보다 큰 차이일 때만 의미가 있다.`;
})();

/* 지표별 카드 */
const cards = document.getElementById("cards");
for (const m of METRICS){
  const rows = DATA.filter(d=>d[m.k]).slice().sort((a,b)=>a[m.k].m-b[m.k].m).slice(0,10);
  const hi = Math.max(...rows.map(r=>r[m.k].hi));
  const cls = m.c==="높음" ? "hi" : (m.c==="낮음" ? "lo" : "");
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="chead"><span class="cname">${m.n}${m.u?` <span style="color:var(--muted);font-weight:400">[${m.u}]</span>`:""}</span>`
    + `<span class="conf ${cls}">신뢰도 ${m.c}</span></div>`
    + `<p class="cdesc">${m.d}</p><div class="rows">`
    + rows.map((r,i)=>{
        const d=r[m.k], L=(d.lo/hi)*100, W=Math.max(((d.hi-d.lo)/hi)*100,1.2), M=(d.m/hi)*100;
        return `<div class="row${i===0?" best":""}"><span class="vname">${r.ver}</span>`
          + `<span class="track"><span class="rng" style="left:${L}%;width:${W}%"></span>`
          + `<span class="mean" style="left:${M}%"></span></span>`
          + `<span class="val">${fmt(m.k,d.m)}</span></div>`;
      }).join("")
    + `</div>`;
  cards.appendChild(el);
}

/* 전체 표 */
const COLS = [["ver","버전"],["watt","일률 W"],["sat","포화 %"],["roll","rollRMS °"],
  ["rrate","roll속"],["fb","앞뒤"],["turn","회전"],["lr","옆"],["fall","낙상 %"],
  ["obey","방향"],["note","바뀐 것"]];
let sortKey="watt", asc=true;
function draw(){
  const rows = DATA.slice().sort((a,b)=>{
    if(sortKey==="ver") return (asc?1:-1)*(parseInt(a.ver.replace(/\D/g,""))-parseInt(b.ver.replace(/\D/g,"")));
    if(sortKey==="obey") return (asc?1:-1)*((a.obey?1:0)-(b.obey?1:0));
    if(sortKey==="note") return 0;
    return (asc?1:-1)*(a[sortKey].m-b[sortKey].m);
  });
  document.getElementById("tbl").innerHTML =
    "<thead><tr>"+COLS.map(([k,l])=>`<th data-k="${k}">${l}${sortKey===k?(asc?" ↑":" ↓"):""}</th>`).join("")+"</tr></thead>"
    + "<tbody>"+rows.map(r=>"<tr>"+COLS.map(([k])=>{
        if(k==="ver") return `<td class="ver">${r.ver}</td>`;
        if(k==="obey") return `<td><span class="pill ${r.obey?"ok":"no"}">${r.obey?"OK":"실패"}</span></td>`;
        if(k==="note") return `<td class="desc">${NOTES[r.ver]||""}</td>`;
        const d=r[k];
        return `<td>${fmt(k,d.m)}<br><span class="spread">${fmt(k,d.lo)}~${fmt(k,d.hi)}</span></td>`;
      }).join("")+"</tr>").join("")+"</tbody>";
  document.querySelectorAll("#tbl th").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k==="note") return;
    if(sortKey===k) asc=!asc; else {sortKey=k; asc=true;} draw();
  });
}
draw();
</script>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default=os.path.expanduser("~/odm_out/rep"))
    ap.add_argument("--out", default=os.path.expanduser("~/odm_out/dashboard.html"))
    a = ap.parse_args()
    build(collect(a.rep), a.out)
