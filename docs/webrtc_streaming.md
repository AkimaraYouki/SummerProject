# Isaac Sim 원격 WebRTC 스트리밍 (랩 PC ↔ 맥북)

랩 PC(`do@192.168.137.111`)는 모니터 없는 SSH 전용 서버라, OnShape 임포터처럼 Isaac Sim GUI가
필요한 작업은 WebRTC로 화면을 맥북에 스트리밍해서 처리한다. 이 문서는 그 절차를 기록한다.

## 사전 준비 (한 번만)

- **맥북**: [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release)에서 macOS용 최신 버전 다운로드 후 설치
- **랩 PC**: Isaac Sim 5.0.0-rc.45가 `~/isaacsim`에 이미 설치돼 있음 (별도 설치 불필요)

## 랩 PC에서 스트리밍 서버 실행

```bash
ssh do@192.168.137.111
nohup ~/isaacsim/isaac-sim.streaming.sh \
    --/app/livestream/publicEndpointAddress=192.168.137.111 \
    > ~/isaac_streaming.log 2>&1 &
disown
```

**`--/app/livestream/publicEndpointAddress=<랩PC의 LAN IP>` 플래그가 필수다.** 랩 PC엔 네트워크
인터페이스가 여러 개(LAN `192.168.137.111`, Tailscale, Docker 브리지)라서, 이 플래그 없이 실행하면
시그널링(8011 포트)은 성공하는데 WebRTC가 클라이언트에게 엉뚱한 인터페이스 주소를 미디어
엔드포인트로 광고해버려서 — 접속은 되는데 화면 로딩에서 멈추는 증상이 생긴다.

## 로드 확인

첫 실행은 셰이더 캐시가 없어 1~2분, 이후 재실행은 캐시가 남아있어 몇 초면 끝난다. 아래 메시지가
로그에 뜨면 준비 완료:

```
Isaac Sim Full Streaming App is loaded.
```

```bash
grep "Isaac Sim Full Streaming App is loaded" ~/isaac_streaming.log
ss -tlnp | grep -E "8011|49100"   # 8011=시그널링, 49100=미디어, 둘 다 열려있어야 함
```

## 맥북에서 접속

Isaac Sim WebRTC Streaming Client 앱 실행 → **Server**란에 `192.168.137.111` 입력 → **Connect**

## 종료 (다 쓰면 꼭 끄기 — 공용 GPU 서버)

```bash
pkill -f "isaacsim.exp.full.streaming"
```

GPU/CPU를 크게 잡아먹는 프로세스라(관측치: CPU 800%+, RAM 16GB+), 안 쓸 때 계속 띄워두면 같은
서버를 쓰는 다른 사람 작업에 지장을 준다.

## 트러블슈팅

- **포트가 8211이 아니라 8011이다.** 예전 NVIDIA 문서나 온라인 예시는 대부분 8211을 쓰는데,
  이 설치본(Isaac Sim 5.0.0-rc.45)은 8011을 쓴다. 버전마다 다르니 `ss -tlnp`로 `kit` 프로세스가
  실제로 뭘 물고 있는지 직접 확인하는 게 제일 확실하다.
- **`http://<ip>:8011`을 브라우저로 직접 열면 `{"detail":"Not Found"}`가 뜬다.** 이 버전은 자체
  브라우저 클라이언트를 안 띄우고 `/v1/streaming/creds` 같은 협상용 API만 노출한다 — 반드시
  전용 클라이언트 앱으로 접속해야 한다.
- **`publicEndpointAddress` 없이 실행하면 로딩 화면에서 멈춘다.** 시그널링(HTTP)까진 성공해서
  헷갈리기 쉽지만, 실제 원인은 WebRTC 미디어 협상(ICE)이 잘못된 인터페이스 IP를 광고해서
  생기는 문제다. 위 플래그로 해결됨.
- SSH로 `pkill` 등을 실행할 때 이따금 세션 자체가 exit code 255로 끊기는 경우가 있었다 —
  재접속해서 `ps aux`로 실제 상태를 확인하면 대개 명령 자체는 정상적으로 실행돼 있었다.

## 참고 — 슈퍼컴퓨터(다른 클러스터)에서의 동일 절차

같은 팀에서 슈퍼컴 L40S 노드(`kitsu02`)에 Docker로 Isaac Sim을 띄울 때도 같은 플래그를 쓴다:

```bash
./runheadless.sh -v --/app/livestream/publicEndpointAddress=172.19.19.200
```

다만 그 환경은 로그인 노드와 GPU 노드가 분리돼 있어서 `socat`으로 포트(49100 TCP, 47998 UDP)를
로그인 노드까지 릴레이해야 한다. 랩 PC는 같은 LAN에서 직접 붙는 구조라 그 릴레이 단계는
필요 없다.
