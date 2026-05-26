# video-caption-generator (`vcg`)

영어 영상의 음성을 OpenAI Whisper로 받아쓰고, OpenAI 채팅 모델로 한국어로 번역한 뒤, ffmpeg로 자막을 영상에 입히는(하드섭) CLI 도구입니다.

번역 결과를 영상에 굽기 전에 사람이 한 번 검토·수정할 수 있도록 **`generate` → (검토/수정) → `burn`** 의 3단계 워크플로로 설계되어 있습니다. 굽기(burn-in)는 되돌릴 수 없기 때문입니다.

## 요구 사항

- Python >= 3.12
- OpenAI API 키
- ffmpeg는 별도 설치가 필요 없습니다 — `imageio-ffmpeg`로 번들되어 제공됩니다.
- 한국어 자막용 폰트: 기본값은 Windows에 기본 설치된 **Malgun Gothic**입니다. 다른 OS에서는 `--font-name`으로 지정하세요.

## 설치

[uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync
```

## 설정

`.env.example`을 복사해 `.env`를 만들고 API 키를 입력합니다.

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...

# 선택: 모델 오버라이드
OPENAI_TRANSCRIBE_MODEL=whisper-1   # 기본값. 세그먼트별 타임스탬프를 주는 유일한 OpenAI 모델
OPENAI_TRANSLATE_MODEL=gpt-4o       # 기본값. 모든 채팅 모델 사용 가능
```

## 사용법

명령은 `uv run vcg ...`로 실행하거나, 가상환경을 활성화한 뒤 `vcg ...`로 실행합니다.

### 1) `generate` — 받아쓰기 + 번역

영상에서 오디오를 추출해 Whisper로 받아쓰고, 한국어로 번역하여 두 개의 SRT를 만듭니다.

```bash
vcg generate input.mp4
vcg generate input.mp4 --language en   # 원본 언어 힌트
```

출력:
- `input.en.srt` — 영어 자막
- `input.ko.srt` — 한국어 자막 (검토·수정 대상)

### 2) `review` — 영/한 나란히 보기

두 SRT를 세그먼트별로 나란히 출력해 번역 품질을 확인합니다.

```bash
vcg review input.en.srt input.ko.srt
```

이후 `input.ko.srt`를 텍스트 에디터로 직접 수정합니다. **타임스탬프는 그대로 두세요.**

### 3) `burn` — 영상에 자막 굽기

한국어 SRT를 영상에 하드코딩해 새 MP4를 만듭니다. (되돌릴 수 없으므로 확인 프롬프트가 표시됩니다.)

```bash
vcg burn input.mp4 input.ko.srt
vcg burn input.mp4 input.ko.srt --font-size 24 --margin-v 20
```

옵션:
- `--output` 출력 MP4 경로 (기본: `<video>.captioned.mp4`)
- `--font-name` 한국어 지원 폰트 (기본: `Malgun Gothic`)
- `--font-size` 폰트 크기 (기본: 22)
- `--margin-v` 하단 여백(libass 단위, 낮을수록 화면 하단에 가까움, 기본: 10)

### 보조 명령

**`translate`** — 기존 영어 SRT만 다시 번역 (Whisper 재호출 없이 번역 모델만 교체하고 싶을 때 유용)

```bash
vcg translate input.en.srt
vcg translate input.en.srt --output custom.ko.srt
```

**`download`** — URL(YouTube 등)에서 영상 다운로드. 해상도 + fps를 골라 받습니다.

```bash
vcg download https://youtu.be/xxxxxxx
vcg download https://youtu.be/xxxxxxx -o downloads/
vcg download https://youtu.be/xxxxxxx -o myclip.mp4
```

**`trim`** — 영상을 `[start, end]` 구간으로 자르기 (프레임 정확도를 위해 재인코딩)

```bash
vcg trim input.mp4 --start 00:05:01.23 --end 00:05:09.23
vcg trim input.mp4 --end 00:01:00 -o intro.mp4
vcg trim input.mp4 --start 00:10:00
```

옵션: `--crf`(기본 20, 낮을수록 고화질·대용량), `--preset`(기본 `medium`)

**`merge`** — 영상 여러 개를 순서대로 하나로 합치기

```bash
vcg merge intro.mp4 main.mp4 outro.mp4 -o full.mp4
vcg merge part1.mp4 part2.mp4 --copy -o joined.mp4
```

- 기본(재인코딩): 각 입력을 **첫 영상 해상도**로 정규화(레터박스)하고 fps·오디오 포맷을 맞춘 뒤 이어붙입니다. 해상도·코덱이 서로 달라도 동작합니다.
- `--copy`: 재인코딩 없이 스트림 복사라 매우 빠르지만, 모든 입력의 코덱·해상도·fps가 동일해야 합니다.
- 옵션: `--crf`, `--preset`(재인코딩 모드에만 적용)

**`speed`** — 영상 배속 변경 (영상 `setpts` + 오디오 `atempo`, 음정 유지)

```bash
vcg speed talk.mp4 --rate 1.25
vcg speed talk.mp4 -r 1.5 -o talk_fast.mp4
```

- `--rate`(필수): 배속 배율. `1` 초과면 빠르게, 미만이면 느리게. 2.0를 넘는 큰 배율은 `atempo` 필터를 체인으로 분해해 처리합니다.
- 기본 출력명 `<stem>.<rate>x.mp4` (예: `talk.1.5x.mp4`)
- 옵션: `--crf`, `--preset`

## 참고 사항

- **긴 영상 처리(자동 청크 분할)**: Whisper API는 25 MB 업로드 상한이 있습니다. 오디오는 16 kHz 모노 WAV(32 KB/s)로 추출되어 한 파일은 약 13분 분량입니다. 이보다 긴 영상은 자동으로 약 13분 단위 청크로 분할해 각각 전사한 뒤, 각 청크의 타임스탬프에 시작 오프셋을 더해 하나의 타임라인으로 다시 합칩니다(별도 옵션 불필요, 결과는 단일 전사와 동일). 청크 경계는 파일 크기로 정확히 계산되어 시간 오차가 누적되지 않습니다.
- **번역 모델**: `OPENAI_TRANSLATE_MODEL`에 지정한 모델명이 OpenAI에서 오류를 내면, 그 오류를 그대로 표시합니다(임의로 다른 모델로 대체하지 않습니다).

## 전체 워크플로 예시

```bash
vcg download https://youtu.be/xxxxxxx -o talk.mp4   # (선택) 영상 받기
vcg trim talk.mp4 --start 00:00:00 --end 00:02:00 -o intro.mp4  # (선택) 구간 자르기
vcg generate intro.mp4                               # 받아쓰기 + 번역
vcg review intro.en.srt intro.ko.srt                 # 검토
# -> intro.ko.srt 를 에디터로 수정
vcg burn intro.mp4 intro.ko.srt                      # 자막 굽기 -> intro.captioned.mp4
```
