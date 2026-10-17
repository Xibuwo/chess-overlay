# Chess Overlay (Windows 11)

Passive, screen-reading chess overlay. It never injects into a process, never sends
mouse or keyboard input, and never reads game memory. It captures screen pixels,
works out the position, asks a local UCI engine, and draws on its own transparent,
click-through window.

## Install

```
pip install -r requirements.txt
python main.py
```

Engines (any you want to use):

* **Stockfish** - download the Windows build, point the "Stockfish exe" box at it.
* **Maia 3** - install it from https://github.com/CSSLab/maia3 so that `maia3-uci`
  works in a terminal, then use the command `maia3-uci --model 5m` (or 3m/23m/79m).
* **Maia (Lc0 + weights)** - `lc0.exe` plus a folder containing `maia-1100.pb.gz` ... `maia-1900.pb.gz`.
* **Lc0** - `lc0.exe` plus any weights file.

## Auto mode (default): 2D and 3D, no setup

Tick **Auto-detect board**. It finds the checkerboard by itself, follows it when the camera moves, works out which way White faces, and reads the game from pixels. It only needs
to see the **starting position once** at the start of a game (or press *Reset game* on one). The
game's own yellow/gold *last-move* highlight is used as an extra cue to find moves reliably.
Requirements: borderless windowed game, board fully on screen at the start.

## Manual setup (per game / skin, only when Auto is off)

1. Run the game in **borderless windowed** mode (exclusive fullscreen hides overlays).
2. Open a game where the board is in the **starting position**.
3. Pick the right "I play Black" / "Board flipped" boxes.
4. **Calibrate board**: click the four outer corners of the 8x8 grid in the order
   top-left, top-right, bottom-right, bottom-left (as seen on screen).
5. **Learn pieces**: captures piece templates from the starting position (saved to `%APPDATA%\ChessOverlay`).
6. Play. The overlay follows the game by matching every frame against legal moves.

## What the slider does

| Engine | Slider behaviour |
|---|---|
| Stockfish | ELO -> Skill Level (0-20), depth limit (1-18), or UCI_LimitStrength + UCI_Elo (pick the mode in the GUI) |
| Maia 3 | ELO sent directly to the engine's `Elo` option, search forced to `nodes=1` |
| Maia (Lc0) | nearest `maia-XXXX` net for the ELO, `nodes=1` |
| Lc0 | ELO -> node limit |

## Overlay legend

* green arrow: best move - blue arrow: second move - red arrow: opponent's threat
  (their best reply if you passed; on their turn it's their predicted move)
* orange square + 🔥 `A2 D1 (-1)`: piece under attack, attackers / defenders / differential
* 🩸 piece that is hung (attacked, undefended) - 💧 piece that loses material in the exchange (`-N` = value)

## Layout

`main.py` wiring - `gui.py` settings window - `overlay.py` transparent overlay -
`winflags.py` Win32 styles - `calibration.py` corner picker - `geometry.py`
homography - `recognition.py` piece templates - `tracker.py` legal-move tracking -
`analysis.py` attackers/defenders/SEE - `engines.py` UCI + strength mapping -
`strength.py` slider maths - `worker.py` background pipeline - `capture.py` mss/bettercam -
`engine_service.py` engine thread + mailbox - `yolo.py` optional YOLO backend -
`logsetup.py` logging - `supervision.py` back-off / restart policy.

## Optional YOLO vision backend

Default is `"vision_backend": "classical"` in `settings.json` - no extra packages, nothing imported.
To try a neural detector instead:

1. `pip install ultralytics` (pulls in torch; install a CUDA torch first if you want the GPU).
2. Put your weights somewhere and set, in `settings.json`:
   `"vision_backend": "yolo"`, `"yolo_piece_weights": "C:/path/pieces.pt"` and optionally
   `"yolo_corner_weights": "C:/path/corners.pt"`.
   Other knobs: `yolo_conf` (0.25), `yolo_device` ("" = auto, "cpu", "cuda:0"), `yolo_piece_imgsz`,
   `yolo_corner_imgsz`, `yolo_corner_every` (run the corner model every N frames).

What the weights must be:

* **Piece model** - an object detector run on the flattened (top-down, 640x640) board. 12 classes,
  named like `white-pawn` ... `black-king` (also `wp`/`bK`/`P`/`k`/`Black_Knight` are understood).
  Its detections become the same `(64, 13)` cost matrix the classical reader produces, so move
  tracking, legality checks and everything after it are unchanged. In manual mode "Learn pieces"
  is no longer needed.
* **Corner model** (optional) - a pose model with >= 4 keypoints (the outer corners of the 8x8 area) or a
  segmentation model of the board. The first detection may come in any order (it is sorted by position);
  after that the corners are matched to the previous fit, so the model's keypoint winding must stay
  consistent from frame to frame; used by Auto mode instead of the checker-lattice search.

Missing package, missing/unloadable weights or a failing inference never stops the app: the reason is
logged once and the classical path takes over. Limits: finding the *starting position* in Auto mode
(and its rotation) still uses the classical reader.

## Engine thread, logs, crash recovery

* The engine runs on its own thread. The pipeline never waits for it: a new position is drawn at once
  (board, tactics, no arrows) and the arrows appear when the engine answers. If the board changes
  meanwhile, only the newest position is analysed (older requests and answers are dropped).
* Logs: `<app dir>/logs/chess_overlay.log` (rotating, 1 MB x 5). `"log_level"` sets the file level,
  `"status_level"` which messages reach the menu's status line (`INFO`, `WARNING`, `ERROR`).
  Set `CHESS_OVERLAY_LOG=DEBUG` for a one-off verbose run.
* A failing frame is logged and shown as "Pipeline error: ..."; the loop retries with a growing pause.
  If the whole pipeline thread ever dies, it is restarted (at most 5 times per 2 minutes).

## Notes and limits

* 3D boards: recognition works best at moderate camera angles. Tall pieces overlapping the
  square behind them can confuse it; `crop_top_extend` and `fg_threshold` in
  `settings.json` can be tuned. The tracker only accepts legal moves, which hides most misreads.
* The overlay is excluded from capture (`WDA_EXCLUDEFROMCAPTURE`) so your own arrows
  never feed back into recognition. This also hides it from screenshots and recordings.
* Multi-monitor setups with mixed DPI scaling are best-effort.
* Using it against other players is against the rules of essentially every online game.
  No overlay is "guaranteed safe" from detection or from reports.
  I do not support any form of cheating, this was made for educational purposes.

## Self-test

`python selftest.py` renders a synthetic angled board and checks recognition, move tracking, tactics and the strength mapping offline (no game, no engine needed). It also covers the YOLO path with fake models, the engine thread, logging and the restart policy - no torch or weights required.
