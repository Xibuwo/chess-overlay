"""Offline self-test: renders a synthetic angled board and checks the whole
vision + tracking + tactics chain. Run: python selftest.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import cv2
import numpy as np

import strength
from analysis import analyse_tactics, static_exchange
from config import (KIND_LC0, KIND_MAIA3, KIND_MAIA_LC0, KIND_STOCKFISH, SF_DEPTH, SF_SKILL)
from geometry import BoardGeometry
from recognition import PieceClassifier, board_labels
from tracker import BoardTracker

LIGHT, DARK = (181, 217, 240), (99, 136, 181)  # BGR


def render_board(board, flipped=False, size=640, highlight=None):
    img = np.zeros((size, size, 3), np.uint8)
    cell = size // 8
    for r in range(8):
        for c in range(8):
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = LIGHT if (r + c) % 2 == 0 else DARK
    if highlight is not None:  # yellow move highlight
        for sq in highlight:
            r, c = BoardGeometry.cell_of(sq, flipped)
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = (90, 230, 230)
    for sq, p in board.piece_map().items():
        r, c = BoardGeometry.cell_of(sq, flipped)
        org = (c * cell + cell // 2 - 15, r * cell + cell // 2 + 15)
        letter = p.symbol().upper()
        fg, bg = ((255, 255, 255), (0, 0, 0)) if p.color == chess.WHITE else ((0, 0, 0), (255, 255, 255))
        cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, bg, 7)
        cv2.putText(img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, fg, 3)
    return img


DST = np.float32([[300, 120], [980, 150], [1050, 650], [230, 620]])  # angled board on a 1280x800 "screen"


def to_screen(img):
    src = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])
    H = cv2.getPerspectiveTransform(src, DST)
    return cv2.warpPerspective(img, H, (1280, 800), borderValue=(60, 60, 60))


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        check.failed += 1


check.failed = 0


def vision_run(flipped):
    geom = BoardGeometry(DST)
    clf = PieceClassifier()
    start = chess.Board()
    ok, msg = clf.learn(geom.warp(to_screen(render_board(start, flipped))), geom.cell, flipped)
    check(ok, f"learn pieces (flipped={flipped}): {msg}")
    D = clf.cost_matrix(geom.warp(to_screen(render_board(start, flipped))), geom.cell, flipped)
    check(np.array_equal(D.argmin(axis=1), board_labels(start)), "free classification of start position")

    tr = BoardTracker()
    b = chess.Board()
    seq = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d3", "Nf6", "Nxe5", "Nxe5", "d4", "Bd6"]

    def feed(board, hl=None, n=3):
        D = clf.cost_matrix(geom.warp(to_screen(render_board(board, flipped, highlight=hl))), geom.cell, flipped)
        for _ in range(n):
            tr.update(D, True)

    feed(b)
    check(tr.board is not None and tr.board.board_fen() == chess.STARTING_BOARD_FEN, "sync on start position")
    all_ok = True
    for i, san in enumerate(seq):
        mv = b.parse_san(san)
        b.push(mv)
        feed(b, hl=[mv.from_square, mv.to_square])
        if tr.board.board_fen() != b.board_fen():
            all_ok = False
            print("   mismatch after", san)
    check(all_ok, f"tracked {len(seq)} plies incl. capture + castling + highlights (flipped={flipped})")
    check(tr.board.fen() == b.fen(), "tracker FEN (turn/castling) equals real game")

    # missed ply: observe two moves at once
    for san in ("Qe2", "Qe7"):
        b.push_san(san)
    feed(b)
    check(tr.board.board_fen() == b.board_fen(), "recovers when one frame skipped two plies")

    # new game detected
    feed(chess.Board())
    check(tr.board.board_fen() == chess.STARTING_BOARD_FEN and not tr.board.move_stack, "detects a new game")

    # mid-game cold sync
    tr2 = BoardTracker()
    mid = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    Dm = clf.cost_matrix(geom.warp(to_screen(render_board(mid, flipped))), geom.cell, flipped)
    for _ in range(4):
        tr2.update(Dm, True)
    check(tr2.board is not None and tr2.board.board_fen() == mid.board_fen(), "cold sync mid-game")


def auto_run(flipped):
    """Hands-free mode: no corner clicks, no templates. Board found by its checker lattice, colours
    learned from the starting position, game followed by legal moves + the yellow last-move tint."""
    from autogame import AutoGame
    g = AutoGame()
    b = chess.Board()
    frame = to_screen(render_board(b, flipped))
    for _ in range(4):
        st = g.step(frame)
    check(st.board is not None and st.board.board_fen() == chess.STARTING_BOARD_FEN,
          f"auto: found the board and the starting position (flipped={flipped})")
    seq = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d3", "Nf6", "Nxe5", "Nxe5", "d4", "Bd6"]
    ok = True
    for san in seq:
        mv = b.parse_san(san)
        b.push(mv)
        frame = to_screen(render_board(b, flipped, highlight=[mv.from_square, mv.to_square]))
        for _ in range(3):
            st = g.step(frame)
        if st.board is None or st.board.board_fen() != b.board_fen():
            ok = False
            print("   auto mismatch after", san)
            break
    check(ok, f"auto: followed {len(seq)} plies from the pixels alone (flipped={flipped})")
    # the camera moves (first-person games): same position, board shifted on screen
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 18], [0, 1, -12]]), (1280, 800), borderValue=(60, 60, 60))
    moved = False
    for _ in range(3):
        st = g.step(shifted)
        moved = moved or st.moved
    check(st.board is not None and st.board.board_fen() == b.board_fen() and moved,
          "auto: follows the board when it moves on screen")


def resync_run(flipped):
    """Mid-game recovery: a few moves go by unseen (hand over the board, dropped frames). The game must be
    picked up again from the pixels + legal moves, without a starting position."""
    from autogame import AutoGame
    g = AutoGame()
    b = chess.Board()
    for _ in range(4):
        g.step(to_screen(render_board(b, flipped)))
    seq = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d3", "Nf6", "Nxe5", "Nxe5", "d4", "Bd6"]
    for san in seq[:10]:
        mv = b.parse_san(san)
        b.push(mv)
        fr = to_screen(render_board(b, flipped, highlight=[mv.from_square, mv.to_square]))
        for _ in range(3):
            g.step(fr)
    for san in seq[10:13]:                      # three plies nobody saw
        mv = b.parse_san(san)
        b.push(mv)
    fr = to_screen(render_board(b, flipped, highlight=[mv.from_square, mv.to_square]))
    for _ in range(40):
        st = g.step(fr)
    check(st.board is not None and st.board.board_fen() == b.board_fen(),
          f"auto: re-syncs after 3 unseen plies, no starting position needed (flipped={flipped})")


def tactics_run():
    t = analyse_tactics(chess.Board("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1"))
    w = {chess.square_name(x.square): x for x in t.weak}
    check("d5" in w and w["d5"].kind == "hung" and w["d5"].net == 3, "undefended knight attacked by pawn = hung (-3)")
    t = analyse_tactics(chess.Board("4k3/8/2p5/3n4/4P3/8/8/4K3 w - - 0 1"))
    w = {chess.square_name(x.square): x for x in t.weak}
    check("d5" in w and w["d5"].kind == "vulnerable" and w["d5"].net == 2, "defended knight attacked by pawn = vulnerable (-2)")
    c = [x for x in t.contested if chess.square_name(x.square) == "d5"][0]
    check((c.attackers, c.defenders, c.diff) == (1, 1, 0), "attacker/defender counts on d5")
    b = chess.Board("4k3/8/8/3p4/8/8/8/3QK3 w - - 0 1")  # queen takes defended? pawn undefended
    check(static_exchange(b, chess.D5, chess.WHITE) == 1, "SEE: queen wins undefended pawn")
    b = chess.Board("4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1")
    check(static_exchange(b, chess.D5, chess.WHITE) == 0, "SEE: queen does not take defended pawn")


def strength_run():
    check(strength.skill_from_elo(800) == 0 and strength.skill_from_elo(3200) == 20, "skill mapping ends")
    check(strength.depth_from_elo(800) == 1 and strength.depth_from_elo(3200) == 18, "depth mapping ends")
    check(strength.nodes_from_elo(800) == 1, "node mapping start")
    check(strength.elo_slider_range(KIND_MAIA_LC0) == (1100, 1900), "maia-lc0 slider range")
    check("nodes=1" in strength.describe(KIND_MAIA3, SF_SKILL, 1500), "Maia 3 label says nodes=1 locked")
    check("Skill Level" in strength.describe(KIND_STOCKFISH, SF_SKILL, 1500), "Stockfish skill label")
    check("depth" in strength.describe(KIND_STOCKFISH, SF_DEPTH, 1500), "Stockfish depth label")


# ---------------------------------------------------------------------------------------------
# request 5: YOLO backend (fake models), async engine, logging, supervision
# ---------------------------------------------------------------------------------------------
class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = (np.asarray(xyxy, np.float32).reshape(-1, 4),
                                          np.asarray(conf, np.float32), np.asarray(cls, np.float32))

    def __len__(self):
        return len(self.conf)


class _FakeResult:
    def __init__(self, names, boxes=None, keypoints=None, masks=None):
        self.names, self.boxes, self.keypoints, self.masks = names, boxes, keypoints, masks


_NAMES = {i: n for i, n in enumerate(
    ["white-pawn", "white-knight", "white-bishop", "white-rook", "white-queen", "white-king",
     "black-pawn", "black-knight", "black-bishop", "black-rook", "black-queen", "black-king"])}


class _FakePieceModel:
    """Pretends to be an ultralytics detector: 'sees' whatever board the test put in `.board`."""

    def __init__(self, tall=False):
        self.board, self.flipped, self.tall, self.calls = chess.Board(), False, tall, 0

    def predict(self, img, **kw):
        self.calls += 1
        cell = img.shape[1] / 8.0
        xyxy, conf, cls = [], [], []
        for sq, p in self.board.piece_map().items():
            r, c = BoardGeometry.cell_of(sq, self.flipped)
            cx, bottom = (c + 0.5) * cell, (r + 0.92) * cell
            if self.tall:
                xyxy.append([cx - 0.3 * cell, bottom - 1.9 * cell, cx + 0.3 * cell, bottom])
            else:
                xyxy.append([cx - 0.35 * cell, (r + 0.5) * cell - 0.35 * cell,
                             cx + 0.35 * cell, (r + 0.5) * cell + 0.35 * cell])
            conf.append(0.9)
            cls.append(p.piece_type - 1 + (0 if p.color == chess.WHITE else 6))
        return [_FakeResult(_NAMES, _FakeBoxes(xyxy, conf, cls))]


class _FakeCornerModel:
    def __init__(self, quad=None):
        self.quad, self.calls = quad, 0

    def predict(self, img, **kw):
        self.calls += 1
        if self.quad is None:
            return [_FakeResult({0: "board"}, _FakeBoxes(np.zeros((0, 4)), [], []))]
        kp = type("KP", (), {"xy": np.asarray(self.quad, np.float32)[None]})()
        return [_FakeResult({0: "board"}, _FakeBoxes([[0, 0, 1, 1]], [0.9], [0]), keypoints=kp)]


def _yolo_on(piece=None, corner=None):
    import yolo
    cfg = yolo.YoloConfig(backend="yolo", piece_weights="fake_pieces.pt" if piece else "",
                          corner_weights="fake_corners.pt" if corner else "")
    yolo._models.clear()
    if piece:
        yolo._models["fake_pieces.pt"] = piece
    if corner:
        yolo._models["fake_corners.pt"] = corner
    yolo.configure(cfg)


def _yolo_off():
    import yolo
    yolo._models.clear()
    yolo.configure(yolo.YoloConfig())


def yolo_run():
    import yolo
    from occupancy import SquareReader, extract
    from config import Settings

    # --- names, defaults, lazy import --------------------------------------------------------
    names = {"white-pawn": 1, "Black_Knight": 8, "wp": 1, "bK": 12, "P": 1, "k": 12, "whiteking": 6,
             "w_queen": 5, "black bishop": 9, "board": None, "x": None}
    check(all(yolo.label_from_name(k) == v for k, v in names.items()), "yolo: class names -> labels")
    check(Settings().vision_backend == "classical", "yolo: default backend is classical")
    check("ultralytics" not in sys.modules and "torch" not in sys.modules,
          "yolo: classical needs no torch / ultralytics (nothing imported)")

    # --- piece backend through PieceClassifier / SquareReader ------------------------------------
    geom = BoardGeometry(DST)
    start = chess.Board()
    clf = PieceClassifier()
    ok, _ = clf.learn(geom.warp(to_screen(render_board(start))), geom.cell, False)
    warped0 = geom.warp(to_screen(render_board(start)))
    f0 = extract(warped0, geom.cell, False)
    rd = SquareReader()
    rd.learn_start(f0)
    D_cl = clf.cost_matrix(warped0, geom.cell, False)
    R_cl = rd.cost_matrix(f0, warped=warped0)

    # weights configured but absent -> classical result, bit for bit
    yolo._models.clear()
    yolo.configure(yolo.YoloConfig(backend="yolo", piece_weights="does_not_exist.pt",
                                   corner_weights="does_not_exist2.pt"))
    check(not yolo.piece_backend_active() and not yolo.corner_backend_active(), "yolo: missing weights = inactive")
    check(np.array_equal(clf.cost_matrix(warped0, geom.cell, False), D_cl), "yolo: no weights -> PieceClassifier identical")
    check(np.array_equal(rd.cost_matrix(f0, warped=warped0), R_cl), "yolo: no weights -> SquareReader identical")
    _yolo_off()
    check(np.array_equal(clf.cost_matrix(warped0, geom.cell, False), D_cl), "yolo: backend 'classical' unchanged")

    for flipped in (False, True):
        for tall in (False, True):
            fm = _FakePieceModel(tall=tall)
            fm.flipped = flipped
            _yolo_on(piece=fm)
            check(yolo.piece_backend_active(), "yolo: fake weights active")
            D = clf.cost_matrix(warped0, geom.cell, flipped)
            check(D.shape == (64, 13) and D.dtype == np.float32 and D.min() >= 0.0,
                  "yolo: PieceClassifier.cost_matrix keeps the (64, 13) contract")
            check(np.array_equal(D.argmin(axis=1), board_labels(start)),
                  f"yolo: argmin reads the start position (flipped={flipped}, tall={tall})")
            tr = BoardTracker()
            b = chess.Board()
            seq = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d3", "Nf6", "Nxe5", "Nxe5", "d4", "Bd6"]
            fm.board = b
            for _ in range(3):
                tr.update(clf.cost_matrix(warped0, geom.cell, flipped), True)
            good = tr.board is not None and tr.board.board_fen() == b.board_fen()
            for san in seq:
                b.push_san(san)
                fm.board = b.copy()
                for _ in range(3):
                    tr.update(clf.cost_matrix(warped0, geom.cell, flipped), True)
                if tr.board is None or tr.board.board_fen() != b.board_fen():
                    good = False
                    print("   yolo mismatch after", san)
                    break
            check(good, f"yolo: BoardTracker follows {len(seq)} plies from detections (flipped={flipped}, tall={tall})")

    fm = _FakePieceModel()
    _yolo_on(piece=fm)
    R = rd.cost_matrix(f0, warped=warped0)
    cls = np.where(np.array([start.piece_at(s) is None for s in range(64)]), 0,
                   np.where(np.array([(start.piece_at(s) or chess.Piece(1, True)).color for s in range(64)]), 1, 2))
    fm.board = start
    R = rd.cost_matrix(f0, warped=warped0)
    am = R.argmin(axis=1)
    got = np.where(am == 0, 0, np.where(am <= 6, 1, 2))
    check(R.shape == (64, 13) and np.array_equal(got, cls), "yolo: SquareReader.cost_matrix(warped=) -> (64, 13), right classes")
    check(rd.adapt(f0, start) == 0, "yolo: SquareReader does not adapt on frames YOLO read")

    # a failing model must not break the pipeline: classical fallback
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("cuda out of memory")
    _yolo_on(piece=Boom())
    check(np.array_equal(clf.cost_matrix(warped0, geom.cell, False), D_cl), "yolo: failing inference -> classical result")
    _yolo_off()

    # --- corner model through BoardLocator ----------------------------------------------------------
    from autodetect import BoardLocator
    frame = to_screen(render_board(start))
    quad = np.roll(DST, 2, axis=0)                      # model returns the corners in any order
    cm = _FakeCornerModel(quad)
    _yolo_on(corner=cm)
    loc = BoardLocator()
    fit = loc.update(frame)
    err = float(np.abs(fit.corners - DST).max()) if fit is not None else 1e9
    check(fit is not None and err < 1.0 and cm.calls == 1, f"yolo: corner model -> BoardFit at the right place (err {err:.2f}px)")
    for _ in range(6):
        fit = loc.update(frame)
    check(fit is not None and float(np.abs(fit.corners - DST).max()) < 3.0 and cm.calls <= 3,
          "yolo: corner model runs only every few frames, fit stays on the board")
    cm.quad = np.roll(DST, 1, axis=0)                   # model's winding changes later: fit keeps its order
    calls = cm.calls
    for _ in range(6):
        fit = loc.update(frame)
    check(cm.calls > calls and float(np.abs(fit.corners - DST).max()) < 3.0,
          "yolo: later detections are matched to the previous fit's corner order")
    _yolo_on(corner=_FakeCornerModel(None))
    loc = BoardLocator()
    fit = loc.update(frame)
    check(fit is not None and float(np.abs(fit.corners - DST).max()) < 4.0,
          "yolo: corner model finds nothing -> classical board search takes over")
    _yolo_off()


class _FakeEngineMgr:
    """EngineManager stand-in: analyse() blocks until the test opens the gate."""

    def __init__(self):
        import threading
        self.gate = threading.Event()
        self.started = threading.Event()
        self.seen, self.closed, self.killed, self.error = [], 0, 0, ""
        self.fail_next = False
        self.block = True

    def analyse(self, board, settings, my_turn):
        from engines import EngineResult
        self.seen.append(board.fen())
        self.started.set()
        if self.block:
            self.gate.wait(10)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("engine died")
        return EngineResult(best=next(iter(board.legal_moves)), eval_text="+0.1")

    def close(self):
        self.closed += 1

    def kill(self):
        self.killed += 1
        self.gate.set()


def engine_run():
    import threading
    import time
    from engine_service import EngineRequest, EngineService

    def req(i):
        b = chess.Board()
        b.push(list(b.legal_moves)[i])
        return EngineRequest((b.fen(), True, i), b, None, True)

    def waitfor(cond, t=3.0):
        end = time.time() + t
        while time.time() < end and not cond():
            time.sleep(0.01)
        return cond()

    # latest position wins; the stale in-flight one is dropped, not delivered
    got, mgr = [], _FakeEngineMgr()
    svc = EngineService(lambda r: got.append(r.key[2]), manager_factory=lambda: mgr)
    svc.start()
    svc.submit(req(0))
    check(mgr.started.wait(2), "engine: first request picked up")
    svc.submit(req(1))
    svc.submit(req(2))                                      # replaces req(1) in the mailbox
    mgr.gate.set()
    check(waitfor(lambda: got == [2]) and len(mgr.seen) == 2,
          f"engine: latest position wins, stale ones dropped (delivered {got}, analysed {len(mgr.seen)})")
    check(svc.dropped_stale >= 2, "engine: stale requests are counted")

    # cancel: a result already being computed is discarded
    mgr.gate.clear()
    mgr.started.clear()
    got.clear()
    svc.submit(req(3))
    mgr.started.wait(2)
    svc.cancel()
    mgr.gate.set()
    time.sleep(0.3)
    check(got == [], "engine: cancel() discards the in-flight answer")

    # an engine error is delivered, the thread lives on and the next request works (auto-restart pattern)
    mgr.block = False
    mgr.fail_next = True
    errs = []
    svc2_got = []
    svc._on_result = lambda r: (svc2_got.append(r), errs.append(r.error))
    svc.submit(req(4))
    check(waitfor(lambda: len(svc2_got) == 1) and "engine died" in errs[0], "engine: failure reported with message")
    check(mgr.closed >= 1 and svc.is_alive(), "engine: broken engine closed, service keeps running")
    svc.submit(req(5))
    check(waitfor(lambda: len(svc2_got) == 2) and svc2_got[1].error == "", "engine: next request served after an error")

    # release (pause) closes the engine process; stop() ends the thread promptly
    n = mgr.closed
    svc.release()
    check(waitfor(lambda: mgr.closed > n), "engine: release() closes the engine")
    t0 = time.time()
    svc.stop(timeout=2.0)
    check(not svc.is_alive() and time.time() - t0 < 2.5, "engine: stop() ends the thread cleanly")

    # stop() while analyse() hangs: the engine process is killed so the thread can end
    mgr2 = _FakeEngineMgr()
    svc3 = EngineService(lambda r: None, manager_factory=lambda: mgr2)
    svc3.start()
    svc3.submit(req(0))
    mgr2.started.wait(2)
    t0 = time.time()
    svc3.stop(timeout=0.4)
    check(not svc3.is_alive() and mgr2.killed == 1 and time.time() - t0 < 3.0, "engine: stop() kills a hung engine")


def logging_run():
    import logging
    import tempfile
    import threading
    import logsetup

    hooks = (sys.excepthook, threading.excepthook)
    try:
        with tempfile.TemporaryDirectory() as d:
            path = logsetup.setup_logging("DEBUG", log_dir=__import__("pathlib").Path(d))
            check(path is not None and path.parent.name == os.path.basename(d), "logging: log file created in the log dir")
            lg = logsetup.get_logger("selftest")
            check(lg.name == "chessoverlay.selftest", "logging: per-module logger names")
            lg.info("hello from the test")
            try:
                1 / 0
            except ZeroDivisionError:
                lg.error("caught", exc_info=True)
            shown = []
            h = logsetup.CallbackHandler(shown.append, logging.WARNING)
            st = logging.getLogger(logsetup.STATUS_LOGGER)
            st.addHandler(h)
            st.info("quiet info")
            st.warning("loud warning")
            h.setLevel(logging.INFO)
            st.info("now visible")
            st.removeHandler(h)
            for hd in logging.getLogger(logsetup.ROOT).handlers:
                hd.flush()
            text = path.read_text(encoding="utf-8")
            check("hello from the test" in text and "ZeroDivisionError" in text, "logging: messages and tracebacks reach the file")
            check(shown == ["loud warning", "now visible"], f"logging: status level filter ({shown})")
            check("quiet info" in text, "logging: the file keeps what the status filter hides")
            check(logsetup.parse_level("warning") == logging.WARNING and logsetup.parse_level("nonsense") == logging.INFO,
                  "logging: level parsing")
            logsetup.reset_for_tests()
    finally:
        sys.excepthook, threading.excepthook = hooks


def supervision_run():
    from supervision import Backoff, RestartPolicy
    b = Backoff(0.5, 4.0)
    seq = [b.next() for _ in range(5)]
    check(seq == [0.5, 1.0, 2.0, 4.0, 4.0] and b.failures == 5, f"supervision: back-off grows and caps {seq}")
    b.reset()
    check(b.next() == 0.5 and b.failures == 1, "supervision: back-off resets")
    t = [0.0]
    p = RestartPolicy(max_restarts=3, window=60.0, clock=lambda: t[0])
    check(p.allow()[0], "supervision: first restart allowed")
    p.record_restart()
    check(p.allow() == (False, "waiting"), "supervision: waits between restarts")
    n = 1
    while n < 3:
        t[0] += 11
        check(p.allow()[0], "supervision: restart allowed after the pause")
        p.record_restart()
        n += 1
    t[0] += 11
    ok, why = p.allow()
    check(not ok and "3 restarts" in why, f"supervision: crash loop is refused ({why})")
    t[0] += 100
    p.note_healthy()
    check(p.allow()[0], "supervision: forgives old crashes after a healthy spell")


if __name__ == "__main__":
    tactics_run()
    strength_run()
    vision_run(False)
    vision_run(True)
    auto_run(False)
    auto_run(True)
    resync_run(False)
    resync_run(True)
    yolo_run()
    engine_run()
    logging_run()
    supervision_run()
    print("\nALL PASSED" if not check.failed else f"\n{check.failed} FAILED")
    sys.exit(1 if check.failed else 0)
