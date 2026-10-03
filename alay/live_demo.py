"""LIVE DEMO: type a message -> encode -> modulate -> through space -> find -> identify (CNN)
-> clean -> decode. Run:  python live_demo.py"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons, Slider, Button
from radio_tools import SPS, add_fog, add_spin, add_delay, conv_encode, viterbi_decode
from psk_tools import TABLES, ORDER, BITS, send_psk, receive_psk, soft_psk, hard_psk, find_burst

HELLO = np.array([int(b) for b in format(0x1ACFFC1D, "032b")])     # CCSDS sync marker
DECODABLE = ["BPSK", "QPSK", "8PSK"]
DEFAULT_SNR = {"BPSK": 4.5, "QPSK": 4.5, "8PSK": 9.0}               # comfortably above each cliff
MAX_CHARS = 60
NAVY, TEAL, AMBER, CORAL, GRAY = "#14213D", "#2A9D8F", "#F29E38", "#E76F51", "#6C757D"

try:
    from classifier_tools import Identifier
    print("Loading the trained CNN ...")
    IDENT = Identifier()
except Exception as err:                       # the demo still runs without the network
    IDENT = None
    print("CNN not available, using sync-word check only:", err)


# ======================= the radio link (all real computation) =======================
def through_space(bits, mod, esn0_db, rng):
    signal, n_sym = send_psk(bits, mod)
    recording = np.concatenate([np.zeros(int(rng.integers(1500, 4000))), signal, np.zeros(3000)])
    recording = add_delay(recording, rng.uniform(0, 1))
    spin = rng.choice([-1, 1]) * rng.uniform(0.0001, 0.0002)          # Doppler: visible rotation
    recording = add_spin(recording, spin, rng.uniform(0, 2 * np.pi))
    return add_fog(recording, esn0_db, rng), n_sym


def bits_to_chars(bits, n_chars):
    bits = np.concatenate([bits, np.zeros(8 * n_chars, dtype=int)])[:8 * n_chars]
    return "".join(chr(b) if 32 <= b < 127 else "·" for b in np.packbits(bits))


def run_link(text, mod, ebn0_db, rng):
    k = BITS[mod]
    msg = np.unpackbits(np.frombuffer(text.encode("ascii", "replace"), dtype=np.uint8)).astype(int)
    coded = conv_encode(msg)
    frame = np.concatenate([HELLO, coded])
    rec, n_sym = through_space(frame, mod, ebn0_db + 10 * math.log10(k / 2), rng)  # same energy per info bit
    first, last = find_burst(rec)

    # Identify with the CNN, then verify each guess with the sync word (like a real receiver)
    if IDENT is not None:
        avg, votes = IDENT(rec[first:last])
        names = list(IDENT.mods)
        order = sorted(DECODABLE, key=lambda m: -votes[names.index(m)])
        cnn_top = names[int(np.argmax(votes))]
    else:
        votes, names, order, cnn_top = None, None, DECODABLE, None
    attempts, infos = [], {}
    for cand in order:                          # the CNN proposes (vote order) ...
        n_cand = math.ceil(len(frame) / BITS[cand])
        infos[cand] = receive_psk(rec, HELLO, n_cand, cand)
        m, n = infos[cand]["sync_match"]
        attempts.append((cand, (m, n)))
    # ... and the sync word confirms: keep the best match (CNN order breaks ties)
    used = max(attempts, key=lambda a: (a[1][0] / a[1][1], -order.index(a[0])))[0]
    info = infos[used]
    soft = soft_psk(info["samples"], used)[len(HELLO):]
    soft = np.concatenate([soft, np.zeros(len(coded))])[:len(coded)]
    decoded = viterbi_decode(soft, len(msg))

    # Comparison: same message, same energy per bit, NO code
    frame_u = np.concatenate([HELLO, msg])
    rec_u, n_u = through_space(frame_u, used, ebn0_db + 10 * math.log10(BITS[used]), rng)
    info_u = receive_psk(rec_u, HELLO, n_u, used)
    hard = hard_psk(info_u["samples"], used)[len(HELLO):]
    hard = np.concatenate([hard, np.zeros(len(msg), dtype=int)])[:len(msg)]

    return dict(text=text, mod=mod, msg=msg, coded=coded, n_sym=n_sym, rec=rec, found=(first, last),
                votes=votes, names=names, cnn_top=cnn_top, attempts=attempts, used=used, info=info,
                coded_text=bits_to_chars(decoded, len(text)), coded_err=int(np.sum(decoded != msg)),
                plain_text=bits_to_chars(hard, len(text)), plain_err=int(np.sum(hard != msg)))


# ======================= the window =======================
fig = plt.figure(figsize=(16, 9))
fig.canvas.manager.set_window_title("Blind Deep-Space Receiver: live demo")
grid = fig.add_gridspec(2, 3, left=0.07, right=0.98, top=0.90, bottom=0.22, hspace=0.42, wspace=0.22)
ax_msg, ax_tx, ax_rec = [fig.add_subplot(grid[0, i]) for i in range(3)]
ax_id, ax_clean, ax_out = [fig.add_subplot(grid[1, i]) for i in range(3)]
status = fig.text(0.5, 0.955, "Type a message, pick a modulation, press TRANSMIT", ha="center",
                  fontsize=17, fontweight="bold", color=NAVY)

def blank(ax, title):
    ax.clear()
    ax.set_title(title, fontsize=13, fontweight="bold", color=NAVY, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#D5DCE4")

def reset_panels():
    for ax, t in [(ax_msg, "1. Message → bits → Voyager code"), (ax_tx, "2. Transmitter"),
                  (ax_rec, "3. Through space"), (ax_id, "4. Find + identify (CNN)"),
                  (ax_clean, "5. Clean: undo Doppler + phase"), (ax_out, "6. Received message")]:
        blank(ax, t)
reset_panels()

box = TextBox(fig.add_axes([0.10, 0.08, 0.42, 0.06]), "Message ", initial="Hello Earth! Greetings from deep space.")
radio = RadioButtons(fig.add_axes([0.555, 0.03, 0.08, 0.14]), DECODABLE, active=1)
slider = Slider(fig.add_axes([0.72, 0.10, 0.14, 0.035]), "SNR per bit (dB)", 0.0, 14.0, valinit=DEFAULT_SNR["QPSK"], valstep=0.5)
button = Button(fig.add_axes([0.885, 0.06, 0.09, 0.08]), "TRANSMIT", color=AMBER, hovercolor="#F7B866")
button.label.set_fontsize(15); button.label.set_fontweight("bold")
radio.on_clicked(lambda label: slider.set_val(DEFAULT_SNR[label]))
busy = [False]
rng = np.random.default_rng()


def pause(t):
    plt.pause(t)


def typed(ax, y, label, shown, truth, color_ok, n_show, width=34):
    ax.text(0.02, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=NAVY, va="top")
    for i, ch in enumerate(shown[:n_show]):
        row, col = divmod(i, width)
        ax.text(0.02 + col * 0.028, y - 0.09 - row * 0.085, ch, transform=ax.transAxes, va="top",
                fontsize=13, family="monospace", fontweight="bold",
                color=color_ok if ch == truth[i] else CORAL)


def errs(n):
    return f"{n} bit error{'' if n == 1 else 's'}"


def animate(r):
    mod, k, M = r["mod"], BITS[r["mod"]], ORDER[r["mod"]]

    # 1. message -> bits -> code
    blank(ax_msg, "1. Message → bits → Voyager code")
    ax_msg.text(0.02, 0.92, f"“{r['text']}”", transform=ax_msg.transAxes, fontsize=12, va="top", wrap=True, color=NAVY)
    bits = "".join(map(str, r["msg"][:64]))
    for n in range(8, 65, 8):
        line = " ".join(bits[i:i + 8] for i in range(0, n, 8))
        txt = ax_msg.text(0.02, 0.70, line[:36] + "\n" + line[36:], transform=ax_msg.transAxes, fontsize=11,
                          family="monospace", va="top", color=TEAL)
        pause(0.08)
        if n < 64:
            txt.remove()
    ax_msg.text(0.02, 0.38, f"{len(r['msg'])} message bits\n→ Voyager code → {len(r['coded'])} code bits\n"
                f"+ 32-bit sync marker → {r['n_sym']} {mod} symbols", transform=ax_msg.transAxes,
                fontsize=12, va="top", color=NAVY)
    pause(0.4)

    # 2. transmitter constellation
    blank(ax_tx, f"2. Transmitter: {mod} ({k} bit{'s' if k > 1 else ''} per symbol)")
    t = TABLES[mod]
    circ = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    ax_tx.plot(circ.real, circ.imag, color="#D5DCE4", lw=1)
    for lab, p in enumerate(t):
        ax_tx.plot(p.real, p.imag, "o", ms=16, color=AMBER)
        ax_tx.text(p.real * 1.32, p.imag * 1.32, format(lab, f"0{k}b"), ha="center", va="center", fontsize=11, color=NAVY)
        pause(0.05)
    ax_tx.set_xlim(-1.6, 1.6); ax_tx.set_ylim(-1.6, 1.6); ax_tx.set_aspect("equal")
    pause(0.3)

    # 3. through space: the recording builds up
    blank(ax_rec, "3. Through space: noise + Doppler + unknown delay")
    W = 64
    n_pieces = len(r["rec"]) // W
    loud = np.mean(np.abs(r["rec"][:n_pieces * W].reshape(n_pieces, W)) ** 2, axis=1)
    ax_rec.set_xlim(0, n_pieces * W); ax_rec.set_ylim(0, loud.max() * 1.15); ax_rec.set_xticks([])
    ax_rec.set_xlabel("time (samples)", fontsize=10)
    ln, = ax_rec.plot([], [], color=TEAL, lw=1.5)
    for f in range(1, 31):
        n = int(n_pieces * f / 30)
        ln.set_data(np.arange(n) * W, loud[:n])
        pause(0.03)

    # 4. find + identify
    a, b = r["found"]
    ax_rec.axvspan(a, b, color=AMBER, alpha=0.25)
    ax_rec.set_title(f"3. Burst found: samples {a:,}–{b:,}", fontsize=13, fontweight="bold", color=NAVY, loc="left")
    blank(ax_id, "4. Identify: CNN votes (128-sample windows)")
    if r["votes"] is not None:
        top = np.argsort(-r["votes"])[:5][::-1]
        labels = [r["names"][i] for i in top]
        vals = r["votes"][top]
        colors = [AMBER if lb == r["cnn_top"] else GRAY for lb in labels]
        for f in range(1, 13):
            ax_id.clear(); blank(ax_id, "4. Identify: CNN votes (128-sample windows)")
            ax_id.barh(labels, vals * f / 12, color=colors)
            ax_id.set_xlim(0, vals.max() * 1.25); ax_id.set_yticks(range(len(labels)), labels, fontsize=11)
            pause(0.04)
        for i, v in enumerate(vals):
            ax_id.text(v + vals.max() * 0.02, i, str(int(v)), va="center", fontsize=11, color=NAVY)
    checks = "   ".join(f"{m}: {s[0]}/{s[1]}{' ✓' if m == r['used'] else ''}" for m, s in r["attempts"])
    ax_id.set_xlabel(f"Sync-word check (bits matching) → {checks}", fontsize=10, color=NAVY)
    pause(0.5)

    # 5. clean: the rotating cloud snaps into clusters
    info = r["info"]
    raw = info["raw"]
    phase = np.unwrap(np.angle(info["all_corrected"] * np.conj(raw)))
    tcol = np.arange(len(raw))
    lim = np.percentile(np.abs(raw), 99) * 1.1
    for f in range(0, 26):
        ax_clean.clear()
        blank(ax_clean, "5. Clean: rotating cloud (colour = time)" if f == 0 else "5. Clean: undoing Doppler + phase …")
        pts = raw * np.exp(1j * phase * f / 25)
        ax_clean.scatter(pts.real, pts.imag, c=tcol, s=5, cmap="viridis")
        ax_clean.set_xlim(-lim, lim); ax_clean.set_ylim(-lim, lim); ax_clean.set_aspect("equal")
        pause(0.6 if f == 0 else 0.04)
    used = r["used"]
    ax_clean.plot(TABLES[used].real, TABLES[used].imag, "+", color=CORAL, ms=18, mew=3)
    ax_clean.set_title("5. Cleaned: clusters locked", fontsize=13, fontweight="bold", color=NAVY, loc="left")
    ax_clean.set_xlabel(f"Doppler {info['spin']:+.2e} cycles/sample · rotation {info['turn'] * 360 // ORDER[used]}°",
                        fontsize=10, color=NAVY)
    pause(0.4)

    # 6. decode: type out both versions
    n_chars = len(r["text"])
    for n in range(0, n_chars + 3, 3):
        blank(ax_out, "6. Received message (same energy per bit)")
        typed(ax_out, 0.95, f"Without coding: {errs(r['plain_err'])}", r["plain_text"], r["text"], NAVY, n)
        typed(ax_out, 0.48, f"With Voyager code: {errs(r['coded_err'])}", r["coded_text"], r["text"], TEAL, n)
        pause(0.03)


def transmit(_event=None):
    if busy[0]:
        return
    busy[0] = True
    try:
        text = (box.text.strip() or "Hello Earth!")[:MAX_CHARS]
        text = text.encode("ascii", "replace").decode("ascii")
        mod, snr = radio.value_selected, float(slider.val)
        reset_panels()
        status.set_text(f"Transmitting “{text[:30]}{'…' if len(text) > 30 else ''}” with {mod} at {snr:.1f} dB …")
        pause(0.05)
        r = run_link(text, mod, snr, rng)
        animate(r)
        verdict = "message recovered perfectly" if r["coded_err"] == 0 else f"{errs(r['coded_err'])} (below the coding cliff)"
        status.set_text(f"{mod} at {snr:.1f} dB: uncoded {errs(r['plain_err'])}  |  Voyager code: {verdict}")
        fig.canvas.draw_idle()
    finally:
        busy[0] = False


button.on_clicked(transmit)
box.on_submit(lambda _t: None)
plt.show()
