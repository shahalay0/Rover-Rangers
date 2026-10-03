"""Telemetry packet format, CRC, randomizer and framing (CCSDS-style, simplified).

Frame on the air:  [ ASM (32 bits) | randomized( payload (13 B) + CRC-16 (2 B) ) ]
"""
import struct
import numpy as np

ASM = 0x1ACFFC1D                     # CCSDS attached sync marker
PAYLOAD_FMT = ">HIhHhB"              # seq, time_ms, temp_cC, bus_mV, current_mA, status
PAYLOAD_LEN = struct.calcsize(PAYLOAD_FMT)          # 13 bytes
BODY_BITS = (PAYLOAD_LEN + 2) * 8                    # payload + CRC
FRAME_BITS = 32 + BODY_BITS
FIELDS = ["seq", "time_ms", "temp_C", "bus_V", "current_A", "status"]


def int_to_bits(value, n):
    return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


ASM_BITS = int_to_bits(ASM, 32)


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _pn_sequence(nbits):
    """LFSR randomizer (x^8+x^7+x^5+x^3+1, seed all ones). Breaks up long runs
    of identical bits so timing/carrier loops stay locked."""
    state = [1] * 8
    out = np.empty(nbits, dtype=np.uint8)
    for i in range(nbits):
        out[i] = state[0]
        fb = state[0] ^ state[3] ^ state[5] ^ state[7]
        state = state[1:] + [fb]
    return out


PN = _pn_sequence(BODY_BITS)


def make_packet(seq, rng):
    """Synthetic spacecraft housekeeping telemetry."""
    t_ms = seq * 1000
    temp = 21.0 + 4.0 * np.sin(seq / 25) + rng.normal(0, 0.1)
    bus_v = 28.0 - 0.002 * seq + rng.normal(0, 0.02)
    current = 1.8 + 0.3 * np.sin(seq / 10) + rng.normal(0, 0.02)
    status = 0b0000_0001 | ((seq % 50 == 0) << 3)        # bit3 = heartbeat flag
    return struct.pack(PAYLOAD_FMT, seq & 0xFFFF, t_ms, int(temp * 100),
                       int(bus_v * 1000), int(current * 1000), status)


def build_frame(payload: bytes) -> np.ndarray:
    body = payload + struct.pack(">H", crc16_ccitt(payload))
    body_bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8)) ^ PN
    return np.concatenate([ASM_BITS, body_bits])


def decode_body(body_bits: np.ndarray):
    """Returns (dict, crc_ok)."""
    raw = np.packbits((body_bits.astype(np.uint8) ^ PN)).tobytes()
    payload, crc = raw[:PAYLOAD_LEN], struct.unpack(">H", raw[PAYLOAD_LEN:])[0]
    ok = crc16_ccitt(payload) == crc
    seq, t_ms, temp, bus, cur, status = struct.unpack(PAYLOAD_FMT, payload)
    return dict(seq=seq, time_ms=t_ms, temp_C=temp / 100, bus_V=bus / 1000,
                current_A=cur / 1000, status=status), ok
