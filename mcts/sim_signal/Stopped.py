"""ZCM type definitions
This file automatically generated by zcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class Stopped(object):
    __slots__ = ["timestamp", "sim_time"]

    IS_LITTLE_ENDIAN = False;
    def __init__(self):
        self.timestamp = 0
        self.sim_time = 0.0

    def encode(self):
        buf = BytesIO()
        buf.write(Stopped._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">qd", self.timestamp, self.sim_time))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != Stopped._get_packed_fingerprint():
            raise ValueError("Decode error")
        return Stopped._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = Stopped()
        self.timestamp, self.sim_time = struct.unpack(">qd", buf.read(16))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if Stopped in parents: return 0
        tmphash = (0x2713164a63e29d58) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + ((tmphash>>63)&0x1)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if Stopped._packed_fingerprint is None:
            Stopped._packed_fingerprint = struct.pack(">Q", Stopped._get_hash_recursive([]))
        return Stopped._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

