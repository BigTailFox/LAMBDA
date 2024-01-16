"""ZCM type definitions
This file automatically generated by zcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class Scenario(object):
    __slots__ = ["flag", "index", "size", "values"]

    IS_LITTLE_ENDIAN = False;
    def __init__(self):
        self.flag = 0
        self.index = 0
        self.size = 0
        self.values = []

    def encode(self):
        buf = BytesIO()
        buf.write(Scenario._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">bqq", self.flag, self.index, self.size))
        buf.write(struct.pack('>%dd' % self.size, *self.values[:self.size]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != Scenario._get_packed_fingerprint():
            raise ValueError("Decode error")
        return Scenario._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = Scenario()
        self.flag, self.index, self.size = struct.unpack(">bqq", buf.read(17))
        self.values = struct.unpack('>%dd' % self.size, buf.read(self.size * 8))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if Scenario in parents: return 0
        tmphash = (0x87d3b81dbd61e7a6) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + ((tmphash>>63)&0x1)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if Scenario._packed_fingerprint is None:
            Scenario._packed_fingerprint = struct.pack(">Q", Scenario._get_hash_recursive([]))
        return Scenario._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

