"""ZCM type definitions
This file automatically generated by zcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

from sim_signal.FI import FI as sim_signal_FI

from sim_signal.Scenario import Scenario as sim_signal_Scenario

class Task(object):
    __slots__ = ["index", "scenario", "fi"]

    IS_LITTLE_ENDIAN = False;
    def __init__(self):
        self.index = 0
        self.scenario = sim_signal_Scenario()
        self.fi = sim_signal_FI()

    def encode(self):
        buf = BytesIO()
        buf.write(Task._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">q", self.index))
        assert self.scenario._get_packed_fingerprint() == sim_signal_Scenario._get_packed_fingerprint()
        self.scenario._encode_one(buf)
        assert self.fi._get_packed_fingerprint() == sim_signal_FI._get_packed_fingerprint()
        self.fi._encode_one(buf)

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != Task._get_packed_fingerprint():
            raise ValueError("Decode error")
        return Task._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = Task()
        self.index = struct.unpack(">q", buf.read(8))[0]
        self.scenario = sim_signal_Scenario._decode_one(buf)
        self.fi = sim_signal_FI._decode_one(buf)
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if Task in parents: return 0
        newparents = parents + [Task]
        tmphash = (0x7cdef7466d1f2cbc+ sim_signal_Scenario._get_hash_recursive(newparents)+ sim_signal_FI._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + ((tmphash>>63)&0x1)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if Task._packed_fingerprint is None:
            Task._packed_fingerprint = struct.pack(">Q", Task._get_hash_recursive([]))
        return Task._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

