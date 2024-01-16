# coding=utf8

import zerocm as zcm
import copy
import threading
import time

FREQ = 0.05


class AMessage:
    def __init__(self, channel, url, msg_type) -> None:
        self.channel = channel
        self.url = url
        self.msg_type = msg_type
        self.recieved = False
        self.msg = None
        self.sub = None
        self.tunnel = zcm.ZCM(url)
        self.mtx = threading.Lock()

    def __handler(self, channel, msg):
        self.mtx.acquire()
        self.recieved = True
        self.msg = msg
        self.mtx.release()

    def publish(self):
        self.tunnel.publish(self.channel, self.msg)

    def subscribe(self):
        self.sub = self.tunnel.subscribe(self.channel, self.msg_type, self.__handler)

    def unsubscribe(self):
        self.tunnel.unsubscribe(self.sub)

    def start(self):
        self.tunnel.start()

    def stop(self):
        self.tunnel.stop()

    def run(self):
        self.tunnel.run()

    def handle(self):
        self.tunnel.handle()

    def close(self):
        self.unsubscribe()
        self.stop()
        self.recieved = False

    def launch(self):
        self.subscribe()
        self.start()

    def snapshot(self):
        self.mtx.acquire()
        ret = copy.deepcopy(self.msg)
        self.mtx.release()
        return ret

    def is_recieved(self):
        self.mtx.acquire()
        ret = self.recieved
        self.mtx.release()
        return ret

    def no_message(self):
        ret = False
        self.mtx.acquire()
        if self.msg is None:
            ret = True
        now = time.time_ns() // 1000
        if (now - self.msg.timestamp) > FREQ * 1000000 * 1.2:
            ret = True
        self.mtx.release()
        return ret
