# coding=utf8

import os
import time
import json
import zerocm as zcm
from task_center.AMessage import AMessage
import sim_signal


FREQ = 0.1


class Task:
    def __init__(self, url=None) -> None:
        if url:
            self.url = url
        else:
            with open(os.getenv('REMOTE_TASK_CLIENT_CONFIG'), "r") as f:
                dom = json.load(f)
                self.url = dom["zcm_url_remote"]

    def listen_task_req(self):
        task_req = AMessage("PullTask", self.url, sim_signal.PullTask)
        task_req.launch()
        while True:
            tp = time.time()
            if task_req.is_recieved():
                task_req.close()
                break
            time.sleep(FREQ - time.time() + tp)
        worker_id = task_req.snapshot().worker_id
        return worker_id

    def publish_task(self, task, worker_id):
        task_push = AMessage("PushTask", self.url, sim_signal.PushTask)
        task_push.msg = sim_signal.PushTask()
        task_push.msg.worker_id = worker_id
        task_push.msg.task = task
        push_resp = AMessage("PushTaskResp", self.url, sim_signal.PushTaskResp)
        push_resp.launch()
        while True:
            tp = time.time()
            task_push.msg.timestamp = time.time_ns() // 1000
            task_push.publish()
            if push_resp.is_recieved() and push_resp.snapshot().worker_id == worker_id:
                push_resp.close()
                break
            time.sleep(FREQ - time.time() + tp)

    def wait_result(self, worker_id):
        result_listen = AMessage("PushResult", self.url, sim_signal.PushResult)
        result_listen.launch()
        while True:
            tp = time.time()
            if (
                result_listen.is_recieved()
                and result_listen.snapshot().worker_id == worker_id
            ):
                result_listen.close()
                break
            time.sleep(FREQ - time.time() + tp)
        resp = AMessage("PushResultResp", self.url, sim_signal.PushResultResp)
        resp.msg = sim_signal.PushResultResp()
        resp.msg.worker_id = worker_id
        while True:
            tp = time.time()
            resp.msg.timestamp = time.time_ns() // 1000
            resp.publish()
            if result_listen.no_message():
                break
            time.sleep(FREQ - time.time() + tp)
        return result_listen.snapshot().result

    def execute(self, scenario, fi, index):
        print("[TaskCenter] task (%d) waits a worker..." % index)
        worker_id = self.listen_task_req()
        task = sim_signal.Task()
        task.fi = fi
        task.scenario = scenario
        task.index = index
        print("[TaskCenter] assign task (%d) to worker (%d)..." % (index, worker_id))
        self.publish_task(task, worker_id)
        print("[TaskCenter] task (%d) is running..." % index)
        ret = self.wait_result(worker_id)
        print("[TaskCenter] task (%d) gets result" % index)
        return ret
