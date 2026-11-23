import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

import wan12_queue as q

root=q.root
state_lock=threading.Lock()
state_file=root/"queue_status.json"

def update(name, state):
    with state_lock:
        states=q.read(state_file)
        states[name]=state
        q.save(state_file,states)

def run(job):
    name=job["job"]
    out=root/name
    out.mkdir(exist_ok=True)
    try:
        if not (out/"trained.json").is_file():
            update(name,"parallel_waiting_or_training_exclusive_gpu")
            q.train(job,out)
        weights=q.read(out/"trained.json")["weights"]["native"]
        assert q.sha(Path(weights["path"]))==weights["sha256"]
        dest=out/"native"
        status=dest/"evaluation_status.json"
        if not status.exists() or q.read(status)["status"]!="evaluated":
            update(name,"parallel_queued_for_generation")
            with (out/"generation_scheduler.log").open("a") as log:
                subprocess.run([sys.executable,str(root/"evaluation_scheduler.py"),"--job",name,"--checkpoint",weights["path"],"--out",str(dest),"--gpu-free","120000"],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert q.read(status)["status"]=="evaluated" and q.read(dest/"alignment/alignment.json")["passed"]
        update(name,"evaluated")
    except Exception as exc:
        update(name,"needs_attention: "+repr(exc))
        traceback.print_exc()

def main():
    with (root/"parallel.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=int((root/"pid").read_text())
        proc=Path("/proc")/str(old)
        handle=os.pidfd_open(old)
        assert str(root/"wan12_queue.py").encode() in (proc/"cmdline").read_bytes().split(b"\0")
        # Freeze only the original dispatcher while it waits for a detached generation scheduler.
        # No training watchdog or generation scheduler may be paused.
        signal.pidfd_send_signal(handle,signal.SIGSTOP)
        try:
            for _ in range(100):
                if (proc/"stat").read_text().split(") ")[1][0]=="T":break
                time.sleep(.02)
            assert (proc/"stat").read_text().split(") ")[1][0]=="T"
            children=(proc/"task"/str(old)/"children").read_text().split()
            assert len(children)==1,children
            args=(Path("/proc")/children[0]/"cmdline").read_bytes().decode().split("\0")
            assert str(root/"evaluation_scheduler.py") in args,args
            active=args[args.index("--job")+1]
            states=q.read(state_file)
            jobs=[j for j in q.read(root/"jobs.json") if states[j["job"]]=="queued_for_training"]
            assert active not in {j["job"] for j in jobs} and jobs
            q.save(root/"parallel_handoff.json",dict(active=active,dispatcher=old,jobs=[j["job"] for j in jobs],pid=os.getpid()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                list(pool.map(run,jobs))
        finally:
            signal.pidfd_send_signal(handle,signal.SIGCONT)
            os.close(handle)

if __name__=="__main__":
    main()

