# driver.py
import subprocess, sys, os,json

SCRIPT = os.path.join(os.path.dirname(__file__), "ubo_replay_gui.py")

var_num = 6
exp_id = "exp1"


for s in range(23,25):
    subject_id = f"sub{s}"

    patient_id = f"p3"
    
    print(f"Starting {exp_id} - {patient_id} - {subject_id}")
    
    print("========================================================\n")

    argv ={
           "init_flags":{"corc" :{"on":True},
                        "xsens" :{"on":True},
                        "gui"   :{
                                "on":True,
                                "freq":60,
                                "3d":True,
                                "force":False},
                        "replay":{"on":True}
                         },
            "session_data":{
                "exp_id":exp_id,
                "patient_id":patient_id,
                "subject_id":subject_id,
                "var_id":"var_1",
                "take_num":1,
            }
           }

    argv = json.dumps(argv)

    rc = subprocess.call(
        [sys.executable, SCRIPT,str(argv)],
        env=os.environ,
    )
    if rc != 0:
        print(f" {exp_id} - {patient_id} - {subject_id} exited with {rc}, stopping.")
        break
    import time
    time.sleep(1)
    print(f"\n=== {exp_id} - {patient_id} - {subject_id} ended ===\n")
