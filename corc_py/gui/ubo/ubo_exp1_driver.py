# driver.py
import subprocess, sys, os,json

SCRIPT = os.path.join(os.path.dirname(__file__), "ubo_gui.py")

var_num = 6
exp_id = "exp1"

subject_id = input("subject id?: sub<>: ")
patient_id = input("patient id?: p<>: ")

print(f"Starting {exp_id} - {patient_id} - {subject_id}")

for var in range(var_num):
    # var = 5
    print("========================================================\n")
    go_on = input(f"Start var_{var+1}?: y/n: ")
    while (go_on != "y" and go_on != "n"):
        go_on = input("Continue?: Reenter y/n: ")
    if go_on == "y":
        p = "Y"
        pass
    elif go_on == "n":
        p = "N"
        break
    print("========================================================\n")
    argv ={
           "init_flags":{"corc":{"on":True,
                                 "ip":"127.0.0.1",
                                 "port":2048},
                        "xsens":{"on":True,
                                 "ip":"0.0.0.0",
                                 "port":9764},
                         "gui":{"on":True,
                                "freq":30,
                                "3d":True,
                                "force":False},
                         "log":{"on":True}
                         },
            "session_data":{
                "exp_id": exp_id,
                "patient_id":patient_id,
                "subject_id":subject_id,
                "var_id":f"var_{var+1}",
                "take_num":0,}
           }

    argv = json.dumps(argv)

    print(f"=== var_{var+1} started ===")
    rc = subprocess.call(
        [sys.executable, SCRIPT,str(argv)],
        env=os.environ,
    )
    if rc != 0:
        print(f" var_{var+1} exited with {rc}, stopping.")
        break
    import time
    time.sleep(1)
    print(f"\n=== var_{var+1} ended ===\n")

    if p == "N":
        break
print("\nExperiment Ended")
print("\nPlease Ask How They Would Interact")
