import json
import numpy as np

def get_subject_file(subject_path):
    json_path = f"{subject_path}/body_meas.json"
    with open(json_path, 'r') as file:
        body_file = json.load(file)
    return body_file
        
def get_subject_params(subject_path,ub_type="ubo"):
    subject = get_subject_file(subject_path)
    body_params = {'torso': subject["torso"]/1000,
                    'clav': subject["clav"]/1000,
                    'ua_l': subject["ua_l"]/1000,
                    'fa_l': subject["fa_l"]/1000,
                    'ha_l': subject["ha_l"]/1000,
                    'm_ua': 2.0,
                    'm_fa': 1.1+0.23+0.6,
                    "shoulder_aa_offset": np.array(subject["shoulder_aa_offset"]),
                    "ft_offsets": subject["ft_offsets"]}
    if ub_type == "ubo":
        gravity = {
          "clav": subject["cuff_grav"]["clav"]+subject["sensor_grav"],
          "ua":   subject["cuff_grav"]["ua"]+subject["sensor_grav"],
          "fa":   subject["cuff_grav"]["fa"]+subject["sensor_grav"]
        }
    elif ub_type=="emu":
        gravity ={
          "fa": subject["top_grav"]["fa"]+subject["sensor_grav"],
        }
    return body_params,gravity,subject["removed_bias"]