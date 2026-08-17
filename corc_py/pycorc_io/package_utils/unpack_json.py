import json
import numpy as np

def get_subject_file(subject_path):
    json_path = f"{subject_path}/body_meas.json"
    with open(json_path, 'r') as file:
        body_file = json.load(file)
    return body_file
        
def get_subject_params(subject_path):
    body_params = get_subject_file(subject_path)
    body_params_rbt = {'torso': body_params["torso"]/1000,
                    'clav': body_params["clav"]/1000,
                    'ua_l': body_params["ua_l"]/1000,
                    'fa_l': body_params["fa_l"]/1000,
                    'ha_l': body_params["ha_l"]/1000,
                    'm_ua': 2.0,
                    'm_fa': 1.1+0.23+0.6,
                    "shoulder_aa_offset": np.array(body_params["shoulder_aa_offset"]),
                    "ft_offsets": body_params["ft_offsets"]}
    
    gravity ={
      "clav": body_params["cuff_grav"]["clav"]+body_params["sensor_grav"],
      "ua":   body_params["cuff_grav"]["ua"]+body_params["sensor_grav"],
      "fa":   body_params["cuff_grav"]["fa"]+body_params["sensor_grav"]
    }
        
    return body_params_rbt,gravity,body_params["removed_bias"]