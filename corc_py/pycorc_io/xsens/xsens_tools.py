import numpy as np

"""
Base Parse Function
"""
def convert_4_bytes(message:bytes,base_offsets,add_offsets,dtype):
    message_array = np.frombuffer(message, dtype=np.uint8)
    converted_msg = message_array[np.add.outer(base_offsets+add_offsets, np.arange(4))].reshape(-1, 4) # extract the required bytes
    converted_msg = np.frombuffer(converted_msg.tobytes(), dtype=dtype)
    return converted_msg

"""
Parse Functions
"""
def parse_string(message:bytes):
    return message.decode('utf-8')
def parse_COM_kinematics(message:bytes):
    COM_kinematics = np.frombuffer(message, dtype='>f4').tolist()    
    return COM_kinematics
def parse_UL_joint_angle(message: bytes):
    # Compute starting offsets for each joint
    JOINT_RIGHT_T4_SHOULDER = 6
    JOINT_RIGHT_SHOULDER = 7
    JOINT_RIGHT_ELBOW = 8
    JOINT_RIGHT_WRIST = 9
    JOINT_LEFT_T4_SHOULDER = 10
    JOINT_LEFT_SHOULDER = 11
    JOINT_LEFT_ELBOW = 12
    JOINT_LEFT_WRIST = 13
    JOINT_TRUNK_A = 25
    JOINT_TRUNK_B = 27

    joints_list = [JOINT_RIGHT_T4_SHOULDER,JOINT_RIGHT_SHOULDER,JOINT_RIGHT_ELBOW,JOINT_RIGHT_WRIST,
                    JOINT_LEFT_T4_SHOULDER,JOINT_LEFT_SHOULDER,JOINT_LEFT_ELBOW,JOINT_LEFT_WRIST,
                    JOINT_TRUNK_A, JOINT_TRUNK_B] # get the order from Analyze Pro
    packet_size = 20
    base_offsets = np.array(joints_list) * packet_size

    x_values = convert_4_bytes(message=message,base_offsets=base_offsets,add_offsets=8,dtype='>f4')
    y_values = convert_4_bytes(message=message,base_offsets=base_offsets,add_offsets=12,dtype='>f4')
    z_values = convert_4_bytes(message=message,base_offsets=base_offsets,add_offsets=16,dtype='>f4')

    # Convert to lists for compatibility
    x_arr = x_values.tolist()
    y_arr = y_values.tolist()
    z_arr = z_values.tolist()

    # extract the joint angles
    JOINT_RIGHT_T4_SHOULDER = 0
    JOINT_RIGHT_SHOULDER = 1
    JOINT_RIGHT_ELBOW = 2
    JOINT_RIGHT_WRIST = 3
    JOINT_LEFT_T4_SHOULDER = 4
    JOINT_LEFT_SHOULDER = 5
    JOINT_LEFT_ELBOW = 6
    JOINT_LEFT_WRIST = 7
    JOINT_TRUNK_A = 8
    JOINT_TRUNK_B = 9

    right_dict = {
        'trunk_ie':y_arr[JOINT_TRUNK_A],
        'trunk_aa':x_arr[JOINT_TRUNK_B],
        'trunk_fe':z_arr[JOINT_TRUNK_B],
        'clav_dep_ev':z_arr[JOINT_RIGHT_T4_SHOULDER],
        'clav_prot_ret':y_arr[JOINT_RIGHT_T4_SHOULDER],
        'shoulder_fe':z_arr[JOINT_RIGHT_SHOULDER],
        'shoulder_aa':x_arr[JOINT_RIGHT_SHOULDER],
        'shoulder_ie':y_arr[JOINT_RIGHT_SHOULDER],
        'elbow_fe':z_arr[JOINT_RIGHT_ELBOW],
        'elbow_ps':y_arr[JOINT_RIGHT_ELBOW],
        'wrist_fe':z_arr[JOINT_RIGHT_WRIST],
        'wrist_dev':x_arr[JOINT_RIGHT_WRIST]
    }

    left_dict = {
        'trunk_ie':y_arr[JOINT_TRUNK_A],
        'trunk_aa':x_arr[JOINT_TRUNK_B],
        'trunk_fe':z_arr[JOINT_TRUNK_B],
        'clav_dep_ev':z_arr[JOINT_LEFT_T4_SHOULDER],
        'clav_prot_ret':y_arr[JOINT_LEFT_T4_SHOULDER],
        'shoulder_fe':z_arr[JOINT_LEFT_SHOULDER],
        'shoulder_aa':x_arr[JOINT_LEFT_SHOULDER],
        'shoulder_ie':y_arr[JOINT_LEFT_SHOULDER],
        'elbow_fe':z_arr[JOINT_LEFT_ELBOW],
        'elbow_ps':y_arr[JOINT_LEFT_ELBOW],
        'wrist_fe':z_arr[JOINT_LEFT_WRIST],
        'wrist_dev':x_arr[JOINT_LEFT_WRIST]
    }
    return {"dict":right_dict,"list":list(right_dict.values())},{"dict":left_dict,"list":list(left_dict.values())}
