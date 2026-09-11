import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.node import Node

from pynput import keyboard
# from pynput.keyboard import Key, KeyCode
import sys
import yaml
import os

from corc_interfaces.msg import StateVar
from transitions import Machine,State

from ament_index_python.packages import get_package_share_directory
package_share = get_package_share_directory('sm_py_nodes')
# get workspace root
if "CORC-LfD-ROS2" in package_share:
    workspace_root = package_share[:package_share.index("CORC-LfD-ROS2") + len("CORC-LfD-ROS2")]
    
class base_node(Node):
    """
    Base ROS2 Node and State Machine Implementation
    - subscribes to EMU Task dynamics and EMU State ID
    - state controlled by CORC State Machine via 'state_id' subscriber
    - Recommend State Transition: use the state published by CORC and the corc_state_transition callback, Q is the only allowed keybord transition (to quit the program)
    """
    def __init__(self,node_name="base"):
        super().__init__(node_name)
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=10)  # Default depth; adjust as needed)
        
        self.states = [
            State(name="init",on_enter=self.in_init),
            State(name="standby",on_enter=self.in_standby),
            State(name="lock",on_enter=self.in_lock),
            State(name="quit",on_enter=self.in_quit)
            ]
        
        # Define FSM and transitions
        self.machine = Machine(
            model=self,
            states=self.states,
            initial="init",
            auto_transitions=False,
        )
        self.machine.add_transition("to_init", "*", "init")
        self.machine.add_transition("to_quit", "*", "quit")
    
        self.machine.add_transition("to_standby", "init", "standby")
        self.machine.add_transition("to_standby", "standby", "standby")
        self.machine.add_transition("to_lock", "standby", "lock")
        self.machine.add_transition("to_standby", "lock", "standby")
        
        # Load YAML config
        self.config = self.load_yaml(node_name)
        self.session_param = {
            "exp_id":self.config["exp_id"],
            "patient_id":self.config["patient_id"],
            "sub_id":self.config["sub_id"],
            }
        self.session_param["directory"] = f'{workspace_root}/logs/ros2_recordings/{self.session_param['exp_id']}/{self.session_param['patient_id']}/{self.session_param['sub_id']}/'

        # Init experiment state subscribers
        self.exp_state_sub = self.create_subscription(StateVar,'state_var',self.corc_state_transition,qos_profile=self.qos)
        self.exp_state_sub
        self.exp_state =self.states[0]
       
        # keyboard input thread to change states
        self.listener = keyboard.Listener(on_press=self.keyboard_listener)
        self.listener.start()
    """
    FSM Hooks 
    """
    def in_init(self):
        self.get_logger().info("In init")
    def in_quit(self):
        # perform base node quit
        self.get_logger().info("In quit")
        rclpy.try_shutdown()   # safe/no-op if already shutting down
    def in_standby(self):
        self.get_logger().info("In standby")
    def in_lock(self):
        self.get_logger().info("In lock")
    """
    Callbacks 
    """
    def keyboard_listener(self, event):
        """Handle key press events immediately when a key is pressed. Usage as state transitions not recommended"""
        if hasattr(event, 'char'):
            if event.char == 'Q':
                self.in_quit()
            else:
                pass
            sys.stdout.write('\033[2K\r')  # Clear entire line and move to start
    def corc_state_transition(self, msg): 
        if msg.state_id == "init":
            self.state_transition_log("init")
            self.to_init()
        elif msg.state_id == "standby":
            self.state_transition_log("standby")
            self.to_standby()
        elif msg.state_id == "lock":
            self.state_transition_log("lock")
            self.to_lock()
        elif msg.state_id == "quit":
            self.state_transition_log("quit")
            self.to_quit()
        else:
            pass
        
    """"
    Helper Functions
    """
    def state_transition_log(self,new_state):
        self.exp_state = new_state
        self.get_logger().info(f"Start transition {self.state} -> {self.exp_state}")
    def load_yaml(self,node_name):
        config_file = os.path.join(
            workspace_root, 'config', f'{node_name}_corc_node.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.get_logger().info(f"Config loaded: {config}")
            return config

def main(args=None):
    base = None
    try:
        rclpy.init(args=args)
        base = base_node()
        rclpy.spin(base)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # exit node safely
        if base is not None:
            base.destroy_node()
        try:
            rclpy.shutdown()   # unconditional + guarded
        except Exception:
            pass
if __name__ == '__main__':
    main()