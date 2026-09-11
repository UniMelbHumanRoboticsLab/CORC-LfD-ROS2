"""Simple PySide6 application example with ROS2 Subscriber"""
import os, sys  
import signal

# import extra libraries within this ros2 package 
#(need to include __init__.py for each new folder)
print("Importing Listener")
sys.path.append(os.path.join(os.path.dirname(__file__)))
from std_msgs.msg import String
from ros_io.ros_listeners import ros_listener

# import extra libraries outside this ros2 package 
# (dont need to include __init__.py for each new folder)
print("Importing GUI")
from ament_index_python.packages import get_package_share_directory
package_share = get_package_share_directory('sm_py_nodes')
# get workspace root to grab the corc_py folder
if "CORC-LfD-ROS2" in package_share:
    workspace_root = package_share[:package_share.index("CORC-LfD-ROS2") + len("CORC-LfD-ROS2")]
    sys.path.append(workspace_root)
from corc_py.gui.base.pycorc_gui import pycorc_gui

print("Importing PySide6")
from PySide6 import QtWidgets
from PySide6.QtGui import QShortcut
from PySide6.QtCore import QThread,Qt,QMetaObject,Slot

class simple_viz(pycorc_gui):
    """
    Initilization
    """
    def __init__(self):
        super().__init__(freq=30,gui_3d=True)

        self.num_closed_threads = 0
        self.num_opened_threads = 0
        
        self.ros_threads={}
        self.ros_workers={}
        self.ros_labels={}
        self.ros_responses={}
        
        self.init_IOs()
        self.init_shortcuts()
        
        # Handle CTRL+C gracefully
        def signal_handler(sig, frame):
            self.close_workers()
        signal.signal(signal.SIGINT, signal_handler)
    """
    Init Sensor Worker / Thread / GUI Helper Functions
    """
    def add_opened_threads(self):
        self.num_opened_threads += 1
        print(f"Threads Opened:{self.num_opened_threads}")
    def init_shortcuts(self):
        close = QShortcut("Q", self)
        close.activated.connect(self.gui_timer.stop)
        close.activated.connect(self.close_workers)
    def init_ros_listener(self,topic,msg_type):
        # init response label
        ros_label = self.init_response_label(size=[525,200])
        self.ros_labels[topic] = ros_label
        # init corc thread and worker
        ros_thread = QThread()
        self.ros_threads[topic] = ros_thread
        ros_worker = ros_listener(topic,msg_type)
        self.ros_workers[topic] = ros_worker
    def init_IOs(self):
        """
        Init IO thread and worker
        """
        self.init_ros_listener("chatter",String)
        self.init_ros_listener("chatter2",String)
        
        """
        Move sensor worker to thread 
        Connect intersensor signals
        Connect thread start to start sensor worker, update number of open threads
        Connect worker stop to stop thread at closeup
        Connect thread finished to close app
        Connect signals to sensor GUI callbacks
        """
        for topic in self.ros_workers:
            # move worker to thread
            self.ros_workers[topic].moveToThread(self.ros_threads[topic])
            # connect thread start to start worker   
            self.ros_threads[topic].started.connect(self.add_opened_threads)
            self.ros_threads[topic].started.connect(self.ros_workers[topic].start_worker)
            # connect worker stop to stop thread at closeup
            self.ros_workers[topic].stopped.connect((self.ros_threads[topic]).exit)
            # connect thread finished to close app
            self.ros_threads[topic].finished.connect(self.close_app)
            # connect to gui
            self.ros_workers[topic].data_ready.connect(self.update_ros,type=Qt.ConnectionType.QueuedConnection)
        """
        Start threads
        """
        for topic in self.ros_workers:
            self.ros_threads[topic].start()
            self.ros_threads[topic].setPriority(QThread.Priority.TimeCriticalPriority)

    """
    Update ROS Listener Callbacks
    """
    @Slot(dict)
    def update_ros(self,ros_response):
        self.ros_responses[ros_response["topic"]] = ros_response
    def update_gui(self):
        super().update_gui()
        for topic in self.ros_workers:
            if topic in self.ros_responses: # check if key exists
                # update ros_worker labels
                ros_data = self.ros_responses[topic]["raw_data"]
                fps = self.ros_responses[topic]["listener_fps"]
                txt = ""
                self.update_response_label(self.ros_labels[topic],f"FPS:{fps}\nROS2 Listener Message: {ros_data}\n{txt}")
    """
    Cleanup
    """
    def close_worker_thread(self,worker):
        QMetaObject.invokeMethod(worker, "stop", Qt.ConnectionType.QueuedConnection)
    def close_workers(self):
        for topic in self.ros_workers:
            self.close_worker_thread(self.ros_workers[topic])
    @Slot()
    def close_app(self):
        self.num_closed_threads += 1
        if self.num_closed_threads == self.num_opened_threads:
            print("Shutting down app...")
            for plt in self.plt_items:
                plt.clear()
                plt.deleteLater()
            self.close()
            self.deleteLater()
            
def main():
    """Main entry point."""
    try:
        # Create Qt application
        app = QtWidgets.QApplication(sys.argv)

        # Create and show window
        window = simple_viz()
        window.show()

        app.exec()
        app.quit()
        import shiboken6
        shiboken6.delete(app)  # Immediate deletion
        sys.exit()
    except Exception as e:
        print(f'Fatal error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
