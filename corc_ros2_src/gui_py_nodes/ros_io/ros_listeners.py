# -*- coding: utf-8 -*-
from PySide6.QtCore import Signal, QObject,QElapsedTimer,Slot,QTimer,Qt
import rclpy
from rclpy.node import Node

class ros_listener(QObject):
    """ROS2 node listens and emits Qt signals."""
    data_ready = Signal(dict)
    stopped = Signal()
    def __init__(self,topic,msg_type):
        super().__init__()
        
        # Initialise ROS2 FIRST, before creating the node
        if not rclpy.ok():
            rclpy.init()
        self.topic = topic
        self.node = Node(f'{topic}_listener')
        self.subscription = self.node.create_subscription(
            msg_type,
            topic,
            self.listen_emit,
            10
        )
        self.executor = None
        self.is_shutdown = False
        self.node.get_logger().info(f'{topic}_listener initialised')
        
        # FPS Calculator
        self.listener_timer = QElapsedTimer() # use the system clock
        self.listener_timer.start()
        self.listener_frame_count = 0
        self.listener_fps = 0
        self.listener_cur_time = self.listener_timer.elapsed()
        self.listener_last_time = self.listener_timer.elapsed()

    def listen_emit(self, msg):
        """Callback for incoming messages."""
        # update FPS
        self.listener_frame_count += 1
        self.listener_cur_time = self.listener_timer.elapsed()
        if self.listener_cur_time-self.listener_last_time >= 500:
            self.listener_fps = self.listener_frame_count * 1000 / (self.listener_cur_time-self.listener_last_time)
            self.listener_last_time = self.listener_cur_time
            self.listener_frame_count = 0
            
        self.data_ready.emit(
            {"msg":msg,
             "topic":self.topic,
             "listener_fps":self.listener_fps})
    """
    Initialization Callback
    """
    def start_worker(self):
        """Execute ROS2 node"""
        if self.is_shutdown:
            return
        if self.executor is None:
            from rclpy.executors import SingleThreadedExecutor
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            
        self.poll_timer = QTimer()
        self.poll_timer.setTimerType(Qt.PreciseTimer)
        self.poll_timer.timeout.connect(self._spin_once)
        self.poll_timer.start(0) 
    def _spin_once(self):
        if self.is_shutdown:
            return
        try: 
            if self.executor is not None:
                self.executor.spin_once(timeout_sec=0.0)
        except:
            # sometimes the timer cannot stop in time, cheat way to stop the warnings
            pass
                
    """
    External Signals Callback
    """
    @Slot()
    def stop(self):
        self.poll_timer.stop()
        """Shutdown ROS2."""
        if self.is_shutdown:
            return
            
        self.is_shutdown = True
        
        try:
            # Stop the executor first
            if self.executor is not None:
                self.executor.shutdown()
                self.executor = None
        except Exception as e:
            print(f'Error shutting down executor: {e}')

        try:
            # Destroy the node
            if self.node is not None:
                self.node.destroy_node()
                self.node = None
        except Exception as e:
            print(f'Error destroying node: {e}')

        try:
            # Finally shutdown ROS2
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f'Error shutting down ROS2: {e}')
            
        
        self.stopped.emit()