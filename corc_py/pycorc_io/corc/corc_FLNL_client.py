import socket
import time
import sys,os
import pickle as pkl
import struct

import numpy as np
np.set_printoptions(
    precision=4,
    linewidth=np.inf,   
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

from PySide6.QtCore import QObject, QThread, Signal,QTimer,Slot,QElapsedTimer, Qt, QMetaObject
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget

def Checksum(b):
    ck = 0
    for i in range(2, len(b)-1):
        ck = ck ^ b[i]
    return ck

class corc_FLNL_client(QObject):
    data_ready = Signal(dict)
    stopped = Signal()
    def __init__(self, ip="127.0.0.1",port=2048):
        super().__init__()
        self.port = port
        self.ip = ip
        self.error = [0 for i in range(19)]
        
        # FPS Calculator
        self.corc_timer = QElapsedTimer() # use the system clock
        self.corc_timer.start()
        self.corc_frame_count = 0
        self.corc_fps = 0
        self.corc_cur_time = self.corc_timer.elapsed()
        self.corc_last_time = self.corc_timer.elapsed()

        self.newCmdRcv = False
        self.newValsRcv = False
        self.CmdRcv = ""
        self.Connected = False
        self.receiving=False

    """
    CORC Connection Functions
    """
    def reconnect(self):
        print("CORC Connecting")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(1)
        server_address = (self.ip, self.port)

        # Connection
        print('CORC FLNLClient: connecting to ('+self.ip+':'+str(self.port)+')...')
        try:
            self.socket.connect(server_address)
            self.sampling = 0
            print('CORC FLNLClient: Client connected!')
            self.connection = self.socket
            self.Connected = True
        except Exception as e:
            print('UBO FLNLClient: Connection failed! (', e, ')')
            self.Connected = False
            self.socket.close()
            self.sampling = 10
            
        return self.Connected

    """
    CORC FLNL Function and Callback
    """
    def SendCmd(self, cmd, vals=None):
        if(self.Connected):
            if vals is None:
                vals = []

            #Build packet header to send:
            tosend=bytearray(255)
            tosend[0]=ord('C')
            tosend[1]=len(vals)

            #Command
            cmd_bytes=cmd.encode()
            i=2
            for byte in cmd_bytes:
                tosend[i]=byte
                i=i+1
            for k in range(4-len(cmd_bytes)):
                tosend[i]=0
                i=i+1

            #Pack double values
            for val in vals:
                val_b=bytearray(struct.pack("d", val))
                for byte in val_b:
                    tosend[i]=byte
                    i=i+1

            tosend[255-1]=Checksum(tosend)

            #send
            try:
                self.connection.sendall(tosend)
            except (BrokenPipeError, ConnectionResetError):
                self.Connected=False
                self.receiving=False
                return False
            except Exception as e:
                print(e)    
                return False

            return True
    def ProcessRcvValues(self, data, nbvals):
        self.ValsRcv=[]
        for i in range(0, 8*nbvals, 8):
            try:
                self.ValsRcv.append(struct.unpack('d', bytearray(data[i:i+8]))[0])
            except Exception as e:
                #If for some reason one value is corrupted, give up all sequence
                print(f"data corrupt {e}")
                nbvals = 0
                self.ValsRcv=[]
                return
    def get_latest(self):
        try:
            data = self.connection.recv(255)
            if data:
                if data[0]==ord('C'):
                    #print('Cmd: '+str(data[2:5]))
                    self.newCmdRcv = True
                    self.CmdRcv = data[2:5].decode("utf-8")
                    nbvals = data[1]
                    self.ProcessRcvValues(data[6:6+8*nbvals], nbvals)
                if data[0]==ord('V'):
                    nbvals = data[1]
                    #print('Values ('+str(nbvals)+')')
                    self.ProcessRcvValues(data[2:2+8*nbvals], nbvals)
                    if nbvals>0:
                        self.newValsRcv = True
        except Exception as e:
            # print(f"Other Error: {e}")
            self.CmdRcv = "???"
            self.ValsRcv=[]

    def read_corc_data(self):
        try:
            # update FPS
            self.corc_frame_count += 1
            self.corc_cur_time = self.corc_timer.elapsed()
            if self.corc_cur_time-self.corc_last_time >= 500:
                self.corc_fps = self.corc_frame_count * 1000 / (self.corc_cur_time-self.corc_last_time)
                self.corc_last_time = self.corc_cur_time
                self.corc_frame_count = 0

            self.get_latest()
            if(self.newValsRcv and len(self.ValsRcv)>0):
                data = {
                    "raw_data":self.ValsRcv,
                    "corc_fps":self.corc_fps,
                }
            else:
                data = {
                    "raw_data":self.error,
                    "corc_fps":self.corc_fps,
                }
            self.data_ready.emit(data)
        except Exception as e:
            data = {
                "raw_data":self.error,
                "corc_fps":self.corc_fps,
            }
            self.data_ready.emit(data)
    """
    Initialization Callback
    """
    def start_worker(self):  
        self.reconnect()
        self.poll_timer = QTimer()
        self.poll_timer.setTimerType(Qt.PreciseTimer)
        self.poll_timer.timeout.connect(self.read_corc_data)
        self.poll_timer.start(self.sampling)
        self.SendCmd("REC")
    
    """
    External Signals Callback
    """
    @Slot()
    def stop(self):
        self.SendCmd("STP")
        if self.Connected:
            self.connection.close()
        self.poll_timer.stop()
        self.stopped.emit()
        
# ----------------------------
# Main application window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread Example")
        
        # Label to display data
        self.label = QLabel("Waiting for data...")
        self.button = QPushButton("Stop Thread")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Setup thread and worker
        self.thread = QThread()
        self.worker = corc_FLNL_client()
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.start_worker)
        self.worker.data_ready.connect(self.update_label)
        self.button.clicked.connect(self.cleanup)

        self.thread.start()

    def update_label(self, text):
        self.label.setText(f"info: {text}")

    def cleanup(self):
        QMetaObject.invokeMethod(self.worker, "stop", Qt.ConnectionType.QueuedConnection)
        self.worker.stopped.connect(self.thread.exit)
        self.label.setText("Thread stopped.")

        import time
        time.sleep(0.3) 
        self.close()

# ----------------------------
# Application entry point
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())