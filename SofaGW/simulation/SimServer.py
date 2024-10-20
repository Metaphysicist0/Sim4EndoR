import subprocess
import os
import sys
import socketserver
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import threading
import queue
import time
import pickle
import numpy as np
from typing import Optional


# <SofaGuidewireNav>/SofaGW/simulation/../../
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../../")
from SofaGW.utils import abspath, datasave, dataload, root_dir, delete_old_files, mkdir
global count
count = 0
class Server():
    def __init__(self, commu_dir, timeout:Optional[int]=None):
        self.port_rpc = None
        self.clientfile = os.path.dirname(os.path.abspath(__file__)) + '/SimClient.py'
        self.timeout = timeout
        self.commu_dir = commu_dir
    def start(self):
        print('start')
        global count
        if count >= 10:
            print(gg)
        count += 1
        self.setport()
        self.data = Server.Data(timeout=self.timeout)
        self.startthread()
    def setport(self):
        # Find a free port to connect a server.
        with socketserver.TCPServer(("localhost", 0), None) as s:
            free_port = s.server_address[1]
            print('free_port :',free_port)
        self.port_rpc = free_port

    class CustomQueue(queue.Queue):
        def __init__(self):
            queue.Queue.__init__(self)
            self.len = 0
        def put(self, item, timeout=None):
            queue.Queue.put(self, item, timeout=timeout)
            self.len += 1
        def get(self, timeout=None):
            res = queue.Queue.get(self, timeout=timeout)
            self.len -= 1
            return res
        def __len__(self):
            return self.len

    class Data():
        """Every method should have non-none return
        """
        def __init__(self, timeout=None):
            self.timeout = timeout
            self.serverdata = Server.CustomQueue() # server -> client
            self.clientdata = Server.CustomQueue() # client -> server
        def serverget(self):
            res = self.serverdata.get(timeout=self.timeout)
            assert len(self.serverdata) == 0
            return res
        def serverput(self,item):
            self.serverdata.put(item, timeout=self.timeout)
            assert len(self.serverdata) == 1
            return 0
        def clientget(self):
            res = self.clientdata.get(timeout=self.timeout)
            assert len(self.clientdata) == 0
            return res
        def clientput(self,item):
            self.clientdata.put(item, timeout=self.timeout)
            assert len(self.clientdata) == 1
            return 0
    def dataput(self, item):
        # Send some data to the client.
        return self.data.serverput(item)
    def dataget(self):
        # Get some data from the client.
        return self.data.clientget()
    
    class SimpleThreadedXMLRPCServer(SimpleXMLRPCServer):
        pass
    # Restrict to a particular path.
    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)
        def log_message(self, format, *args):
            pass
    def startthread(self):
        # Register functions
        def dispatch(port_rpc, dataqueue):
            with self.SimpleThreadedXMLRPCServer(('localhost', port_rpc), requestHandler=self.RequestHandler) as s:
                s.register_instance(dataqueue)
                s.serve_forever()
        # Starts the server thread with the context.
        server_thread = threading.Thread(target=dispatch, args=(self.port_rpc,self.data))
        server_thread.daemon = True
        server_thread.start()
    def runclient(self, vessel_filename):
        # Run the client
        def deferredStart(path, port_rpc, vessel_filename, commu_dir):
            vessel_filename = abspath(vessel_filename)
            print('123123')
            try:
                print(' '.join([sys.executable, path, str(port_rpc), vessel_filename, commu_dir]))
                res = subprocess.run([sys.executable, path, str(port_rpc), vessel_filename, commu_dir],
                                check=True, capture_output=True, text=True, stderr=subprocess.STDOUT)
                return
            except:
                print('err')
            print(gg)
            print(res)

       # max_connections = 1  # 定义最大线程数
       # pool_sema = threading.BoundedSemaphore(max_connections)
       # self.first_worker_thread = threading.Thread(target=deferredStart, args=(self.clientfile, self.port_rpc, vessel_filename, self.commu_dir))
       # self.first_worker_thread.daemon = True
       # self.first_worker_thread.start()
       # time.sleep(1)

        deferredStart(self.clientfile, self.port_rpc, vessel_filename, self.commu_dir)
    def waitclientclose(self):
        # exit command was aleady sent
        self.first_worker_thread.join() # Wait until the client be finished.
        

class SimController():
    def __init__(self, vessel_filename, timeout=None):
        self.vessel_filename = vessel_filename
        self.sim_opened = False
        self.commu_dir = root_dir + '/_cache_'
        self.timeout = timeout
        self.server = Server(commu_dir=self.commu_dir,timeout=self.timeout)
        self.server.start()
        self.open(vessel_filename=vessel_filename)
    def exchange(self, item):
        self.server.dataput(item)
        return self.server.dataget()
    def reset(self, vessel_filename=None):
        """
        input param
            vessel_filename : (str | None) If None, use the last file name.
        """
        if vessel_filename is None:
            vessel_filename = self.vessel_filename
        self.close()
        self.open(vessel_filename=vessel_filename)
    def close(self):
        # Close the client.
        if self.sim_opened:
            orderdict = {'order': 'close',
                        'info': dict()}
            self.exchange(orderdict)
            self.server.waitclientclose()
            self.sim_opened = False
    def open(self, vessel_filename):
        """Run the client.
        """
        mkdir(directory=self.commu_dir)
        print(self.commu_dir)
        delete_old_files(directory=self.commu_dir, seconds_old=self.timeout+1)
        self.server.runclient(vessel_filename=vessel_filename)
        self.sim_opened = True
    def action(self, translation=0, rotation=0):
        translation = float(translation)
        rotation = float(rotation)
        orderdict = {'order':'action',
                    'info': {'translation':translation,
                            'rotation':rotation}}
        self.exchange(orderdict)
    def step(self, realtime=False):
        orderdict = {'order': 'step',
                    'info': {'realtime':realtime}}
        response = self.exchange(orderdict)
        print(response)
        errclose = response['data']['errclose']
        if errclose: self.sim_opened = False
        return errclose
    def GetImage(self) -> np.ndarray:
        orderdict = {'order': 'GetImage',
                     'info': dict()}
        response = self.exchange(orderdict)
        filename = response['data']['filename']
        image = dataload(filename=filename)
        return image
    def getData(self, function_name:list):
        """
        ------
        input
            function_name : get_GW_position, get_GW_velocity
        """
        orderdict = {'order': 'getData',
                     'info': dict()}
        orderdict['info']['function_name'] = function_name
        filename = self.exchange(orderdict)['data']['filename']
        data = dataload(filename=filename)
        return data
    
    def get_GW_position(self) -> np.ndarray:
        orderdict = {'order': 'get_GW_position',
                     'info': dict()}
        response = self.exchange(orderdict)
        filename = response['data']['filename']
        GW_position = dataload(filename=filename)
        return GW_position
    def get_GW_velocity(self) -> np.ndarray:
        orderdict = {'order': 'get_GW_velocity',
                     'info': dict()}
        response = self.exchange(orderdict)
        filename = response['data']['filename']
        GW_velocity = dataload(filename=filename)
        return GW_velocity
    def move_camera(self, position=None, lookAt=None, orientation=None):
        filename = self.commu_dir + f'/{self.server.port_rpc}_{str(time.time())[7:]}.pkl'
        orderdict = {'order': 'move_camera',
                     'info': {'filename': filename}
                    }
        data = {'position':position,
                'lookAt':lookAt,
                'orientation':orientation}
        datasave(item=data, filename=filename)
        self.exchange(orderdict)

