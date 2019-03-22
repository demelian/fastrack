from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from multiprocessing.pool import Pool

import multiprocessing

import os
import numpy as np
import pandas as pd
import ctypes
from ctypes import cdll

class Model():
    def __init__(self):
        super().__init__()
        
        dirName = os.path.dirname(__file__)
        
        geomPath = os.path.join(dirName, 'geometry.bin')
        connPath = os.path.join(dirName, 'connections.bin')
        libPath  = os.path.join(dirName, 'libModel.so')
        
        geom_string = geomPath.encode('utf-8')
        conn_string = connPath.encode('utf-8')
        
        p1D_float = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
        p1D_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='CONTIGUOUS')
        p1D_int    = np.ctypeslib.ndpointer(dtype='int32', ndim=1, flags='CONTIGUOUS')
        
        self.tmlLib = cdll.LoadLibrary(libPath)
        
        self.tmlLib.createModel.restype = ctypes.c_void_p
        self.tmlLib.createModel.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.tmlLib.deleteModel.argtypes = [ctypes.c_void_p]
        
        self.tmlLib.importHits.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                           p1D_int,
                                           p1D_float, p1D_float, p1D_float,
                                           p1D_int, p1D_int, p1D_int]

        self.tmlLib.importCells.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            p1D_int, p1D_int, p1D_int, p1D_float]
        
        self.tmlLib.findTracks.argtypes = [ctypes.c_void_p, p1D_int]
                                           
        self.tmlModel = self.tmlLib.createModel(geom_string, len(geom_string), conn_string, len(conn_string))
    
    def __del__(self) :
        self.tmlLib.deleteModel(self.tmlModel)
    
    def predict_one_event(self, event_id, hits, cells):
        
        hit_id= hits.hit_id.values
        hit_x = hits.x.values
        hit_y = hits.y.values
        hit_z = hits.z.values
        vol_id= hits.volume_id.values
        lay_id= hits.layer_id.values
        mod_id= hits.module_id.values

        cells_hit_id = cells.hit_id.values
        cells_ch0    = cells.ch0.values
        cells_ch1    = cells.ch1.values
        cells_value  = cells.value.values
        
        nHits = hit_x.shape[0]
        nCells = cells.hit_id.shape[0]
        
        #hit_id,ch0,ch1,value
        #1,259,732,0.297276
        #2,306,1097,0.297281
        #3,268,995,0.00778383
        #3,267,995,0.118674
        #3,267,996,0.194891
        
        self.tmlLib.importHits(self.tmlModel, ctypes.c_int(hits.x.values.shape[0]),
                               hit_id, hit_x, hit_y, hit_z, vol_id, lay_id, mod_id)

        self.tmlLib.importCells(self.tmlModel, ctypes.c_int(cells.hit_id.shape[0]),
                                cells_hit_id, cells_ch0, cells_ch1, cells_value)
        
        labels = np.zeros(shape = hit_id.shape, dtype = np.int32)
        
        self.tmlLib.findTracks(self.tmlModel, labels)
        
        sub = pd.DataFrame(data=np.column_stack((hits.hit_id.values, labels)), columns=["hit_id", "track_id"]).astype(int)
        sub['event_id'] = event_id
        return sub
