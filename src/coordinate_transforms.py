import numpy as np
import os
import json
from pathlib import Path
import cv2 as cv


class BoxToPoint:
    def __init__(self, det_struct:dict, **kwargs)->None:
        self.__det_struct = det_struct # JSON representation of the detections object
        self.__xy = None # <x,y> coordinates of the object.
        self.__init()

    def __init(self):
        x1 = self.__det_struct.get('bbox').get("x1")
        y1 = self.__det_struct.get('bbox')["y1"]
        x2 = self.__det_struct.get('bbox')["x2"]
        y2 = self.__det_struct.get('bbox')["y2"]
        width = x2 - x1;
        self.__xy = (int(x2-(width/2)), int(y2))
        self.__det_struct["coordinates"] = self.__xy

    def get_struct(self):
        return self.__det_struct
    
    
class CoordinatesFilter:
    def __init__(self, polygon:list, detections:dict)->None:
        self.__polygon = polygon
        self.__detections = detections

class BTransformations:
    def __init__(self):
        pass

    def transform(self, detections:list[dict])->list[dict]:
        return detections


class PerspectiveTransform(BTransformations):
    def __init__(self, src_poly:list, dst_pts, id = 0, **kwargs)->None:
        super().__init__()
        self.__src_poly = src_poly
        self.__dst_poly = dst_pts
        self.__pers_matrix = None
        self.__top_offset = 0
        self.__left_offset = 0
        self.__dst_width = 0
        self.__dst_height = 0
        self.__width = int(2590*0.98)
        self.__height = int(1942*0.7)
        self.__id = id
        self.__init()
        

    def __init(self)->None:
        # get the transformation matrix
        self.__pers_matrix = cv.getPerspectiveTransform(np.array(self.__src_poly, dtype=np.float32), np.array(self.__dst_poly, dtype=np.float32))
    
    def recalculate_perspective_matrix(self)->None:
          # get the transformation matrix
        self.__pers_matrix = cv.getPerspectiveTransform(np.array(self.__src_poly, dtype=np.float32), np.array(self.__dst_poly, dtype=np.float32))

    def update_src_poly(self, src_poly)->None:
        self.__src_poly = src_poly
    
    def get_src_poly(self)->list[tuple]:
        return self.__src_poly

   
    def transform(self, detections:list[dict])->list[dict]:
        super().transform(detections)
        src_pts = []
        bbox_transforms = []
        box_widths = []
        box_heights = []
        boxes = False
        for det in detections:
            if det.get('coordinates') is not None:
                src_pts.append(det.get("coordinates"))
                box = det.get('bbox')
                if box is not None:
                    bbox_transforms.append((box['x1'], box['y1']))
                    b_width = box['x2'] - box['x1']
                    b_height = box['y2'] - box['y1']
                    box_widths.append(b_width)
                    box_heights.append(b_height)
                    boxes = True

                if box is None:
                    boxes = False
        # apply the perspective transform here..
        result = []
        if len(src_pts) > 0:
            trans = cv.perspectiveTransform(np.array(src_pts, dtype=np.float32)[None, :, :], self.__pers_matrix)
            if boxes:
                box_trans = cv.perspectiveTransform(np.array(bbox_transforms, dtype=np.float32)[None, :, :], self.__pers_matrix)
                for det_ , t_b, w, h in zip(detections, box_trans[0], box_widths, box_heights):
                    x1 = t_b[0]
                    y1 = t_b[1]
                    det_['t_box'] = {'x1': x1, "y1":y1, "x2":x1+w, "y2":y1+h}

            for transformed_point, det_  in zip(trans[0], detections):
                det_["coordinates"] = (int(transformed_point[0]), int(transformed_point[1]))
                if transformed_point[0] >= 0 and transformed_point[1] >=0:
                    result.append((int(transformed_point[0]), int(transformed_point[1])))
        return detections, result
    
    def getDstPts(self)->list:
        return self.__dst_poly
        
    def get_offsets(self)->tuple:
       return self.__left_offset, self.__top_offset, self.__dst_width, self.__dst_height


class Transformer:
    def __init__(self, width, height, pitch_coordinates:dict, id=0)->None:
        self.__mini_boudary = []
        self.__mini_boundary_transformed = [] 
        self.__stream_id = id
        self.__centre_point = None
        self.__pers_transformer = None
        self.__width = width
        self.__height = height
        self.__pitch_coordinates = pitch_coordinates
        self.__centre_point = self.__pitch_coordinates.get('center_pt')
        self.__src_pts = self.__pitch_coordinates.get('src_pts')
        self.__dst_pts = self.__pitch_coordinates.get('dst_pts')
        self.__mini_boudary = self.__pitch_coordinates.get('mini_boundary')
        self.init()


    def init(self):
        self.__pers_transformer = PerspectiveTransform(self.__src_pts, self.__dst_pts, id = self.__stream_id)
        self.__is_init = True
        return
        
    def is_init(self)->bool:
        return self.__is_init

    def getDstPts(self):
        if self.__pers_transformer:
            return self.__transform_dst_pts()
        return []
    
    def get_mini_boudary(self)->list[dict]:
        self.__mini_boundary_transformed = self.__transform_boundary_pts(self.__mini_boudary)
        return self.__mini_boundary_transformed
    
    def getDstPtsRaw(self)->list:
        return self.__pers_transformer.getDstPts()
    
    def getSrcPts(self)->list:
        return self.__pitch_coordinates
    
    def get_offsets(self)->tuple:
        if self.__pers_transformer:
            return self.__pers_transformer.get_offsets()
        return None, None, None, None
    

    def set_perspective_transform(self, pers_trans_obj:PerspectiveTransform)->None:
        self.__pers_transformer = pers_trans_obj
        self.__pitch_coordinates = pers_trans_obj.get_src_poly()
            
    def __normalize_coordinates(self, width, height, detections_t)->dict:
        detections_n = []
        offsets = self.__pers_transformer.get_offsets()
        
        for detection in detections_t:
            coord = detection.get('coordinates')
            x_n = ((coord[0] - offsets[0]) / offsets[2])
            y_n = ((coord[1]  - offsets[1])/ offsets[3])
            detection['coordinates'] = (x_n, y_n)
            detections_n.append(detection)
        return detections_n
        
    def normalize_one(self, det:list[dict])->tuple:
        dets = self.__normalize_coordinates(0,0, det)
        if len(dets) == 1:
            return dets[0].get('coordinates')
        return dets

    def transform_one(self, det)->tuple[list[dict], list[dict]]:
        return self.__pers_transformer.transform(det)
    
    def __transform_dst_pts(self)->list[dict]:
        dst_pts = self.__src_pts
        dst_dict = []
    
        if len(dst_pts) > 0:
            for point in dst_pts:
                res = {
                    'coordinates':point,
                    'track_id': self.__stream_id + 0x10,
                    'boundary_point':True,
                    'color_l': (int(255*self.__stream_id/2), int(255*self.__stream_id/2), int(0*self.__stream_id/2))
                }
                dst_dict.append(res)
            trans_, _ = self.transform_one(dst_dict)
            n_pts = self.normalize_one(trans_)
            return n_pts
        return []
    
    def __transform_boundary_pts(self, boundary:list[tuple])->list[dict]:
        dst_pts = boundary
        dst_dict = []

        if len(dst_pts) > 0:
            for point in dst_pts:
                res = {
                    'coordinates':point,
                    'track_id': self.__stream_id + 0x10,
                    'boundary_point':True,
                    'color_l': (int(255*self.__stream_id/2), int(255*self.__stream_id/2), int(0*self.__stream_id/2))
                }
                dst_dict.append(res)
            trans_, _ = self.transform_one(dst_dict)
            n_pts = self.normalize_one(trans_)
            return n_pts
        return []
            
    def transform(self, detections:list[dict])->list[dict]:
        detections_t = detections
        res_vector = None
        if self.__pers_transformer is not None:
            detections_t, res_vector = self.__pers_transformer.transform(detections) 
        return detections_t



def convert_box_2_points(dets:list)->list:
    # a_img = img
    a_list = []
    for det in dets:
        bp = BoxToPoint(det)
        # a_img = bp.draw_point(a_img)
        a_list.append(bp.get_struct())
    return a_list
        
