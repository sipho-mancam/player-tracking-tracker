import cv2 as cv
import numpy as np
import json
from tracker.bot_sort import BoTSORT, STrack
from cfg.config_ import TrackingConf
import time
from tracker.kalman_associator import AssociationsManager
from pprint import pprint

tracker = BoTSORT(TrackingConf(), 10)
frame_count = 0

def convert_to_output_results(arr):
    num_entries = len(arr)
    output_results = np.zeros((num_entries, 12))  # Assuming there are 10 elements in total
    for idx, entry in enumerate(arr):
        # Extract values from each dictionary entry
        x1, y1, x2, y2 = entry['t_box']['x1'], entry['t_box']['y1'], entry['t_box']['x2'], entry['t_box']['y2']
        confidence = entry['confidence']
        cls = entry['class']
        x_1, y_1, x_2, y_2 = entry['box']['x1'], entry['box']['y1'], entry['box']['x2'], entry['box']['y2']
        mm_coordinates, bbox = entry['coordinates'], None
        # Assign values to the output_results array
        output_results[idx, :4] =  [x_1, y_1, x_2, y_2]
        output_results[idx, 4] = confidence
        output_results[idx, 5] = cls
        output_results[idx, 6:8] = mm_coordinates
        output_results[idx, 8:] = [x1, y1, x2, y2]
        # Assign placeholder values for features 3 to 9
        # output_results[idx, 8:] = np.nan  # or any other placeholder value you prefer
    return output_results


def convert_one(det):
    x1, y1= det['coordinates']
    return np.array([x1, y1])

def coordinates_association(det, res_list)->dict:
    coord_list = convert_one(det)
    test_list = []
    for r in res_list:
        res = r.coordinates
        test_list.append(res)
        if res[0] == coord_list[0] and res[1] == coord_list[1]:
            det['track_id'] = r.track_id
            return det
    return det

def associate_dets_with_ids(dets:list[dict], track_res:list)->list[dict]:
    for det in dets:
        det = coordinates_association(det, track_res)
        if det.get('has_child'):
            det['child'] = coordinates_association(det['child'], track_res)
    return dets

def draw_bbox(frame, det)->cv.Mat:
    frame_c = frame
    x1 , y1, w, h = det._tlwh
    x2 = x1+w
    y2 = y1+h
    frame_c = cv.rectangle(frame_c, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    frame_c = cv.putText(frame_c, f"{det.track_id}", (int(x1), int(y1-2)), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return frame_c

def draw_bbox_2(frame, entry)->cv.Mat:
    frame_c = frame
    x1, y1, x2, y2 = entry['box']['x1'], entry['box']['y1'], entry['box']['x2'], entry['box']['y2']
    frame_c = cv.rectangle(frame_c, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    return frame_c


class TrackedObject:
    def __init__(self, track_object:dict, id, **kwargs)->None:
        self.__track_object = track_object
        self.__track_child = { }
        self.__track_id = id
        self.__parent_is_active = False
        self.__child_is_active = False
        self.__track_lost = False

    def get_id(self)->int:
        return self.__track_id

    def handover(self)->None:
        '''
        This is supposed to perform the hand over frame parent to child, 
        turning the child to a parent
        '''
        # This means the parent is activated on this iteration
        if not self.__parent_is_active and not self.__child_is_active:
            self.__track_lost = True
            return 
        
        # if len(self.__track_child.keys()) > 0:
        #     self.__track_object = self.__track_child

        if (not self.__parent_is_active and self.__child_is_active):
            self.__track_object = self.__track_child
            self.__parent_is_active = False
            self.__child_is_active = False
            self.__track_child = {}
            self.__track_lost = False

        self.__parent_is_active = False
        self.__child_is_active = False

    def is_track_lost(self)->bool:
        return self.__track_lost
    
    def to_dict(self)->dict:
        res = self.__track_object.copy()
        res['global_id'] = self.__track_id
        return res

    def get_original_object(self)->dict:
        return self.__track_object

    def update(self, tracked_object:dict)->bool: 
        '''
        1. Check if the object lives in the overlap
        2. Check if the object has a child in the overlap
        3. compare the parents id_ with the objects ID, 
        4. Compare the tracked object with the child ID.
        '''
        # Check if this detection is not held as the child in this object.
        if self.__track_child.get('track_id') is not None and tracked_object.get('track_id') == self.__track_child.get('track_id'):
            self.__child_is_active = True
            self.__track_child = tracked_object
            return True
       
        o_id = tracked_object.get('track_id')
        child = tracked_object.get('child')
        
        # check if the detection is the parent object in this container
        if o_id == self.__track_object.get('track_id'):
            self.__track_object = tracked_object
            self.__parent_is_active = True

        # Check if detection's child didn't initialize the container first. 
        elif child is not None and child.get('track_id') == self.__track_object.get('track_id'):
            self.__track_child = child
            self.__track_object = tracked_object
            self.__parent_is_active = True
            self.__child_is_active = True
            return True
        else:
            return False


        if tracked_object.get('is_overlap'):
            if 'child' in tracked_object:
                self.__track_child = tracked_object['child']
                self.__child_is_active = True
        return True



class TrackObjectsManager:
    def __init__(self)->None:
        self.__activated_tracks = []
        self.__tracks_pool = []
        self.__lost_tracks = []
        self.__ids_count = 1

    def find_track_object(self, det)->TrackedObject|None:
        for t in self.__tracks_pool:
            if t.update(det):
                return t
        return None

    def create_track_object(self, det)->TrackedObject:
        ob = TrackedObject(det, self.__ids_count)
        self.__ids_count += 1
        self.__tracks_pool.append(ob)
        return ob
    
    def add_track(self, track:TrackedObject)->None:
        for t in self.__activated_tracks:
            if track.get_id() == t.get_id():
                return 
        self.__activated_tracks.append(track)
    
    def update(self, dets:list[dict])->list[dict]:
        for det in dets:
            res = self.find_track_object(det) # This is also where we update the state of the tracked objects
            if res is None:
                new_object = self.create_track_object(det)
                new_object.update(det)
                self.add_track(new_object)
            else:
                self.add_track(res)
        
        self.handover()
        result = self.get_tracks_dict()
        self.__activated_tracks = []
        return result

    
    def handover(self):
        for tracklet in self.__tracks_pool:
            tracklet.handover()

        for active_track in self.__activated_tracks:
            active_track.handover()
    
    def get_tracks_dict(self)->list[dict]:
        res  = []
        for track in self.__activated_tracks:
            res.append(track.to_dict())
        return res
    
    def get_orig_dict(self)->dict:
        res  = []
        for track in self.__activated_tracks:
            res.append(track.get_original_object())
        return res

b_c1 =  ( 90.3    ,  103.52    ,  243.37)#(164.37 ,  133.26 , 142.19) #(229, 152, 105)
b_c2 =  (104.19   ,   108.46    ,  105.73)#(249.18 , 251.15 , 246.8)
 
tracks_manager = TrackObjectsManager()
associations_manager = AssociationsManager((b_c1, b_c2))

def filter_list(full_list:list, comp_list:list)->list:
    for det in comp_list:
        try:
            full_list = list(filter(lambda fdet: (fdet['coordinates'][0] != det['coordinates'][0] 
                                                  and fdet['coordinates'][1] != det['coordinates'][1]), 
                                                  full_list))
        except ValueError as ve:
            pass

    comp_list.extend(full_list)
    return comp_list

def track2(detections:list):
    # print(f"Kit Detector Time: {kit_detector.get_execution_time()} ms \t For Detections {len(detections)}")
    det_output = convert_to_output_results(detections)
    global tracker
    global tracks_manager

    s_time = time.time()
    online_tracks = tracker.update(det_output)
    e_time = time.time()
    tracking_results = {}
    res = []
    o_detections = detections
    detections = associate_dets_with_ids(detections, online_tracks)
    # detections = tracks_manager.update(detections)
    # print("Detections received for tracking", len(detections))
    associations_manager.update(o_detections)
    dets = associations_manager.get_dets()
 
    for _, det in enumerate(dets):
        res.append({
            'coordinates':det['coordinates'],
            'tracking-id': det.get('guid') if det.get('guid') else -1,
            'bbox': det['box'],
            'conf': det['confidence'], 
            'kit_color':det.get('kit_color'),
            'alert':det.get('alert')
        })
            
    tracking_results['tracks'] = res
    o_detections.extend(dets)
    return o_detections, tracking_results



def track_raw(detections:list):
    # print(f"Kit Detector Time: {kit_detector.get_execution_time()} ms \t For Detections {len(detections)}")
    res = []
    tracking_results = {}
    for _, det in enumerate(detections):
        res.append({
            'coordinates':det['coordinates'],
            'tracking-id': 0, #det.get('guid') if det.get('guid') else -1,
            'bbox': det['box'],
            'conf': det['confidence'], 
            'kit_color':det.get('kit_color'),
            'alert':det.get('alert')
        })
            
    tracking_results['tracks'] = res
    return detections, tracking_results
