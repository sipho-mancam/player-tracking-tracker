
import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
from .proximity_calculator import Point, ProximityCalculator, JumpsInvestigator

"""
This module implement the second layer of our Data Association problem
1. AssociationsManager
    a. This implements the main object that handles all the tracked objects
    b. Implements the 'global update method
    c. Implements the To JSON method for output.
    d. Manages associations bins (Found and Unfound)

2. Tracklet:
    Object that is being tracked
    1. Implements ID Associations
    2. Calls Kalman Filter for predict, update and project
    3. Implements Euclidean distance calc and perform secondary associations with detections,
    4. If detection has ID, update current ID
    5. If there's a matched Association, update the Kalman Filter with the update
    6. Else update it with it's current state
    7. Exit

3. Kalman Filter
    Implements the kalman filter associated with object.

"""

class KalmanFilter:
    def __init__(self, dt, state_dim, meas_dim, Q, R, id=0):
        self.dt = dt  # Time step
        self.state_dim = state_dim  # Dimension of the state vector
        self.meas_dim = meas_dim  # Dimension of the measurement vector
        self.__id = id

        # Initial state vector
        self.x = np.zeros(state_dim)

        # Initial state covariance matrix
        self.P = np.eye(state_dim)*0

        # Process noise covariance matrix
        self.Q = Q

        # Measurement noise covariance matrix
        self.R = R

        # Measurement matrix
        self.H = np.zeros((meas_dim, state_dim))
        self.H[0, 0] = 1  # x position
        self.H[1, 1] = 1  # y position

        # State transition matrix
        self.F = np.eye(state_dim)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        self.x = (np.eye(self.state_dim) - K@self.H)@self.x + (K @ z)
        self.x[:2] = z

    def get_state(self):
        return self.x


class TrackLet:
    def __init__(self, global_id,  det:dict, color=None)->None:
        self.dt = 1
        self.state_dim = 4
        self.meas_dim = 2
        self.global_id = global_id
        self.Q = np.eye(self.state_dim) * 0.01
        self.R = np.eye(self.meas_dim) * 0.1

        self.__det_raw = det
        self.__tracked_id = None
        self.__past_track_ids = []
        self.__coordinates = None       
        self.__color = color
        self.__vanished = False
        self.__kalman_filter = KalmanFilter(self.dt, self.state_dim, self.meas_dim, self.Q, self.R, self.global_id)
        self.__is_updated = False
        self.__life_span_reset = 50
        self.__life_span = self.__life_span_reset
        self.__temporary_state = None
        self.__sig_jump_dist = 0.03
        self.__flagged  = False   
        self.__obstructed = False
        self.__ob_dist = 0.04
        self.__jump_distance = 0.0
        self.__alert_state = False
        self.__alert_distance = 0.03
        self.init()

    @staticmethod
    def detection_to_point(det:dict)->Point:
        coord = det.get('coordinates')
        x, y = coord
        point = Point(x, y, 'X', 10, id=det.get('id'))
        point.extras = ('det', det)
        return point
    
    @staticmethod
    def euclidean_distance(point1, point2)->float:
        return np.sqrt(((point1[0]-point2[0])**2)+((point1[1] - point2[1])**2))
    
    @property
    def temp_state(self)->dict|None:
        return self.__temporary_state
    
    @property
    def life_span(self)->float:
        return self.__life_span
    
    @life_span.setter
    def life_span(self, span)->None:
        self.__life_span = span
    
    @property
    def obstructed(self)->bool:
        return self.__obstructed
    
    @obstructed.setter
    def obstructed(self, ob)->None:
        self.__obstructed = ob

    @property
    def guid(self):
        return self.global_id
    
    @property
    def death_percentage(self):
        return self.__life_span/self.__life_span_reset
    
    @property
    def object_vanished(self):
        return self.__vanished
    
    @object_vanished.setter
    def object_vanished(self, vaished)->None:
        self.__vanished = vaished

    @property
    def flagged(self)->bool: # This property tells us that the object attempted to commit and illegal jump.
        return self.__flagged

    @property
    def death_span(self):
        return self.__life_span
    
    @property
    def coordinates(self)->tuple:
        return self.__coordinates

    @coordinates.setter
    def coordinates(self, coord)->None:
        self.__coordinates = coord
    
    def __find_closest_detection(self, pred_coordinates:list, dets:list)->dict|None:
        closest_det = None
        min_dist = math.inf
        min_idx = math.inf
        dist_list = []
        for idx, det in enumerate(dets):
            coord = det.get('coordinates')
            dist = self.__euclidean_dist(pred_coordinates, coord)
            dist_list.append(dist)
            if dist < min_dist and dist < 0.05:
                min_dist = dist
                closest_det = det
                min_idx = idx
        if len(dets) > 0 and min_idx is not math.inf:
            closest_det = dets.pop(min_idx)
        return closest_det

    def __find_closest_detection_reassign(self, dets:list)->dict|None:
        min_idx = math.inf
        min_dist = math.inf
        closest_det = None
        for ix, det in enumerate(dets):
            coord = det.get('coordinates')
            dist = self.__euclidean_dist(coord, self.__coordinates)
            if dist < min_dist:
                min_dist = dist
                min_idx = ix
        if min_idx is not math.inf:
            closest_det = dets.pop(min_idx)
        return closest_det

    def __euclidean_dist(self, point1, point2)->float:
        return np.sqrt(((point1[0]-point2[0])**2)+((point1[1] - point2[1])**2))
    
    def euclidean_dist(self, point)->float:
        point1 = self.__coordinates
        return self.__euclidean_dist(point1, point)

    def associate_dead_tracklet(self, dets)->bool:
        closest = self.__find_closest_detection_reassign(dets)
        if closest is not None:
            closest['reassigned'] = True
            self.reassigned = True
            self.__coordinates = closest.get('coordinates')
            self.__tracked_id = closest.get('track_id')
            self.__det_raw = closest
            return True
        return False
    
    def init(self)->None:
        self.__coordinates = self.__det_raw.get('coordinates')
        self.__tracked_id = self.__det_raw.get('track_id')
        self.__past_track_ids.append(self.__tracked_id)

    # ID Associations
    def associate_id(self, dets:list)->bool:
        for idx, det in enumerate(dets):
            id = det.get('track_id')
            if id is None:
                continue

            if self.__tracked_id == id:
                self.__coordinates = det.get('coordinates')
                self.__tracked_id = det.get('track_id')
                self.__det_raw = dets.pop(idx)
                self.__life_span = self.__life_span_reset
                
                if self.__alert_state:
                    self.__det_raw['alert'] = self.__alert_state
                return True


        return False

    def associate_prediction(self, dets:list)->bool:
        """
        1. Get a Kalman Prediction
        2. find the closest detection
        3. associate closest detection with tracklet and update tracklet.
        4. Update Kalman filter with new measurement
        """
        self.__kalman_filter.predict()
        pred_coord = self.__kalman_filter.get_state()[:2]
        closest_det = self.__find_closest_detection(pred_coord, dets)
        # update closest found det
        if closest_det is not None:
            self.__coordinates = closest_det.get('coordinates')
            self.__tracked_id = closest_det.get('track_id')
            self.__det_raw = closest_det
            self.__life_span = self.__life_span_reset
            return True
        return False

    def any(self, ite, var)->bool:
        for i in ite:
            if i == var:
                return True
        return False

    def predict_assign(self)->None:
        self.__det_raw['color'] = (0, 0, 255)
        self.__det_raw['missed'] = True

    def get_distance_from_x(self, x)->float:
        coord = x.get('coordinates')
        return self.__euclidean_dist(self.__coordinates, coord)
    
    def closer_than_x(self, x, det)->tuple[bool, float]:
        dist = self.get_distance_from_x(det)
        if dist < x:
            return (True, dist)
        return (False, None)
    
    def find_closer_than_det(self, x, dets:list[dict])->tuple[bool, dict]:
        min_closest = math.inf
        min_index = math.inf

        for idx, det in enumerate(dets):
            _, dist = self.closer_than_x(x, det)
            if _ and dist < min_closest:
                min_closest = dist
                min_index = idx

        if min_closest is not math.inf:
            return (True, dets[min_index])
        return (False, None)

    def find_closest(self, dets:list[dict])->tuple[float, dict]:
        min_idx = math.inf
        min_dist = math.inf
        closest_det = None

        for ix, det in enumerate(dets):
            coord = det.get('coordinates')
            dist = self.__euclidean_dist(coord, self.__coordinates)
            if dist < min_dist:
                min_dist = dist
                min_idx = ix

        if min_idx is not math.inf:
            closest_det = dets[min_idx]

        return (min_dist, closest_det)
    
    def commit(self)->None:
        if self.__temporary_state is None:
            return
        
        if self.__jump_distance > 0.2 and self.__life_span > 0: 
            # print(self.global_id, self.__jump_distance, self.__life_span)
            self.__life_span -=1
            return
        #Commit changes if the object was under investigation
        self.__det_raw = self.__temporary_state
        self.__coordinates = self.__det_raw['coordinates']
        self.__tracked_id = self.__det_raw.get('track_id')
        self.__vanished = False
        self.__life_span = self.__life_span_reset
        self.__is_updated = True
        self.__flagged = False

        if self.__jump_distance > self.__alert_distance:
            self.__det_raw['alert'] = True
            

    def assign_det(self, det)->None:
        if det is None:
            return 
        
        self.__temporary_state = det
        self.__det_raw = det
        self.__coordinates = self.__det_raw['coordinates']
        self.__tracked_id = self.__det_raw.get('track_id')
        self.__vanished = False
        self.__life_span = self.__life_span_reset
        self.__is_updated = True
        self.__flagged = False

        if self.__jump_distance > self.__alert_distance:
            self.__det_raw['alert'] = True
        return
    
    def to_point(self)->Point:
        self.__det_raw['id'] = self.global_id
        return TrackLet.detection_to_point(self.__det_raw)

    def set_found(self)->None:
        self.__is_updated = True
    
    def clear_found(self)->None:
        self.__is_updated = False

    def get_id(self)->int:
        return self.__tracked_id
        
    def set_color(self, color)->None:
        self.__color = color
    
    def is_on_edge(self)->bool:
        return (self.__coordinates[0] <= 0.003 or self.__coordinates[0]>= 0.997  or self.__coordinates[1] <= 0.003 or self.__coordinates[1]>=0.997)
    
    def get_dead_span(self)->int:
        return self.__life_span
    
    def decrease_dead_span(self):
        self.__life_span -= 1
    
    def reset_dead_span(self)->None:
        self.__life_span = self.__life_span_reset

    def awaken_the_dead(self, dets:list[dict])->bool:
        """
        1. If you are dead, but are not on the edge
        2. If you are dead and there's someone close to you, we reassign
        """
        # self.__det_raw['color'] = (0, 0, 255)
        # self.__det_raw['missed'] = True
        # print(self.__coordinates, self.global_id)
        # return
        if  self.is_on_edge() and self.__life_span > 0:
            self.__det_raw['color'] = (0, 0, 255)
            self.__det_raw['missed'] = True
            self.__life_span -= 1
            return False

        if self.__life_span > 20:
            self.__life_span -= 1
            return False

        shortest_distance = math.inf
        closest_index = math.inf
        for i, det in enumerate(dets):
            coord = det.get('coordinates')
            dist = self.__euclidean_dist(coord, self.__coordinates)
            if dist < shortest_distance:
                shortest_distance = dist
                closest_index = i

        if shortest_distance is not math.inf:
            self.__det_raw = dets.pop(closest_index)
            self.__tracked_id = self.__det_raw.get('track_id')
            self.__coordinates = self.__det_raw.get('coordinates')
            self.__det_raw['color'] = (0, 245, 185)
            self.__det_raw['awakened'] = True
            self.__life_span = self.__life_span_reset
            return True
        return False
    
    def found(self)->bool:
        return self.__is_updated

    def get_dict(self)->dict:
        self.__det_raw['guid'] = self.global_id
        # self.__det_raw['kit_color'] = self.__color # if self.found() else (0, 0, 255)#(255, 0, 0) if self.__det_raw.get('missed') is None else self.__det_raw['color']
        self.__det_raw['coordinates'] = self.__kalman_filter.get_state()[:2].tolist()
        self.__det_raw['dead'] = not self.found()
        self.__det_raw['death_percentage'] = self.death_percentage

        if not self.found():
            self.__life_span -= 1
        return self.__det_raw

    def update(self)->None:
        self.__kalman_filter.update(np.array(self.__coordinates))

    def predict_current_state(self)->None:
        self.__coordinates = self.__kalman_filter.get_state()[:2]

    def __eq__(self, track_id: object) -> bool:
        return self.__tracked_id == track_id
    
    def __ge__(self, obj)->bool:
        return self.__life_span >= obj.__life_span

    def __gt__(self, obj)->bool:
        return self.__life_span > obj.__life_span

    def __lt__(self, obj)->bool:
        return self.__life_span < obj.__life_span

    def __le__(self, obj)->bool:
        return self.__life_span <= obj.__life_span
    
    def __sub__(self, obj)->float:
        return self.__life_span - obj.__life_span
    
    def __add__(self, obj)->float:
        return self.__life_span + obj.__life_span

    def __str__(self)->str:
        return self.__det_raw.__str__()
    
    

class AssociationsManager:
    def __init__(self, teams_colors:tuple)->None:
        self.__associated_tracklets = []
        self.__unfound_tracklets = []
        self.__guid_counter = 1
        self.__tracklets_pool = [] # Full list of tracklets
        self.__reset_tracks = [] # This contains a list of the tracks that need to be reset.
        self.__tracklets_limit = 26
        self.__team_ids_track = [{'ids_track':0, 'tracklets':[], 'color':teams_colors[0], 'init':False, 'id':2, 'guid':0}, 
                                 {'ids_track':0, 'tracklets':[], 'color':teams_colors[1], 'init':False, 'id':1, 'guid':0}]
        self.__teams_init = False
        self.colors = [
                (0, 0, 0),        # Black
                (255, 255, 255),  # White
                (255, 0, 0),      # Blue
                (0, 255, 0),      # Green
                (0, 0, 255),      # Red
                (0, 255, 255),    # Yellow
                (255, 0, 255),    # Magenta
                (255, 255, 0),    # Cyan
                (128, 128, 128),  # Gray
                (128, 0, 0),      # Maroon
                (128, 128, 0),    # Olive
                (0, 128, 0),      # Dark Green
                (128, 0, 128),    # Purple
                (0, 128, 128),    # Teal
                (0, 0, 128),      # Navy
                (192, 192, 192),  # Silver
                (255, 165, 0),    # Orange
                (255, 105, 180),  # Hot Pink
                (147, 112, 219),  # Medium Purple
                (173, 216, 230),  # Light Blue
                (32, 178, 170),   # Light Sea Green
                (135, 206, 250),  # Light Sky Blue
                (240, 230, 140),  # Khaki
                (72, 61, 139),    # Dark Slate Blue
                (210, 105, 30),   # Chocolate
                (255, 20, 147),   # Deep Pink
                (0, 191, 255),    # Deep Sky Blue
                (34, 139, 34),    # Forest Green
                (255, 69, 0),     # Orange Red
                (123, 104, 238)   # Medium Slate Blue
            ]

    def remove_children_with_parents(self, dets)->list[dict]:
        parents_list = []
        children_list = []
        result = []

        for det in dets:
            if det.get('has_child'):
                parents_list.append(det)
                continue

            if det.get('is_child'):
                children_list.append(det)
                continue

            result.append(det)

        for idx, child in enumerate(children_list):
            c_coord = child.get('coordinates')
            for parent in parents_list:
                marker = parent.get('child').get('marker_id')
                if marker == child.get('marker_id'):
                    try:
                        # children_list.remove(child)
                        del child
                        break
                        # del child
                    except Exception as e:
                        pass
        # result.extend(children_list)
        result.extend(parents_list)
        return result
    
    def __colors_match(self, color1:tuple, color2:tuple)->bool:
        return (color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2])
    
    def __get_team(self, det:dict)->dict:
        color = det.get('kit_color')

        if color is None:
            return None
   
        if not self.__teams_init:
            if not self.__team_ids_track[0]['init']:
                return self.__team_ids_track[0]
            elif self.__colors_match(color, self.__team_ids_track[0]['color']):
                return self.__team_ids_track[0]
            else:
                self.__teams_init = True
                return self.__team_ids_track[1]
            
        for team in self.__team_ids_track:
            
            if self.__colors_match(color,  team['color']):
                return team
            
    
    def remove_detections_without_ids(self, dets:list[dict])->list[dict]:
        result = []
        for det in dets:
            if det.get('track_id') is None:
                del det
                continue
            result.append(det)
        return result
    
    def spilt_objects_by_color(self, dets)->tuple[list]:
        team_a = []  # Color a
        team_b = [] # color b
        color_a , color_b = (self.__team_ids_track[0]['color'], self.__team_ids_track[1]['color'])
        for i, det in enumerate(dets):
            if self.__colors_match(color_a, det.get('kit_color')):
                team_a.append(dets.pop(i))
            elif self.__colors_match(color_b, det.get('kit_color')):
                team_b.append(dets.pop(i))
        return team_a, team_b
    
    def get_unfound_tracks(self)->tuple[list]:
        team_a, team_b = (self.__team_ids_track[0]['tracklets'], self.__team_ids_track[1]['tracklets'])
        team_a_result, team_b_result = [] , []

        for track in team_a:
            if not track.found():
                team_a_result.append(track)

        for track in team_b:
            if not track.found():
                team_b_result.append(track)
        
        return team_a_result , team_b_result
    
    def print_tracklets(self, tracklets:list[TrackLet])->str:
        res = ""
        for track in tracklets:
            res += track.__str__() + "\n"
        return res
    
    def proximity_association(self, dets, tracks_list, min_dist=True)->None:
          # Convert the remaining detections to points using
        o_list = []
        x_list = []
        for idx, det in enumerate(dets):
            det['id'] = idx 
            point = TrackLet.detection_to_point(det)
            o_list.append(point)
    
        for track in tracks_list:
            x_list.append(track.to_point())

        pc = ProximityCalculator(x_list, o_list, min_dist)
        pc.compute()
        x_list, _ = pc.get_associated_points()
        for x in x_list:
            for t in tracks_list:
                if x.id == t.guid:
                    if x.extras.get('found_det') is not None:
                        t.assign_det(x.extras['found_det'])
                        i = dets.index(x.extras['found_det'])
                        dets.pop(i)
                        # print(f"Detection Index: {i}")
                        t.set_found()
                    break

        # Check all the tracks that have been flagged and let's investigate.
        if not min_dist:
            ji = JumpsInvestigator(tracks_list)
            commits = ji.get_tracks()
            for track in commits:
                track.commit()


    
    def update(self, dets:list)->None:
        """
        1. Associate with IDS
        2. Associate using secondary state prediction
        3. Initialize new tracklets if tracklets are < x_num_of_players
        """
        for tracklet in self.__tracklets_pool:
            if tracklet.associate_id(dets):
                self.__associated_tracklets.append(tracklet)
                tracklet.set_found()
            else:
                self.__unfound_tracklets.append(tracklet)
                tracklet.clear_found()


        if self.__teams_init:      
            self.proximity_association(dets, self.__unfound_tracklets)
            # Associate the available detections with the tracklets that are ready to be reset
            self.proximity_association(dets, self.__reset_tracks, False)
            
            # Remove the tracks that are reset.
            for i, track in enumerate(self.__reset_tracks):
                if track.found():
                    i = self.__reset_tracks.index(track)
                    self.__reset_tracks.pop(i)  


        if self.__tracklets_limit % 2 != 0:
            self.__tracklets_limit += 1

        if len(dets) > 0 and len(self.__tracklets_pool) < self.__tracklets_limit:
            for i, det in enumerate(dets):
                
                if det.get('track_id') is None:
                    continue

                if len(self.__tracklets_pool) <= self.__tracklets_limit:
                    self.__teams_init = True
                    
                    kit_color = det.get('kit_color')
                    team = self.__get_team(det)

                    # if team is None:
                    #     continue

                    id = self.__guid_counter+1#team['guid'] + team['id'] if team['id'] == 0 else (len(team['tracklets']) * 2) + (team['id'])
                        
                    # team['guid'] = id
                    track =  TrackLet(id, det, kit_color)
                    
                    # if not team['init']:
                    #     team['init'] = True
                    #     # team['color'] = kit_color
                    #     team['tracklets'].append(track)
                    #     team['ids_track'] += 1
                    # else:
                    #     team['tracklets'].append(track)
                    #     team['ids_track'] += 1 

                    self.__tracklets_pool.append(
                       track
                    )
                    self.__guid_counter += 1

        # update the Kalman filters in the tracklets found and asscoiated
        for tracklet in self.__tracklets_pool:
            tracklet.update()

        self.__associated_tracklets = []
        self.__unfound_tracklets = []

    def get_dets(self)->list:
        result = []
        for tracklet in self.__tracklets_pool:
            if tracklet.life_span > 0:
                result.append(tracklet)
            else:
                try:
                    self.__reset_tracks.index(tracklet)
                except ValueError as ve:
                    self.__reset_tracks.append(tracklet)

        return [track.get_dict() for track in result]
    
    def whos_closer_than_x(self, x:float, det:dict, tracklets:list[TrackLet])->tuple[list[float], list[TrackLet]]:
        closer_tracker = []
        distance_list = []
        min_dist = math.inf
        for track in tracklets:
            flag, dist = track.closer_than_x(x, det)
            if flag:
                if dist < min_dist:
                    min_dist = dist
                    distance_list.insert(0, dist)
                    closer_tracker.insert(0, track)
                else:
                    index = self.__smaller_than_x_index(dist, distance_list[1:])
                    if index is not None:
                        distance_list.insert(index, dist)
                        closer_tracker.insert(index, track)
                    else:
                        distance_list.append(dist)
                        closer_tracker.append(track)

        return (distance_list, closer_tracker)

    
    def __smaller_than_x_index(self, x, l:list[float])->float:
        for idx, elem in enumerate(l):
            if elem < x:
                return idx
        return None

    def closest_associations(self, dets:list, unfound_tracks:list[TrackLet])->tuple[list[TrackLet], list[TrackLet]]:
        un_assigned = []
        assigned_tracks = []
        skip = False
        for idx, track in enumerate(unfound_tracks):
            if track in assigned_tracks:
                continue

            dist, closest_det = track.find_closest(dets)
            if dist <= 0.02: # Close enough to just assign
                track.assign_det(closest_det)
                dets.remove(closest_det)
                assigned_tracks.append(track)
                continue
            elif not track.object_vanished:
                track.object_vanished = True
                
            # Check if your object vanished first circle of doubt (Our object may have reappeared a bit far from us).
            if track.object_vanished and dist <= 0.05: 
                track.assign_det(closest_det)
                dets.remove(closest_det)
                assigned_tracks.append(track)
            else:
                track.decrease_dead_span()
                un_assigned.append(track)

        return un_assigned, assigned_tracks
            


