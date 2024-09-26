import numpy as np
from tracker.proximity_calculator import ProximityCalculator
import math
from dataloader import StateLoader, Point


class State:
    """
    1. A state object is an object with a collection of points
    2. It manages the objects and exposes points, while also exposing an indexing method for the points
    """
    MAX_STATE_VOL = 50
    def __init__(self) -> None:
        self.__current_state = []
        self.__next_state = []

    def update_state_data(self, data:list[dict])->list[dict]:
        pts = StateLoader.load_points(data)
        self.update_state(pts)
        return self.get_next_state()

    def update_state(self, state:list[Point])->None:
        if len(self.__next_state) > 0:
            self.__current_state = self.__next_state
            self.__next_state = state
        else:
            self.__next_state = state
            self.__current_state = state

    def get_dicts(self)->list[dict]:
        result = []
        for point in self.__next_state:
            result.append(point.data)
        return result

    def get_next_state(self)->list[dict]:
        for idx, point in enumerate(self.__current_state):
            point.extras.clear()
            point.extras = ('det', point.data)
            point.id = idx 

        if len(self.__current_state) <= round(len(self.__next_state)*1.5) :
            self.__match_states()

        return self.get_dicts()
    

    def __match_states(self)->None:
        """
        1. Take State 0 and Compare it with State 1
        2. Find the holes in state one from all the points that didn't match state 0
        3. Update State 1 with the left over points from State 0
        """
        prox_calc = ProximityCalculator(self.__current_state, self.__next_state, True)
        prox_calc.compute()
        # X list -- > O list
        self.__current_state, self.__next_state = prox_calc.get_associated_points()
        missed_dets = []
        for point in self.__current_state:
            if 'found_det' not in point.extras:
                p = point.copy()
                missed_dets.append(p)
        
        # print(f"Missed Detections: {len(missed_dets)}")
        self.__next_state.extend(missed_dets)            

