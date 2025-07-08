from kafka import KProducer
import json
import time
from pathlib import Path
from cfg.paths_config import __BASE_DIR__, __TRACKING_DATA_DIR__, __KAFKA_CONFIG__
from pprint import pprint
import math


class DetectionsOutput:
    TRACK_JUMP_DISTANCE = 0.02
    def __init__(self, broker:str, output_topic)->None:
        self.__detections = None
        self.__output = None
        self.__kafka_producer = KProducer(broker)
        self.__output_dir = __TRACKING_DATA_DIR__
        self.__frame_count = 0
        self.__output_topic = output_topic
        self.__kafka_producer.send_message("tracking-core-events", json.dumps({}))
        
        self._on_air_flag = False
        self._on_air_plot_ids = []
        self._on_air_plot_ids_map = {} # Keeps {id:<float, float>} coordinates
        self._on_air_state = None
        
        self._id_range = 26
        self._tracked_ids = []

    def process_output_event(self, event:dict)->None:
        if len(event) == 0:
            return 
        print(event)
        if event["event_name"] == "set_on_air":
            data = event['event_data']
            self._on_air_flag = data["on_air_mode"]

            if not self._on_air_flag:
                self._on_air_plot_ids.clear()
                self._on_air_state = None
                self._on_air_plot_ids_map.clear()
        
        if not self._on_air_flag:
            return
        
        if event["event_name"] == "enable_id_plot":
            data = event["event_data"]
            # This is an id that can be plotted
            self._on_air_plot_ids.append(data["id"])
            self._on_air_plot_ids_map[data["id"]] = [0.5, 0.5]
        elif event['event_name'] == "update_idxy":
            data = event["event_data"]
            id = data['id']
            coordinates= data['coordinates']
            if self._on_air_plot_ids_map.get(id) is not None:
                self._on_air_plot_ids_map[id] = coordinates
                print(f"Updated ID: {id} --> Coordinates: {coordinates}")
        elif event["event_name"] == "id_track_correct":
            data = event["event_data"]
            id  = data["id"]
            self._on_air_plot_ids.append(id)
            self._on_air_plot_ids_map[data["id"]] = [0.5, 0.5]
            print(f"Update ID: {id} for Track Correction")

    def update_untracked_ids(self, tracks:dict)->None:
        self._tracked_ids = []
        for id in range(1, self._id_range):
            tracked = False
            for track in tracks["tracks"]:
                if track["tracking-id"] == id:
                    tracked = True

            if not tracked:
                self._tracked_ids.append(id)

    def track_output_state_update(self, tracks:dict)->None:
        if not self._on_air_flag:
            tracks["on_air_mode"] = False
            self.update_untracked_ids(tracks)
            return tracks
        '''
        if we are in the On Air Mode:
            1. Check if we have a stored state if not, update the stored state and leave
            2. Check if the incoming state matches the stored state.
            3. Fill in, any missing ids with those from the stored state.
            4. Perform jump checks on the final state and correct any jumps
            5. update our approved stored state
            return the modified state.
        '''
        if self._on_air_state is None:
            self._on_air_state = tracks["tracks"]
            return tracks
        # First check if out stored state reflects all the stored ids
       
        for id in self._on_air_plot_ids:
            exists = False
            for track in self._on_air_state:
                if id == track["tracking-id"]:
                    exists = True
                    track["coordinates"] = self._on_air_plot_ids_map[id]
                    break
            
            if not exists:
                self._on_air_state.append(
                    {
                        "coordinates": self._on_air_plot_ids_map[id],
                        "tracking-id": id,
                        "bbox": { "x1": 0, "y1": 0, "x2": 100, "y2": 100},
                        "conf":0,
                        "kit_color": [0.0, 0.0, 0.0],
                        "alert": None,
                        "plotted": True # This flag indicates that we plotted this item it's not tracked fully
                    }
                )
        # check if all the  stored IDs are available in the incoming state, if it is, check distance and update stored
        incoming_tracks = tracks["tracks"]
        for stored_track in self._on_air_state:
            exists = False
            for track in incoming_tracks:
                if track["tracking-id"] == stored_track["tracking-id"]:
                    point1, point2 = track["coordinates"], stored_track["coordinates"]
                    if self.__calculate_track_distance(point1, point2) > DetectionsOutput.TRACK_JUMP_DISTANCE:
                        track["coordinates"] = point2
                    else:
                        stored_track["coordinates"] = point1
                    exists = True

            # The incoming state is missing this ID
            if not exists:
                incoming_tracks.append(stored_track)
    
        tracks["tracks"] = incoming_tracks
        tracks["on_air_mode"] = True
        self.update_untracked_ids(tracks)
        return tracks

    def __calculate_track_distance(self, point1, point2)->float:
        return math.sqrt(((point1[0] - point2[0])**2)+(point1[1]-point2[1])**2)
        
    def update(self, data:dict)->None:
        self.__output = self.track_output_state_update(data)
        self.__output['frame_number'] = self.__frame_count
        self.__frame_count += 1

    def write_to_kafka(self):
        if self.__output is not None:
            self.__kafka_producer.send_message(self.__output_topic, json.dumps(self.__output))
            
            event = {}
            event["event_name"] = "update_untracked_ids"
            event["event_data"] = {"untracked_ids":self._tracked_ids}
            self.__kafka_producer.send_message('tracker-gui-events', json.dumps(event))

    def write_to_file(self):
        if self.__output is not None:
            with open(self.__output_dir / Path(f'track_data_{time.time()}.json'), 'w') as fp:
                json.dump(self.__output, fp)
        