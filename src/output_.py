from kafka import KProducer
import json
import time
from pathlib import Path
from cfg.paths_config import __BASE_DIR__, __TRACKING_DATA_DIR__, __KAFKA_CONFIG__
from pprint import pprint



class DetectionsOutput:
    def __init__(self)->None:
        self.__detections = None
        self.__output = None
        self.__kafka_producer = KProducer(__KAFKA_CONFIG__)
        self.__output_dir = __TRACKING_DATA_DIR__
        self.__frame_count = 0

    def update(self, data:dict)->None:
        self.__output = data
        self.__output['frame_number'] = self.__frame_count
        self.__frame_count += 1

    def write_to_kafka(self):
        if self.__output is not None:
            # pprint(self.__output)
            self.__kafka_producer.send_message('tracking-core', json.dumps(self.__output))

    def write_to_file(self):
        if self.__output is not None:
            
            with open(self.__output_dir / Path(f'track_data_{time.time()}.json'), 'w') as fp:
                json.dump(self.__output, fp)
        