from dataloader import DataLoader
from input import InputData, FileInputData
from transformer import SpaceTransformer
from space_merger import SpaceMerger
from pprint import pprint
from botsort_tracker import track2, track_raw
from output_ import DetectionsOutput
import time
from pathlib import Path
import json

running = True
TESTING = False


def initialize():
    status = 0
    # Request init config and allocate start up resources
    return status

def __load_config__(dataPath: Path = Path(r"C:\ProgramData\Player Tracking Software\config.json"))->dict:
    with open(dataPath, 'r') as fp:
        return json.load(fp)
    return {}

def main_loop():
    try:
        global running

        config = __load_config__()
        if len(config) == 0:
            raise FileNotFoundError("Config file not found.")
        
        service_name = "tracking_core"
        topic = config[service_name]["kafka"]["topic"]
        group_id = config[service_name]["kafka"]["group_id"]
        broker = config["system_settings"]["kafka_server_address"]

        # Output
        output = DetectionsOutput(broker, config[service_name]["output_topic"])

        #input data from external source
        if not config[service_name]["testing"]:
            input_data = InputData(broker, topic, group_id)
        else:
             DATA_SOURCE_DIR = Path(config[service_name]["testing_config"]["data_directory"])
             input_data = FileInputData(DATA_SOURCE_DIR)

        
        if config[service_name]["testing"]:
            output = DetectionsOutput(broker, config[service_name]["output_topic"])
            while True:
                start_time = time.time()
                data = input_data.wait_for_data() 
                output.update(data)
                output.write_to_kafka()
                time.sleep(0.12)

                end_time = time.time()
                print(f"Processing Time: {round(1e3*(end_time - start_time))} ms")
        

        # If calibration data doesn't exist, send a message to the UI and wait till it exist.
        config_data = DataLoader().load_config_data()

        # Transformer
        f_width = config["system_settings"]["frame_width"]
        f_height  = config["system_settings"]["frame_height"]
        space_transformer = SpaceTransformer(f_width, f_height, config_data['cams_config'])

        # Space Merger
        # transformer = space_transformer.get_transformer(1)
        # mini_boundary = transformer.get_mini_boudary()
        # main_boundary = transformer.getDstPts()
        mini_boundary = []
        main_boundary = []
        space_merger = SpaceMerger(main_boundary, mini_boundary)

        while running:
            start_time = time.time()
            # This is raw detections json converted data coming from either the detector or the kit-detector, depending on the sport.
            in_topic, data  = input_data.wait_for_data()

            if topic == in_topic:
                transformed_data = space_transformer.apply_transform(data)
                merged_data = space_merger.merge(transformed_data)
                # Apply the tracking algorithm here ....
                map_data, tracked_data = track2(merged_data)

                # Send data tagged with Tracking IDs here ....
                output.update(tracked_data)
                output.write_to_kafka()    
                # output.write_to_file()    
                end_time = time.time()
                print(f"Processing Time: {round(1e3*(end_time - start_time))} ms")
            
        input_data.stop()
        return 0
    except KeyboardInterrupt as ke:
        running = False
        input_data.stop()


def clean_up():
    status = 0


    return status

        

if __name__ == "__main__":
        main_loop()
    