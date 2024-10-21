from dataloader import DataLoader
from input import InputData
from transformer import SpaceTransformer
from space_merger import SpaceMerger
from pprint import pprint
from botsort_tracker import track2, track_raw
from output_ import DetectionsOutput
import time

running = True;

def initialize():
    status = 0
    # Request init config and allocate start up resources
    return status

def main_loop():
    try:
        global running
        #input data from external source
        input_data = InputData()

        # If calibration data doesn't exist, send a message to the UI and wait till it exist.
        config_data = DataLoader().load_config_data()

        # Transformer
        f_width = 2590
        f_height  = 1942
        space_transformer = SpaceTransformer(f_width, f_height, config_data['cams_config'])

        # Space Merger
        # transformer = space_transformer.get_transformer(1)
        # mini_boundary = transformer.get_mini_boudary()
        # main_boundary = transformer.getDstPts()
        mini_boundary = []
        main_boundary = []
        space_merger = SpaceMerger(main_boundary, mini_boundary)

        # Output
        output = DetectionsOutput()

        while running:
            start_time = time.time()
            # This is raw detections json converted data coming from either the detector or the kit-detector, depending on the sport.
            data  = input_data.wait_for_data()
              
            transformed_data = space_transformer.apply_transform(data)
            merged_data = space_merger.merge(transformed_data)
            # Apply the tracking algorithm here ....
            map_data, tracked_data = track2(merged_data)

            # Send data tagged with Tracking IDs here ....
            output.update(tracked_data)
            output.write_to_kafka()        
            end_time = time.time()
            
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
    