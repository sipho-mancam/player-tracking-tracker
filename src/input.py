
"""
The input class does a few things,
1. Get 3 Camera Detections Data
"""
import time
import threading
import confluent_kafka
from confluent_kafka import Consumer, KafkaException
import json
from pprint import pprint
from pre_transform import PreDetectionsTransform
from coordinate_transforms import convert_box_2_points

class KafkaConsumer:
    def __init__(self, brokers, group_id, topic):
        self.brokers = brokers
        self.group_id = group_id
        self.topic = topic
        self.run = True
        self.message = None
        self.cv = threading.Condition()

        self.conf = {
            'bootstrap.servers': brokers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }

        self.consumer = Consumer(self.conf)
        self.consumer.subscribe([topic])

    def start(self):
        self.consumer_thread = threading.Thread(target=self.consume)
        self.consumer_thread.start()

    def consume(self):
        while self.run:
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == confluent_kafka.KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            else:
                with self.cv:
                    self.message = msg.value().decode('utf-8')
                    self.cv.notify_all()

    def wait_for_message(self):
        with self.cv:
            self.cv.wait_for(lambda: self.message is not None)
            # print(f"Received message: {self.message}")
            temp = self.message
            self.message = None
            return temp

    def stop(self):
        self.run = False
        self.consumer.close()


class InputData:
    def __init__(self, broker="172.21.243.238:9092", topic = "kit-detector-topic", group_id = "tracking_core_consumer_1") -> None:
        self.__id = 0
        self.__kafka_consumer = KafkaConsumer(broker, group_id, topic)
        self.__kafka_consumer.start()
    
    def stop(self):
        self.__kafka_consumer.stop()

    def wait_for_data(self)->list[list[dict]]:
        
        data = self.__kafka_consumer.wait_for_message()

        # pass the received data to a dictionary to be used inside the program
        data = json.loads(data)
        #  Perform the format transform (x_c, y_c, width, height) -> (x1, y1, x2, y2)
        PreDetectionsTransform.xywh2xyxy(data)
        res_list = PreDetectionsTransform.cams2list(data)
        # Perform the coordinate transform (Turn the box to a point for every detection and throw away all the other data we are not using.)
        for cam_data in res_list:
            convert_box_2_points(cam_data)
        return res_list 

    


if __name__ == "__main__":
    try:
        input_data = InputData()
        input_data
        while True:
            start_time = time.time()
            data = input_data.wait_for_data()
            # print(data)
            end_time = time.time()
            print(f"Waiting time is: {round((end_time-start_time)*1e3)} ms")
    
    except KeyboardInterrupt as ke:
        input_data.stop()

