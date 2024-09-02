from configparser import ConfigParser
from confluent_kafka import Consumer, OFFSET_BEGINNING,  Producer
from threading import Thread, Event
import json
import pprint
import re

reset = False
def reset_offset(consumer, partitions):
    if reset:
        for p in partitions:
            p.offset = OFFSET_BEGINNING
        consumer.assign(partitions)

class KConsumer:
    def __init__(self, config_path:str):
        self.__config_path = config_path
        self.__config_parser = ConfigParser()
        self.__config = None
        self.__consumer = None
        self.__event = Event()
        self.__worker_thread = Thread(target=self.__run)
        self.__data_event = Event()
        self.__data_queue = [] # for all data
        self.__tracking_data_queue = []
        self.__clear_to_leave = Event()
        self.__init()

    def toggle_offsetReset(self):
        reset = not reset

    def __init(self):
        with open(self.__config_path) as fp:
            self.__config_parser.read_file(fp)
        self.__config = dict(self.__config_parser['default'])
        self.__config.update(self.__config_parser['consumer'])
         # Create Consumer instance
        self.__consumer = Consumer(self.__config)

    def __run(self):
        # this method runs the listener and stores the messages in a queue
        # Poll for new messages from Kafka and print them.
        try:
            while not self.__event.is_set():
                msg = self.__consumer.poll(0.1)
                if msg is None:
                   pass
                #    print("waiting ...")
                elif msg.error():
                    # you need to publish to the error log
                    print("ERROR: %s".format(msg.error()))
                else:
                    message = msg.value().decode('utf-8')
                    self.__data_queue.append(message)
                    self.__tracking_data_queue.append(message)
                    self.__data_event.set()

            print("Cleaning up and exiting ...")
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.__event.set()
            self.__clear_to_leave.set()
            print(e)
        finally:
            # Leave group and commit final offsets
            self.__consumer.close()
            self.__clear_to_leave.set()

    def start(self):
        #self.__run()
        try:
            # this will start the consumer listening procedure
            self.__worker_thread.start()
        except Exception as e:
            pass

    def stop(self):
        self.__event.set()
        self.__clear_to_leave.wait()


    def subscribe(self, topic):
        self.__consumer.subscribe([topic], on_assign=reset_offset)

    def waitForEvent(self):
        # this method will signal the main thread of data arrival for update
        self.__data_event.wait(1)
        self.__data_event.clear()

    def getTrackingData(self)->dict:
        # you can only call this method once.
        if len(self.__tracking_data_queue) <= 0:
            return None
        data_piece = self.__tracking_data_queue.pop(0)
        return data_piece



class KProducer:
    def __init__(self, config_file):
        self.config = ConfigParser()
        self.config.read(config_file)
        conf = {
            'bootstrap.servers': self.config.get('default', 'bootstrap.servers'),
            'client.id': 'tracking_core_producer'
        }
        self.producer = Producer(conf)

    def delivery_report(self, err, msg):
        # return
        if err is not None:
            print(f"Message delivery failed: {err} Message : {msg}")
        # else:
            # print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def send_message(self, topic, message):
        self.producer.produce(topic, message.encode('utf-8'),'tracking-data'.encode('utf-8'), callback=self.delivery_report)
        self.producer.poll(0)

    def close(self):
        self.producer.flush()