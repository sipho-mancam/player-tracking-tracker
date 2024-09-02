import os
from pathlib import Path


__BASE_DIR__ = Path(r"./src").resolve()
__TRACKING_DATA_DIR__ = (__BASE_DIR__ / Path(r'tracking_data_files')).resolve()
__KAFKA_CONFIG__ = (__BASE_DIR__ / Path(r'cfg/tracking_core_kafka_config.ini')).resolve()
__CALIBRATION_CFG_DIR__ = (__BASE_DIR__ / Path(r'calibration')).resolve()
__MODELS_DIR__ = (__BASE_DIR__ / Path(r'model/weights')).resolve()
__ENGINE_FILE__ = (__BASE_DIR__ / Path(r'model/weights/best.engine')).resolve()
__VIDEO_REC_OUTPUT_DIR__ = Path(r"E:\Tracking Footage")
__WHITE_BG__ = (__BASE_DIR__ / Path(r"../assets/white_bg.jpg")).resolve()
__ASSETS_DIR__ =  (__BASE_DIR__ / Path(r"../assets")).resolve()
__MAP_BG_PATH__ = (__ASSETS_DIR__ / Path("soccer_pitch_poles.png")).resolve()

# print(os.path.exists(__MAP_BG_PATH__), __MAP_BG_PATH__.as_posix())       