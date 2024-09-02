

class PreDetectionsTransform:
    def __init__(self)->None:
        pass

    @staticmethod
    def xywh2xyxy(cams_detections_object:dict)->dict:
        cams_data = cams_detections_object['cams']
        keys = cams_data.keys()
        for camera in keys:
            c_data = cams_data[camera]
            for det in c_data['detections']:               
                bbox = det.get('bbox')
                x = bbox['x']
                y = bbox['y']
                width  = bbox['width']
                height = bbox['height']
                x2 = x + width
                y2  = y + height
                bbox['x2'] = x2
                bbox['y2'] = y2
                bbox["x1"] = x
                bbox['y1'] = y
                det['box'] = {"x1":x, "y1":y, "x2":x2, "y2":y2}

    @staticmethod
    def cams2list(cams_detections_object:dict)->list:
        cams_data = cams_detections_object['cams']
        keys = cams_data.keys()
        result = []
        for camera in keys:
            c_data = cams_data[camera]
            result.append(c_data['detections'])

        return result
