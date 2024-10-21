from coordinate_transforms import Transformer

class SpaceTransformer:
    def __init__(self, width:int, height:int, pitch_coord:list[dict]) -> None:
        self.__width = width
        self.__height = height
        self.__pitch_coordinates = pitch_coord
        self.__transformers = [ Transformer(width, height, pitch_coord[0], 0), # Cam 1 Transformer
                                Transformer(width, height, pitch_coord[1], 1), # Cam 2 Transformer
                                Transformer(width, height, pitch_coord[2], 2)] # Cam 3 Transformer
        
    def get_transformer(self, index:int)->Transformer:
        return self.__transformers[index]
        
    
    def apply_transform(self, cams_detections_lists:list[list])->list[list]:
        """
        1. This function receives detections in the format
            [
                [det, det, det],
                [det, det, det],
                [det, det, det]
            ]
            Each list corresponds to a camera input streams from the video input system
            It applies perspective transforms to the detections coordinates and returns 
        """
        results = []
        for idx, cam_detections in enumerate(cams_detections_lists):
            res = self.__transformers[idx].transform(cam_detections)    
            results.append(res)
        return results