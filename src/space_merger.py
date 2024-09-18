from pprint import pprint
import math


class SpaceMerger:
    def __init__(self, main_boundary:list, inner_boundary:list)->None:
        self.__stream_results = [] # a list of tuples containing the current results
        self.__left_wing = 7/20
        self.__right_wing = 7/20
        self.__middle = 6/20
        self.__m_overlap = (1/20)*0.3
        self.__is_init = False

        # Overlap settings
        self.__overlaping_points = []
        self.__frame_width = 1
        self.__frame_height = 1
        self.__one_frame_width = 1
        self.__m_left_overlap = 0
        self.__m_right_overlap = 0
        self.__main_boundary = main_boundary # These relate to cam 2's transformer.
        self.__mini_boundary = inner_boundary # Along with this one

        self.init()


    def init(self)->None:
        self.__m_left_overlap = (self.__mini_boundary[0]['coordinates'][0] - self.__main_boundary[0]['coordinates'][0])*self.__left_wing
        self.__m_right_overlap = (self.__main_boundary[1]['coordinates'][0]- self.__mini_boundary[1]['coordinates'][0])*self.__left_wing

    def is_init(self)->bool:
        return self.__is_init

    def align_frame_points(self, dts:list[dict], stream_id)->list[dict]:
        for det in dts:
            det = self.align_detection(det, stream_id)
        return dts

    def align_detection(self, det, stream_id):
        x, y = det['coordinates']
        x = self.align_x(x, stream_id)
        det['coordinates'] = (x, y)
        return det

    def align_x(self, x_coord, stream_id):
        if stream_id == 0:
            x_scaled = x_coord * self.__left_wing
            return x_scaled
        elif stream_id ==1:
            x_scaled = x_coord * (self.__middle + self.__m_left_overlap + self.__m_right_overlap)
            x_shifted = x_scaled + (self.__left_wing - self.__m_left_overlap)
            return x_shifted
        elif stream_id == 2:
            x_scaled = x_coord * self.__right_wing
            x_shifted = x_scaled + (self.__left_wing + self.__middle)
            return x_shifted
        return x_coord

    def overlap_has_child(self, det, cam_dets:list[dict])->dict:
        '''
        1. Look for the child associated with the overlap center detection, if any.
        2. return that child or not if no child exists

        Child finding Algorithm
        1. Check if the current detection is y units below the center detection point
        2. check if the current detections x offset lives with 0 < x < dist.
        3. if both the conditions are met, we assign it and mark it. and return
        4. The child must be in the overlapped region as well
        '''
        child_dist = 0.035
        det['has_child'] = False
        center_coord = det.get('coordinates')
        closest_distance = math.inf
        child_index = math.inf

        for idx, c in enumerate(cam_dets):
            coord =  c.get('coordinates')
            if not c.get('is_overlap') and c.get('is_child'):
                continue

            dist = math.sqrt(((center_coord[0]-coord[0])**2)+((center_coord[1] - coord[1])**2))
            if dist <= child_dist:
                distance = dist
                if distance < closest_distance:
                    closest_distance = distance
                    child_index = idx

        if closest_distance is not math.inf:
            child = cam_dets.pop(child_index)
            child['is_child'] = True
            child['marker_id'] = id(det)
            det['child'] = child
            det['has_child'] = True
            det['marker_id'] =  id(det)
            del child
        return det

    def is_in_overlap(self, det):
        coord = det.get('coordinates')
        det['is_overlap'] = False
        # This is the left overlap region
        error_constant = 0.01
        overlap_start = self.__left_wing - self.__m_left_overlap - error_constant
        overlap_end  = self.__left_wing + error_constant
        if coord[0] >=overlap_start and coord[0] <=overlap_end:
            det['is_overlap'] = True
            det['overlap_side'] = 0 # cam 0
            # print(det, overlap_start, overlap_end)
            return det

        # check overlap in the right region
        error_constant_2 = 0.01
        overlap_start = (self.__left_wing + self.__middle) - error_constant_2
        overlap_end  = (self.__left_wing + self.__middle) +  self.__m_right_overlap + error_constant
        if coord[0] >= overlap_start and coord[0] <= overlap_end:
            det['is_overlap'] = True
            det['overlap_side'] = 2 # cam 3
            # print(det, overlap_start, overlap_end)
            return det

        return det

    # stream1 = 0-30%; stream2 = 31-70%; stream3 = 71-100%
    def merge(self, cams_detections:list[list[dict]])->list[dict]:
        unified_space = []
        width = 2590
        cam1_space = []
        cam2_space = []
        cam3_space = []
        offset = 0.004
        for idx, dets_group in enumerate(cams_detections):
           for det in dets_group:
                coord = det['coordinates']
               # left wing
                if idx == 0:
                    
                    if (coord[0] >= 0 and coord[0]<=1) and (coord[1]>=0 and coord[1] <=1):
                        x_scaled = self.align_x(coord[0], idx)#coord[0] * self.__left_wing
                        det['coordinates'] = (x_scaled, coord[1])
                        det = self.is_in_overlap(det)
                        det['camera'] = idx
                        cam1_space.append(det)
                # middle
                elif idx == 1:
                    if (coord[0]>=0 and coord[0]<=1-offset) and (coord[1]>=0 and coord[1]<=1):
                        x_shifted = self.align_x(coord[0], idx)
                        det['coordinates'] = (x_shifted, coord[1])
                        det = self.is_in_overlap(det)
                        det['box']['x1'] += width
                        det['box']['x2'] += width
                        det['camera'] = idx
                        cam2_space.append(det)

                # Right wing
                elif idx==2:
                    if (coord[0] >= 0 and coord[0]<=1) and (coord[1]>=0 and coord[1] <=1):
                        x_shifted = self.align_x(coord[0], idx)
                        det['coordinates'] =  (x_shifted, coord[1])
                        det = self.is_in_overlap(det)
                        det['box']['x1'] += 2*width
                        det['box']['x2'] += 2*width
                        det['camera'] = idx
                        cam3_space.append(det)

        for c_2_det in cam2_space:
            if c_2_det.get('is_overlap'):
                cam_side = c_2_det.get('overlap_side')
                if cam_side == 0:
                    c_2_det = self.overlap_has_child(c_2_det, cam1_space)
                elif cam_side == 2:
                    c_2_det = self.overlap_has_child(c_2_det, cam3_space)

        unified_space.extend(cam1_space)
        unified_space.extend(cam2_space)
        unified_space.extend(cam3_space)
        return unified_space


