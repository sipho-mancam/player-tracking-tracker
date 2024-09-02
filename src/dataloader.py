import os
from pathlib import Path
import json

from pprint import pprint


class DataLoader:
    def __init__(self) -> None:
        self.__root = Path(os.getcwd() + "/calib_data")    
        print(os.path.exists(self.__root), self.__root.as_posix())

    def load_config_data(self)->dict:
        """
        1. This function loads the transformer and space merger config data
        """
        result  = {}
        for file in os.scandir(self.__root):
            with open(file, 'r') as fp:
                data = json.load(fp)
                name = file.name.split('.')[0]
                result[name] = data
        result['cams_config'] = []
        for key in result:
            if 'cam' in key:
                result['cams_config'].append(result[key])
        return result

        

        


if __name__ == "__main__":
    loader  = DataLoader()

    pprint(loader.load_config_data())

    