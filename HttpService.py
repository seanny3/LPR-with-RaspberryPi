import datetime
import os
import requests
import json
import cv2
import configparser
from requests_toolbelt import MultipartEncoder

class HttpService:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')
        self.url = self.config.get('SERVER', 'URL')
        self.headers = {'Content-type': 'application/json; charset=utf-8'}
    
    def check_validation(self, lp_str):
        #REST API
        api = self.config.get('SERVER', 'CHECK_VALIDATION')
        try:
            response = requests.post(self.url+api, headers=self.headers, data=json.dumps({"plateNumber": lp_str, "parkingLotId": 1}))
            response.raise_for_status()
            is_valid = json.loads(response.text)["valid"]
            
        except requests.exceptions.RequestException as e:
            print("[v-HTTP]: HTTP request is not possible at the moment!")
            is_valid = False
            
        return is_valid
    
    def send_image(self, lp_str, img):
        # save detected lp image
        file_name = f'{lp_str}_'+ datetime.datetime.now().strftime('%Y-%m-%d-%d-%H-%M-%S') + '.jpg'
        folder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
        file_dir = os.path.join(folder_dir, file_name)
        cv2.imwrite(file_dir, img)
        
        # REST API
        self.api = self.config.get('SERVER', 'SEND_IMAGE')
        files = {'image': (file_name, open(file_dir, 'rb'), 'multipart/form-data', {'Expires': '0'})}
        try:
            response = requests.post(self.url+api, files=files)
            response.raise_for_status()
            print('[i-HTTP]: The detected LP has been successfully transmitted!')
            
        except:
            print('[i-HTTP]: HTTP request is not possible at the moment!')
