import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
 
import uuid
import json
import time
import requests
import io
import os
import platform
import configparser


class LPBboxDetectionModel:
    MODEL_PATH='./data/lp_bbox.tflite'
    IMG_HEIGHT=192
    IMG_WIDTH=256
    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
        }[platform.system()]

    def __init__(self):
        model_file, *device = self.MODEL_PATH.split('@')
        self.interpreter = tflite.Interpreter(model_path=model_file, experimental_delegates=[tflite.load_delegate(self.EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def frame2input(self, frame):   # (__x__x3)->(192x256x3)->(1x192x256x3) - normalization
        cvt = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        cvt = np.expand_dims(cvt, axis=0)
        cvt = np.float32(cvt)
        cvt = cvt[..., ::-1]
        cvt /= 255.0
        return cvt

    def get_bbox(self, frame):  # ipnut:(1x192x256x3), output:(1x3024x6)->(2x5)
        input = self.frame2input(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_result = np.expand_dims(output_data[0], axis=0)

        max_conf = detection_result[0][0][4]
        
        idx = 0
        for i in range(3024):
            if max_conf < detection_result[0][i][4]:
                max_conf = detection_result[0][i][4]
                idx = i
        detection_result[0][idx][0] *= 256
        detection_result[0][idx][1] *= 192
        detection_result[0][idx][2] *= 256
        detection_result[0][idx][3] *= 192

        x1 = detection_result[0][idx][0] - detection_result[0][idx][2]/2
        y1 = detection_result[0][idx][1] - detection_result[0][idx][3]/2
        x2 = detection_result[0][idx][0] + detection_result[0][idx][2]/2
        y2 = detection_result[0][idx][1] + detection_result[0][idx][3]/2

        ret = np.zeros((2, 5), dtype=np.float32)
        ret[0][0] = 0
        ret[0][1] = 0
        ret[0][2] = 0
        ret[0][3] = 0
        ret[0][4] = 0
        ret[1][4] = max_conf

        gain_w = float(256.0 / frame.shape[1])
        gain_h = float(192.0 / frame.shape[0])

        pad = []
        pad.append(0)
        pad.append(0)
        a =  ((x1-pad[0]) / gain_w )
        b = ((y1-pad[1]) / gain_h)
        c = ((x2-pad[0]) / gain_w)
        d = ((y2-pad[1]) / gain_h)
        x_c = (a+c)/2
        y_c = (b+d)/2
        w = c-a
        h = d-b

        ret[1][0] = x_c/frame.shape[1]
        ret[1][1] = y_c/frame.shape[0]
        ret[1][2] = w/frame.shape[1]
        ret[1][3] = h/frame.shape[0]

        return ret

class LPVertexDetectionModel:
    MODEL_PATH='./data/lp_vertex.tflite'
    IMG_HEIGHT=128
    IMG_WIDTH=128
    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
        }[platform.system()]

    def __init__(self):
        model_file, *device = self.MODEL_PATH.split('@')
        self.interpreter = tflite.Interpreter(model_path=model_file, experimental_delegates=[tflite.load_delegate(self.EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def frame2input(self, frame):   # (128x128x3)->(1x3x128x128) - normalization
        cvt = cv2.resize(frame, (self.IMG_WIDTH, self.IMG_HEIGHT))
        cvt = np.moveaxis(cvt, source=-1, destination=0)
        cvt = np.expand_dims(cvt, axis=0)
        cvt = np.float32(cvt)
        cvt = cvt[..., ::-1]
        cvt /= 255.0
        return cvt

    def get_coordinate(self, frame):    # input:(1x3x128x128), output:(1x8)->(8)
        input = self.frame2input(frame)

        self.interpreter.set_tensor(self.input_details[0]['index'], input)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        detection_result = np.squeeze(output_data)

        return detection_result

class LPOcrModel:

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')
        self.api_url = self.config.get('OCR', 'API_URL')
        self.secret_key = self.config.get('OCR', 'SECRET_KEY')

        self.request_json = {'images': [{'format': 'jpg',
                                        'name': 'demo'
                                    }],
                            'requestId': str(uuid.uuid4()),
                            'version': 'V2',
                            'timestamp': int(round(time.time() * 1000))
                        }
        
        self.payload = {'message': json.dumps(self.request_json).encode('UTF-8')}
        
        self.headers = {
            'X-OCR-SECRET': self.secret_key,
        }

    def get_image_handler(self, frame):
        ret, img_encode = cv2.imencode('.jpg', frame)
        str_encode = img_encode.tobytes()
        img_byteio = io.BytesIO(str_encode)
        img_byteio.name = './demo.png'
        reader = io.BufferedReader(img_byteio)
        return reader

    def get_string(self, frame):
        files = self.get_image_handler(frame)
        response = requests.request("POST", self.api_url, headers=self.headers, data=self.payload, files=[('file', files)])
        result = response.json()
        return result
