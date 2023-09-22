import cv2
import time, datetime
import numpy as np
import re
from PIL import ImageFont, ImageDraw, Image
import threading
import pigpio

from DetectionModel import *
from HttpService import HttpService

class Main:
    def __init__(self):
        
        # Initialize detection model
        self.lp_bbox_model = LPBboxDetectionModel()
        self.lp_vertex_model = LPVertexDetectionModel()
        self.lp_ocr_model = LPOcrModel()
        self.http_service = HttpService()
        
        # cropped frame & detected string
        self.lp_frame = cv2.imread('./data/no_lp_frame.png')
        self.lp_str = ""
        
        # Initialize false_latency timer
        self.latency_start = time.time()
        self.latency = 0

        # valid_True count: probabilistic counting
        self.valid_count = 0
        self.check_num = 1
        
        # Thread
        self.barrier_control_thread_join_flag = False
        self.lpr_thread_start = False
        
        # 
        self.roi = []
        self.valid = False
        
    def lpr_thread(self, frame):
        self.lpr_thread_start = True
        self.result_frame = frame.copy()
        '''
        1. License plate ROI inference
        '''
        detected_bbox = self.lp_bbox_model.get_bbox(frame)
        bbox_score = detected_bbox[1][4]
        
        # 1-1. if the recognition rate is less than 50 percent
        
        if  bbox_score < 0.5:
            # Initailize detected results, if the false_latency is over than 2 sec.
            self.latency = time.time() - self.latency_start
            if self.latency > 1:
                self.is_valid = False
                self.valid_count = 0
                self.lp_frame = cv2.imread('./data/no_lp_frame.png')
                self.lp_str = ""
                self.roi = []
                self.valid = False
        else:
            # 1-2. if the recognition rate is more than 50 percent            
            print("[LPR]: The ROI has been succesfully detected!")
            if self.valid_count < self.check_num:
                self.latency_start = time.time()    # reset the false_latency timer
                
                w = frame.shape[1]
                h = frame.shape[0]

                bbox_coord = []
                bbox_x_center= detected_bbox[1][0]
                bbox_y_center = detected_bbox[1][1]
                bbox_width = detected_bbox[1][2]
                bbox_height = detected_bbox[1][3]

                bbox_coord.append(bbox_x_center - 0.5*bbox_width)
                bbox_coord.append(bbox_y_center - 0.5*bbox_height)
                bbox_coord.append(bbox_x_center + 0.5*bbox_width)
                bbox_coord.append(bbox_y_center + 0.5*bbox_height)

                w_ = int(0.01 * w)
                h_ = int(0.01 * h)
                
                pt1_x = max(int(w*bbox_coord[0] - w_), 0)
                pt1_y = max(int(h*bbox_coord[1] - h_), 0)
                pt3_x = min(int(w*bbox_coord[2] - w_), w)
                pt3_y = min(int(h*bbox_coord[3] - h_), h)

                new_w = int(pt3_x - pt1_x)
                new_h = int(pt3_y - pt1_y)

                self.roi = [pt1_x, pt1_y, new_w, new_h]

                '''
                2. License plate Morphology 
                '''
                cropped_lp_frame = frame[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
                cropped_lp_frame = cv2.resize(cropped_lp_frame, (128, 128))
                detected_vertex = self.lp_vertex_model.get_coordinate(cropped_lp_frame) # get detected lp's vertex coordinate
                print("[LPR]: LP has been succesfully detected!")
                
                #sigmoid
                vertex_coord = []
                for i in range(8):
                    value = float(np.exp(-detected_vertex[i]) + 1)
                    value = 1 / value
                    value = value * 128
                    vertex_coord.append(int(value))       

                #perspective transformation
                startM = np.float32(
                            [[vertex_coord[2], vertex_coord[1]],
                            [vertex_coord[0], vertex_coord[3]],
                            [vertex_coord[6], vertex_coord[5]],
                            [vertex_coord[4], vertex_coord[7]]]
                        )
                endM = np.float32(
                            [[255, 0],
                            [0, 0],
                            [0, 127],
                            [255,127]]
                        )
                matrix = cv2.getPerspectiveTransform(startM, endM)
                warp_lp_frame = cv2.warpPerspective(cropped_lp_frame, matrix, (256, 128), flags=cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
                self.lp_frame = warp_lp_frame
                
                '''
                3. Character Recognition (NAVER CLOVA OCR API)
                '''
                detected_lp_str_json = self.lp_ocr_model.get_string(frame)
                str = ""
                if detected_lp_str_json:    # recognize only the charaters within the bbox
                    for field in detected_lp_str_json['images'][0]['fields']:
                        ver = field['boundingPoly']['vertices']
                        f1_x = ver[0]['x']
                        f1_y = ver[0]['y']
                        if f1_x > self.roi[0] and f1_x < self.roi[0]+self.roi[2] and f1_y > self.roi[1] and f1_y < self.roi[1]+self.roi[3]:
                            str += field['inferText']
                    str = str.replace(" " , "")
                    self.lp_str = re.sub('[^ㄱ-힣0-9]', '', str)
                print("[LPR]: LP-string has been succesfully detected!")
                
                '''
                4. Check if the vehicle exists
                '''
                self.is_valid = self.http_service.check_validation(self.lp_str)
                if self.is_valid:   # Increase by 1, if the vehicle exists. 
                    self.valid_count = self.valid_count + 1
                
                print(f"[v-HTTP]: {self.lp_str} is {'a registered vehicle!' if self.is_valid else 'an unregistered vehicle!'} [{self.valid_count}]")
                
                '''
                4-1. Save result frame
                '''
                if self.valid_count == self.check_num:
                    self.http_service.upload_lp(self.lp_str, self.lp_frame)
        
        self.lpr_thread_start = False
    
    def barrier_control_thread(self):     # open/close the barrier
        servo_pin = 17

        red_pin = 16
        green_pin = 20
        blue_pin = 21

        pi = pigpio.pi()
        LED_MIN_DUTY = 0
        LED_MAX_DUTY = 255
        SERVO_MIN_DUTY = 500
        SERVO_MAX_DUTY = 2500
        
        def set_color(r, g, b):
            red_duty = int((r/255)*(LED_MAX_DUTY-LED_MIN_DUTY)+LED_MIN_DUTY)
            green_duty = int((g/255)*(LED_MAX_DUTY-LED_MIN_DUTY)+LED_MIN_DUTY)
            blue_duty = int((b/255)*(LED_MAX_DUTY-LED_MIN_DUTY)+LED_MIN_DUTY)
            
            pi.set_PWM_dutycycle(red_pin, red_duty)
            pi.set_PWM_dutycycle(green_pin, green_duty)
            pi.set_PWM_dutycycle(blue_pin, blue_duty)

        def set_angle(angle):
            duty = int((angle/180)*(SERVO_MAX_DUTY-SERVO_MIN_DUTY)+SERVO_MIN_DUTY)
            pi.set_servo_pulsewidth(servo_pin, duty)

        try:
            open_flag = True
            close_flag = True
            set_angle(10)
            set_color(0, 0, 0)
            while True:
                if self.barrier_control_thread_join_flag:
                    raise KeyboardInterrupt("[SENSOR]: terminated!")

                if self.valid_count >= self.check_num:    # Open the door, if the vehicle exists
                    close_flag = True
                    set_color(0, 255, 0)

                    if open_flag:
                        print("[SENSOR]: the barrier is open!")
                        set_angle(85)
                        time.sleep(0.7)
                        pi.set_servo_pulsewidth(servo_pin, 0)
                        open_flag = False

                    time.sleep(1)
                else:   # Close the door, if the vehicle is not found in the DB
                    open_flag = True
                    set_color(0, 0, 0)
                    
                    if close_flag:
                        print("[SENSOR]: the barrier is close!")
                        set_angle(10)
                        time.sleep(0.7)
                        pi.set_servo_pulsewidth(servo_pin, 0)
                        close_flag = False

                    time.sleep(1)

        except KeyboardInterrupt:
            set_color(0, 0, 0)
            set_angle(10)
            time.sleep(0.7)
            pi.set_servo_pulsewidth(servo_pin, 0)
            pi.stop()
    
    
    def run(self):
        # start barrier control thread
        threading.Thread(target=self.barrier_control_thread).start()
        
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("LPR Monitor", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("LPR Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while(True):
            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                           
            
            if not self.lpr_thread_start:
                threading.Thread(target=self.lpr_thread, args=(frame, )).start()
                
            '''
            Layout
            '''
            result = frame.copy()
            
            # Draw ROI
            if self.roi:
                cv2.rectangle(result, (self.roi[0], self.roi[1]), (self.roi[2]+self.roi[0], self.roi[3]+self.roi[1]), (0, 255, 0), 1)
            
            result[:self.lp_frame.shape[0], :self.lp_frame.shape[1]] = self.lp_frame
            if self.lp_str:
                font = ImageFont.truetype('fonts/MALGUN.TTF', 40)
                if self.valid:
                    font_color = (0,255,0)
                else:
                    font_color = (0,0,255)
                result = Image.fromarray(result)
                ImageDraw.Draw(result).text((10, 128), self.lp_str, font_color, font=font)
                result = np.array(result)
            cv2.imshow('LPR Monitor', result)

        print("[SYSTEM]: terminating...")
        self.barrier_control_thread_join_flag = True
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main = Main()
    print("[SYSTEM]: starting...")
    main.run()
