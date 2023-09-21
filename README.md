#라즈베리파이에서 차량번호인식을 이용한 주차차단기 프로젝트

### settings.ini
```
[SERVER]
URL = <your host>
CHECK_VALIDATION = <vehicle's number check api>
UPLOAD = <upload api>

[OCR]
API_URL = <your naver clova ocr api url>
SECRET_KEY = <your secret key>
```

### requirements
- python: 3.9
- raspberrypi 3 B+ (aarch64)

```
sudo apt-get update && sudo apt-get install pigpio python-pigpio python3-pigpio
sudo pigpiod

git clone [repository]
cd [repository]

python3.9 -m pip install virtualenv
python3.9 -m virtualenv venv
source venv/bin/activate

pip install pigpio
pip install opencv-python
pip install Pillow
pip install requests requests_toolbet
pip install tensorflow tflite-runtime

python Main.py
```

### fonts
```
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
echo "set encoding=utf-8" >> ~/.vimrc
sudo apt-get install fonts-nanum
sudo dpkg-reconfigure locales
```
