import os
import random
import colorsys
from PIL import Image
import threading

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

# 이미지 저장 기본 경로 및 이미지 설정
output_dir = os.path.join(base_dir, "synthetic_color_images")
img_size = (224, 224)
num_images_per_class = 500

# HSV 기준값과 증강 범위 설정s
# 색상(Hue), 채도(Saturation), 명도(Value), h, s, v 모두 [0,1] 범위
# h_range, s_range, v_range는 각각 ±범위

color_specs = {
    'red':    {'h': 0  /360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.03, 's_range': 0.6,  'v_range': 0.5},
    'orange': {'h': 30 /360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.02, 's_range': 0.6,  'v_range': 0.5},
    'yellow': {'h': 60 /360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.02, 's_range': 0.6,  'v_range': 0.5},
    'green':  {'h': 120/360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.06, 's_range': 0.6,  'v_range': 0.5},
    'blue':   {'h': 210/360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.06, 's_range': 0.6,  'v_range': 0.5},
    'navy':   {'h': 240/360.0, 's': 1.0, 'v': 0.4, 'h_range': 0.02, 's_range': 0.6,  'v_range': 0.1},
    'purple': {'h': 280/360.0, 's': 1.0, 'v': 1.0, 'h_range': 0.03, 's_range': 0.6,  'v_range': 0.5},
    'black':  {'h': 0  /360.0, 's': 0.0, 'v': 0.0, 'h_range': 1.0,  's_range': 0.04, 'v_range': 0.3},
    'gray':   {'h': 0  /360.0, 's': 0.0, 'v': 0.5, 'h_range': 1.0,  's_range': 0.04, 'v_range': 0.2},
    'white':  {'h': 0  /360.0, 's': 0.0, 'v': 1.0, 'h_range': 1.0,  's_range': 0.02, 'v_range': 0.1}
}

def random_value(base, var_range):
    """
    base를 중심으로 ±var_range 내에서 무작위 값을 반환합니다.
    반환 값은 0과 1 사이로 클램핑됩니다.
    """
    low = max(0, base - var_range)
    high = min(1, base + var_range)
    return random.uniform(low, high)

def generate_images_for_color(color_name, spec):
    """
    지정된 색상에 대해 num_images_per_class 장의 이미지를 생성하여 저장합니다.
    """
    # 색상별 하위 디렉토리 생성 (예: synthetic_color_images/red)
    class_dir = os.path.join(output_dir, color_name)
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(num_images_per_class):
        # HSV 각 요소에 대해 무작위 값 생성 (증강 효과)
        h = random_value(spec['h'], spec['h_range'])
        s = random_value(spec['s'], spec['s_range'])
        v = random_value(spec['v'], spec['v_range'])
        
        # HSV에서 RGB로 변환 (colorsys는 h, s, v가 [0,1] 범위라고 가정)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # 0~255 정수값으로 변환
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        
        # 지정된 색상으로 이미지 생성
        img = Image.new('RGB', img_size, (r, g, b))
        
        # 파일명 지정 후 이미지 저장 (예: synthetic_color_images/red/red_01.png)
        img_filename = os.path.join(class_dir, f"{color_name}_{i+1:02d}.png")
        img.save(img_filename)
    
    print(f"{color_name} 이미지 생성 완료!")

def main():
    # 전체 결과를 저장할 기본 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    threads = []
    # 각 색상 클래스마다 별도의 스레드 생성
    for color_name, spec in color_specs.items():
        thread = threading.Thread(target=generate_images_for_color, args=(color_name, spec))
        threads.append(thread)
        thread.start()
    
    # 모든 스레드가 완료될 때까지 대기
    for thread in threads:
        thread.join()
    
    print("모든 합성 이미지 생성 완료!")

if __name__ == '__main__':
    main()
