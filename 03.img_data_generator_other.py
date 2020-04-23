import os
import random
import pathlib
import shutil
import glob
import cv2
import numpy as np

def load_name_images(image_path_pattern):
    name_images = []
    # 지정한 경로에서 이미지 파일 취득
    image_paths = glob.glob(image_path_pattern)
    # 파일별로 읽기
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # 파일 경로
        fullpath = str(path.resolve())
        print(f"이미지 파일(절대 경로):{fullpath}")
        # 파일명
        filename = path.name
        print(f"이미지 파일(파일명):{filename}")
        # 이미지 읽기
        image = cv2.imread(fullpath)
        if image is None:
            print(f"이미지 파일[{fullpath}]을 읽을 수가 없습니다.")
            continue
        name_images.append((filename, image))
    return name_images

def scratch_image(image, use_median=True, use_threshold=True, use_gaussian=True, use_bilateral=True):
    # 어떤 증가규칙을 사용할 것인지 설정 (Flip or 밝기 or 화소 ...)
    methods = [use_median, use_threshold, use_gaussian, use_bilateral]
    # 흐린 필터 작성
    filter1 = np.ones((3, 3))
    # 오리지날 이미지를 배열로 저장
    images = [image] # 배열로 만들어서 저장

    # 증가 규칙을 위한 함수
    scratch = np.array([
        # 또 다른 블러 처리
        lambda x: cv2.medianBlur(x,5),
        # 밝기 처리
        # 람다식을 통해 랜덤으로 크기등을 다르게 할것
        lambda x: cv2.threshold(x, 100, 255, cv2.THRESH_TOZERO)[1],
        # 블러 처리
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        # Bilateral 블러 처리
        lambda x: cv2.bilateralFilter(x,9,75,75),
    ])

    # 이미지 증가
    created_images = lambda f, img : np.r_[img, [f(i) for i in img]] 
    for func in scratch[methods]:
        images = created_images(func, images)
    
    return images

def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Test Image Directory
TEST_IMAGE_PATH = "./test_image"
# Face Image Directory
IMAGE_PATH_PATTERN = "./fontsss/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./fonts_inc"

def main():
    print("===================================================================")
    print("이미지 증가를 위한 OpenCV 이용")
    print("지정한 이미지 파일의 수를 증가 (Flip, 임계값 등의 작업으로 8배 증가)")
    print("===================================================================")

    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_IMAGE_DIR):
        os.mkdir(OUTPUT_IMAGE_DIR)
    # 디렉토리 내 파일 제거
    delete_dir(OUTPUT_IMAGE_DIR, False)

    # 디렉토리 작성
    if not os.path.isdir(TEST_IMAGE_PATH):
        os.mkdir(TEST_IMAGE_PATH)
    # 디렉토리 내 파일 제거
    delete_dir(TEST_IMAGE_PATH, False)

    # 대상 이미지에 20% 정보를 테스트용 파일로 구분
    image_files = glob.glob(IMAGE_PATH_PATTERN)
    random.shuffle(image_files)
    for i in range(len(image_files) // 5):
        shutil.move(str(image_files[i]), TEST_IMAGE_PATH)


    # 대상 이미지 읽기
    name_images = load_name_images(IMAGE_PATH_PATTERN)

    # 대상 이미지 별로 증가 작업
    for name_image in name_images:
        filename, extension = os.path.splitext(name_image[0])
        image = name_image[1]
        # 이미지 수 증가 함수 호출
        scratch_face_image = scratch_image(image)
        #대상 이미지 저장
        for (idx, image) in enumerate(scratch_face_image):
            output_path = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_{str(idx)}{extension}")
            print(f"출력 파일(절대 경로):{output_path}")
            cv2.imwrite(output_path, image)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()