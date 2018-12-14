import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argumnet('image_file_list')
parser.add_argument('image_dir')
args = parser.parse_args()


with open(args.image_file_list, 'r') as image_file_list:
    image_file_list = image_file_list.readlines()

for image_file in image_file_list:

    image_path = os.path.join(args.image_dir, image_file)
    image = cv2.imread(args.image, -1)
    print(image.shape)
    if image.shape[0] != 227 or image.shape[1] != 277:
	print(image_file, 'the shape is not correct!!')


