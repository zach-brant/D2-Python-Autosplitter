import cv2, mss, time, os
import numpy as np
import datetime as dt
import pyautogui as pag
import logging as log

pag.FAILSAFE = False

time_var = 0

sift = cv2.SIFT_create()

def get_sift_features(img):
    # Convert the region to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to get binary image
    ret, img_bin = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    # apply some blur
    img_blur = cv2.GaussianBlur(img_bin, (5,5), 0)
    # apply a second threshold function to binarize again
    ret, img_bin = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY)
    # Detect SIFT features
    keypoints, descriptors = sift.detectAndCompute(img_bin, None)
    # Return the feature descriptors as a numpy array
    # cv2.imwrite("binary_image_compare.png", img_bin)
    return descriptors

def looking_for(split_image_location, split_image_descriptors):
    global time_var
    highest, time_var = 0, 0
    sct = mss.mss()
    log.basicConfig(level=log.DEBUG, format='%(message)s', handlers=[log.StreamHandler()])
    # Build an index for the split image descriptors using the pyflann library
    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    while True:
        # Get the SIFT features of the current image
        sct_img = np.array(sct.grab(split_image_location))
        current_image_descriptors = get_sift_features(sct_img)
        # Use the pyflann library to find the matches between the current image and the split image
        matches = flann.knnMatch(current_image_descriptors, split_image_descriptors, k=2)
        # Calculate the number of good matches
        good_matches = [match for match in matches if len(match) == 2 and match[0].distance < 0.8 * match[1].distance]
        percent_correct = round(len(good_matches) / len(split_image_descriptors), 5)
        # Increment time_var to keep track of how many loops we've done
        time_var += 1
        # Log the % correct and highest % correct
        if percent_correct > highest: 
            highest = percent_correct
        log_message = str(f"{percent_correct:<7}") + " | " + str(f"{highest:<7}")
            # log.debug(log_message)

        if time_var % 10 == 0: log.debug(log_message)
        # Break if % correct >= 95
        if (percent_correct >= 0.85):
            log.debug(log_message)
            break
    return percent_correct

def find_split(image_file):
    # get the split image we will be comparing to
    split_image = cv2.imread(image_file)
    
    # Get the screen and image resolution
    screen_width, screen_height = pag.size()
    image_height, image_width = split_image.shape[:2]

    # Get the info from the file name
    file_name = image_file.split("/")[-1]#.split(".")[0]
    top_left_col, top_left_row, bottom_right_col, bottom_right_row, _ = file_name.split("_")

    # Scale the coordinates based on the screen resolution
    top_left_col = int(int(top_left_col) * screen_width/image_width)
    top_left_row = int(int(top_left_row) * screen_height/image_height)
    bottom_right_col = int(int(bottom_right_col) * screen_width/image_width)
    bottom_right_row = int(int(bottom_right_row) * screen_height/image_height)

    # Calculate the split image location
    split_image_location = {"top": top_left_row, "left": top_left_col, "width": bottom_right_col - top_left_col, "height": bottom_right_row - top_left_row}
        
    # get the split image we will be comparing to
    split_image = cv2.resize(split_image, (screen_width, screen_height), interpolation=cv2.INTER_NEAREST)[top_left_row:bottom_right_row, top_left_col:bottom_right_col]
    split_image_features = get_sift_features(split_image)

    # Convert the region to grayscale
    img_gray = cv2.cvtColor(split_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get binary image
    ret, img_bin = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binary_image.png", img_bin)

    # start the timer and see how long it takes to find the split
    start_time = dt.datetime.today().timestamp()
    print(f"Loking for: {image_file}")
    percent_correct = looking_for(split_image_location, split_image_features)
    time_diff = dt.datetime.today().timestamp() - start_time

    # split found, press split key, print data, wait 7 seconds
    pag.press('add')
    print(f"Split!\nThe image was found with a {round((percent_correct*100), 3)}% match.\nThe average fps was {round((time_var / time_diff), 1)}.")
    time.sleep(7)
    return

def main():
    split_files_folder = "Split Files"
    split_images_folder = "Split Images"
    # Check if Split Files and Split Images folders exist
    if not os.path.exists(split_files_folder) or not os.path.exists(split_images_folder):
        raise Exception(f"Either {split_files_folder} or {split_images_folder} folder does not exist in the current working directory.")
    
    split_files = [f for f in os.listdir(split_files_folder) if f.endswith('.txt')]
    split_files_dict = {f[:-4]: f for f in split_files}
    print("Select an image from the following options:")
    for i, key in enumerate(split_files_dict.keys()):print(f"{i+1}. {key}")

    selected_num = int(input("Select an image by entering its number: "))
    selected_split_file = split_files_dict[list(split_files_dict.keys())[selected_num - 1]]

    # Read the image names from the Split Files folder
    split_image_names = []
    with open(f"./{split_files_folder}/{selected_split_file}", "r") as f:
        split_image_names = [line.strip() for line in f.readlines()]
    
    for image_name in split_image_names:
        image_file = f"./{split_images_folder}/{image_name}"
        if not os.path.exists(image_file):
            raise Exception(f"Image {image_file} does not exist.")
        find_split(image_file)

if __name__ == '__main__':
    main()

# def main():
#     image_folder = "./Split Images"
#     images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
#     image_dict = {f.rsplit("_", 1)[-1][:-4]: f for f in images}
#     print("Select an image from the following options:")
#     for i, key in enumerate(image_dict.keys()):print(f"{i+1}. {key}")

#     selected_num = int(input("Select an image by entering its number: "))
#     selected_image = os.path.join(image_folder, image_dict[list(image_dict.keys())[selected_num - 1]])

#     print("You selected:", selected_image)
#     time.sleep(1.5)
#     find_split(selected_image)
#     return

# if __name__ == '__main__':
#     main()
