import cv2

class VideoToImageSaver:

    def __init__(self, vid_path, start_frame = 0, num_frames = None):
        self.vid_path = vid_path
        self.start_frame = start_frame
        self.num_frames = num_frames

    def save_frames(self, image_dir_path):
        vidcap = cv2.VideoCapture(self.vid_path)
        cap_success, cap_image = vidcap.read()
        frame_count = 0
        cap_success = True
        while cap_success:
            cap_success, cap_image = vidcap.read()
            print("Read a new frame: ", cap_success)
            if frame_count > self.start_frame:
                if self.num_frames == None:
                    cv2.imwrite(image_dir_path + str(frame_count) + ".png", cap_image)
                else:
                    if frame_count - self.start_frame < self.num_frames:
                        cv2.imwrite(image_dir_path + str(frame_count) + ".png", cap_image)
                    else:
                        cap_success = False
            frame_count += 1
