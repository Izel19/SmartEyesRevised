from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import fast_colorthief as cf
import numpy as np
import time
import supervision as sv
import cv2
import os
import csv

# color directory for color search (nearest color) algo
colors = {
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (128, 0, 0): "red",
    (0, 128, 0): "green",
    (128, 0, 128): "purple",
    (0, 0, 128): "blue",
    (128, 128, 0): "yellow",
    (255, 165, 0): "orange",
    (255, 140, 0): "orange",
    (139, 0, 0): "red",
    (165, 42, 42): "brown",
    (220, 20, 60): "red",
    (25, 25, 112): "blue",
    (205, 92, 92): "red",
    (143, 188, 143): "green",
    (34, 139, 32): "green",
    (60, 60, 60): "grey",
    (60, 60, 0): "yellow",
    (60, 60, 30): "yellow",
    (60, 0, 0): "red",
    (0, 60, 0): "green",
    (30, 60, 30): "green",
    (0, 0, 60): "blue",
    (30, 30, 30): "black"
}

# stores processed human-clothes tracks
processed = []

# array for csv output
to_csv = []


# Directories
# home_dir = Path.home()
# main_dir = home_dir / "Music\\Outputs"
#
# mainFolder = Path(main_dir / "")
# outputPath.mkdir(parents=True, exist_ok=True)

home_dir = Path.home()
main_dir = Path("C:\\Users\\User\\Music\\Outputs")
main_dir.mkdir(parents=True, exist_ok=True)

outputPath = Path(main_dir / "Output Video")
outputPath.mkdir(parents=True, exist_ok=True)

screenshotPath = Path(main_dir / "Screenshots")
screenshotPath.mkdir(parents=True, exist_ok=True)

croppedPath = Path(main_dir / "Screenshots\\Clothes")
croppedPath.mkdir(parents=True, exist_ok=True)

csvPath = Path(main_dir / "Logs")
csvPath.mkdir(parents=True, exist_ok=True)
ids = 0
num = 0


class SMART_EYES:
    def __init__(self, i_path):
        self.i_path = i_path
        self.o_path = outputPath
        self.model = YOLO(model="C:\\Users\\User\\Downloads\\Nano-Best-Run-1200-DT-60-Epochs.pt")

    def initializePaths(self):
        global outputPath, screenshotPath, croppedPath, num
        num = len([x for x in os.listdir(outputPath) if os.path.isdir(outputPath)]) + 1
        os.mkdir(os.path.join(outputPath, str(num)))
        outputPath = Path(f"{outputPath}/{num}")
        os.mkdir(os.path.join(screenshotPath, str(num)))
        screenshotPath = Path(f"{screenshotPath}/{num}")
        os.mkdir(os.path.join(croppedPath, str(num)))
        croppedPath = Path(f"{croppedPath}/{num}")
        print(outputPath, screenshotPath, croppedPath)

    def writeToCSV(self, data, path):
        fields = ['ID', 'Timestamp', 'Color Name', 'RGB Value', 'Difference', 'Photo Name']
        with open(f"{path}/{num}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            writer.writerows(data)

    # sets labels for detection
    def setLabels(self, detections):
        if detections.tracker_id.any():
            labels = [f"#{tracker_id} {self.model.model.names[class_id]} {confidence:.2f}" for
                      tracker_id, class_id, confidence in
                      zip(detections.tracker_id, detections.class_id, detections.confidence)]
        else:
            labels = [f"{self.model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in
                      zip(detections.class_id, detections.confidence)]
        return labels

    # cretes bounding box for detections
    def box_annotator(self, video_info):
        thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
        box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
        return box_annotator

    # annotates labels from setLabels to predictions
    def label_annotator(self, video_info):
        label_annotator = sv.LabelAnnotator()
        return label_annotator

        # gets timestamps for stored data

    def getTimestamp(self, current_frame_number, video_info):
        timestamp_seconds = current_frame_number / video_info.fps
        timestamp_seconds = time.strftime("%H:%M:%S", time.gmtime(timestamp_seconds))
        return timestamp_seconds

    # creates a cropped version of detected clothes
    def cropAndSave(self, x1, y1, x2, y2, frame):

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_frame = frame[y1:y2, x1:x2]
        cv2.imwrite(f"{croppedPath}/croppedImage_ID-{ids}.jpg", cropped_frame) # Revisions accdg. to Dr. Arguson
        return (f"{croppedPath}/croppedImage_ID-{ids}.jpg")

    # gets dominant color from cropped image of clothes
    def thief(self, path):
        dominant = cf.get_dominant_color(path, quality=10)
        return dominant

    # returns the color nearest to color directory
    def nearest_colour(self, query, file_name_main, cropped_file_name):
        old_path_sc = screenshotPath / f"{file_name_main}.jpg"
        old_path_cropped = croppedPath / f"{cropped_file_name}.jpg"
        color_file_name_sc = ""
        color_file_name_cropped = ""
        min_color = min(colors.keys(), key=lambda color: sum((s - q) ** 2 for s, q in zip(color, query)))
        min_distance = sum((s - q) ** 2 for s, q in zip(min_color, query))

        if min_distance > 1000:
            color = "Undetected"
            color_file_name_sc = f"{file_name_main}-{color}.jpg"
            color_file_name_cropped = f"{cropped_file_name}-{color}.jpg"
            new_path_sc = old_path_sc.with_name(color_file_name_sc)
            new_path_cropped = old_path_cropped.with_name(color_file_name_cropped)
        else:
            color = colors[min_color]
            color_file_name_sc = f"{file_name_main}-{color}.jpg"
            color_file_name_cropped = f"{cropped_file_name}-{color}.jpg"
            new_path_sc = old_path_sc.with_name(color_file_name_sc)
            new_path_cropped = old_path_cropped.with_name(color_file_name_cropped)

        os.rename(old_path_sc, new_path_sc)
        os.rename(old_path_cropped, new_path_cropped)

        return color, min_distance, color_file_name_sc
        # return min(subject.keys(),key = lambda key: sum((s - q) ** 2 for s, q in zip(subject[key], query))) # calculates the distance between color values

    # processes detections
    def processFrame(self, detections, frame, frameNumber, video_info):
        global ids
        if detections.tracker_id.any():  # checks if detections stayed to have trackers be set to it.
            class0 = detections[detections.class_id == 0]  # clothes
            class1 = detections[detections.class_id == 1]  # human
            for i in class0:  # clothes xyxy
                ix1, iy1, ix2, iy2 = i[0]
                for j in class1:  # human xyxy
                    jx1, jy1, jx2, jy2 = j[0]
                    if (jx1 <= ix1 <= ix2 <= jx2) and (
                            jy1 <= iy1 <= iy2 <= jy2):  # checks if clothes box coordinates are inside human box
                        c = i[4]
                        h = j[4]
                        for point in processed:
                            if point[0] == c:
                                return
                        f_copy = frame  # creates copy of current frame
                        f_copy = f_copy[int(jy1):int(jy2), int(jx1):int(jx2)]  # crops human bounding box from frame
                        cv2.imwrite(f"{screenshotPath}/screenshot_ID-{ids}.jpg", f_copy)
                        file_ref_name_main = f"screenshot_ID-{ids}"
                        file_ref_name_cropped = f"croppedImage_ID-{ids}"
                        croppedPath = self.cropAndSave(ix1, iy1, ix2, iy2, frame)
                        shirtColor = self.thief(croppedPath)
                        cname, value_diff, img_name = self.nearest_colour(shirtColor, file_ref_name_main, file_ref_name_cropped)
                        to_csv.append([ids, self.getTimestamp(frameNumber, video_info), cname, shirtColor, value_diff, img_name])
                        processed.append([c, h])  # appends processed values to processed
                        ids += 1
                        return

    def process_video(self):
        video_info = sv.VideoInfo.from_video_path(self.i_path)  # gets video information from input
        box_annotator = self.box_annotator(video_info)
        label_annotator = self.label_annotator(video_info)

        tracker = sv.ByteTrack(track_activation_threshold=0.1,
                               frame_rate=video_info.fps)  # setups tracker for detections

        generator = sv.get_video_frames_generator(self.i_path)  # gathers frames to be processed
        with sv.VideoSink(f"{outputPath}/Output_{num}.mp4",
                          video_info) as sink:  # supervision's video writer, similar to cv2
            for current_frame_number, frame in tqdm(enumerate(generator), total=video_info.fps):
                res = self.model(frame, conf=0.55, vid_stride=1, device='cpu')[0]

                detections = sv.Detections.from_ultralytics(res)
                detections = tracker.update_with_detections(detections)
                labels = self.setLabels(detections)
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                self.processFrame(detections, frame, current_frame_number, video_info)
                sink.write_frame(annotated_frame)  # writes frames into videos

    def findInCSV(self, color, Path=csvPath):
        global csvPath
        rows = []
        counter = 0
        if Path == csvPath:
            csvPath = f"{csvPath}/{num}.csv"
        with open(csvPath, 'r') as f:
            reader = csv.reader(f)

            next(reader)

            for row in reader:
                if color == row[2]:
                    rows.append(row)
                    counter += 1

        print(f"Found {counter} matches to color {color}!")
        return rows

    def summary_generator(self, Path=csvPath):
        global csvPath
        unique_items = set()
        color_count = 0
        undetected_count = 0

        # Update the csvPath if needed
        if Path == csvPath:
            csvPath = f"{csvPath}/{num}.csv"

        # Open and read the CSV file
        with open(csvPath, 'r') as r:
            reader = csv.reader(r)
            headers = next(reader)  # Skip the header row

            for row in reader:
                if len(row) < 3:
                    continue  # Skip rows that don't have enough columns

                unique_items.add(row[0])

                if row[2].strip() == "Undetected":
                    undetected_count += 1
                else:
                    color_count += 1

        num_of_detected = len(unique_items)
        total_detected = f"Number of detected items: {num_of_detected}"
        color_counted_num = f"Total colors detected: {color_count}"
        undectected_counted_num = f"Undetected count {undetected_count}"

        print(total_detected)
        print(color_counted_num)
        print(undectected_counted_num)

        return total_detected, color_counted_num, undectected_counted_num



    def main(self):
        self.initializePaths()
        self.process_video()
        self.writeToCSV(to_csv, csvPath)
        self.summary_generator(csvPath)


# if __name__ == "__main__":
#     SE = SMART_EYES("./vids/test2.mp4")
#     SE.main()
#     print(SE.findInCSV("grey"))