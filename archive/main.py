import os
import cv2
import torch
import datetime
import numpy as np
import logging
from collections import defaultdict
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define command line flags
flags.DEFINE_string("video", "test1.mp4", "Path to input video or webcam index (0)")
flags.DEFINE_string("output", "./output/output.mp4", "Path to output video")
flags.DEFINE_float("conf", 0.10, "Confidence threshold")
flags.DEFINE_integer("blur_id", None, "Class ID to apply Gaussian Blur")
flags.DEFINE_integer("class_id", 2, "Class ID to track")

FLAGS = flags.FLAGS
lanes = {}
lane_count = {}
detected_objects = {}
FEED_WIDTH = 0
FEED_HEIGHT = 0


def initialize_video_capture(video_input):
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        logger.error("Error: Unable to open video source.")
        raise ValueError("Unable to open video source")

    return cap


def initialize_model():
    model_path = "./weights/yolov10x.pt"
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found at {model_path}")
        raise FileNotFoundError("Model weights file not found")

    model = YOLO("yolov10x.pt")

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")

    model.to(device)
    logger.info(f"Using {device} as processing device")
    return model


def load_class_names():
    classes_path = "./configs/coco.names"
    if not os.path.exists(classes_path):
        logger.error(f"Class names file not found at {classes_path}")
        raise FileNotFoundError("Class names file not found")

    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names


def process_frame(frame, model, tracker, class_names, colors):
    results = model(frame, verbose=False)[0]
    detections = []
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if FLAGS.class_id is None:
            if confidence < FLAGS.conf:
                continue
        else:
            if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                continue

        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks


def draw_tracks(
    frame, tracks, class_names, colors, lane_colors, class_counters, track_class_mapping
):
    tracked_objects = {}
    global lane_count
    global FEED_HEIGHT
    global FEED_WIDTH
    lane_count = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)
        centre_x = (x1 + x2) // 2
        centre_y = (y1 + y2) // 2
        color = colors[class_id]
        B_, G_, R_ = map(int, color)

        # Assign a new class-specific ID if the track_id is seen for the first time
        if track_id not in track_class_mapping:
            class_counters[class_id] += 1
            track_class_mapping[track_id] = class_counters[class_id]

        class_specific_id = track_class_mapping[track_id]
        text = f"ID: {class_specific_id}"
        tracked_objects[class_specific_id] = (centre_x, centre_y)
        ################## clustering for cars based on distance to lane functions
        if centre_y >= (FEED_HEIGHT // 2 + 100):
            for lane_id, lane_x in lanes.items():
                if abs(centre_x - lane_x) <= 100:
                    lane_color = lane_colors[lane_id]
                    l_b, l_g, l_r = map(int, lane_color)
                    cv2.circle(frame, (centre_x, centre_y), 5, (l_b, l_g, l_r), -1)
                    if lane_id not in lane_count:
                        lane_count[lane_id] = [(centre_x, centre_y)]
                    else:
                        lane_count[lane_id].append((centre_x, centre_y))
        else:
            # cluster based on centre distance to lines
            lane_equations = get_lane_coeff(lane_count)
            # print(f"lane_equations: {lane_equations}, {class_specific_id}")
            for lane_id, (m, b, x, y) in lane_equations.items():
                A = m
                B = -1
                C = b
                # Distance formula for point to line
                distance = abs(A * centre_x + B * centre_y + C) / np.sqrt(A**2 + B**2)
                if distance <= 50:
                    lane_color = lane_colors[lane_id]
                    l_b, l_g, l_r = map(int, lane_color)
                    cv2.circle(frame, (centre_x, centre_y), 5, (l_b, l_g, l_r), -1)
                    if lane_id not in lane_count:
                        lane_count[lane_id] = [(centre_x, centre_y)]
                    else:
                        lane_count[lane_id].append((centre_x, centre_y))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B_, G_, R_), 2)
        cv2.rectangle(
            frame,
            (x1 - 1, y1 - 20),
            (x1 + len(text) * 9, y1),
            (B_, G_, R_),
            thickness=-2,
            lineType=cv2.LINE_AA,
        )
        # draw a line at frame height // 2 + 100
        cv2.line(
            frame,
            (0, FEED_HEIGHT // 2 + 100),
            (FEED_WIDTH, FEED_HEIGHT // 2 + 100),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 4)
    return frame, tracked_objects


def get_lanes(tracked_objects):
    # sorted_objects is a sorted list of all centre x values of each vehicle
    """
    ADD
    1. incorperate thershold offset based on distance (area of bounding box) of the object
    2. prespective ratification
    3. dynamic threshold based on the average size of the bounding box (width) of the mean x

    tracked_objects:{
    1: (x1, y1),
    2: (x2, y2),
    }
    """
    global lanes
    global FEED_WIDTH
    global FEED_HEIGHT
    lanes = {}
    queue = []
    lane_id = 0
    threshold = 125
    sorted_objects = sorted(
        [
            centre_point[0]
            for centre_point in tracked_objects.values()
            if centre_point[1] >= (FEED_HEIGHT // 2 + 100)
        ]
    )
    for idx, cur_val in enumerate(sorted_objects):
        queue.append(cur_val)
        if idx == len(sorted_objects) - 1:
            mean = np.mean(queue)
            lanes[lane_id] = mean
            lane_id += 1
            queue = []
            break
        if abs(sorted_objects[idx + 1] - np.mean(queue)) >= threshold:
            mean = np.mean(queue)
            lanes[lane_id] = mean
            lane_id += 1
            queue = []
    return lanes


def get_lane_coeff(lane_count):
    lane_equations = {}
    for lane_id, lane_points in lane_count.items():
        if len(lane_points) > 1:
            lane_points = np.array(lane_points, dtype=np.float32)
            line = cv2.fitLine(lane_points, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = line
            vx, vy, x, y = (
                float(vx.item()),
                float(vy.item()),
                float(x.item()),
                float(y.item()),
            )
            m = vy / vx
            b = y - m * x
            lane_equations[lane_id] = (m, b, x, y)
    return lane_equations


def cluster_objects(frame):
    pass


def main(_argv):
    try:
        global lanes
        global lane_count
        global FEED_WIDTH
        global FEED_HEIGHT
        cap = initialize_video_capture(FLAGS.video)
        model = initialize_model()
        class_names = load_class_names()

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FEED_WIDTH = frame_width
        FEED_HEIGHT = frame_height
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

        tracker = DeepSort(max_age=30, n_init=3)

        np.random.seed(15)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))
        lane_colors = np.random.randint(0, 255, size=(10, 3))

        class_counters = defaultdict(int)
        track_class_mapping = {}
        frame_count = 0

        while True:
            start = datetime.datetime.now()
            ret, frame = cap.read()
            if not ret:
                break
            tracks = process_frame(frame, model, tracker, class_names, colors)
            frame, tracked_objects = draw_tracks(
                frame,
                tracks,
                class_names,
                colors,
                lane_colors,
                class_counters,
                track_class_mapping,
            )

            lanes = get_lanes(tracked_objects)
            total_count = 0
            lane_equations = get_lane_coeff(lane_count)
            for m, b, x, y in lane_equations.values():
                y2 = int(((frame.shape[1] - x) * m) + y)
                frame = cv2.line(
                    frame, (frame.shape[1] - 1, y2), (0, int(b)), (255, 255, 255), 1
                )
            if lanes != {} and lane_count != {}:
                for count, (lane_id, lane_x) in enumerate(lanes.items()):
                    lane_text = f"LANE {lane_id}"
                    cv2.putText(
                        frame,
                        lane_text,
                        (int(lane_x) - len(lane_text) * 9, frame_height - 50),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    try:
                        total_count += len(lane_count[lane_id])
                        cv2.putText(
                            frame,
                            f"LANE {lane_id}: {len(lane_count[lane_id])}",
                            (20, 100 + (count * 40)),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                        )
                    except:
                        continue
            cv2.putText(
                frame,
                f"TOTAL: {total_count}",
                (20, 65),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 0),
                1,
            )

            end = datetime.datetime.now()
            frame_count += 1

            fps_text = f"FPS {1 / (end - start).total_seconds():.2f}"
            cv2.putText(
                frame,
                fps_text,
                (20, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 0),
                1,
            )

            writer.write(frame)
            cv2.imshow("FEED", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        logger.info("Class counts:")
        for class_id, count in class_counters.items():
            logger.info(f"{class_names[class_id]}: {count}")

    except Exception as e:
        logger.exception("An error occurred during processing")q
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass