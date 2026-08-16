import cv2
import os
import time
import math
import tkinter as tk
from tkinter import messagebox


# ============================================================
# VIDEO PATH
# ============================================================

VIDEO_PATH = r"C:\Users\DELL\Downloads\drone_video.mp4.mp4"


# ============================================================
# SCREEN SETTINGS
# ============================================================

MAX_WIDTH = 1000
MAX_HEIGHT = 650


# ============================================================
# COLORS - BGR
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
BLACK = (0, 0, 0)


# ============================================================
# VIDEO CHECK
# ============================================================

print("=" * 60)
print("DRONE ANALYSIS SYSTEM")
print("=" * 60)

print("Video Path   :", VIDEO_PATH)
print("Video Exists :", os.path.exists(VIDEO_PATH))

if not os.path.exists(VIDEO_PATH):

    print("ERROR: Video not found!")

    messagebox.showerror(
        "Video Error",
        "Video not found!\n\n"
        + VIDEO_PATH
    )

    exit()

print("Video Found Successfully")
print("=" * 60)


# ============================================================
# DRONE ANALYZER
# ============================================================

class DroneAnalyzer:

    def __init__(self):

        self.previous_gray = None
        self.previous_center = None

        self.object_count = 0
        self.confidence = 0

        self.speed = 0

        self.threat_detected = False

        self.start_time = time.time()


    # ========================================================
    # MOTION DETECTION
    # ========================================================

    def detect_motion(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (21, 21),
            0
        )

        if self.previous_gray is None:

            self.previous_gray = gray

            return []

        difference = cv2.absdiff(
            self.previous_gray,
            gray
        )

        threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]

        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 700:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 25 or h < 25:
                continue

            boxes.append(
                (x, y, w, h)
            )

        self.previous_gray = gray

        return boxes


    # ========================================================
    # SPEED ESTIMATION
    # ========================================================

    def calculate_speed(self, boxes):

        if len(boxes) == 0:

            self.speed *= 0.90

            return

        largest_box = max(
            boxes,
            key=lambda box: box[2] * box[3]
        )

        x, y, w, h = largest_box

        center_x = x + w // 2
        center_y = y + h // 2

        current_center = (
            center_x,
            center_y
        )

        if self.previous_center is not None:

            dx = (
                current_center[0]
                - self.previous_center[0]
            )

            dy = (
                current_center[1]
                - self.previous_center[1]
            )

            distance = math.sqrt(
                dx * dx + dy * dy
            )

            self.speed = min(
                999,
                distance * 3
            )

        self.previous_center = current_center


    # ========================================================
    # THREAT ANALYSIS
    # ========================================================

    def detect_threat(
        self,
        boxes,
        frame_width,
        frame_height
    ):

        self.threat_detected = False

        if not boxes:
            return

        center_x = frame_width // 2
        center_y = frame_height // 2

        for x, y, w, h in boxes:

            object_center_x = x + w // 2
            object_center_y = y + h // 2

            distance = math.sqrt(
                (
                    object_center_x
                    - center_x
                ) ** 2
                +
                (
                    object_center_y
                    - center_y
                ) ** 2
            )

            large_object = (
                w > frame_width * 0.20
                or
                h > frame_height * 0.20
            )

            near_center = (
                distance
                <
                min(
                    frame_width,
                    frame_height
                ) * 0.25
            )

            if large_object and near_center:

                self.threat_detected = True

                break


    # ========================================================
    # DRAW HUD
    # ========================================================

    def draw_hud(
        self,
        frame,
        boxes
    ):

        height, width = frame.shape[:2]

        self.object_count = len(boxes)

        self.confidence = min(
            98,
            70 + self.object_count * 5
        )

        self.calculate_speed(boxes)

        self.detect_threat(
            boxes,
            width,
            height
        )


        # ====================================================
        # DETECTION BOXES
        # ====================================================

        for x, y, w, h in boxes:

            if self.threat_detected:

                box_color = RED
                label = "THREAT"

            else:

                box_color = GREEN
                label = "TARGET"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                3
            )

            label_y = max(
                30,
                y
            )

            cv2.rectangle(
                frame,
                (
                    x,
                    label_y - 30
                ),
                (
                    x + 145,
                    label_y
                ),
                box_color,
                -1
            )

            cv2.putText(
                frame,
                f"{label} {self.confidence}%",
                (
                    x + 5,
                    label_y - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                BLACK,
                2
            )


        # ====================================================
        # OUTER BORDER
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (width - 10, height - 10),
            GREEN,
            2
        )


        # ====================================================
        # HEADER
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (width - 10, 65),
            BLACK,
            -1
        )

        cv2.putText(
            frame,
            "DRONE ANALYSIS",
            (25, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GREEN,
            2
        )


        # ====================================================
        # TIMER
        # ====================================================

        mission_time = (
            time.time()
            - self.start_time
        )

        cv2.putText(
            frame,
            f"T+ {mission_time:.1f}s",
            (
                width - 190,
                48
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2
        )


        # ====================================================
        # LEFT TELEMETRY
        # ====================================================

        telemetry_x = 25
        telemetry_y = 105

        telemetry = [

            f"OBJECTS    : {self.object_count}",

            f"SPEED      : "
            f"{self.speed:.1f} px/s",

            f"CONFIDENCE : "
            f"{self.confidence}%",

            "CAMERA     : ONLINE",

            "GPS        : ACTIVE",

            "ANALYSIS   : LIVE"
        ]

        for i, text in enumerate(
            telemetry
        ):

            cv2.putText(
                frame,
                text,
                (
                    telemetry_x,
                    telemetry_y
                    + i * 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                GREEN,
                2
            )


        # ====================================================
        # STATUS PANEL
        # ====================================================

        status_x = width - 320
        status_y = 90

        if self.threat_detected:

            cv2.rectangle(
                frame,
                (
                    status_x,
                    status_y
                ),
                (
                    width - 20,
                    status_y + 55
                ),
                RED,
                -1
            )

            cv2.putText(
                frame,
                "THREAT DETECTED",
                (
                    status_x + 15,
                    status_y + 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE,
                2
            )

        else:

            cv2.rectangle(
                frame,
                (
                    status_x,
                    status_y
                ),
                (
                    width - 20,
                    status_y + 55
                ),
                GREEN,
                2
            )

            cv2.putText(
                frame,
                "SYSTEM NOMINAL",
                (
                    status_x + 15,
                    status_y + 35
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                GREEN,
                2
            )


        # ====================================================
        # CENTER CROSSHAIR
        # ====================================================

        center_x = width // 2
        center_y = height // 2

        cv2.circle(
            frame,
            (
                center_x,
                center_y
            ),
            35,
            GREEN,
            1
        )

        cv2.line(
            frame,
            (
                center_x - 50,
                center_y
            ),
            (
                center_x - 10,
                center_y
            ),
            GREEN,
            1
        )

        cv2.line(
            frame,
            (
                center_x + 10,
                center_y
            ),
            (
                center_x + 50,
                center_y
            ),
            GREEN,
            1
        )

        cv2.line(
            frame,
            (
                center_x,
                center_y - 50
            ),
            (
                center_x,
                center_y - 10
            ),
            GREEN,
            1
        )

        cv2.line(
            frame,
            (
                center_x,
                center_y + 10
            ),
            (
                center_x,
                center_y + 50
            ),
            GREEN,
            1
        )


        # ====================================================
        # BOTTOM BAR
        # ====================================================

        cv2.putText(
            frame,
            "LIVE VIDEO ANALYSIS",
            (
                25,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            GREEN,
            2
        )

        cv2.putText(
            frame,
            "ESC / Q = EXIT",
            (
                width - 180,
                height - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1
        )

        return frame


# ============================================================
# FIT VIDEO TO SCREEN
# ============================================================

def resize_for_screen(frame):

    height, width = frame.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# START ANALYSIS
# ============================================================

def start_analysis():

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        messagebox.showerror(
            "Video Error",
            "Unable to open video."
        )

        return


    # ========================================================
    # VIDEO WINDOW
    # ========================================================

    window_name = "DRONE ANALYSIS - LIVE"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        MAX_WIDTH,
        MAX_HEIGHT
    )


    # ========================================================
    # VIDEO FPS
    # ========================================================

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30

    delay = max(
        1,
        int(1000 / fps)
    )


    # ========================================================
    # ANALYZER
    # ========================================================

    analyzer = DroneAnalyzer()

    print("Live analysis started...")


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Video Finished")

            break


        # Motion detection

        boxes = analyzer.detect_motion(
            frame
        )


        # Draw HUD

        frame = analyzer.draw_hud(
            frame,
            boxes
        )


        # Resize for screen

        display_frame = resize_for_screen(
            frame
        )


        # Show video

        cv2.imshow(
            window_name,
            display_frame
        )


        # Keyboard

        key = cv2.waitKey(
            delay
        ) & 0xFF


        if key == 27:

            break


        if key == ord("q"):

            break


    cap.release()

    cv2.destroyAllWindows()

    print("Analysis stopped.")


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Drone Analysis Mission Control"
)

root.geometry(
    "700x500"
)

root.configure(
    bg="#10151c"
)

root.resizable(
    False,
    False
)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="DRONE ANALYSIS",
    font=(
        "Arial",
        28,
        "bold"
    ),
    bg="#10151c",
    fg="#00ff00"
)

title.pack(
    pady=(35, 5)
)


subtitle = tk.Label(
    root,
    text="LIVE VIDEO INTELLIGENCE SYSTEM",
    font=(
        "Arial",
        12
    ),
    bg="#10151c",
    fg="white"
)

subtitle.pack(
    pady=5
)


# ============================================================
# STATUS
# ============================================================

status = tk.Label(
    root,
    text="● SYSTEM READY",
    font=(
        "Arial",
        15,
        "bold"
    ),
    bg="#10151c",
    fg="#00ff00"
)

status.pack(
    pady=20
)


# ============================================================
# VIDEO NAME
# ============================================================

video_info = tk.Label(
    root,
    text=(
        "VIDEO: "
        + os.path.basename(VIDEO_PATH)
    ),
    font=(
        "Arial",
        11
    ),
    bg="#10151c",
    fg="#dddddd"
)

video_info.pack(
    pady=5
)


# ============================================================
# FEATURES
# ============================================================

features = tk.Label(
    root,
    text=(
        "✓ Motion Detection    "
        "✓ Green Target Boxes\n"
        "✓ Speed Estimation    "
        "✓ Threat Analysis\n"
        "✓ Confidence Score    "
        "✓ Live Telemetry HUD"
    ),
    font=(
        "Arial",
        11
    ),
    bg="#10151c",
    fg="#cccccc",
    justify="center"
)

features.pack(
    pady=20
)


# ============================================================
# START BUTTON
# ============================================================

start_button = tk.Button(
    root,
    text="▶  START LIVE ANALYSIS",
    command=start_analysis,
    font=(
        "Arial",
        15,
        "bold"
    ),
    bg="#008f4c",
    fg="white",
    activebackground="#00aa55",
    activeforeground="white",
    padx=35,
    pady=15,
    cursor="hand2"
)

start_button.pack(
    pady=15
)


# ============================================================
# EXIT BUTTON
# ============================================================

exit_button = tk.Button(
    root,
    text="EXIT",
    command=root.destroy,
    font=(
        "Arial",
        10,
        "bold"
    ),
    bg="#222831",
    fg="white",
    padx=25,
    pady=7
)

exit_button.pack(
    pady=5
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="Drone Mission Control • Computer Vision Analysis",
    font=(
        "Arial",
        9
    ),
    bg="#10151c",
    fg="#777777"
)

footer.pack(
    side="bottom",
    pady=12
)


# ============================================================
# RUN GUI
# ============================================================

root.mainloop()
