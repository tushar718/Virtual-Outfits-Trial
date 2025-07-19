import cv2
import mediapipe as mp
import numpy as np

def overlay_tshirt(frame, tshirt_img, body_points):
    if len(body_points) < 7:
        print("Not enough points for a good overlay.")
        return frame

    points_array = np.array(body_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points_array)

    # Adjust bounding box to improve t-shirt fit
    x -= int(0.05 * w)
    w += int(0.1 * w)
    y -= int(0.1 * h)  # Adjust upwards to include neck
    h += int(0.2 * h)  # Increase height to include full t-shirt length

    # Ensure bounding box is completely within the frame
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    if w <= 0 or h <= 0:
        print(f"Invalid dimensions w: {w}, h: {h}")
        return frame

    try:
        tshirt_resized = cv2.resize(tshirt_img, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha_s = tshirt_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_s * tshirt_resized[:, :, c] +
                                      alpha_l * frame[y:y+h, x:x+w, c])
    except Exception as e:
        print(f"Error during overlay: {e}")

    return frame

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    image_path = r"C:/Users/Shreyas/Pictures/Resources-1/Resources-1/Resources/Shirts/1.png"
    tshirt_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if tshirt_img is None:
        print("Failed to load t-shirt image")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            body_points = [
                (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])),
                (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0])),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0])),
                (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0])),
                (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0]))
            ]

            # Calculate neck position as midpoint between shoulders
            neck_x = (body_points[0][0] + body_points[1][0]) // 2
            neck_y = (body_points[0][1] + body_points[1][1]) // 2
            body_points.append((neck_x, neck_y))

            frame = overlay_tshirt(frame, tshirt_img, body_points)

        cv2.imshow('Virtual T-Shirt Try-On', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
