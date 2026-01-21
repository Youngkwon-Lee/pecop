"""
Extract frames from PD4T MP4 videos
"""
import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir):
    """Extract frames from MP4 video"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = os.path.join(output_dir, f'img_{frame_count:05d}.jpg')
        cv2.imwrite(frame_path, frame)

    cap.release()
    return frame_count

def main(args):
    """Extract frames from all PD4T videos"""
    video_root = args.video_root
    output_root = args.output_root
    tasks = ['Gait', 'Hand movement', 'Finger tapping', 'Leg agility']

    total_videos = 0
    total_frames = 0

    for task in tasks:
        task_video_dir = os.path.join(video_root, task)
        task_output_dir = os.path.join(output_root, task)

        if not os.path.exists(task_video_dir):
            print(f"Warning: Task directory not found: {task_video_dir}")
            continue

        print(f"\nProcessing: {task}")

        for patient_id in sorted(os.listdir(task_video_dir)):
            patient_video_dir = os.path.join(task_video_dir, patient_id)

            if not os.path.isdir(patient_video_dir):
                continue

            for video_file in sorted(os.listdir(patient_video_dir)):
                if not video_file.endswith('.mp4'):
                    continue

                video_path = os.path.join(patient_video_dir, video_file)
                video_name = video_file.replace('.mp4', '')
                frame_output_dir = os.path.join(task_output_dir, patient_id, video_name)

                if os.path.exists(frame_output_dir) and len(os.listdir(frame_output_dir)) > 0:
                    print(f"  OK: {patient_id}/{video_name}")
                    continue

                print(f"  -> {patient_id}/{video_name}")
                frame_count = extract_frames(video_path, frame_output_dir)
                total_videos += 1
                total_frames += frame_count
                print(f"     Extracted {frame_count} frames")

    print(f"\nDone!")
    print(f"Total videos: {total_videos}")
    print(f"Total frames: {total_frames}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str,
                       default='D:\\Hawkeye\\data\\raw\\PD4T\\PD4T\\PD4T\\Videos',
                       help='PD4T video root path')
    parser.add_argument('--output_root', type=str,
                       default='D:\\parkinson\\PECoP\\datasets\\PD4T_frames',
                       help='Output frames path')

    args = parser.parse_args()
    main(args)
