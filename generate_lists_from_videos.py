#!/usr/bin/env python3
"""
Generate data lists from actual video files in directory structure
Usage: python generate_lists_from_videos.py --video_root /path/to/PD4T_Videos --output_dir ./data_lists
"""
import os
import cv2
import argparse
from pathlib import Path


def get_video_frame_count(video_path):
    """Get frame count from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"  Warning: Could not get frame count for {video_path}: {e}")
        return 0


def generate_lists(video_root, output_dir):
    """Generate data lists from video files"""
    os.makedirs(output_dir, exist_ok=True)

    tasks = ['Gait', 'Hand movement', 'Finger tapping', 'Leg agility']

    for task in tasks:
        task_dir = os.path.join(video_root, task)

        if not os.path.isdir(task_dir):
            print(f"Warning: {task_dir} not found")
            continue

        print(f"\nProcessing: {task}")

        train_list = []
        test_list = []

        # Walk through patient folders
        for patient_folder in sorted(os.listdir(task_dir)):
            patient_path = os.path.join(task_dir, patient_folder)

            if not os.path.isdir(patient_path):
                continue

            # Get all video files
            videos = sorted([f for f in os.listdir(patient_path) if f.endswith('.mp4')])

            for video_file in videos:
                video_path = os.path.join(patient_path, video_file)
                visit_id = video_file.replace('.mp4', '')  # e.g., "15-001760"

                # Get frame count
                frame_count = get_video_frame_count(video_path)

                if frame_count == 0:
                    print(f"  Skipping {video_file} (frame count = 0)")
                    continue

                # Create entry: visit_id score num_frames patient_id
                # score is dummy (0) since we don't have ground truth
                entry = f"{visit_id} 0 {frame_count} {patient_folder}\n"

                # Split: 80% train, 20% test
                if hash(video_file) % 10 < 8:
                    train_list.append(entry)
                else:
                    test_list.append(entry)

        # Write train list
        if train_list:
            train_file = os.path.join(output_dir, f'{task}_train.list')
            with open(train_file, 'w') as f:
                f.writelines(train_list)
            print(f"  Created: {train_file} ({len(train_list)} samples)")

        # Write test list
        if test_list:
            test_file = os.path.join(output_dir, f'{task}_test.list')
            with open(test_file, 'w') as f:
                f.writelines(test_list)
            print(f"  Created: {test_file} ({len(test_list)} samples)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data lists from video files')
    parser.add_argument('--video_root', type=str, default='./PD4T_Videos',
                       help='Root directory containing video files')
    parser.add_argument('--output_dir', type=str, default='./data_lists',
                       help='Output directory for data lists')

    args = parser.parse_args()

    print(f"Video root: {args.video_root}")
    print(f"Output dir: {args.output_dir}")

    generate_lists(args.video_root, args.output_dir)
    print("\nDone!")
