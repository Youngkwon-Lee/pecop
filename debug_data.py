#!/usr/bin/env python3
"""Debug script to test data loading"""
import os
import sys

data_list_file = "data_lists/Gait_train.list"
video_root = os.path.expanduser("~/PD4T_Videos")

print(f"Data list: {data_list_file}")
print(f"Video root: {video_root}\n")

if not os.path.exists(data_list_file):
    print(f"Error: Data list not found: {data_list_file}")
    sys.exit(1)

# 첫 3개 항목 확인
with open(data_list_file) as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        parts = line.strip().split()
        visit_id = parts[0]
        patient_id = parts[3]

        print(f"Entry {i+1}: visit_id={visit_id}, patient_id={patient_id}")

        # Gait 폴더에서 검색
        task_dir = os.path.join(video_root, 'Gait')
        video_path = os.path.join(task_dir, patient_id, f'{visit_id}.mp4')

        if os.path.exists(video_path):
            print(f"  ✓ Found: {video_path}")
        else:
            print(f"  ✗ Not found at: {video_path}")

            # 실제 파일 찾기
            if os.path.isdir(os.path.join(task_dir, patient_id)):
                files = os.listdir(os.path.join(task_dir, patient_id))[:5]
                print(f"    Sample files in {patient_id}: {files}")
            else:
                print(f"    Patient folder {patient_id} does not exist in {task_dir}")

        print()
