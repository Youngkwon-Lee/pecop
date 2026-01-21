"""
Create training data lists for PD4T dataset
"""
import os
import pandas as pd

def create_data_lists(annotations_root, output_dir):
    """Create data lists from PD4T annotations CSV files"""
    os.makedirs(output_dir, exist_ok=True)

    tasks = ['Gait', 'Hand movement', 'Finger tapping', 'Leg agility']

    for task in tasks:
        task_anno_dir = os.path.join(annotations_root, task)

        if not os.path.exists(task_anno_dir):
            print(f"Warning: {task_anno_dir} not found")
            continue

        print(f"\nProcessing: {task}")

        # Read train.csv
        train_csv = os.path.join(task_anno_dir, 'train.csv')
        if os.path.exists(train_csv):
            df = pd.read_csv(train_csv, header=None)

            output_file = os.path.join(output_dir, f'{task}_train.list')
            with open(output_file, 'w') as f:
                for idx, row in df.iterrows():
                    # Format: video_name num_frames score patient_id
                    video_name = row[0]
                    num_frames = row[1]
                    score = row[2]

                    # Extract patient ID from video_name
                    if task == 'Gait':
                        # Format: visit_patient_id
                        parts = video_name.split('_')
                        patient_id = parts[-1]
                    else:
                        # Format: visit_position_patient_id
                        parts = video_name.split('_')
                        patient_id = parts[-1]

                    # Write to list
                    f.write(f"{video_name} {score} {num_frames} {patient_id}\n")

            print(f"  Created: {output_file} ({len(df)} samples)")

        # Read test.csv
        test_csv = os.path.join(task_anno_dir, 'test.csv')
        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv, header=None)

            output_file = os.path.join(output_dir, f'{task}_test.list')
            with open(output_file, 'w') as f:
                for idx, row in df.iterrows():
                    video_name = row[0]
                    num_frames = row[1]
                    score = row[2]

                    if task == 'Gait':
                        parts = video_name.split('_')
                        patient_id = parts[-1]
                    else:
                        parts = video_name.split('_')
                        patient_id = parts[-1]

                    f.write(f"{video_name} {score} {num_frames} {patient_id}\n")

            print(f"  Created: {output_file} ({len(df)} samples)")

if __name__ == '__main__':
    annotations_root = 'D:\\Hawkeye\\data\\raw\\PD4T\\PD4T\\PD4T\\Annotations'
    output_dir = 'D:\\parkinson\\PECoP\\data_lists'

    create_data_lists(annotations_root, output_dir)
    print("\nDone!")
