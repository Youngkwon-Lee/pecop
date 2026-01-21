"""
Simplified PECoP training without torchvideotransforms dependency
Using MP4 video loading directly
"""
import os
import time
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import cv2
from models.i3d_adapter import I3D, Unit3Dpy
from tensorboardX import SummaryWriter

class PD4T_Dataset(Dataset):
    """Simple PD4T dataset loader - loads MP4 videos directly"""

    def __init__(self, data_list, video_root, clip_len=32, max_sr=5, max_segment=4, fr=3):
        with open(data_list, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            parts = line.strip().split()
            video_name = parts[0]
            score = int(parts[1])
            num_frames = int(parts[2])
            patient_id = parts[3]
            self.samples.append((video_name, score, num_frames, patient_id))

        self.video_root = video_root
        self.clip_len = clip_len
        self.max_sr = max_sr
        self.max_segment = max_segment
        self.fr = fr
        self.data_list = data_list

        # Extract task name from data_list path
        basename = os.path.basename(data_list)  # Get "Gait_train.list"
        if 'Hand' in basename:
            self.task = 'Hand movement'
        elif 'Finger' in basename:
            self.task = 'Finger tapping'
        elif 'Leg' in basename:
            self.task = 'Leg agility'
        else:
            self.task = 'Gait'

    def __len__(self):
        return len(self.samples)

    def load_video_frames(self, video_path, start_frame, num_frames_to_load, sample_rate):
        """Load frames from MP4 video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        frame_idx = 0
        current_frame = 0

        while len(frames) < num_frames_to_load:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                ret, frame = cap.read()
                if not ret:
                    break

            if current_frame >= start_frame and (current_frame - start_frame) % sample_rate == 0:
                frames.append(frame)

            current_frame += 1
            # Reset to start if we reach near the end
            if current_frame > total_frames - 10:
                current_frame = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()

        # Pad if necessary
        if len(frames) < num_frames_to_load:
            while len(frames) < num_frames_to_load:
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((480, 854, 3), dtype=np.uint8))

        return np.array(frames[:num_frames_to_load])

    def preprocess_frames(self, frames):
        """Simple preprocessing: resize, crop, normalize"""
        # Resize to 256x256
        resized = []
        for frame in frames:
            resized.append(cv2.resize(frame, (256, 256)))
        frames = np.array(resized)

        # Random crop to 224x224
        h, w = frames[0].shape[:2]
        top = np.random.randint(0, h - 224 + 1) if h > 224 else 0
        left = np.random.randint(0, w - 224 + 1) if w > 224 else 0
        frames = frames[:, top:top + 224, left:left + 224, :]

        # Convert to torch tensor (C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()

        # Normalize ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames / 255.0 - mean) / std

        return frames

    def __getitem__(self, idx):
        video_name, score, num_frames, patient_id = self.samples[idx]

        # video_name format: visit_patient_id (e.g., "15-001760_009")
        # Extract visit_id (e.g., "15-001760")
        parts = video_name.rsplit('_', 1)
        visit_id = parts[0] if len(parts) > 1 else video_name

        # Build video path
        video_path = self._find_video_path(visit_id, patient_id)

        if not os.path.exists(video_path):
            raise RuntimeError(f"Video not found: {video_path}")

        # Random sampling
        sample_rate = np.random.randint(1, self.max_sr + 1)
        while sample_rate == self.fr:
            sample_rate = np.random.randint(1, self.max_sr + 1)

        segment = np.random.randint(1, self.max_segment + 1)

        start_frame = np.random.randint(0, max(1, num_frames - self.clip_len))
        frames = self.load_video_frames(video_path, start_frame, self.clip_len, sample_rate)

        # Preprocess
        frames_tensor = self.preprocess_frames(frames)

        label_speed = sample_rate - 1
        label_segment = segment - 1

        return frames_tensor, np.array([label_speed, label_segment])

    def _find_video_path(self, visit_id, patient_id):
        """Find video path in the correct task directory

        For Gait: Videos/Gait/PatientID/visit_id.mp4
        For others: Videos/Task/PatientID/visit_id_l|r.mp4
        """
        task_dir = os.path.join(self.video_root, self.task)

        if not os.path.isdir(task_dir):
            raise RuntimeError(f"Task directory not found: {task_dir}")

        # First try with the provided patient_id
        if patient_id and os.path.isdir(os.path.join(task_dir, patient_id)):
            video_path = os.path.join(task_dir, patient_id, f'{visit_id}.mp4')
            if os.path.exists(video_path):
                return video_path

            # Try with suffixes for non-Gait tasks
            if self.task != 'Gait':
                for suffix in ['_l', '_r']:
                    video_path = os.path.join(task_dir, patient_id, f'{visit_id}{suffix}.mp4')
                    if os.path.exists(video_path):
                        return video_path

        # If not found, search all patient folders in this task
        for patient_folder in sorted(os.listdir(task_dir)):
            patient_path = os.path.join(task_dir, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            # Try without suffix (Gait)
            video_path = os.path.join(patient_path, f'{visit_id}.mp4')
            if os.path.exists(video_path):
                return video_path

            # Try with suffixes (other tasks)
            if self.task != 'Gait':
                for suffix in ['_l', '_r']:
                    video_path = os.path.join(patient_path, f'{visit_id}{suffix}.mp4')
                    if os.path.exists(video_path):
                        return video_path

        raise RuntimeError(f"Cannot find video: {visit_id} in {task_dir}")


class VSPP(nn.Module):
    """VSPP model with I3D backbone and 3D-Adapters"""

    def __init__(self, num_classes_p=5, num_classes_s=4, pretrained_i3d_path=None):
        super(VSPP, self).__init__()

        self.num_classes_s = num_classes_s
        self.num_classes_p = num_classes_p

        self.dropout = nn.Dropout(p=0.5)

        # Load I3D with 3D-Adapters
        self.model = I3D(num_classes=400, dropout_prob=0.5)

        if pretrained_i3d_path:
            cp = torch.load(pretrained_i3d_path, map_location='cpu')
            self.model.load_state_dict(cp, strict=False)

        # Freeze original weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze adapter layers
        adapter_modules = ['mixed_3b', 'mixed_3c', 'mixed_4b', 'mixed_4c',
                          'mixed_4d', 'mixed_4e', 'mixed_4f', 'mixed_5b', 'mixed_5c']

        for module_name in adapter_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                if hasattr(module, 'tuning_module'):
                    for param in module.tuning_module.parameters():
                        param.requires_grad = True

        # Prediction heads
        self.logits_p = Unit3Dpy(
            in_channels=1024,
            out_channels=num_classes_p,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

        self.logits_s = Unit3Dpy(
            in_channels=1024,
            out_channels=num_classes_s,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    def forward(self, x):
        x = self.model(x)
        x_p = self.logits_p(self.dropout(x))
        x_s = self.logits_s(self.dropout(x))

        l_p = x_p.squeeze(3).squeeze(3)
        l_s = x_s.squeeze(3).squeeze(3)
        l_p = torch.mean(l_p, 2)
        l_s = torch.mean(l_s, 2)

        return l_p, l_s


def train(args):
    torch.backends.cudnn.benchmark = True

    exp_name = f'{args.dataset}_sr_{args.max_sr}_{args.model}_lr_{args.lr}_len_{args.clip_len}_sz_{args.crop_sz}'
    print(f"Experiment: {exp_name}")

    pretrain_cks_path = os.path.join('pretrain_cks', exp_name)
    log_path = os.path.join('visual_logs', exp_name)

    os.makedirs(pretrain_cks_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Dataset and DataLoader
    print("Loading dataset...")
    train_dataset = PD4T_Dataset(
        args.data_list,
        args.video_root,
        clip_len=args.clip_len,
        max_sr=args.max_sr,
        max_segment=args.max_segment,
        fr=args.fr)

    print(f"Dataset size: {len(train_dataset)}")
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if args.device != 'cpu' else False)

    # Model
    print("Building model...")
    device = torch.device(args.device)
    model = VSPP(
        num_classes_p=args.max_sr,
        num_classes_s=args.max_segment,
        pretrained_i3d_path=args.pretrained_i3d_weight)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=0,
        weight_decay=1e-4,
        nesterov=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=22,
        eta_min=args.lr / 1000)

    writer = SummaryWriter(log_dir=log_path)
    iterations = 1

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(args.epoch):
        total_loss = 0.0
        correct = 0
        it = 0

        for i, (rgb_clip, labels) in enumerate(dataloader):
            rgb_clip = rgb_clip.to(device, dtype=torch.float)
            label_speed = labels[:, 0].to(device, dtype=torch.long)
            label_segment = labels[:, 1].to(device, dtype=torch.long)

            optimizer.zero_grad()
            out1, out2 = model(rgb_clip)

            loss1 = criterion(out1, label_speed)
            loss2 = criterion(out2, label_segment)
            loss = loss1 + loss2

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            probs_segment = nn.Softmax(dim=1)(out2)
            preds_segment = torch.max(probs_segment, 1)[1]
            accuracy_seg = torch.sum(preds_segment == label_segment.data).float()

            probs_speed = nn.Softmax(dim=1)(out1)
            preds_speed = torch.max(probs_speed, 1)[1]
            accuracy_speed = torch.sum(preds_speed == label_speed.data).float()

            accuracy = ((accuracy_speed + accuracy_seg) / 2) / args.bs
            correct += ((accuracy_speed + accuracy_seg) / 2).item() / args.bs

            iterations += 1
            it += 1

            if i % args.pf == 0:
                writer.add_scalar('data/train_loss', loss, iterations)
                writer.add_scalar('data/Acc', accuracy, iterations)
                print(f"[Epoch{epoch + 1}/{args.epoch}] Iter{i} Loss: {loss.item():.4f} Acc: {accuracy.item():.4f}")

        avg_loss = total_loss / it
        avg_acc = correct / it
        print(f'[Epoch {epoch + 1}] loss: {avg_loss:.3f}, acc: {avg_acc:.3f}')

        scheduler.step()

    writer.close()
    print("Training finished!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cpu')
    parser.add_argument('--height', type=int, default=256, help='resize height')
    parser.add_argument('--width', type=int, default=256, help='resize width')
    parser.add_argument('--clip_len', type=int, default=32, help='input clip length')
    parser.add_argument('--crop_sz', type=int, default=224, help='crop size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--epoch', type=int, default=2, help='total epoch')
    parser.add_argument('--max_sr', type=int, default=5, help='max sampling rate')
    parser.add_argument('--max_segment', type=int, default=4, help='max segments')
    parser.add_argument('--fr', type=int, default=1, help='base frame rate')
    parser.add_argument('--pf', type=int, default=5, help='print frequency')
    parser.add_argument('--dataset', type=str, default='PD4T', help='dataset name')
    parser.add_argument('--model', type=str, default='i3d', help='model name')
    parser.add_argument('--pretrained_i3d_weight', type=str,
                       default='./pretrained_models/model_rgb.pth',
                       help='Path to pretrained I3D weights')
    parser.add_argument('--data_list', type=str,
                       default='./data_lists/Gait_train.list',
                       help='Path to data list file')
    parser.add_argument('--video_root', type=str,
                       default='D:\\Hawkeye\\data\\raw\\PD4T\\PD4T\\PD4T\\Videos',
                       help='Path to PD4T videos')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    train(args)
