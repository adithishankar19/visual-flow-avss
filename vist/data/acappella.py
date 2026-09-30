import os
import torch
import numpy as np
import librosa
import random
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

# --- Utility Functions ---

def pad_or_truncate(audio, target_length):
    if audio.size(0) > target_length:
        return audio[:target_length]
    elif audio.size(0) < target_length:
        return torch.nn.functional.pad(audio, (0, target_length - audio.size(0)))
    return audio

def normalize_keypoints_2d(kp):
    kp = kp.astype(np.float32)
    T = kp.shape[0]
    # Landmark indices for eyes
    left_eye, right_eye = kp[:, 36], kp[:, 45]
    deltas = right_eye - left_eye
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    
    # Rotation matrix to level the eyes
    c, s = np.cos(-angles), np.sin(-angles)
    R = np.zeros((T, 2, 2))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
    
    kp = np.einsum('tik,tkj->tij', kp, R)
    kp -= kp[:, 30:31, :] # Center on nose bridge
    
    eye_dist = np.linalg.norm(kp[:, 36] - kp[:, 45], axis=1, keepdims=True)
    kp /= (eye_dist[:, :, np.newaxis] + 1e-8)
    kp[:, :, 1] *= -1 
    kp -= np.mean(kp, axis=0, keepdims=True)
    return kp

def normalize_keypoints_2d11(keypoints, scale_factor=1.0):
    """Center and scale landmarks to [-1, 1]."""
    keypoints = np.array(keypoints).astype(np.float32) 
    if keypoints.size == 0:
        return keypoints
        
    # Temporal Mean centering (Head-pose invariant normalization)
    mean_point = np.mean(np.mean(keypoints, axis=1), axis=0)
    keypoints -= mean_point
    
    # Max distance normalization
    max_distance = np.max(np.linalg.norm(keypoints, axis=2)) + 1e-8
    return (keypoints / max_distance) * scale_factor

def normalize_rms(audio, target_rms=0.1):
    """Normalizes an audio tensor to a precise baseline RMS value."""
    rms = torch.sqrt(torch.mean(audio**2)) + 1e-8
    return audio * (target_rms / rms)

# --- Data Handler ---

class AcapellaData:
    def __init__(self, audiorate, data_path, fps=25):
        self.arate = audiorate
        self.data_path = Path(data_path)
        self.fps = fps

    def load_audio(self, lang, gender, stem, offset, duration):
        path = self.data_path / 'audio' / lang / gender / f"{stem}.wav"
        # Ensure offset and duration don't exceed file bounds
        audio, _ = librosa.load(str(path), sr=self.arate, offset=offset, duration=duration)
        return pad_or_truncate(torch.from_numpy(audio).float(), 65535)

    def load_landmarks1(self, lang, gender, stem, offset, duration):
        path = self.data_path / 'landmarks' / lang / gender / f"{stem}.npy"
        try:
            landmarks = np.load(str(path)).astype(np.float32)
        except Exception:
            # Must match the [T, 2, 68] layout returned on the success path below.
            # The previous [T, 68, 2] fallback silently fed transposed keypoints
            # to the visual encoder whenever a landmark file failed to load.
            return torch.zeros((100, 2, 68))

        start_f = int(offset * self.fps)
        num_f = int(duration * self.fps)
        window = landmarks[start_f : start_f + num_f]
        
        if window.shape[0] < num_f:
            padding = np.repeat(landmarks[-1:], num_f - window.shape[0], axis=0)
            window = np.concatenate([window, padding], axis=0) if window.shape[0] > 0 else padding
        
        window = normalize_keypoints_2d(window)
        window = torch.from_numpy(window).float()
        
        # Target 100 frames for 4 seconds at 25fps
        target_f = 100 
        if window.shape[0] > target_f: window = window[:target_f]
        elif window.shape[0] < target_f:
            padding = torch.zeros((target_f - window.shape[0], 68, 2), dtype=window.dtype)
            window = torch.cat([window, padding], dim=0)
        
        return window.permute(0, 2, 1).float() # [T, 2, 68]

# --- Main Dataset Class ---

class AcapellaDataset(Dataset):
    def __init__(self,
                 data_path,
                 musdb_path,
                 audioset_path,
                 audiorate=16384,
                 framerate=25,
                 n=1,
                 languages= ['English', 'Hindi', 'Spanish', 'Others'],
                 is_train=True,
                 snr_range=(-5, 5),
                 interference_prob=1.0,
                 identity_shuffle_prob=0.5,
                 deterministic=False,
                 seed=1234,
                 exclude_same_stem_interferer=False,
                 target_rms_range=(0.1, 0.1),
                 global_gain_db_range=(0.0, 0.0),
                 clip_samples=65535):
        super().__init__()

        self.data_path = Path(data_path)
        self.musdb_path = Path(musdb_path)
        self.audioset_path = Path(audioset_path)
        self.arate = audiorate
        self.fps = framerate
        self.duration = 4.0 * n
        self.is_train = is_train
        self.snr_range = snr_range
        self.interference_prob = float(interference_prob)
        self.identity_shuffle_prob = float(identity_shuffle_prob)
        # `deterministic=True` derives every random choice for item i from
        # (seed, i) instead of the global `random` module, so a whole split
        # becomes a fixed manifest: identical mixtures on every pass and across
        # checkpoints.  Without this, each evaluation re-draws the interferer,
        # the accompaniment and both SNRs, which is what produces the
        # 0.2-0.4 dB spread when the same checkpoint is evaluated twice.
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        # Defaults to False so existing runs keep their exact data distribution.
        # Turning it on is a deliberate distribution change, not a free fix:
        # enable it for new training runs, and keep it off when re-evaluating an
        # old checkpoint so that the only thing that changed is determinism.
        self.exclude_same_stem_interferer = bool(exclude_same_stem_interferer)
        self.target_rms_range = tuple(float(x) for x in target_rms_range)
        self.global_gain_db_range = tuple(float(x) for x in global_gain_db_range)
        if len(self.target_rms_range) != 2 or self.target_rms_range[0] <= 0 or self.target_rms_range[1] < self.target_rms_range[0]:
            raise ValueError("target_rms_range must be (low, high) with 0 < low <= high")
        if len(self.global_gain_db_range) != 2 or self.global_gain_db_range[1] < self.global_gain_db_range[0]:
            raise ValueError("global_gain_db_range must be (low_db, high_db) with low <= high")
        # NOTE: every tensor in this file is cropped/padded to `clip_samples`,
        # historically the literal 65535 (not 16384*4 = 65536).  Keeping the odd
        # value is deliberate: it makes the STFT frame count 256 instead of 257,
        # which keeps the 4-level U-Net's time axis a clean power of two.
        self.clip_samples = int(clip_samples)
        self.target_length = self.clip_samples

        self.data_handler = AcapellaData(self.arate, str(self.data_path), fps=self.fps)
        self.list_args = []

        # 1. Index Vocal Segments
        print(f"--- Scanning Acapella dataset ---")
        for lang in languages:
            audio_lang_dir = self.data_path / 'audio' / lang
            if not audio_lang_dir.exists(): continue

            for wav_file in audio_lang_dir.glob("**/*.wav"):
                rel_path = wav_file.relative_to(audio_lang_dir)
                gender_sub = rel_path.parent
                stem = wav_file.stem
                lm_file = self.data_path / 'landmarks' / lang / gender_sub / f"{stem}.npy"
                
                if lm_file.exists():
                    try:
                        full_dur = librosa.get_duration(path=str(wav_file))
                        curr_offset = 0.0
                        while curr_offset + self.duration <= full_dur:
                            self.list_args.append({
                                'lang': lang, 'gender': str(gender_sub),
                                'stem': stem, 'offset': curr_offset
                            })
                            curr_offset += self.duration
                    except: continue

        # 2. Index Accompaniments (MUSDB .wav + AudioSet .ogg)
        self.musdb_files = list(self.musdb_path.glob("*.wav"))
        self.audioset_files = list(self.audioset_path.glob("**/*.ogg"))
        
        self.all_acc_files = self.musdb_files + self.audioset_files
        
        print(f"Dataset ready: {len(self.list_args)} vocal segments.")
        print(f"Pool: {len(self.musdb_files)} MUSDB tracks, {len(self.audioset_files)} AudioSet clips.")

    def __len__(self):
        return len(self.list_args)

    def _rng(self, index):
        """Per-item RNG.

        In deterministic mode the stream depends only on (seed, index), so the
        split is a reproducible manifest.  Otherwise fall back to the global
        `random` module, which PyTorch reseeds per worker per epoch.
        """
        if self.deterministic:
            return random.Random((self.seed * 1_000_003) ^ (index + 1))
        return random

    def _pick_interferer(self, index, rng):
        """Choose an interferer index that is not the same source recording.

        Segments of one `stem` are consecutive entries in `list_args`, so the
        old `(index + randint(1, N-1)) % N` rule regularly paired a singer with
        another 4 s window of *their own* take.  That makes the target/
        interferer distinction ill-posed and inflates the apparent difficulty.
        """
        n = len(self.list_args)
        if n < 2:
            return None
        if not self.exclude_same_stem_interferer:
            return (index + rng.randint(1, n - 1)) % n
        this = self.list_args[index]
        for _ in range(20):
            idx2 = (index + rng.randint(1, n - 1)) % n
            other = self.list_args[idx2]
            if other['stem'] != this['stem']:
                return idx2
        return (index + rng.randint(1, n - 1)) % n

    def __getitem__(self, index):
        rng = self._rng(index)

        # 1. Load Primary Singer (A)
        target_info = self.list_args[index]
        s1_audio = self.data_handler.load_audio(target_info['lang'], target_info['gender'], target_info['stem'], target_info['offset'], self.duration)

        # 2. Chance for Interference Singer (B)
        s2_audio = torch.zeros(self.clip_samples)
        inter_info = None
        if rng.random() < self.interference_prob:
            idx2 = self._pick_interferer(index, rng)
            if idx2 is not None:
                inter_info = self.list_args[idx2]
                s2_audio = self.data_handler.load_audio(inter_info['lang'], inter_info['gender'], inter_info['stem'], inter_info['offset'], self.duration)

        # 3. Load random Accompaniment (MUSDB or AudioSet)
        acc_file = rng.choice(self.all_acc_files)
        try:
            total_acc_dur = librosa.get_duration(path=str(acc_file))
            start_t = rng.uniform(0, max(0, total_acc_dur - self.duration))
            acc_audio, _ = librosa.load(str(acc_file), sr=self.arate, offset=start_t, duration=self.duration)
        except Exception:
            acc_audio = np.zeros(self.clip_samples) # Fallback for corrupted files

        acc_audio = pad_or_truncate(torch.from_numpy(acc_audio).float(), self.clip_samples)

        # 4. Mixing and RMS Normalization
        # Sample the target RMS instead of pinning every training target to 0.1.
        # Accompaniment/interferer are normalized to the same reference before
        # the explicit SNR draw, preserving the intended SNR semantics while
        # removing absolute target loudness as a shortcut.
        base_rms = rng.uniform(self.target_rms_range[0], self.target_rms_range[1])
        s1_audio = normalize_rms(s1_audio, target_rms=base_rms)
        acc_audio = normalize_rms(acc_audio, target_rms=base_rms)
        if inter_info is not None:
            s2_audio = normalize_rms(s2_audio, target_rms=base_rms)

        # Apply SNR adjustments relative to the uniform 0.1 RMS target baseline
        snr_mus = rng.uniform(self.snr_range[0], self.snr_range[1])
        snr_v2 = rng.uniform(self.snr_range[0], self.snr_range[1])

        mus_factor = 10 ** (-snr_mus / 20)
        v2_factor = 10 ** (-snr_v2 / 20)
        
        # Build raw mix and generate decoupled components
        mixture_raw = s1_audio + (acc_audio * mus_factor) + (s2_audio * v2_factor)
        s1_target = s1_audio
        s2_target = s2_audio * v2_factor

        # Global level augmentation changes absolute loudness without changing
        # target/interferer/accompaniment ratios.
        gain_db = rng.uniform(self.global_gain_db_range[0], self.global_gain_db_range[1])
        global_gain = 10 ** (gain_db / 20.0)
        mixture_raw = mixture_raw * global_gain
        s1_target = s1_target * global_gain
        s2_target = s2_target * global_gain

        # Final safety check to ensure overall scaling doesn't clip
        max_val = mixture_raw.abs().max() + 1e-8
        if max_val > 1.0:
            mixture = mixture_raw / max_val
            s1_target = s1_target / max_val
            s2_target = s2_target / max_val
        else:
            mixture = mixture_raw

        # 5. Identity Shuffling & Gating
        if self.is_train and inter_info is not None and rng.random() < self.identity_shuffle_prob:
            face_ld = self.data_handler.load_landmarks1(inter_info['lang'], inter_info['gender'], inter_info['stem'], inter_info['offset'], self.duration)
            target_audio = s2_target
        else:
            face_ld = self.data_handler.load_landmarks1(target_info['lang'], target_info['gender'], target_info['stem'], target_info['offset'], self.duration)
            target_audio = s1_target

        return [mixture, face_ld], target_audio