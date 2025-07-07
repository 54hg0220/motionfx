"""
Simple motion blur blend methods
"""

import numpy as np
import cv2
from PIL import Image


def load_image(img_path):
    """Load image from file path"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image, output_path):
    """Save image to file"""
    if image.ndim == 3:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(output_path, image)


def avg_blur(image, motion_vector, num_frames=10):
    """Average motion blur"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        shift_x = int(dx * t)
        shift_y = int(dy * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        frames.append(shifted.astype(np.float32))
    
    result = np.mean(frames, axis=0)
    return np.clip(result, 0, 255).astype(np.uint8)


def flow_blur(image, motion_vector, strength=10):
    """Flow-based motion blur"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    # Normalize direction
    magnitude = np.sqrt(dx*dx + dy*dy)
    if magnitude > 0:
        dx, dy = dx/magnitude, dy/magnitude
    
    frames = []
    num_samples = max(5, int(strength / 2))
    
    for i in range(num_samples):
        t = (i - num_samples/2) / num_samples
        shift_x = int(dx * strength * t)
        shift_y = int(dy * strength * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        frames.append(shifted.astype(np.float32))
    
    result = np.mean(frames, axis=0)
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_blur(image, motion_vector, num_frames=15):
    """Gaussian weighted motion blur"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    frames = []
    weights = []
    
    for i in range(num_frames):
        t = (i - (num_frames-1)/2) / (num_frames-1)
        shift_x = int(dx * t)
        shift_y = int(dy * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        frames.append(shifted.astype(np.float32))
        
        # Gaussian weight
        weight = np.exp(-t*t / 0.5)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    result = np.zeros_like(frames[0])
    for frame, weight in zip(frames, weights):
        result += frame * weight
    
    return np.clip(result, 0, 255).astype(np.uint8)


def linear_blur(image, motion_vector, num_frames=10):
    """Linear weighted motion blur"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    frames = []
    weights = []
    
    for i in range(num_frames):
        t = (i - (num_frames-1)/2) / (num_frames-1)
        shift_x = int(dx * t)
        shift_y = int(dy * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        frames.append(shifted.astype(np.float32))
        
        # Linear weight (center-weighted)
        weight = 1.0 - abs(t)
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    result = np.zeros_like(frames[0])
    for frame, weight in zip(frames, weights):
        result += frame * weight
    
    return np.clip(result, 0, 255).astype(np.uint8)


def max_blur(image, motion_vector, num_frames=12):
    """Maximum value motion blur (light trails)"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        shift_x = int(dx * t)
        shift_y = int(dy * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        
        # Apply decay for trail effect
        decay = 0.9 ** i
        shifted = shifted * decay
        frames.append(shifted.astype(np.float32))
    
    result = np.maximum.reduce(frames)
    return np.clip(result, 0, 255).astype(np.uint8)


def weighted_blur(image, motion_vector, custom_weights=None, num_frames=10):
    """Custom weighted motion blur"""
    h, w = image.shape[:2]
    dx, dy = motion_vector
    
    if custom_weights is None:
        custom_weights = [1.0] * num_frames
    
    if len(custom_weights) != num_frames:
        raise ValueError("Number of weights must match number of frames")
    
    # Normalize weights
    weights = np.array(custom_weights) / np.sum(custom_weights)
    
    frames = []
    for i in range(num_frames):
        t = (i - (num_frames-1)/2) / (num_frames-1)
        shift_x = int(dx * t)
        shift_y = int(dy * t)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        frames.append(shifted.astype(np.float32))
    
    result = np.zeros_like(frames[0])
    for frame, weight in zip(frames, weights):
        result += frame * weight
    
    return np.clip(result, 0, 255).astype(np.uint8)