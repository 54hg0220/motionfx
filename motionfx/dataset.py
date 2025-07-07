"""
Dataset creation tools for motion blur research

This module helps create paired datasets of sharp/blurred images for training
deep learning models, particularly for video deblurring applications.

Used in research: "Parallel scale de-blur net for sharpening video images 
for remote clinical assessment of hand movements"
https://doi.org/10.1016/j.eswa.2023.121093
"""

import os
import random
import numpy as np
from PIL import Image
import cv2
from .blend_methods import (
    load_image, save_image, avg_blur, flow_blur, gaussian_blur, linear_blur, max_blur
)



class MotionBlurDatasetGenerator:
    """Generate motion blur datasets for training deblurring models"""
    
    def __init__(self, output_dir="motion_blur_dataset"):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Directory to save the dataset
        """
        self.output_dir = output_dir
        self.sharp_dir = os.path.join(output_dir, "sharp")
        self.blurred_dir = os.path.join(output_dir, "blurred")
        
        # Create directories
        os.makedirs(self.sharp_dir, exist_ok=True)
        os.makedirs(self.blurred_dir, exist_ok=True)
    
    def process_single_image(self, image_path, num_variants=5):
        """
        Create multiple motion blur variants from a single sharp image
        
        Args:
            image_path: Path to sharp input image
            num_variants: Number of blurred variants to create
        """
        # Load sharp image
        sharp_image = load_image(image_path)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save sharp image
        sharp_output = os.path.join(self.sharp_dir, f"{base_name}.jpg")
        save_image(sharp_image, sharp_output)
        
        print(f"Processing: {base_name}")
        
        # Generate blur variants
        for i in range(num_variants):
            # Random motion parameters
            motion_params = self._generate_random_motion()
            
            # Apply random blur method
            blur_method = random.choice(['avg', 'gaussian', 'linear', 'max'])
            blurred_image = self._apply_blur(sharp_image, blur_method, motion_params)
            
            # Save blurred variant
            blur_output = os.path.join(self.blurred_dir, f"{base_name}_blur_{i+1}.jpg")
            save_image(blurred_image, blur_output)
            
            print(f"  Created blur variant {i+1}: {blur_method} method")
    
    def process_directory(self, input_dir, num_variants=5, image_extensions=('.jpg', '.jpeg', '.png')):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing sharp images
            num_variants: Number of blur variants per image
            image_extensions: Supported image file extensions
        """
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images to process")
        
        for i, filename in enumerate(image_files, 1):
            image_path = os.path.join(input_dir, filename)
            print(f"\n[{i}/{len(image_files)}] Processing {filename}")
            
            try:
                self.process_single_image(image_path, num_variants)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        
        print(f"\nDataset creation completed!")
        print(f"Sharp images: {self.sharp_dir}")
        print(f"Blurred images: {self.blurred_dir}")
    
    def create_video_sequence_dataset(self, video_path, frame_interval=30, num_variants=3):
        """
        Create dataset from video frames (useful for clinical hand movement analysis)
        
        Args:
            video_path: Path to input video
            frame_interval: Extract every N frames
            num_variants: Blur variants per frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_count = 0
        
        print(f"Processing video: {os.path.basename(video_path)}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save as temporary image and process
                temp_path = f"temp_frame_{extracted_count}.jpg"
                save_image(frame_rgb, temp_path)
                
                try:
                    self.process_single_image(temp_path, num_variants)
                    extracted_count += 1
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {extracted_count} frames from video")
    
    def _generate_random_motion(self):
        """Generate random motion parameters"""
        # Random motion vector (clinical hand movements are typically 5-30 pixels)
        motion_magnitude = random.uniform(5, 25)
        motion_angle = random.uniform(0, 2 * np.pi)
        
        dx = motion_magnitude * np.cos(motion_angle)
        dy = motion_magnitude * np.sin(motion_angle)
        
        # Random number of frames
        num_frames = random.randint(8, 15)
        
        return {
            'motion_vector': (dx, dy),
            'num_frames': num_frames,
            'strength': random.uniform(8, 20)  # For flow method
        }
    
    def _apply_blur(self, image, method, params):
        """Apply blur method with given parameters"""
        motion_vector = params['motion_vector']
        num_frames = params['num_frames']
        strength = params['strength']
        
        if method == 'avg':
            return avg_blur(image, motion_vector, num_frames)
        elif method == 'gaussian':
            return gaussian_blur(image, motion_vector, num_frames)
        elif method == 'linear':
            return linear_blur(image, motion_vector, num_frames)
        elif method == 'max':
            return max_blur(image, motion_vector, num_frames)
        elif method == 'flow':
            return flow_blur(image, motion_vector, strength)
        else:
            return avg_blur(image, motion_vector, num_frames)
    
    def create_training_splits(self, train_ratio=0.8, val_ratio=0.1):
        """
        Create train/val/test splits for machine learning
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set (rest goes to test)
        """
        # Get all sharp images
        sharp_images = [f for f in os.listdir(self.sharp_dir) if f.endswith('.jpg')]
        random.shuffle(sharp_images)
        
        n_total = len(sharp_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Create split directories
        splits = {
            'train': sharp_images[:n_train],
            'val': sharp_images[n_train:n_train+n_val],
            'test': sharp_images[n_train+n_val:]
        }
        
        for split_name, image_list in splits.items():
            split_sharp_dir = os.path.join(self.output_dir, split_name, "sharp")
            split_blur_dir = os.path.join(self.output_dir, split_name, "blurred")
            
            os.makedirs(split_sharp_dir, exist_ok=True)
            os.makedirs(split_blur_dir, exist_ok=True)
            
            for img_name in image_list:
                # Copy sharp image
                src_sharp = os.path.join(self.sharp_dir, img_name)
                dst_sharp = os.path.join(split_sharp_dir, img_name)
                Image.open(src_sharp).save(dst_sharp)
                
                # Copy corresponding blurred images
                base_name = os.path.splitext(img_name)[0]
                for blur_file in os.listdir(self.blurred_dir):
                    if blur_file.startswith(base_name + "_blur_"):
                        src_blur = os.path.join(self.blurred_dir, blur_file)
                        dst_blur = os.path.join(split_blur_dir, blur_file)
                        Image.open(src_blur).save(dst_blur)
        
        print(f"Created dataset splits:")
        print(f"  Train: {len(splits['train'])} images")
        print(f"  Val: {len(splits['val'])} images") 
        print(f"  Test: {len(splits['test'])} images")


def create_clinical_hand_dataset(input_dir, output_dir="clinical_hand_dataset", num_variants=5):
    """
    Create motion blur dataset specifically for clinical hand movement analysis
    
    This function creates datasets suitable for training models as described in:
    "Parallel scale de-blur net for sharpening video images for remote clinical assessment of hand movements"
    https://doi.org/10.1016/j.eswa.2023.121093
    
    Args:
        input_dir: Directory with sharp hand movement images/videos
        output_dir: Output directory for dataset
        num_variants: Number of blur variants per image
    """
    generator = MotionBlurDatasetGenerator(output_dir)
    
    print("Creating clinical hand movement motion blur dataset...")
    print("Based on research: Parallel scale de-blur net for sharpening video images")
    print("Paper: https://doi.org/10.1016/j.eswa.2023.121093\n")
    
    # Process images
    generator.process_directory(input_dir, num_variants)
    
    # Create ML-ready splits
    generator.create_training_splits()
    
    print("\nDataset ready for training deblurring models!")
    return generator