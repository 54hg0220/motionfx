"""
Simple CLI for MotionFX
"""

import argparse
import sys
from .blend_methods import (
    load_image, save_image, avg_blur, flow_blur, gaussian_blur, linear_blur, max_blur, weighted_blur
)


def parse_vector(vector_str):
    """Parse motion vector from 'x,y' format"""
    try:
        x, y = vector_str.split(',')
        return float(x), float(y)
    except:
        raise ValueError(f"Invalid vector format: {vector_str}. Use 'x,y' format.")


def parse_weights(weights_str):
    """Parse weights from 'w1,w2,w3' format"""
    try:
        return [float(w) for w in weights_str.split(',')]
    except:
        raise ValueError(f"Invalid weights format: {weights_str}")


def main():
    parser = argparse.ArgumentParser(description="Simple Motion Blur Effects")
    parser.add_argument('method', choices=['avg', 'flow', 'gaussian', 'linear', 'max', 'weighted'],
                       help='Motion blur method')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    parser.add_argument('-v', '--vector', required=True, help='Motion vector "x,y"')
    parser.add_argument('-f', '--frames', type=int, default=10, help='Number of frames')
    parser.add_argument('-s', '--strength', type=float, default=10, help='Blur strength (for flow method)')
    parser.add_argument('-w', '--weights', help='Custom weights "w1,w2,w3" (for weighted method)')
    
    args = parser.parse_args()
    
    try:
        # Load image
        print(f"Loading image: {args.input}")
        image = load_image(args.input)
        
        # Parse motion vector
        motion_vector = parse_vector(args.vector)
        print(f"Motion vector: {motion_vector}")
        
        # Apply blur method
        print(f"Applying {args.method} blur...")
        
        if args.method == 'avg':
            result = avg_blur(image, motion_vector, args.frames)
        elif args.method == 'flow':
            result = flow_blur(image, motion_vector, args.strength)
        elif args.method == 'gaussian':
            result = gaussian_blur(image, motion_vector, args.frames)
        elif args.method == 'linear':
            result = linear_blur(image, motion_vector, args.frames)
        elif args.method == 'max':
            result = max_blur(image, motion_vector, args.frames)
        elif args.method == 'weighted':
            weights = None
            if args.weights:
                weights = parse_weights(args.weights)
                if len(weights) != args.frames:
                    print(f"Warning: {len(weights)} weights provided for {args.frames} frames")
                    args.frames = len(weights)
            result = weighted_blur(image, motion_vector, weights, args.frames)
        
        # Save result
        print(f"Saving result: {args.output}")
        save_image(result, args.output)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()