import argparse
from src.pipeline import DetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Food & Vessel Detection Pipeline")
    parser.add_argument("--path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--distance", type=float, default=None, help="Distance to the object in cm (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--fov", type=float, default=None, help="Camera Field of View in degrees (optional override)")
    
    args = parser.parse_args()
    
    pipeline = DetectionPipeline(device=args.device)
    pipeline.run(args.path, distance=args.distance, fov=args.fov)

if __name__ == "__main__":
    main()
