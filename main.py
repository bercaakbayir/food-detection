import argparse
from src.pipeline import DetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Food & Vessel Detection Pipeline")
    parser.add_argument("--path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--distance", type=float, default=None, help="Distance to the object in cm (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    pipeline = DetectionPipeline(device=args.device)
    pipeline.run(args.path, args.distance)

if __name__ == "__main__":
    main()

