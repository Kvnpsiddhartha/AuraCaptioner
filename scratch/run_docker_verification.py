import os
import json
import subprocess
from pathlib import Path

def main():
    base_dir = Path("/home/siddhartha/personal/video-captioning-agent-phase8-9/video-captioning-agent")
    test_dir = base_dir / "docker_test"
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create tasks.json with dropped/varied styles as requested
    tasks = [
        {
            "task_id": "v1",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
            "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
        },
        {
            "task_id": "v2",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
            "styles": ["formal", "humorous_non_tech"]
        },
        {
            "task_id": "v3",
            "video_url": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
            "styles": ["sarcastic", "humorous_tech", "humorous_non_tech"]
        }
    ]
    
    tasks_file = input_dir / "tasks.json"
    tasks_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"Created tasks.json at {tasks_file}")
    
    # Clean output if present
    results_file = output_dir / "results.json"
    if results_file.exists():
        results_file.unlink()
        print("Removed existing results.json")
        
    # 2. Build Docker Image
    print("Building Docker image...")
    build_cmd = ["docker", "build", "-t", "video-captioning-agent:latest", str(base_dir)]
    subprocess.run(build_cmd, check=True)
    print("Docker image built successfully!")
    
    # 3. Run Docker Container
    print("Running Docker container...")
    run_cmd = [
        "docker", "run", "--rm",
        "-e", "FIREWORKS_API_KEY=fw_WZKKzcPMxV2jezsXg5YB5j",
        "-v", f"{input_dir}:/input",
        "-v", f"{output_dir}:/output",
        "video-captioning-agent:latest"
    ]
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    
    print("\n--- Container Stdout ---")
    print(result.stdout)
    print("--- Container Stderr ---")
    print(result.stderr)
    
    if result.returncode != 0:
        print(f"Container failed with return code {result.returncode}")
        exit(1)
        
    print("Container finished successfully!")
    
    # 4. Verify results.json
    if results_file.exists():
        print("\n=== results.json generated successfully ===")
        print(results_file.read_text(encoding="utf-8"))
    else:
        print("\nERROR: results.json was not created!")
        exit(1)

if __name__ == "__main__":
    main()
