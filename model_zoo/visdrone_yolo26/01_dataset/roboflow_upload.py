import os
import roboflow
# Provide your key via env var (do NOT hard-code secrets in a tracked file):
#   export ROBOFLOW_API_KEY=xxxx   (get it from app.roboflow.com -> Settings -> API)
rf = roboflow.Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
ws = rf.workspace("Levi Pereira")

# Create (once) a project, then upload the sliced YOLO dataset.
project = ws.project("visdrone-sliced-416")            # or ws.create_project(...)
project.upload_dataset(
    dataset_path="/dataset/dfine/dataset/yolo_slice",  # folder with images + .txt labels (+ data yaml)
    num_workers=16,
    project_license="CC BY-NC-SA 4.0",                 # VisDrone is non-commercial research
    project_type="object-detection",
)
