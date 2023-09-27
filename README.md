# Work_task
## Introduction
Task is to implement a modified Octree algorithm. The algorithm should construct a standard Octree, but we want you to embed a sphere in each of the Octree cubes. All the points within a sphere should further be subdivided into eight cubes (with spheres inside). All the points that are outside of the sphere can be discarded. Visualize the results.
The data is provided in las format.

## How to run
1. Pull source code from Git repository and open directory:
```bash
https://github.com/vladas333/Work_task.git
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/Scripts/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```
4. Run the the code:
```bash
python app.py
```