# mdok-style @ POLAR

The mdok-style systems submitted to the [SemEval-2026 Task 9](https://github.com/Polar-SemEval) shared task.

## Cite
If you use the data, code, or the information in this repository, cite the following paper.

TBA

## Source Code Structure
| File | Description |
| :- | :- |
|polar1.py|the script for training and inference of the mdok-style submitted to subtask 1 of POLAR|
|polar2.py|the script for training and inference of the mdok-style submitted to subtask 2 of POLAR|

## Installation
Clone and install the [IMGTB framework](https://github.com/kinit-sk/IMGTB), activate the conda environment.
   ```
   git clone https://github.com/kinit-sk/IMGTB.git
   cd IMGTB
   conda env create -f environment.yaml
   conda activate IMGTB
   ```

## Code Usage
1. To retrain the Qwen3-32B or Gemma-3-27b model for subtasks, run the enclosed code (refer to polar1.py for subtask 1 and polar2.py for subtask 2, for subtask 3 just adjust the classes in the polar2.py script).

2. To run just inference, append ```--test_only``` option to the above mentioned script.
