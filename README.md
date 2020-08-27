# Sudoku Solver

This program serves as a way to calculate the solution to any 9x9 sudoku puzzle via webcam.
It identifies the puzzle through the webcam, processes it uses OpenCV, runs against a neural network to predict the digits, 
and runs an efficient sudoku solver to determine the answer. It then displays the answer on the same frame if it is solvable.

Demonstration: [YouTube Video](https://www.youtube.com/watch?v=O6WfZRyatcY)

<img src="test_imgs/screen.png"> 

Tested using `Python 3.6` (newer versions may or may not work)

Relevant Packages:

- `opencv-python`: 4.3.0.36
- `numpy`: 1.19.1
- `tensorflow`: 2.2.0
- `sklearn`: 0.0
- `keras`: 2.3.1

```bash
python app.py
```

