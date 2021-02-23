Hereafter is the hierarchy tree and main introduction for this Sudoku project:
-


`code` is the root directory for Sudoku project, in order to make sure this project runs smoothly on 
your computer, you should at least pre-install all the packages in the package list below:
- torchvision (ver > 0.5.0)
- pytorch (ver > 1.4.0)
- opencv (ver > 4.4.0)
- imutils (ver > 0.5.0)
- numpy (ver > 1.18.5)

--------
Based on these libraries, you could input `python solve_sudoku_puzzle.py --model output/torch128.ckpt --image img/1-1.png`
in the command line to check the results of `1-1.png`, or change the model and image as you like.
----
`./img` is the directory for Sudokus  
`./output` is the directory for results generated in the process
---
Finally, the main program is in file `solve_sudoku_puzzle.py`, which leverages the Sudoku solving program in
`sudoku.py` and optimized LeNet-5 neural network in `./pyimagesearch/models/lenet.py`.
