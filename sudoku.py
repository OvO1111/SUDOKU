import numpy as np
from utils import ocr

sudokuExample = [[8, 0, 0, 0, 1, 0, 0, 0, 9],
                 [0, 5, 0, 8, 0, 7, 0, 1, 0],
                 [0, 0, 4, 0, 9, 0, 7, 0, 0],
                 [0, 6, 0, 7, 0, 1, 0, 2, 0],
                 [5, 0, 8, 0, 6, 0, 1, 0, 7],
                 [0, 1, 0, 5, 0, 2, 0, 9, 0],
                 [0, 0, 7, 0, 4, 0, 6, 0, 0],
                 [0, 8, 0, 3, 0, 9, 0, 4, 0],
                 [3, 0, 0, 0, 5, 0, 0, 0, 3]]  # (9, 1) 3 -> 5 -> 3
sudokuExample = np.array(sudokuExample)
sudokuErrorExample = sudokuExample[np.argwhere(sudokuExample == 8)[2]]


class Sudoku:
    __DIMENSION = 3
    __MAX_VARIABLE = __DIMENSION * __DIMENSION
    __TOTAL_CELL_NUM = __MAX_VARIABLE * __MAX_VARIABLE
    __VARIABLES = [i for i in range(1, __MAX_VARIABLE + 1)]
    
    def __init__(self, gridMap, avo=0):
        """
        :param gridMap: gridMap is a 2-d array containing the entries of the sudoku puzzle
        :param avo: avo is the conflicted numbers left
        """
        self._emptyCells = 0
        self._rows = []
        self._cells = []
        self._error = 3
        self._blocks = []
        self._columns = []
        self._errorLog = []
        self._functionBuffer = {}
        self._raw = gridMap
        self._grids = self._raw.copy()
        self.unfillable = None
        
        if self._raw.shape != (Sudoku.__MAX_VARIABLE, Sudoku.__MAX_VARIABLE):
            self._error = 0
            # self._errorHandle(errorFlag=0)
        perm = self.permutation()
        if not perm:
            candy = self._solve()
            if not np.all(candy != 0):
                self._error = 2
                # self._errorHandle(errorFlag=2, params=self._grids)
            else:
                self.solution = candy
        else:
            self._error = 1
            # self._errorHandle(errorFlag=1, params=perm)
        
        if self._error != 3:
            print(f"No sol, error = {self._error}")
        
    def __restore(self):
        self._emptyCells = 0
        self._cells = []
        self._rows = []
        self._columns = []
        self._blocks = []
        self._grids = self._raw.copy()
    
    def _solve(self):
        """
        Main solve function
        :return updated grid if solvable else
        :errorHandle() jump to error handling
        """
        rnd, delta = 0, 0
        self._checkUndone()
        while self._emptyCells:
            for cell in zip(self._checkUndone()[0], self._checkUndone()[1]):
                if len(self._cells[cell].val) == 0:
                    self.solution = self._grids
                    self.unfillable = cell
                    return self._grids
                
                elif len(self._cells[cell].val) == 1:
                    self._grids[cell] = self._cells[cell].val[0]
                    # print(rnd, ": ", cell, "->", self._cells[cell].val[0])
                    rnd += 1
                
                self._cellUpdate()
                
            if delta == self._emptyCells:  # every cell has more than 1 candidates, the sudoku has more than 1 sol
                undecided = np.argwhere(self._grids == 0)
                self._grids = self._traverse(undecided)
                return self._grids
            
            delta = self._emptyCells
        
        return self._grids
    
    def permutation(self):  # run at the start of the program, in the first part _cells is list
        for x, rows in enumerate(self._grids):
            self._cells.append([])
            for y, entry in enumerate(rows):
                cell = (x, y)
                value = self._grids[x, y] if self._grids[x, y] in Sudoku.__VARIABLES else 0
                candidate = self._available(cell)
                if value != 0 and value not in candidate:
                    # self.re_OCR_pos, self.reOCR = collision(value)
                    self.solution = self._grids
                    self.__restore()
                    return value, cell
                
                newCell = Cell(cell, candidate)
                self._cells[-1].append(newCell)
        
        self._cells = np.array(self._cells)
        for cell in self._cells.flat:
            cell.row = self._cells[cell.pos[0]]
            cell.column = self._cells[:, cell.pos[1]]
            cell.block = self._getblock(cell.pos, self._cells)
        
        # self.reOCR = -1
        return False
    
    def _traverse(self, blanks):
        """
        use traverse and backtracking algorithm to solve a multi-solution sudoku
        :param: current_grid: an partly-filled grid
        :return: one possible solution of the sudoku
        """
        cellStack, cellPtr, availPtr = {}, 0, 0
        recursed = False
        while cellPtr != len(blanks):
            cur = tuple(blanks[cellPtr])
            candy = self._available(cur)
            
            if len(candy) == 0:
                cellPtr -= 1
                recursed = True
                continue
            
            if not recursed:
                cellStack[cur] = [candy, availPtr]
            else:
                cellStack[cur] = [candy, cellStack[cur][1]+1]
            
            try:
                self._grids[cur] = candy[cellStack[cur][1]]
            except IndexError:
                if cur == tuple(blanks[0]):
                    self._error = 2
                    self.solution = self._grids
                    # self._errorHandle(errorFlag=2)
                    # self._grids = self._raw.copy()
                    break
                
                cellStack[cur] = [candy, 0]
                self._grids[cur] = 0
                recursed = True
                cellPtr -= 1
                continue
            
            cellPtr += 1
            recursed = False
            
        return self._grids
    
    def _available(self, cell):
        """
        Check a given cell's candidate number
        :return an 1-d np.array of the cell's candidate nums
        """
        cellX, cellY = cell
        temp, avail = [], []
        for i in range(Sudoku.__MAX_VARIABLE):
            temp.append(self._grids[cellX, i])  # row consistency
            temp.append(self._grids[i, cellY])  # column consistency
        for i in self._getblock(cell, self._grids):
            temp.append(i)  # block consistency
        for i in range(Sudoku.__DIMENSION):
            temp.remove(self._grids[cell])
        for num in Sudoku.__VARIABLES:
            if num not in temp:
                avail.append(num)
        avail = np.array(avail)
        return avail
    
    def _checkUndone(self):
        checkTmp = np.where(self._grids == 0)
        self._emptyCells = len(checkTmp[0])
        return checkTmp
    
    def _getblock(self, cell, grid):
        cellX, cellY = cell
        block_x_min = (cellX // Sudoku.__DIMENSION) * Sudoku.__DIMENSION
        block_x_max = (cellX // Sudoku.__DIMENSION + 1) * Sudoku.__DIMENSION - 1
        block_y_min = (cellY // Sudoku.__DIMENSION) * Sudoku.__DIMENSION
        block_y_max = (cellY // Sudoku.__DIMENSION + 1) * Sudoku.__DIMENSION - 1
        return grid[block_x_min:block_x_max + 1, block_y_min:block_y_max + 1].flatten()
    
    def _cellUpdate(self):
        for cell in self._cells.flat:
            pos = cell.pos
            cell.val = self._available(pos)
            cell.row, cell.column, cell.block = \
                self._grids[pos[0]], self._grids[:, pos[1]], self._getblock(pos, self._cells)
    
    def show_full(self):
        print("\n----Sudoku Result----")
        print("---------------------")
        for row in self._grids:
            print('|', end=' ')
            for entry in row:
                print(entry, end=' ')
            print('|')
    
    def show_candidate(self, rnd=-1):
        print(f"\nCandidates in Round {rnd}")
        print("----------------------")
        for row in self._cells:
            print('|', end=' ')
            for entry in row:
                print(entry.val, end=' ')
            print('|')
    
    @classmethod
    def setSudoku(cls, params=3):
        cls.__DIMENSION = params
        return cls.__DIMENSION


class Cell:
    def __init__(self,
                 pos,  # positions
                 candidates,  # candidate cells
                 ):
        self.pos = pos
        self.row, self.column, self.block = None, None, None
        self.val = candidates if len(candidates) == 1 else candidates


if __name__ == '__main__':
    a = Sudoku(sudokuExample)
    p = a.solution
    # print(p)
