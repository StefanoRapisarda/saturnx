import sys
from fitting_app.application import FitApp

sys.setrecursionlimit(1000000)

if __name__ == '__main__':
    app = FitApp()
    app.mainloop()

