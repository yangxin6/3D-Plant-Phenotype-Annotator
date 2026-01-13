from PyQt5 import QtWidgets
from ui.main_window import LeafAnnotatorWindow


def main():
    app = QtWidgets.QApplication([])
    w = LeafAnnotatorWindow()
    w.resize(1500, 850)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
