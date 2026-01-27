from PyQt5 import QtWidgets
from ui.main_window import LeafAnnotatorWindow


def main():
    app = QtWidgets.QApplication([])
    app.setStyleSheet("QWidget { font-size: 18px; }")
    w = LeafAnnotatorWindow()
    w.resize(1920, 1080)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
