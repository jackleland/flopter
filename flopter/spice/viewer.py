# import sys
# import PyQt5.QtWidgets as qtw
# import PyQt5.QtGui as qtg
#
#
# class Spice2Viewer(qtw.QWidget):
#
#     def __init__(self):
#         super().__init__()
#         self.init_ui()
#
#     def init_ui(self):
#         open_button = qtw.QPushButton('Open', self)
#         open_button.setToolTip('Open a spice input-file')
#
#         quit_button = qtw.QPushButton('Quit', self)
#         quit_button.clicked.connect(qtw.QApplication.instance().quit)
#         quit_button.setToolTip('Quit the program')
#         # btn.resize(btn.sizeHint())
#         # btn.move(25, 25)
#
#         hbox = qtw.QHBoxLayout()
#         hbox.addStretch(1)
#         hbox.addWidget(open_button)
#         hbox.addWidget(quit_button)
#
#         vbox = qtw.QVBoxLayout()
#         vbox.addStretch(1)
#         vbox.addLayout(hbox)
#
#         self.setLayout(vbox)
#
#         self.resize(600, 440)
#         self.setWindowTitle('SPICE Viewer')
#         self.center()
#         self.show()
#
#     def center(self):
#         qr = self.frameGeometry()
#         cp = qtw.QDesktopWidget().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())
#
#
# if __name__ == '__main__':
#     app = qtw.QApplication(sys.argv)
#     ex = Spice2Viewer()
#     sys.exit(app.exec_())

import sys
import time

import numpy as np

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(static_canvas)
        self.addToolBar(NavigationToolbar(static_canvas, self))

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea,
                        NavigationToolbar(dynamic_canvas, self))

        self._static_ax = static_canvas.figure.subplots()
        t = np.linspace(0, 10, 501)
        self._static_ax.plot(t, np.tan(t), ".")

        self._dynamic_ax = dynamic_canvas.figure.subplots()
        self._timer = dynamic_canvas.new_timer(
            100, [(self._update_canvas, (), {})])
        self._timer.start()

    def _update_canvas(self):
        self._dynamic_ax.clear()
        t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        self._dynamic_ax.plot(t, np.sin(t + time.time()))
        self._dynamic_ax.figure.canvas.draw()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
