from QtGUI import SanSui
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Infermain import InferMain

if __name__ == '__main__':
    infermain = InferMain()

    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = SanSui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    ui.pushButton.clicked.connect(infermain.infer)
    ui.pushButton_2.clicked.connect(infermain.end)
    sys.exit(app.exec_())

