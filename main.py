import sys
import os
import re
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QFileDialog, QListWidget, QTextEdit, QMessageBox, 
                             QProgressBar, QLineEdit, QScrollArea, QLayout, QCheckBox)
from PyQt6.QtGui import QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal
from PIL import Image

# --- CLASSE CUSTOMIZADA: FlowLayout ---
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=5, spacing=5):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing()
            spaceY = self.spacing()
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

# --- CLASSE CUSTOMIZADA: Botão de Tag ---
class TagButton(QPushButton):
    rightClicked = pyqtSignal(str)
    leftClicked = pyqtSignal(str)

    def __init__(self, text, is_selected=True, parent=None):
        super().__init__(text, parent)
        self.tag_text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if is_selected:
            self.setProperty("tagState", "selected")
        else:
            self.setProperty("tagState", "available")
            
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self.tag_text)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.leftClicked.emit(self.tag_text)
            super().mousePressEvent(event)

# --- APLICAÇÃO PRINCIPAL ---
class BatchEditorAero(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Editor - Frutiger Aero (Visual Tags)")
        self.setMinimumSize(1100, 700)
        
        # State Variables
        self.current_folder = ""
        self.current_image_path = ""
        self.global_vocabulary = set()
        self.current_selected_tags = []
        
        # Load Stylesheet
        self.load_stylesheet()

        # Main Layout Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Tagger (Viewer & Visual Tags)
        self.tab_tagger = QWidget()
        self.setup_tagger_tab()
        self.tabs.addTab(self.tab_tagger, "Visualizador e Anotador (Tagger)")

        # Tab 2: Image Batch
        self.tab_image = QWidget()
        self.setup_image_batch_tab()
        self.tabs.addTab(self.tab_image, "Flip de Imagens em Lote")

        # Tab 3: Text Batch
        self.tab_text = QWidget()
        self.setup_text_batch_tab()
        self.tabs.addTab(self.tab_text, "Anotações em Lote")

    def load_stylesheet(self):
        qss_path = Path(__file__).parent / "style.qss"
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            print("style.qss não encontrado, usando tema padrão.")

    # ==========================================
    # --- TAB 1: TAGGER (VISUAL TAGS) ---
    # ==========================================
    def setup_tagger_tab(self):
        layout = QHBoxLayout(self.tab_tagger)
        
        # Left Panel: List
        left_panel = QVBoxLayout()
        self.btn_load_folder = QPushButton("Carregar Pasta de Imagens")
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.list_images = QListWidget()
        self.list_images.currentItemChanged.connect(self.on_image_selected)
        left_panel.addWidget(self.btn_load_folder)
        left_panel.addWidget(self.list_images, stretch=1)
        
        # Middle Panel: Image Preview
        middle_panel = QVBoxLayout()
        self.lbl_image_preview = QLabel("Selecione uma imagem...")
        self.lbl_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_preview.setMinimumSize(400, 400)
        middle_panel.addWidget(self.lbl_image_preview, stretch=1)

        # Right Panel: Tags Manager
        right_panel = QVBoxLayout()
        
        # Checkbox Separator
        self.chk_comma_separator = QCheckBox("Separar tags por vírgula (Illustrious vs Flux)")
        self.chk_comma_separator.setChecked(True)
        self.chk_comma_separator.stateChanged.connect(self.refresh_tags_ui)
        right_panel.addWidget(self.chk_comma_separator)

        # Input Area
        input_layout = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Adicionar tag...")
        self.tag_input.returnPressed.connect(self.add_new_tag_from_input)
        self.btn_add_tag = QPushButton("+")
        self.btn_add_tag.clicked.connect(self.add_new_tag_from_input)
        input_layout.addWidget(self.tag_input)
        input_layout.addWidget(self.btn_add_tag)
        
        right_panel.addWidget(QLabel("Adicionar Nova Tag:"))
        right_panel.addLayout(input_layout)

        # Selected Tags Area
        right_panel.addWidget(QLabel("Tags Selecionadas (Salvas no txt):"))
        self.scroll_selected = QScrollArea()
        self.scroll_selected.setWidgetResizable(True)
        self.widget_selected = QWidget()
        self.layout_selected = FlowLayout(self.widget_selected)
        self.scroll_selected.setWidget(self.widget_selected)
        right_panel.addWidget(self.scroll_selected, stretch=1)

        # Available / Removed Tags Area
        lbl_avail = QLabel("Tags Disponíveis / Removidas:")
        lbl_avail.setToolTip("Clique Esquerdo: Adicionar\nClique Direito: Excluir Globalmente")
        right_panel.addWidget(lbl_avail)
        self.scroll_available = QScrollArea()
        self.scroll_available.setWidgetResizable(True)
        self.widget_available = QWidget()
        self.layout_available = FlowLayout(self.widget_available)
        self.scroll_available.setWidget(self.widget_available)
        right_panel.addWidget(self.scroll_available, stretch=1)

        layout.addLayout(left_panel, 1)
        layout.addLayout(middle_panel, 2)
        layout.addLayout(right_panel, 2)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecione a Pasta")
        if folder:
            self.current_folder = folder
            self.list_images.clear()
            self.global_vocabulary.clear()
            
            valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
            images = [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in valid_exts)]
            
            # Scan global vocabulary from all txt files
            for f in os.listdir(folder):
                if f.lower().endswith(".txt"):
                    with open(os.path.join(folder, f), "r", encoding="utf-8") as txt_file:
                        content = txt_file.read()
                        tags = self.parse_tags(content)
                        for t in tags:
                            self.global_vocabulary.add(t)

            for img in images:
                self.list_images.addItem(img)
            
            if self.list_images.count() > 0:
                self.list_images.setCurrentRow(0)

    def parse_tags(self, text):
        if not text.strip(): return []
        if self.chk_comma_separator.isChecked():
            return [t.strip() for t in text.split(",") if t.strip()]
        else:
            return [t.strip() for t in text.split() if t.strip()]

    def format_tags(self, tags_list):
        if self.chk_comma_separator.isChecked():
            return ", ".join(tags_list)
        else:
            return " ".join(tags_list)

    def on_image_selected(self, current, previous):
        if not current:
            return

        image_name = current.text()
        self.current_image_path = os.path.join(self.current_folder, image_name)
        
        # Load Image
        pixmap = QPixmap(self.current_image_path)
        scaled_pixmap = pixmap.scaled(self.lbl_image_preview.size(), 
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.lbl_image_preview.setPixmap(scaled_pixmap)

        # Load Text and Tags
        txt_path = self.get_txt_path_for_image(image_name)
        self.current_selected_tags = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                self.current_selected_tags = self.parse_tags(f.read())
                for t in self.current_selected_tags:
                    self.global_vocabulary.add(t)

        self.refresh_tags_ui()

    def refresh_tags_ui(self):
        # Clear layouts
        self.clear_layout(self.layout_selected)
        self.clear_layout(self.layout_available)

        # Populate Selected
        for tag in self.current_selected_tags:
            btn = TagButton(tag, is_selected=True)
            btn.leftClicked.connect(self.on_tag_remove_click)
            self.layout_selected.addWidget(btn)

        # Populate Available
        available_tags = sorted(list(self.global_vocabulary - set(self.current_selected_tags)))
        for tag in available_tags:
            btn = TagButton(tag, is_selected=False)
            btn.leftClicked.connect(self.on_tag_add_click)
            btn.rightClicked.connect(self.on_tag_delete_global)
            self.layout_available.addWidget(btn)
            
        # Re-apply styles as dynamic properties might have changed
        self.widget_selected.setStyleSheet(self.styleSheet())
        self.widget_available.setStyleSheet(self.styleSheet())

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_txt_path_for_image(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        return os.path.join(self.current_folder, base_name + ".txt")

    def save_current_tags(self):
        if not self.current_image_path: return
        image_name = os.path.basename(self.current_image_path)
        txt_path = self.get_txt_path_for_image(image_name)
        content = self.format_tags(self.current_selected_tags)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

    def add_new_tag_from_input(self):
        text = self.tag_input.text()
        if not text.strip(): return
        
        new_tags = self.parse_tags(text)
        for t in new_tags:
            if t not in self.current_selected_tags:
                self.current_selected_tags.append(t)
            self.global_vocabulary.add(t)
            
        self.tag_input.clear()
        self.save_current_tags()
        self.refresh_tags_ui()

    def on_tag_remove_click(self, tag_text):
        if tag_text in self.current_selected_tags:
            self.current_selected_tags.remove(tag_text)
            self.save_current_tags()
            self.refresh_tags_ui()

    def on_tag_add_click(self, tag_text):
        if tag_text not in self.current_selected_tags:
            self.current_selected_tags.append(tag_text)
            self.save_current_tags()
            self.refresh_tags_ui()

    def on_tag_delete_global(self, tag_text):
        reply = QMessageBox.question(self, "Confirmar Exclusão", 
                                     f"Deseja excluir a tag '{tag_text}' do vocabulário global?\n(Isso não a removerá dos arquivos .txt antigos, apenas desta lista rápida)", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if tag_text in self.global_vocabulary:
                self.global_vocabulary.remove(tag_text)
            self.refresh_tags_ui()


    # ==========================================
    # --- TAB 2: IMAGE BATCH (FLIP) ---
    # ==========================================
    def setup_image_batch_tab(self):
        layout = QVBoxLayout(self.tab_image)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_title = QLabel("Espelhar (Inverter Horizontalmente) Imagens em Lote")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(lbl_title)

        self.btn_select_src = QPushButton("1. Selecionar Pasta de Origem")
        self.btn_select_src.clicked.connect(self.select_batch_src)
        self.lbl_src = QLabel("Origem: Nenhuma")
        
        self.btn_select_dst = QPushButton("2. Selecionar Pasta de Destino (Opcional, salva em subpasta se vazio)")
        self.btn_select_dst.clicked.connect(self.select_batch_dst)
        self.lbl_dst = QLabel("Destino: Nenhuma")

        self.btn_run_flip = QPushButton("3. INICIAR FLIP EM LOTE")
        self.btn_run_flip.setStyleSheet("background-color: #3b82f6; color: white;")
        self.btn_run_flip.clicked.connect(self.run_batch_flip)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout.addWidget(self.btn_select_src)
        layout.addWidget(self.lbl_src)
        layout.addWidget(self.btn_select_dst)
        layout.addWidget(self.lbl_dst)
        layout.addSpacing(20)
        layout.addWidget(self.btn_run_flip)
        layout.addWidget(self.progress_bar)

        self.batch_src = ""
        self.batch_dst = ""

    def select_batch_src(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Origem")
        if folder:
            self.batch_src = folder
            self.lbl_src.setText(f"Origem: {folder}")

    def select_batch_dst(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Destino")
        if folder:
            self.batch_dst = folder
            self.lbl_dst.setText(f"Destino: {folder}")

    def run_batch_flip(self):
        if not self.batch_src:
            QMessageBox.warning(self, "Aviso", "Selecione a pasta de origem primeiro!")
            return

        dst_folder = self.batch_dst
        if not dst_folder:
            dst_folder = os.path.join(self.batch_src, "Flip_Output")
            os.makedirs(dst_folder, exist_ok=True)
            self.lbl_dst.setText(f"Destino: {dst_folder}")

        valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        images = [f for f in os.listdir(self.batch_src) if any(f.lower().endswith(ext) for ext in valid_exts)]
        
        if not images:
            QMessageBox.information(self, "Info", "Nenhuma imagem encontrada na origem.")
            return

        self.progress_bar.setMaximum(len(images))
        self.progress_bar.setValue(0)

        for i, img_name in enumerate(images):
            src_path = os.path.join(self.batch_src, img_name)
            dst_path = os.path.join(dst_folder, img_name)
            try:
                with Image.open(src_path) as img:
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_img.save(dst_path)
            except Exception as e:
                print(f"Erro processando {img_name}: {e}")
            
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        QMessageBox.information(self, "Concluído", "Todas as imagens foram invertidas com sucesso!")


    # ==========================================
    # --- TAB 3: TEXT BATCH ---
    # ==========================================
    def setup_text_batch_tab(self):
        layout = QVBoxLayout(self.tab_text)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        lbl_title = QLabel("Anotação em Lote (Aplicar texto a todas as imagens da pasta)")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(lbl_title)

        self.btn_select_txt_folder = QPushButton("1. Selecionar Pasta com Imagens (Cria .txt para cada uma)")
        self.btn_select_txt_folder.clicked.connect(self.select_batch_txt_folder)
        self.lbl_txt_folder = QLabel("Pasta: Nenhuma")

        lbl_input = QLabel("Texto para inserir/sobrescrever nos arquivos:")
        self.txt_batch_input = QTextEdit()
        self.txt_batch_input.setMaximumHeight(150)

        self.btn_append_txt = QPushButton("Adicionar texto (Append) em todos os .txt")
        self.btn_append_txt.clicked.connect(lambda: self.run_batch_text("append"))
        
        self.btn_overwrite_txt = QPushButton("Sobrescrever todos os .txt com este texto")
        self.btn_overwrite_txt.clicked.connect(lambda: self.run_batch_text("overwrite"))

        layout.addWidget(self.btn_select_txt_folder)
        layout.addWidget(self.lbl_txt_folder)
        layout.addSpacing(10)
        layout.addWidget(lbl_input)
        layout.addWidget(self.txt_batch_input)
        layout.addWidget(self.btn_append_txt)
        layout.addWidget(self.btn_overwrite_txt)

        self.batch_txt_folder = ""

    def select_batch_txt_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder:
            self.batch_txt_folder = folder
            self.lbl_txt_folder.setText(f"Pasta: {folder}")

    def run_batch_text(self, mode):
        if not self.batch_txt_folder:
            QMessageBox.warning(self, "Aviso", "Selecione a pasta primeiro!")
            return

        text_to_write = self.txt_batch_input.toPlainText()
        if not text_to_write and mode == "append":
            QMessageBox.warning(self, "Aviso", "Digite algum texto para adicionar.")
            return

        valid_exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        images = [f for f in os.listdir(self.batch_txt_folder) if any(f.lower().endswith(ext) for ext in valid_exts)]
        
        count = 0
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(self.batch_txt_folder, base_name + ".txt")
            
            if mode == "overwrite":
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_to_write)
            elif mode == "append":
                prefix = "\n" if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 else ""
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(prefix + text_to_write)
            count += 1
            
        QMessageBox.information(self, "Concluído", f"Operação aplicada em {count} arquivos de texto.")

    def resizeEvent(self, event):
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BatchEditorAero()
    window.show()
    sys.exit(app.exec())
