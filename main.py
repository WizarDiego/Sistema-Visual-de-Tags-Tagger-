import os
import re
import sys
import traceback
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QFrame,
    QSizePolicy,
    QGridLayout,
)

try:
    import ai_analyzer
except ImportError as error:
    ai_analyzer = None
    print("Aviso: ai_analyzer nao pode ser importado. Falta instalar dependencias?", error)


class CopyableMessageDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 420)

        layout = QVBoxLayout(self)

        description = QLabel(
            "Mensagem completa abaixo. Voce pode copiar o texto para pesquisar, pedir ajuda ou guardar no projeto."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlainText(message)
        layout.addWidget(self.text_box, stretch=1)

        buttons = QHBoxLayout()
        copy_button = QPushButton("Copiar Mensagem")
        copy_button.clicked.connect(self.copy_message)
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        buttons.addWidget(copy_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

    def copy_message(self):
        QApplication.clipboard().setText(self.text_box.toPlainText())


class AITestWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, models_dir, device_preference="auto", model_key="microsoft-large"):
        super().__init__()
        self.models_dir = models_dir
        self.device_preference = device_preference
        self.model_key = model_key

    def run(self):
        try:
            if not ai_analyzer:
                self.error.emit("Modulo ai_analyzer nao encontrado.")
                return

            manager = ai_analyzer.AIModelManager(
                self.models_dir,
                device_preference=self.device_preference,
                model_key=self.model_key,
            )
            self.finished.emit(manager.preload_models())
        except Exception as error:
            traceback.print_exc()
            self.error.emit(f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}")


class AIWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    single_result = pyqtSignal(str)

    def __init__(
        self,
        task,
        bbox_format,
        images_list,
        models_dir=None,
        ignored_words="",
        auto_save=True,
        caption_prompt="<CAPTION>",
        caption_tokens=96,
        tags_tokens=48,
        bbox_output_dir="",
        device_preference="auto",
        model_key="microsoft-large",
    ):
        super().__init__()
        self.task = task
        self.bbox_format = bbox_format
        self.images_list = images_list
        self.models_dir = models_dir
        self.ignored_words = [word.strip().lower() for word in ignored_words.split(",") if word.strip()]
        self.auto_save = auto_save
        self.is_running = True
        self.yolo_class_map = {}
        self.caption_prompt = caption_prompt
        self.caption_tokens = caption_tokens
        self.tags_tokens = tags_tokens
        self.bbox_output_dir = bbox_output_dir
        self.device_preference = device_preference
        self.model_key = model_key

    def run(self):
        try:
            if not ai_analyzer:
                self.error.emit("Modulo ai_analyzer nao encontrado. Verifique as dependencias.")
                return

            manager = ai_analyzer.AIModelManager(
                self.models_dir,
                device_preference=self.device_preference,
                model_key=self.model_key,
            )

            for index, image_path in enumerate(self.images_list, start=1):
                if not self.is_running:
                    break

                base_name = os.path.splitext(image_path)[0]
                result_text = self.process_image(manager, image_path, base_name)

                if not self.auto_save and index == 1:
                    self.single_result.emit(result_text)

                self.progress.emit(index)

            self.finished.emit("Processamento concluido!")
        except Exception as error:
            traceback.print_exc()
            self.error.emit(f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}")

    def process_image(self, manager, image_path, base_name):
        if self.task == "caption":
            caption = manager.generate_caption(
                image_path,
                task_prompt=self.caption_prompt,
                max_new_tokens=self.caption_tokens,
            )
            for word in self.ignored_words:
                caption = re.sub(r"\b" + re.escape(word) + r"\b", "", caption, flags=re.IGNORECASE)
            caption = re.sub(r"\s+", " ", caption).strip(" ,")
            if self.auto_save:
                with open(base_name + ".txt", "w", encoding="utf-8") as file_obj:
                    file_obj.write(caption)
            return caption

        if self.task == "tags":
            tags = manager.generate_tags(image_path, max_new_tokens=self.tags_tokens)
            parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            parsed_tags = [tag for tag in parsed_tags if tag.lower() not in self.ignored_words]
            tags_text = ", ".join(parsed_tags)
            if self.auto_save:
                with open(base_name + ".txt", "w", encoding="utf-8") as file_obj:
                    file_obj.write(tags_text)
            return tags_text

        if self.task == "analysis":
            caption = manager.generate_prompt_analysis(
                image_path,
                max_new_tokens=self.caption_tokens,
            )
            for word in self.ignored_words:
                caption = re.sub(r"\b" + re.escape(word) + r"\b", "", caption, flags=re.IGNORECASE)
            caption = re.sub(r"\s+", " ", caption).strip(" ,")
            if self.auto_save:
                with open(base_name + ".txt", "w", encoding="utf-8") as file_obj:
                    file_obj.write(caption)
            return caption

        if self.task == "bbox":
            result = manager.generate_bounding_boxes(image_path)
            if "<OD>" in result:
                result = result["<OD>"]
            if "bboxes" not in result or "labels" not in result:
                raise ValueError(f"Resposta inesperada do Florence-2 para bounding boxes: {result}")

            bboxes = result["bboxes"]
            labels = result["labels"]
            output_dir = self.resolve_bbox_output_dir(image_path)
            os.makedirs(output_dir, exist_ok=True)
            file_stem = os.path.splitext(os.path.basename(image_path))[0]
            if self.bbox_format == "yolo":
                with Image.open(image_path) as image_obj:
                    width, height = image_obj.size
                yolo_path = os.path.join(output_dir, file_stem + "_yolo.txt")
                self.yolo_class_map = ai_analyzer.save_yolo_txt(
                    bboxes,
                    labels,
                    width,
                    height,
                    yolo_path,
                    self.yolo_class_map,
                )
                classes_path = ai_analyzer.write_yolo_classes(output_dir, self.yolo_class_map)
                return f"YOLO salvo em:\n{yolo_path}\n\nClasses salvas em:\n{classes_path}"

            output_image = os.path.join(output_dir, file_stem + "_bboxes.png")
            ai_analyzer.draw_bounding_boxes(image_path, bboxes, labels, output_image)
            return f"Imagem com bounding boxes salva em:\n{output_image}"

        raise ValueError(f"Tarefa de IA invalida: {self.task}")

    def stop(self):
        self.is_running = False

    def resolve_bbox_output_dir(self, image_path):
        if self.bbox_output_dir:
            return self.bbox_output_dir
        return os.path.join(os.path.dirname(image_path), "Box_Research")


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=5, spacing=5):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.item_list = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.item_list.append(item)

    def count(self):
        return len(self.item_list)

    def itemAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.item_list):
            return self.item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(2 * margins.top(), 2 * margins.top())
        return size

    def do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0

        for item in self.item_list:
            next_x = x + item.sizeHint().width() + self.spacing()
            if next_x - self.spacing() > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + self.spacing()
                next_x = x + item.sizeHint().width() + self.spacing()
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()


class TagButton(QPushButton):
    rightClicked = pyqtSignal(str)
    leftClicked = pyqtSignal(str)

    def __init__(self, text, is_selected=True, parent=None):
        super().__init__(text, parent)
        self.tag_text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("tagState", "selected" if is_selected else "available")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self.tag_text)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.leftClicked.emit(self.tag_text)
        super().mousePressEvent(event)


class BatchEditorAero(QMainWindow):
    VALID_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch Editor - Frutiger Aero (Visual Tags)")
        self.setMinimumSize(1120, 760)
        self.configure_fonts()

        self.current_folder = ""
        self.current_image_path = ""
        self.global_vocabulary = set()
        self.current_selected_tags = []

        self.batch_src = ""
        self.batch_dst = ""
        self.batch_txt_folder = ""
        self.ai_folder = ""
        self.models_folder = ""
        self.bbox_output_folder = ""
        self.ai_worker = None
        self.ai_test_worker = None
        self.last_ai_preview_path = ""
        self.ai_model_ready = False

        self.load_stylesheet()
        self.setup_ui()
        self.refresh_model_status()

    def configure_fonts(self):
        app = QApplication.instance()
        if app is None:
            return

        base_font = QFont("Segoe UI", 10)
        app.setFont(base_font)

    def load_stylesheet(self):
        qss_path = Path(__file__).parent / "style.qss"
        if qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as file_obj:
                self.setStyleSheet(file_obj.read())

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_tagger = QWidget()
        self.setup_tagger_tab()
        self.tabs.addTab(self.tab_tagger, "Visualizador e Anotador")

        self.tab_image = QWidget()
        self.setup_image_batch_tab()
        self.tabs.addTab(self.tab_image, "Flip de Imagens em Lote")

        self.tab_text = QWidget()
        self.setup_text_batch_tab()
        self.tabs.addTab(self.tab_text, "Anotacoes em Lote")

        self.tab_ai = QWidget()
        self.setup_ai_analysis_tab()
        self.tabs.addTab(self.tab_ai, "Auto Analise IA")

    def setup_tagger_tab(self):
        layout = QHBoxLayout(self.tab_tagger)

        left_panel = QVBoxLayout()
        self.btn_load_folder = QPushButton("Carregar Pasta de Imagens")
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.list_images = QListWidget()
        self.list_images.currentItemChanged.connect(self.on_image_selected)
        left_panel.addWidget(self.btn_load_folder)
        left_panel.addWidget(self.list_images, stretch=1)

        middle_panel = QVBoxLayout()
        self.lbl_image_preview = QLabel("Selecione uma imagem...")
        self.lbl_image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_preview.setMinimumSize(400, 400)
        middle_panel.addWidget(self.lbl_image_preview, stretch=1)

        right_panel = QVBoxLayout()
        self.chk_comma_separator = QCheckBox("Separar tags por virgula")
        self.chk_comma_separator.setChecked(True)
        self.chk_comma_separator.stateChanged.connect(self.refresh_tags_ui)
        right_panel.addWidget(self.chk_comma_separator)

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

        right_panel.addWidget(QLabel("Tags Selecionadas (.txt atual):"))
        self.scroll_selected = QScrollArea()
        self.scroll_selected.setWidgetResizable(True)
        self.widget_selected = QWidget()
        self.layout_selected = FlowLayout(self.widget_selected)
        self.scroll_selected.setWidget(self.widget_selected)
        right_panel.addWidget(self.scroll_selected, stretch=1)

        lbl_avail = QLabel("Tags Disponiveis / Removidas:")
        lbl_avail.setToolTip("Clique esquerdo: adicionar\nClique direito: excluir da lista global")
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

    def setup_image_batch_tab(self):
        layout = QVBoxLayout(self.tab_image)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Espelhar Imagens em Lote")
        title.setProperty("role", "sectionTitle")
        title.style().unpolish(title)
        title.style().polish(title)
        layout.addWidget(title)

        self.btn_select_src = QPushButton("1. Selecionar Pasta de Origem")
        self.btn_select_src.clicked.connect(self.select_batch_src)
        self.lbl_src = QLabel("Origem: Nenhuma")

        self.btn_select_dst = QPushButton("2. Selecionar Pasta de Destino (opcional)")
        self.btn_select_dst.clicked.connect(self.select_batch_dst)
        self.lbl_dst = QLabel("Destino: Nenhuma")

        self.btn_run_flip = QPushButton("3. Iniciar Flip em Lote")
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

    def setup_text_batch_tab(self):
        layout = QVBoxLayout(self.tab_text)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Anotacao em Lote")
        title.setProperty("role", "sectionTitle")
        title.style().unpolish(title)
        title.style().polish(title)
        layout.addWidget(title)

        self.btn_select_txt_folder = QPushButton("1. Selecionar Pasta com Imagens")
        self.btn_select_txt_folder.clicked.connect(self.select_batch_txt_folder)
        self.lbl_txt_folder = QLabel("Pasta: Nenhuma")

        self.txt_batch_input = QTextEdit()
        self.txt_batch_input.setMaximumHeight(150)

        self.btn_append_txt = QPushButton("Adicionar texto em todos os .txt")
        self.btn_append_txt.clicked.connect(lambda: self.run_batch_text("append"))
        self.btn_overwrite_txt = QPushButton("Sobrescrever todos os .txt")
        self.btn_overwrite_txt.clicked.connect(lambda: self.run_batch_text("overwrite"))

        layout.addWidget(self.btn_select_txt_folder)
        layout.addWidget(self.lbl_txt_folder)
        layout.addSpacing(10)
        layout.addWidget(QLabel("Texto para inserir/sobrescrever:"))
        layout.addWidget(self.txt_batch_input)
        layout.addWidget(self.btn_append_txt)
        layout.addWidget(self.btn_overwrite_txt)

    def setup_ai_analysis_tab(self):
        layout = QVBoxLayout(self.tab_ai)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Fluxo de IA para Dataset")
        title.setProperty("role", "sectionTitle")
        title.style().unpolish(title)
        title.style().polish(title)
        layout.addWidget(title)

        subtitle = QLabel(
            "Objetivo atual: ler imagens de tamanhos variados, analisar com Florence-2 e salvar descricao, tags ou bounding boxes."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.combo_task = QComboBox()
        self.combo_task.addItems(
            [
                "Florence: CAPTION",
                "Florence: PROMPT_GEN_TAGS",
                "Florence: ANALYZE / prompt longo",
                "Florence: BOX / deteccao de objetos",
            ]
        )
        self.combo_task.currentIndexChanged.connect(self.update_ai_task_ui)

        self.combo_florence_model = QComboBox()
        self.combo_florence_model.addItems(
            [
                "Microsoft Florence-2 Large",
                "PromptGen Large v2.0",
            ]
        )
        self.combo_florence_model.currentIndexChanged.connect(lambda: self.refresh_model_status())

        self.combo_bbox = QComboBox()
        self.combo_bbox.addItems(
            [
                "Imagem com caixas",
                "YOLO .txt + classes.txt",
            ]
        )

        self.lbl_task_hint = QLabel()
        self.lbl_task_hint.setWordWrap(True)
        self.lbl_task_hint.setProperty("role", "aiHint")

        self.btn_select_bbox_output = QPushButton("Selecionar Pasta BOX")
        self.btn_select_bbox_output.setProperty("role", "compactAction")
        self.btn_select_bbox_output.clicked.connect(self.select_bbox_output_folder)
        self.lbl_bbox_output = QLabel("Saida BOX: subpasta automatica Box_Research")
        self.lbl_bbox_output.setWordWrap(True)
        self.lbl_bbox_output.setProperty("role", "aiHint")

        self.spin_caption_tokens = QSpinBox()
        self.spin_caption_tokens.setRange(1, 9999)
        self.spin_caption_tokens.setValue(1024)
        self.spin_caption_tokens.setAccelerated(True)

        self.spin_tags_tokens = QSpinBox()
        self.spin_tags_tokens.setRange(1, 9999)
        self.spin_tags_tokens.setValue(256)
        self.spin_tags_tokens.setAccelerated(True)

        top_cards = QHBoxLayout()

        output_card = QGroupBox("Florence")
        output_card.setProperty("role", "aiTopCard")
        output_layout = QGridLayout()
        output_layout.addWidget(QLabel("Modelo"), 0, 0)
        output_layout.addWidget(self.combo_florence_model, 0, 1)
        output_layout.addWidget(QLabel("Modo"), 1, 0)
        output_layout.addWidget(self.combo_task, 1, 1)
        output_layout.addWidget(QLabel("Tokens desc."), 2, 0)
        output_layout.addWidget(self.spin_caption_tokens, 2, 1)
        output_layout.addWidget(QLabel("Tokens tags"), 3, 0)
        output_layout.addWidget(self.spin_tags_tokens, 3, 1)
        output_layout.addWidget(QLabel("BOX"), 4, 0)
        output_layout.addWidget(self.combo_bbox, 4, 1)
        output_layout.addWidget(self.btn_select_bbox_output, 5, 0)
        output_layout.addWidget(self.lbl_bbox_output, 5, 1)
        output_layout.addWidget(self.lbl_task_hint, 6, 0, 1, 2)
        output_card.setLayout(output_layout)
        top_cards.addWidget(output_card, 3)

        model_card = QGroupBox("Modelo")
        model_card.setProperty("role", "aiTopCard")
        model_layout = QGridLayout()
        self.lbl_quick_device = QLabel("Dispositivo: verificando")
        self.lbl_quick_device.setWordWrap(True)
        model_layout.addWidget(self.lbl_quick_device, 0, 0, 1, 2)
        self.btn_model_status = QPushButton("Modelo: verificando...")
        self.btn_model_status.setEnabled(False)
        self.btn_model_status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        model_layout.addWidget(self.btn_model_status, 1, 0, 1, 2)
        self.btn_select_models_folder = QPushButton("Selecionar Pasta para Modelos")
        self.btn_select_models_folder.setProperty("role", "compactAction")
        self.btn_select_models_folder.clicked.connect(self.select_models_folder)
        self.lbl_models_folder = QLabel("Pasta: Padrao do sistema")
        self.lbl_models_folder.setWordWrap(True)
        self.lbl_models_folder.setProperty("role", "aiHint")
        self.btn_check_model = QPushButton("Verificar Modelo")
        self.btn_check_model.setProperty("role", "compactAction")
        self.btn_check_model.clicked.connect(self.check_ai_model)
        self.btn_download_model = QPushButton("Baixar / Preparar Florence-2")
        self.btn_download_model.setProperty("role", "compactAction")
        self.btn_download_model.clicked.connect(self.test_ai_models)
        model_layout.addWidget(self.btn_select_models_folder, 2, 0)
        model_layout.addWidget(self.lbl_models_folder, 2, 1)
        model_layout.addWidget(self.btn_check_model, 3, 0)
        model_layout.addWidget(self.btn_download_model, 3, 1)
        model_card.setLayout(model_layout)
        top_cards.addWidget(model_card, 2)

        layout.addLayout(top_cards)

        self.ai_subtabs = QTabWidget()

        self.tab_ai_single = QWidget()
        self.setup_ai_single_tab()
        self.ai_subtabs.addTab(self.tab_ai_single, "Imagem Atual")

        self.tab_ai_batch = QWidget()
        self.setup_ai_batch_tab()
        self.ai_subtabs.addTab(self.tab_ai_batch, "Pasta em Lote")

        self.tab_ai_config = QWidget()
        self.setup_ai_config_tab()
        self.ai_subtabs.addTab(self.tab_ai_config, "Configuracao")

        layout.addWidget(self.ai_subtabs)
        self.update_ai_task_ui()

    def setup_ai_single_tab(self):
        layout = QVBoxLayout(self.tab_ai_single)

        preview_layout = QHBoxLayout()
        original_layout = QVBoxLayout()
        analyzed_layout = QVBoxLayout()

        original_layout.addWidget(QLabel("Imagem carregada:"))
        self.lbl_ai_source_preview = QLabel("Selecione uma imagem no visualizador")
        self.lbl_ai_source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ai_source_preview.setMinimumSize(440, 300)
        self.lbl_ai_source_preview.setProperty("role", "imagePanel")
        original_layout.addWidget(self.lbl_ai_source_preview)

        analyzed_layout.addWidget(QLabel("Imagem analisada / resultado visual:"))
        self.lbl_ai_result_preview = QLabel("A saida analisada aparecera aqui")
        self.lbl_ai_result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ai_result_preview.setMinimumSize(440, 300)
        self.lbl_ai_result_preview.setProperty("role", "imagePanel")
        analyzed_layout.addWidget(self.lbl_ai_result_preview)

        preview_layout.addLayout(original_layout, 1)
        preview_layout.addLayout(analyzed_layout, 1)
        layout.addLayout(preview_layout)

        status_row = QHBoxLayout()
        self.lbl_ai_current_image = QLabel("Nenhuma imagem selecionada")
        self.lbl_ai_current_image.setWordWrap(True)
        self.lbl_ai_current_task = QLabel("-")
        self.lbl_ai_current_task.setWordWrap(True)
        self.lbl_ai_processing_text = QLabel("IA parada")
        self.lbl_ai_processing_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(self.lbl_ai_current_image, 2)
        status_row.addWidget(self.lbl_ai_current_task, 2)
        status_row.addWidget(self.lbl_ai_processing_text, 1)
        layout.addLayout(status_row)

        self.ai_processing_bar = QFrame()
        self.ai_processing_bar.setProperty("role", "processingBar")
        self.ai_processing_bar.setProperty("state", "idle")
        self.ai_processing_bar.setMinimumHeight(10)
        self.ai_processing_bar.setMaximumHeight(10)
        layout.addWidget(self.ai_processing_bar)

        self.txt_ai_ignored_single = QLineEdit()
        self.txt_ai_ignored_single.setPlaceholderText("Palavras ignoradas: blurry, watermark")
        layout.addWidget(self.txt_ai_ignored_single)

        buttons_row = QHBoxLayout()
        self.btn_run_ai_single = QPushButton("Analisar Imagem Atual")
        self.btn_run_ai_single.setProperty("role", "primaryAction")
        self.btn_run_ai_single.clicked.connect(self.run_ai_single)
        buttons_row.addWidget(self.btn_run_ai_single)

        self.btn_save_ai_single = QPushButton("Salvar Resultado no .txt")
        self.btn_save_ai_single.setProperty("role", "secondaryAction")
        self.btn_save_ai_single.clicked.connect(self.save_ai_single_txt)
        buttons_row.addWidget(self.btn_save_ai_single)
        layout.addLayout(buttons_row)

        hints_row = QHBoxLayout()
        self.lbl_run_ai_hint = QLabel("Usa a imagem selecionada no visualizador.")
        self.lbl_run_ai_hint.setProperty("role", "aiHint")
        self.lbl_run_ai_hint.setWordWrap(True)
        self.lbl_save_ai_hint = QLabel("Salva no .txt com o mesmo nome da imagem.")
        self.lbl_save_ai_hint.setProperty("role", "aiHint")
        self.lbl_save_ai_hint.setWordWrap(True)
        hints_row.addWidget(self.lbl_run_ai_hint)
        hints_row.addWidget(self.lbl_save_ai_hint)
        layout.addLayout(hints_row)

        text_labels_row = QHBoxLayout()
        text_labels_row.addWidget(QLabel("Texto atual da imagem (.txt existente):"))
        text_labels_row.addWidget(QLabel("Resultado gerado pela IA:"))
        layout.addLayout(text_labels_row)

        text_boxes_row = QHBoxLayout()
        self.txt_ai_original = QTextEdit()
        self.txt_ai_original.setReadOnly(True)
        self.txt_ai_original.setMaximumHeight(120)
        self.txt_ai_generated = QTextEdit()
        self.txt_ai_generated.setMaximumHeight(120)
        text_boxes_row.addWidget(self.txt_ai_original)
        text_boxes_row.addWidget(self.txt_ai_generated)
        layout.addLayout(text_boxes_row)

    def setup_ai_batch_tab(self):
        layout = QVBoxLayout(self.tab_ai_batch)

        folder_layout = QHBoxLayout()
        self.btn_select_ai_folder = QPushButton("Selecionar Pasta para Lote")
        self.btn_select_ai_folder.clicked.connect(self.select_ai_folder)
        self.lbl_ai_folder = QLabel("Pasta: Nenhuma")
        folder_layout.addWidget(self.btn_select_ai_folder)
        folder_layout.addWidget(self.lbl_ai_folder)
        layout.addLayout(folder_layout)

        layout.addWidget(QLabel("Palavras ignoradas para todas as imagens:"))
        self.txt_ai_ignored_batch = QLineEdit()
        self.txt_ai_ignored_batch.setPlaceholderText("ex: blurry, signature")
        layout.addWidget(self.txt_ai_ignored_batch)

        self.btn_run_ai_batch = QPushButton("Iniciar Processo em Lote")
        self.btn_run_ai_batch.clicked.connect(self.run_ai_batch)
        layout.addWidget(self.btn_run_ai_batch)

        self.ai_progress = QProgressBar()
        self.ai_progress.setValue(0)
        layout.addWidget(self.ai_progress)

        layout.addWidget(QLabel("Status e mensagens:"))
        self.txt_ai_status = QTextEdit()
        self.txt_ai_status.setReadOnly(True)
        self.txt_ai_status.setMaximumHeight(150)
        layout.addWidget(self.txt_ai_status)

    def setup_ai_config_tab(self):
        layout = QVBoxLayout(self.tab_ai_config)

        guide_card = QGroupBox("Orientacao de Uso")
        guide_card.setProperty("role", "configCard")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. Verifique se o modelo esta pronto.\n"
            "2. Escolha CPU ou GPU.\n"
            "3. Ajuste tokens quando precisar mais velocidade ou mais detalhes.\n"
            "4. Use BOX para pesquisa visual e validacao de qualidade."
        )
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_text)
        guide_card.setLayout(guide_layout)
        layout.addWidget(guide_card)

        runtime_card = QGroupBox("Motor de Processamento")
        runtime_card.setProperty("role", "configCard")
        runtime_layout = QVBoxLayout()

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Processamento:"))
        self.combo_processing_device = QComboBox()
        self.combo_processing_device.addItems(["Auto", "GPU", "CPU"])
        self.combo_processing_device.currentIndexChanged.connect(lambda: self.refresh_model_status())
        device_row.addWidget(self.combo_processing_device)
        runtime_layout.addLayout(device_row)

        self.lbl_runtime_device = QLabel("Dispositivo atual: aguardando verificacao")
        self.lbl_runtime_device.setWordWrap(True)
        runtime_layout.addWidget(self.lbl_runtime_device)

        self.lbl_device_hint = QLabel("Auto usa GPU quando disponivel.")
        self.lbl_device_hint.setWordWrap(True)
        runtime_layout.addWidget(self.lbl_device_hint)
        runtime_card.setLayout(runtime_layout)
        layout.addWidget(runtime_card)

        hardware_card = QGroupBox("Memoria e Modelo")
        hardware_card.setProperty("role", "configCard")
        hardware_layout = QVBoxLayout()
        keep_loaded_text = QLabel(
            "O Florence-2 fica carregado na memoria durante o uso para acelerar as proximas analises."
        )
        keep_loaded_text.setWordWrap(True)
        hardware_layout.addWidget(keep_loaded_text)
        self.lbl_gpu_install_hint = QLabel("")
        self.lbl_gpu_install_hint.setWordWrap(True)
        hardware_layout.addWidget(self.lbl_gpu_install_hint)
        hardware_card.setLayout(hardware_layout)
        layout.addWidget(hardware_card)

        layout.addStretch(1)

    def parse_tags(self, text):
        if not text.strip():
            return []
        if self.chk_comma_separator.isChecked():
            return [tag.strip() for tag in text.split(",") if tag.strip()]
        return [tag.strip() for tag in text.split() if tag.strip()]

    def format_tags(self, tags_list):
        return ", ".join(tags_list) if self.chk_comma_separator.isChecked() else " ".join(tags_list)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecione a Pasta")
        if not folder:
            return

        self.current_folder = folder
        self.list_images.clear()
        self.global_vocabulary.clear()

        images = [name for name in os.listdir(folder) if self.is_valid_image(name)]
        for file_name in os.listdir(folder):
            if file_name.lower().endswith(".txt"):
                txt_path = os.path.join(folder, file_name)
                try:
                    with open(txt_path, "r", encoding="utf-8") as file_obj:
                        for tag in self.parse_tags(file_obj.read()):
                            self.global_vocabulary.add(tag)
                except Exception as error:
                    self.show_copyable_dialog(
                        "Erro ao Ler TXT",
                        f"Arquivo: {txt_path}\n\n{type(error).__name__}: {error}",
                    )

        for image_name in images:
            self.list_images.addItem(image_name)

        if self.list_images.count() > 0:
            self.list_images.setCurrentRow(0)

    def on_image_selected(self, current, previous):
        if not current:
            return

        image_name = current.text()
        self.current_image_path = os.path.join(self.current_folder, image_name)
        self.lbl_ai_current_image.setText(image_name)

        pixmap = QPixmap(self.current_image_path)
        scaled = pixmap.scaled(
            self.lbl_image_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_image_preview.setPixmap(scaled)
        self.update_ai_preview_labels(source_path=self.current_image_path, analyzed_path="")

        txt_path = self.get_txt_path_for_image(image_name)
        self.current_selected_tags = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as file_obj:
                self.current_selected_tags = self.parse_tags(file_obj.read())
                for tag in self.current_selected_tags:
                    self.global_vocabulary.add(tag)

        self.refresh_tags_ui()
        self.load_current_txt_preview()

    def refresh_tags_ui(self):
        self.clear_layout(self.layout_selected)
        self.clear_layout(self.layout_available)

        for tag in self.current_selected_tags:
            button = TagButton(tag, is_selected=True)
            button.leftClicked.connect(self.on_tag_remove_click)
            self.layout_selected.addWidget(button)

        available_tags = sorted(self.global_vocabulary - set(self.current_selected_tags))
        for tag in available_tags:
            button = TagButton(tag, is_selected=False)
            button.leftClicked.connect(self.on_tag_add_click)
            button.rightClicked.connect(self.on_tag_delete_global)
            self.layout_available.addWidget(button)

        self.widget_selected.style().unpolish(self.widget_selected)
        self.widget_selected.style().polish(self.widget_selected)
        self.widget_available.style().unpolish(self.widget_available)
        self.widget_available.style().polish(self.widget_available)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def get_txt_path_for_image(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        return os.path.join(self.current_folder, base_name + ".txt")

    def save_current_tags(self):
        if not self.current_image_path:
            return

        txt_path = self.get_txt_path_for_image(os.path.basename(self.current_image_path))
        with open(txt_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(self.format_tags(self.current_selected_tags))
        self.load_current_txt_preview()

    def add_new_tag_from_input(self):
        text = self.tag_input.text()
        if not text.strip():
            return

        for tag in self.parse_tags(text):
            if tag not in self.current_selected_tags:
                self.current_selected_tags.append(tag)
            self.global_vocabulary.add(tag)

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
        reply = QMessageBox.question(
            self,
            "Confirmar Exclusao",
            (
                f"Deseja excluir a tag '{tag_text}' do vocabulario global?\n"
                "Ela nao sera removida dos .txt antigos, apenas da lista rapida."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes and tag_text in self.global_vocabulary:
            self.global_vocabulary.remove(tag_text)
            self.refresh_tags_ui()

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
            QMessageBox.warning(self, "Aviso", "Selecione a pasta de origem primeiro.")
            return

        dst_folder = self.batch_dst or os.path.join(self.batch_src, "Flip_Output")
        os.makedirs(dst_folder, exist_ok=True)
        self.lbl_dst.setText(f"Destino: {dst_folder}")

        images = [name for name in os.listdir(self.batch_src) if self.is_valid_image(name)]
        if not images:
            QMessageBox.information(self, "Info", "Nenhuma imagem encontrada na origem.")
            return

        self.progress_bar.setMaximum(len(images))
        self.progress_bar.setValue(0)

        failures = []
        for index, image_name in enumerate(images, start=1):
            src_path = os.path.join(self.batch_src, image_name)
            dst_path = os.path.join(dst_folder, image_name)
            try:
                with Image.open(src_path) as image_obj:
                    flipped = image_obj.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    flipped.save(dst_path)
            except Exception as error:
                failures.append(f"{image_name}: {type(error).__name__}: {error}")

            self.progress_bar.setValue(index)
            QApplication.processEvents()

        if failures:
            self.show_copyable_dialog(
                "Flip em Lote com Erros",
                (
                    f"Concluido com falhas.\n\n"
                    f"Sucessos: {len(images) - len(failures)}\n"
                    f"Falhas: {len(failures)}\n\n"
                    + "\n".join(failures)
                ),
            )
            return

        QMessageBox.information(self, "Concluido", f"{len(images)} imagens foram invertidas com sucesso.")

    def select_batch_txt_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder:
            self.batch_txt_folder = folder
            self.lbl_txt_folder.setText(f"Pasta: {folder}")

    def run_batch_text(self, mode):
        if not self.batch_txt_folder:
            QMessageBox.warning(self, "Aviso", "Selecione a pasta primeiro.")
            return

        text_to_write = self.txt_batch_input.toPlainText()
        if not text_to_write and mode == "append":
            QMessageBox.warning(self, "Aviso", "Digite algum texto para adicionar.")
            return

        images = [name for name in os.listdir(self.batch_txt_folder) if self.is_valid_image(name)]
        count = 0

        for image_name in images:
            txt_path = os.path.join(self.batch_txt_folder, os.path.splitext(image_name)[0] + ".txt")
            if mode == "overwrite":
                with open(txt_path, "w", encoding="utf-8") as file_obj:
                    file_obj.write(text_to_write)
            else:
                prefix = "\n" if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0 else ""
                with open(txt_path, "a", encoding="utf-8") as file_obj:
                    file_obj.write(prefix + text_to_write)
            count += 1

        QMessageBox.information(self, "Concluido", f"Operacao aplicada em {count} arquivos de texto.")

    def select_models_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta para Modelos")
        if folder:
            self.models_folder = folder
            self.lbl_models_folder.setText(f"Pasta: {folder}")
            self.refresh_model_status()

    def select_bbox_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta para Saida BOX")
        if folder:
            self.bbox_output_folder = folder
            self.lbl_bbox_output.setText(f"Saida BOX: {folder}")

    def check_ai_model(self):
        self.refresh_model_status()
        if self.ai_model_ready:
            QMessageBox.information(self, "Modelo", "O Florence-2 ja esta disponivel localmente.")
        else:
            QMessageBox.warning(
                self,
                "Modelo",
                "O Florence-2 ainda nao foi encontrado localmente. Use o botao de baixar/preparar antes da primeira analise.",
            )

    def test_ai_models(self):
        if self.ai_test_worker and self.ai_test_worker.isRunning():
            return

        self.btn_download_model.setEnabled(False)
        self.btn_download_model.setText("Baixando / Preparando...")
        self.append_ai_status("Preparando Florence-2. Isso pode demorar na primeira vez.")

        self.ai_test_worker = AITestWorker(
            self.models_folder if self.models_folder else None,
            device_preference=self.get_processing_device_preference(),
            model_key=self.get_florence_model_key(),
        )
        self.ai_test_worker.finished.connect(self.on_test_ai_finished)
        self.ai_test_worker.error.connect(self.on_test_ai_error)
        self.ai_test_worker.start()

    def on_test_ai_finished(self, message):
        self.btn_download_model.setEnabled(True)
        self.btn_download_model.setText("Baixar / Preparar Florence-2")
        self.append_ai_status(message)
        self.refresh_model_status(force_ready=True)
        QMessageBox.information(self, "Sucesso", message)

    def on_test_ai_error(self, error_message):
        self.btn_download_model.setEnabled(True)
        self.btn_download_model.setText("Baixar / Preparar Florence-2")
        self.append_ai_status(error_message)
        self.refresh_model_status()
        self.show_copyable_dialog("Erro ao Carregar Florence-2", error_message)

    def select_ai_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta")
        if folder:
            self.ai_folder = folder
            self.lbl_ai_folder.setText(f"Pasta: {folder}")

    def run_ai_single(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "Aviso", "Nenhuma imagem selecionada no visualizador.")
            return

        self.refresh_model_status()
        if not self.ai_model_ready:
            QMessageBox.warning(
                self,
                "Modelo nao preparado",
                "Prepare o Florence-2 primeiro no botao 'Baixar / Preparar Florence-2'.",
            )
            return

        self.load_current_txt_preview()
        self.txt_ai_generated.setPlainText("Analisando...")
        self.append_ai_status(f"Iniciando analise da imagem atual: {self.current_image_path}")
        self.update_ai_preview_labels(source_path=self.current_image_path, analyzed_path="")
        self.set_ai_processing_state("processing")
        self.start_ai_worker(
            [self.current_image_path],
            ignored_words=self.txt_ai_ignored_single.text(),
            auto_save=False,
        )

    def save_ai_single_txt(self):
        if self.get_ai_task() == "bbox":
            QMessageBox.warning(
                self,
                "Aviso",
                "Bounding boxes nao devem ser salvas no .txt. Use apenas a saida de imagem ou YOLO.",
            )
            return

        if not self.current_image_path:
            return

        txt_path = self.get_txt_path_for_current_image()
        with open(txt_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(self.txt_ai_generated.toPlainText().strip())

        QMessageBox.information(self, "Salvo", "Resultado salvo com sucesso no .txt.")
        self.on_image_selected(self.list_images.currentItem(), None)

    def run_ai_batch(self):
        if not self.ai_folder:
            QMessageBox.warning(self, "Aviso", "Selecione a pasta de lote primeiro.")
            return

        self.refresh_model_status()
        if not self.ai_model_ready:
            QMessageBox.warning(
                self,
                "Modelo nao preparado",
                "Prepare o Florence-2 primeiro no botao 'Baixar / Preparar Florence-2'.",
            )
            return

        images = [
            os.path.join(self.ai_folder, name)
            for name in os.listdir(self.ai_folder)
            if self.is_valid_image(name)
        ]
        if not images:
            QMessageBox.information(self, "Info", "Nenhuma imagem encontrada na pasta.")
            return

        self.append_ai_status(f"Iniciando lote com {len(images)} imagens.")
        self.set_ai_processing_state("processing")
        self.start_ai_worker(images, ignored_words=self.txt_ai_ignored_batch.text(), auto_save=True)

    def start_ai_worker(self, images_list, ignored_words="", auto_save=True):
        if self.ai_worker and self.ai_worker.isRunning():
            QMessageBox.warning(self, "Aviso", "O processamento ja esta rodando.")
            return

        task = self.get_ai_task()
        bbox_format = "image" if self.combo_bbox.currentIndex() == 0 else "yolo"
        caption_prompt = self.get_caption_prompt()
        caption_tokens = self.spin_caption_tokens.value()
        tags_tokens = self.spin_tags_tokens.value()
        bbox_output_dir = self.get_bbox_output_dir(images_list[0] if images_list else "")
        device_preference = self.get_processing_device_preference()
        model_key = self.get_florence_model_key()

        self.ai_progress.setMaximum(len(images_list))
        self.ai_progress.setValue(0)

        self.btn_run_ai_single.setEnabled(False)
        self.btn_run_ai_batch.setEnabled(False)

        models_dir = self.models_folder if self.models_folder else None
        self.ai_worker = AIWorker(
            task,
            bbox_format,
            images_list,
            models_dir=models_dir,
            ignored_words=ignored_words,
            auto_save=auto_save,
            caption_prompt=caption_prompt,
            caption_tokens=caption_tokens,
            tags_tokens=tags_tokens,
            bbox_output_dir=bbox_output_dir,
            device_preference=device_preference,
            model_key=model_key,
        )
        self.ai_worker.progress.connect(self.ai_progress.setValue)
        self.ai_worker.finished.connect(self.on_ai_finished)
        self.ai_worker.error.connect(self.on_ai_error)
        self.ai_worker.single_result.connect(self.on_ai_single_result)
        self.ai_worker.start()

    def on_ai_single_result(self, result_text):
        self.txt_ai_generated.setPlainText(result_text)
        if self.get_ai_task() == "bbox":
            analyzed_path = self.extract_first_generated_path(result_text)
            self.update_ai_preview_labels(source_path=self.current_image_path, analyzed_path=analyzed_path)
            self.append_ai_status("Bounding boxes geradas. Revise os arquivos salvos indicados acima.")
        else:
            self.update_ai_preview_labels(source_path=self.current_image_path, analyzed_path=self.current_image_path)

    def on_ai_finished(self, message):
        self.btn_run_ai_single.setEnabled(True)
        self.btn_run_ai_batch.setEnabled(True)
        self.append_ai_status(message)
        self.set_ai_processing_state("done")
        QMessageBox.information(self, "Concluido", message)

        if (
            self.ai_worker
            and self.ai_worker.auto_save
            and len(self.ai_worker.images_list) == 1
            and self.current_image_path == self.ai_worker.images_list[0]
        ):
            self.on_image_selected(self.list_images.currentItem(), None)

        if self.ai_worker and self.ai_worker.auto_save and self.ai_worker.images_list:
            latest_image = self.ai_worker.images_list[-1]
            if self.get_ai_task() == "bbox":
                bbox_dir = self.get_bbox_output_dir(latest_image)
                image_name = os.path.splitext(os.path.basename(latest_image))[0]
                suffix = "_bboxes.png" if self.combo_bbox.currentIndex() == 0 else ""
                preview_path = os.path.join(bbox_dir, image_name + suffix) if suffix else latest_image
                self.update_ai_preview_labels(source_path=latest_image, analyzed_path=preview_path)
            else:
                self.update_ai_preview_labels(source_path=latest_image, analyzed_path=latest_image)

    def on_ai_error(self, error_message):
        self.btn_run_ai_single.setEnabled(True)
        self.btn_run_ai_batch.setEnabled(True)
        self.txt_ai_generated.setPlainText(f"Erro:\n{error_message}")
        self.append_ai_status(error_message)
        self.set_ai_processing_state("error")
        self.show_copyable_dialog("Erro no Processamento IA", error_message)

    def update_ai_task_ui(self):
        task = self.get_ai_task()
        is_bbox = task == "bbox"
        is_caption = task == "caption"
        is_tags = task == "tags"
        is_analysis = task == "analysis"

        self.combo_bbox.setEnabled(is_bbox)
        self.btn_save_ai_single.setEnabled(not is_bbox)
        self.spin_caption_tokens.setEnabled(is_caption or is_analysis)
        self.spin_tags_tokens.setEnabled(is_tags)
        self.btn_select_bbox_output.setEnabled(is_bbox)

        if task == "caption":
            current_task_text = "Saida: descricao Florence para .txt"
            self.lbl_task_hint.setText(
                "CAPTION gera uma descricao direta e curta, sem modo detalhado."
            )
            self.txt_ai_generated.setPlaceholderText("A descricao da imagem aparecera aqui.")
        elif task == "tags":
            current_task_text = "Saida: PROMPT_GEN_TAGS em linha unica"
            self.lbl_task_hint.setText(
                "PROMPT_GEN_TAGS cria tags separadas por virgula para dataset e prompts compactos."
            )
            self.txt_ai_generated.setPlaceholderText("As tags geradas aparecerao aqui.")
        elif task == "analysis":
            current_task_text = "Saida: ANALYZE com descricao longa"
            self.lbl_task_hint.setText(
                "Use o modelo PromptGen para descricoes longas no estilo ComfyUI."
            )
            self.txt_ai_generated.setPlaceholderText("A analise longa da imagem aparecera aqui.")
        else:
            current_task_text = "Saida: BOX visual ou YOLO"
            self.lbl_task_hint.setText(
                "BOX cria imagem marcada ou YOLO/classes em pasta separada. Nao salva no .txt principal."
            )
            self.txt_ai_generated.setPlaceholderText("Os caminhos dos arquivos gerados aparecerao aqui.")

        self.lbl_ai_current_task.setText(current_task_text)

        if task != "bbox" and self.current_image_path:
            self.update_ai_preview_labels(source_path=self.current_image_path, analyzed_path=self.current_image_path)

    def get_ai_task(self):
        task_index = self.combo_task.currentIndex()
        if task_index == 0:
            return "caption"
        if task_index == 1:
            return "tags"
        if task_index == 2:
            return "analysis"
        return "bbox"

    def get_caption_prompt(self):
        return "<CAPTION>"

    def get_florence_model_key(self):
        if self.combo_florence_model.currentIndex() == 1:
            return "promptgen-large-v2"
        return "microsoft-large"

    def get_bbox_output_dir(self, first_image_path=""):
        if self.bbox_output_folder:
            return self.bbox_output_folder
        if first_image_path:
            return os.path.join(os.path.dirname(first_image_path), "Box_Research")
        return ""

    def append_ai_status(self, message):
        self.txt_ai_status.append(message)

    def load_current_txt_preview(self):
        txt_path = self.get_txt_path_for_current_image()
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as file_obj:
                self.txt_ai_original.setPlainText(file_obj.read())
        else:
            self.txt_ai_original.clear()

    def get_txt_path_for_current_image(self):
        if not self.current_image_path:
            return ""
        if self.current_folder:
            return self.get_txt_path_for_image(os.path.basename(self.current_image_path))
        return os.path.splitext(self.current_image_path)[0] + ".txt"

    def show_copyable_dialog(self, title, message):
        dialog = CopyableMessageDialog(title, message, self)
        dialog.exec()

    def is_valid_image(self, file_name):
        return any(file_name.lower().endswith(extension) for extension in self.VALID_EXTENSIONS)

    def refresh_model_status(self, force_ready=False):
        manager = (
            ai_analyzer.AIModelManager(
                self.models_folder if self.models_folder else None,
                model_key=self.get_florence_model_key(),
            )
            if ai_analyzer
            else None
        )
        self.ai_model_ready = force_ready or (manager.is_florence_available_locally() if manager else False)
        runtime_device = self.get_runtime_device_label()
        self.lbl_runtime_device.setText(f"Dispositivo atual: {runtime_device}")
        self.lbl_quick_device.setText(f"Dispositivo: {runtime_device}")
        self.lbl_device_hint.setText(self.get_device_hint_text())
        if ai_analyzer and not ai_analyzer.AIModelManager.gpu_available():
            self.lbl_gpu_install_hint.setText(
                "GPU indisponivel neste ambiente porque o PyTorch instalado e CPU-only. "
                "Para ativar GPU, sera necessario reinstalar o torch com suporte CUDA no venv."
            )
        else:
            self.lbl_gpu_install_hint.setText("GPU detectada. Voce pode selecionar Auto ou GPU para acelerar o processamento.")
        if self.ai_model_ready:
            self.btn_model_status.setText(f"Modelo: pronto | {runtime_device}")
            self.btn_model_status.setStyleSheet("background-color: #2563eb; color: white;")
        else:
            self.btn_model_status.setText(f"Modelo: nao preparado | {runtime_device}")
            self.btn_model_status.setStyleSheet("background-color: #dc2626; color: white;")

    def update_ai_preview_labels(self, source_path="", analyzed_path=""):
        self.set_preview_on_label(self.lbl_ai_source_preview, source_path, "Selecione uma imagem no visualizador")
        fallback = source_path if analyzed_path and not os.path.exists(analyzed_path) else analyzed_path
        self.set_preview_on_label(
            self.lbl_ai_result_preview,
            analyzed_path if analyzed_path and os.path.exists(analyzed_path) else fallback,
            "A saida analisada aparecera aqui",
        )
        self.last_ai_preview_path = analyzed_path if analyzed_path else source_path

    def set_preview_on_label(self, label, image_path, empty_text):
        if image_path and os.path.exists(image_path) and self.is_valid_image(image_path):
            pixmap = QPixmap(image_path)
            target_size = label.size()
            if target_size.width() <= 1 or target_size.height() <= 1:
                target_size = label.minimumSize()
            scaled = pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled)
            label.setText("")
        else:
            label.setPixmap(QPixmap())
            label.setText(empty_text)

    def extract_first_generated_path(self, text):
        for line in text.splitlines():
            candidate = line.strip()
            if candidate and os.path.exists(candidate):
                return candidate
        return ""

    def get_processing_device_preference(self):
        choice = self.combo_processing_device.currentText().lower()
        if choice == "gpu":
            return "gpu"
        if choice == "cpu":
            return "cpu"
        return "auto"

    def get_runtime_device_label(self):
        preference = self.get_processing_device_preference()
        if preference == "gpu":
            return "GPU" if ai_analyzer and ai_analyzer.AIModelManager.gpu_available() else "CPU (GPU indisponivel)"
        if preference == "cpu":
            return "CPU"
        return "GPU" if ai_analyzer and ai_analyzer.AIModelManager.gpu_available() else "CPU"

    def get_device_hint_text(self):
        preference = self.get_processing_device_preference()
        if preference == "gpu" and ai_analyzer and not ai_analyzer.AIModelManager.gpu_available():
            return "GPU foi selecionada, mas este ambiente esta com torch CPU-only. O processamento vai cair para CPU."
        if preference == "cpu":
            return "CPU selecionada manualmente. Mais compativel, porem mais lenta."
        if ai_analyzer and ai_analyzer.AIModelManager.gpu_available():
            return "Auto usa GPU quando disponivel."
        return "Auto esta usando CPU porque a GPU nao esta disponivel neste ambiente."

    def set_ai_processing_state(self, state):
        self.ai_processing_bar.setProperty("state", state)
        self.ai_processing_bar.style().unpolish(self.ai_processing_bar)
        self.ai_processing_bar.style().polish(self.ai_processing_bar)
        if state == "processing":
            self.lbl_ai_processing_text.setText("IA processando")
        elif state == "done":
            self.lbl_ai_processing_text.setText("Processamento concluido")
        elif state == "error":
            self.lbl_ai_processing_text.setText("Erro no processamento")
        else:
            self.lbl_ai_processing_text.setText("IA parada")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BatchEditorAero()
    window.show()
    sys.exit(app.exec())
