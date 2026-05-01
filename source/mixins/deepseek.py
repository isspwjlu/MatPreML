"""DeepSeek AI对话Mixin模块 - MatPreML"""

import os
import json
import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QLineEdit
)
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QKeyEvent

from ..config import DEFAULT_DEEPSEEK_CONFIG, CONFIG_DIR
from ..threads import ConnectionTestThread, DeepSeekAPIThread


class DeepSeekMixin:
    """DeepSeek AI对话Mixin类 - 聊天界面、API调用、配置管理"""

    def show_deepseek_dialog(self):
        """显示DeepSeek对话界面"""
        dialog = QDialog(self)
        dialog.setWindowTitle('DeepSeek AI 问答')
        dialog.resize(900, 700)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        title_label = QLabel('DeepSeek AI 智能助手')
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title_label.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        info_label = QLabel(
            '基于DeepSeek API的智能问答助手'
            ' - 支持材料科学与机器学习领域问答'
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_font = info_label.font()
        info_font.setPointSize(10)
        info_label.setFont(info_font)
        info_label.setStyleSheet("color: #666666; padding: 5px;")
        layout.addWidget(info_label)

        self.connection_status_label = QLabel('连接状态: 已连接')
        self.connection_status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        status_font = self.connection_status_label.font()
        status_font.setPointSize(9)
        self.connection_status_label.setFont(status_font)
        self.connection_status_label.setStyleSheet(
            "color: #4CAF50; padding: 2px;"
        )
        layout.addWidget(self.connection_status_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        layout.addWidget(self.chat_display, stretch=7)

        input_layout = QVBoxLayout()

        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(120)
        self.user_input.setMinimumHeight(80)
        self.user_input.setPlaceholderText(
            "请输入您的问题...\n按Ctrl+Enter换行，按Enter发送"
        )
        self.user_input.setStyleSheet("""
            QTextEdit {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
            QTextEdit:focus {
                border-color: #45a049;
            }
        """)

        self.user_input_key_filter = self.UserInputKeyFilter(self)
        self.user_input.installEventFilter(self.user_input_key_filter)

        input_layout.addWidget(self.user_input)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        send_button = QPushButton('发送 (Enter)')
        send_button.clicked.connect(self.send_message)
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 11pt;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        button_layout.addWidget(send_button)

        clear_button = QPushButton('清空')
        clear_button.clicked.connect(self.clear_chat)
        clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 11pt;
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        button_layout.addWidget(clear_button)

        button_layout.addStretch()
        input_layout.addLayout(button_layout)

        layout.addLayout(input_layout, stretch=1)

        self.deepseek_dialog = dialog

        self.test_api_connection()

        welcome_msg = (
            "您好！我是基于DeepSeek API的智能助手。\n"
            "我可以帮助您解答关于材料科学、机器学习等方面的问题。\n"
            "您可以询问软件使用方法、模型选择建议、数据分析技巧等。\n"
            "请输入您的问题，我会尽力为您解答！"
        )
        self.display_message("AI助手", welcome_msg)

        self.user_input.setFocus()

        dialog.exec()

    def test_api_connection(self):
        """测试API连接"""
        self.connection_test_thread = ConnectionTestThread()
        self.connection_test_thread.connection_result.connect(
            self.handle_connection_result
        )
        self.connection_test_thread.start()

    def handle_connection_result(self, success, message):
        """处理连接测试结果"""
        if success:
            self.connection_status_label.setText('连接状态: 已连接')
            self.connection_status_label.setStyleSheet(
                "color: #4CAF50; padding: 2px;"
            )
        else:
            self.connection_status_label.setText('连接状态: 连接失败')
            self.connection_status_label.setStyleSheet(
                "color: #f44336; padding: 2px;"
            )
            self.display_message("系统", f"API连接测试失败: {message}")

    class UserInputKeyFilter(QObject):
        """用户输入框的按键事件过滤器"""
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.KeyPress:
                key_event = event
                if key_event.key() in (
                    Qt.Key.Key_Return, Qt.Key.Key_Enter
                ):
                    if not (
                        key_event.modifiers()
                        & Qt.KeyboardModifier.ControlModifier
                    ):
                        self.parent.send_message()
                        return True
            return False

    def send_message(self):
        """发送消息到DeepSeek API"""
        user_message = self.user_input.toPlainText().strip()
        if not user_message:
            return

        self.display_message("您", user_message)
        self.user_input.clear()
        self.get_deepseek_response(user_message)

    def clear_chat(self):
        """清空聊天记录"""
        self.chat_display.clear()
        self.chat_history = []
        welcome_msg = (
            "您好！我是基于DeepSeek API的智能助手。\n"
            "我可以帮助您解答关于材料科学、机器学习等方面的问题。\n"
            "您可以询问软件使用方法、模型选择建议、数据分析技巧等。\n"
            "请输入您的问题，我会尽力为您解答！"
        )
        self.display_message("AI助手", welcome_msg)

    def display_message(self, sender, message):
        """在聊天界面显示消息"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        if not hasattr(self, 'chat_history'):
            self.chat_history = []
        self.chat_history.append((sender, message))

        if sender == "您":
            formatted_message = (
                '<div style="margin: 10px 0; text-align: right;">'
                '<div style="display: inline-block; '
                'background-color: #e3f2fd; border-radius: 10px; '
                'padding: 10px; max-width: 80%;">'
                '<div style="font-weight: bold; color: #1976d2;">'
                f'{sender}</div>'
                '<div style="margin-top: 5px;">'
                f'{message.replace(chr(10), "<br>")}</div>'
                '<div style="font-size: 0.8em; color: #999; '
                'margin-top: 5px; text-align: right;">'
                f'{timestamp}</div></div></div>'
            )
        elif sender == "AI助手":
            formatted_message = (
                '<div style="margin: 10px 0; text-align: left;">'
                '<div style="display: inline-block; '
                'background-color: #f1f8e9; border-radius: 10px; '
                'padding: 10px; max-width: 80%;">'
                '<div style="font-weight: bold; color: #388e3c;">'
                f'{sender}</div>'
                '<div style="margin-top: 5px;">'
                f'{message.replace(chr(10), "<br>")}</div>'
                '<div style="font-size: 0.8em; color: #999; '
                'margin-top: 5px;">'
                f'{timestamp}</div></div></div>'
            )
        else:
            formatted_message = (
                '<div style="margin: 10px 0;">'
                '<div style="font-weight: bold;">'
                f'[{timestamp}] {sender}:</div>'
                '<div style="margin-top: 5px;">'
                f'{message.replace(chr(10), "<br>")}</div></div>'
            )

        self.chat_display.append(formatted_message)
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def get_deepseek_response(self, user_message):
        """获取DeepSeek API响应"""
        thinking_msg = "正在思考..."
        self.display_message("AI助手", thinking_msg)

        learning_materials = self.load_learning_materials()
        if len(learning_materials) > 3000:
            learning_materials = (
                learning_materials[:3000]
                + "\n\n... (内容已截断以提高响应速度)"
            )

        self.api_thread = DeepSeekAPIThread(
            user_message, learning_materials
        )
        self.api_thread.response_ready.connect(self.handle_api_response)
        self.api_thread.error_occurred.connect(self.handle_api_error)
        self.api_thread.start()

    def handle_api_response(self, response_text):
        """处理API响应"""
        self.connection_status_label.setText('连接状态: 已连接')
        self.connection_status_label.setStyleSheet(
            "color: #4CAF50; padding: 2px;"
        )

        self.remove_last_ai_message("正在思考...")
        self.display_message("AI助手", response_text)

    def handle_api_error(self, error_msg):
        """处理API错误"""
        if "网络" in error_msg or "超时" in error_msg:
            self.connection_status_label.setText('连接状态: 连接失败')
            self.connection_status_label.setStyleSheet(
                "color: #f44336; padding: 2px;"
            )
        else:
            self.connection_status_label.setText('连接状态: 已连接')
            self.connection_status_label.setStyleSheet(
                "color: #4CAF50; padding: 2px;"
            )

        self.remove_last_ai_message("正在思考...")

        if ("网络" in error_msg or "超时" in error_msg
                or "连接" in error_msg):
            network_error_msg = (
                "糟糕！deepseek似乎没有联网，"
                "请关闭对话窗口再试一次，"
                "或检查您的网络是否正常！"
            )
            self.display_message("AI助手", network_error_msg)
        else:
            self.display_message("AI助手", error_msg)

    def remove_last_ai_message(self, message_to_remove):
        """移除最后一条AI助手消息"""
        if not hasattr(self, 'chat_history'):
            self.chat_history = []

        if (self.chat_history
                and self.chat_history[-1][0] == "AI助手"
                and self.chat_history[-1][1] == message_to_remove):
            self.chat_history.pop()

        self.chat_display.clear()
        temp_history = self.chat_history
        self.chat_history = []
        for sender, message in temp_history:
            self.display_message(sender, message)

    def load_learning_materials(self):
        """加载本地学习资料"""
        materials = ""

        try:
            with open(
                os.path.join(CONFIG_DIR, 'readme.txt'), 'r', encoding='utf-8'
            ) as f:
                content = f.read(1500)
                if content.strip():
                    materials += "使用说明：\n" + content + "\n\n"
                else:
                    materials += "使用说明文件为空。\n\n"
        except FileNotFoundError:
            materials += "使用说明文件未找到。\n\n"
        except Exception as e:
            materials += f"读取使用说明时出错: {str(e)}\n\n"

        try:
            with open(
                os.path.join(CONFIG_DIR, 'tutorial.txt'), 'r', encoding='utf-8'
            ) as f:
                content = f.read(1500)
                if content.strip():
                    materials += "教程：\n" + content + "\n\n"
                else:
                    materials += "教程文件为空。\n\n"
        except FileNotFoundError:
            materials += "教程文件未找到。\n\n"
        except Exception as e:
            materials += f"读取教程时出错: {str(e)}\n\n"

        return materials

    def load_deepseek_config(self):
        """加载DeepSeek配置"""
        config_file = os.path.join(CONFIG_DIR, 'deepseek_config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            except Exception as e:
                print(f"加载DeepSeek配置时出错: {str(e)}")
                return dict(DEFAULT_DEEPSEEK_CONFIG)
        else:
            default_config = dict(DEFAULT_DEEPSEEK_CONFIG)
            self.save_deepseek_config(default_config)
            return default_config

    def save_deepseek_config(self, config):
        """保存DeepSeek配置"""
        config_dir = CONFIG_DIR
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_file = os.path.join(config_dir, 'deepseek_config.json')
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"保存DeepSeek配置时出错: {str(e)}")

    def show_deepseek_settings(self):
        """显示DeepSeek设置对话框"""
        config = self.load_deepseek_config()

        dialog = QDialog(self)
        dialog.setWindowTitle('DeepSeek API 设置')
        dialog.setModal(True)
        dialog.resize(500, 200)

        layout = QVBoxLayout(dialog)

        url_label = QLabel('API网址:')
        url_edit = QLineEdit()
        url_edit.setText(config.get("api_url", ""))
        layout.addWidget(url_label)
        layout.addWidget(url_edit)

        key_label = QLabel('API Key:')
        key_edit = QLineEdit()
        key_edit.setText(config.get("api_key", ""))
        key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(key_label)
        layout.addWidget(key_edit)

        button_layout = QHBoxLayout()
        ok_button = QPushButton('确定')
        cancel_button = QPushButton('取消')

        def save_settings():
            new_config = {
                "api_url": url_edit.text().strip(),
                "api_key": key_edit.text().strip()
            }
            self.save_deepseek_config(new_config)
            dialog.accept()

        ok_button.clicked.connect(save_settings)
        cancel_button.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.exec()
