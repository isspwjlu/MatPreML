"""注册管理Mixin模块 - MatPreML"""

import os
import json
import subprocess
import platform
import uuid
import hashlib
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QTextEdit, QMessageBox
)
from PyQt6.QtGui import QFont


from ..license_core import verify_registration_code as _verify_registration_code
from ..config import CONFIG_DIR, REGISTRATION_FILE


class RegistrationMixin:
    """注册管理Mixin类 - 机器ID获取、注册码验证、注册状态管理"""

    def show_register_dialog(self):
        """显示注册对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle('注册专业版')
        dialog.setModal(True)
        dialog.resize(400, 250)

        layout = QVBoxLayout(dialog)

        machine_id = self.get_machine_id()

        machine_id_label = QLabel('机器ID:')
        font = machine_id_label.font()
        font.setPointSize(int(font.pointSize() * 1.15))
        machine_id_label.setFont(font)

        machine_id_text = QTextEdit()
        machine_id_text.setReadOnly(True)
        machine_id_text.setMaximumHeight(30)
        machine_id_text.setText(machine_id)
        text_font = machine_id_text.font()
        text_font.setPointSize(int(text_font.pointSize() * 1.15))
        machine_id_text.setFont(text_font)

        layout.addWidget(machine_id_label)
        layout.addWidget(machine_id_text)

        reg_code_label = QLabel('请输入注册码:')
        reg_code_label.setFont(font)

        self.reg_code_input = QLineEdit()
        self.reg_code_input.setPlaceholderText("请输入您的注册码")
        input_font = self.reg_code_input.font()
        input_font.setPointSize(int(input_font.pointSize() * 1.25))
        self.reg_code_input.setFont(input_font)

        layout.addWidget(reg_code_label)
        layout.addWidget(self.reg_code_input)

        button_layout = QHBoxLayout()
        register_button = QPushButton('注册')
        cancel_button = QPushButton('取消')

        button_font = register_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.15))
        register_button.setFont(button_font)
        cancel_button.setFont(button_font)

        register_button.setFixedHeight(
            int(register_button.sizeHint().height() * 1.15)
        )
        cancel_button.setFixedHeight(
            int(cancel_button.sizeHint().height() * 1.15)
        )

        register_button.clicked.connect(
            lambda: self.register_product(machine_id, dialog)
        )
        cancel_button.clicked.connect(dialog.reject)

        button_layout.addWidget(register_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        request_label = QLabel(
            '  \n'
            '             请将用户姓名、单位、Email、机器ID号\n'
            '             发送至: wjlu@issp.ac.cn 获取专业版注册码\n'
            '             专业版专享功能：高精度超参调优、'
            '自动化特征工程'
        )
        request_font = QFont()
        request_font.setPointSize(int(font.pointSize() * 1.05))
        request_label.setFont(request_font)
        request_label.setWordWrap(True)
        layout.addWidget(request_label)

        dialog.exec()

    def get_machine_id(self):
        """获取机器ID（CPU序列号优先，失败则使用硬盘序列号）"""
        try:
            system = platform.system()
            if system == "Windows":
                try:
                    cmd = (
                        "cmd /c wmic cpu get ProcessorId"
                        " | findstr /V ProcessorId"
                    )
                    result = subprocess.run(
                        cmd, shell=True,
                        capture_output=True, text=True, timeout=10
                    )
                    if (result.returncode == 0
                            and result.stdout.strip()):
                        lines = [
                            line.strip()
                            for line in result.stdout.strip().split('\n')
                            if line.strip()
                        ]
                        if lines and lines[0]:
                            return lines[0]
                except Exception:
                    pass

            elif system == "Linux":
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        for line in f:
                            if (line.startswith('vendor_id')
                                    or line.startswith('model name')):
                                cpu_info = (
                                    line.split(':')[1].strip()
                                )
                                if cpu_info:
                                    return hashlib.md5(
                                        cpu_info.encode()
                                    ).hexdigest()[:16].upper()
                except Exception:
                    pass

            if system == "Windows":
                try:
                    cmd = (
                        "cmd /c wmic diskdrive get SerialNumber"
                        " | findstr /V SerialNumber"
                    )
                    result = subprocess.run(
                        cmd, shell=True,
                        capture_output=True, text=True, timeout=10
                    )
                    if (result.returncode == 0
                            and result.stdout.strip()):
                        lines = [
                            line.strip()
                            for line in result.stdout.strip().split('\n')
                            if line.strip()
                        ]
                        if lines and lines[0]:
                            return lines[0]
                except Exception:
                    pass

            elif system == "Linux":
                try:
                    result = subprocess.run(
                        ["lsblk", "-o", "SERIAL", "-r"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            disk_id = lines[1].strip()
                            if disk_id:
                                return disk_id
                except Exception:
                    pass

            try:
                machine_uuid = uuid.getnode()
                machine_id = hashlib.md5(
                    str(machine_uuid).encode()
                ).hexdigest()[:16].upper()
                return machine_id
            except Exception:
                pass

            return "UNKNOWN_MACHINE_ID"

        except Exception as e:
            return "UNKNOWN_MACHINE_ID"

    def register_product(self, machine_id, dialog):
        """注册产品"""
        reg_code = self.reg_code_input.text().strip()
        if not reg_code:
            QMessageBox.warning(self, '警告', '请输入注册码')
            return

        if self.verify_registration_code(machine_id, reg_code):
            self.save_registration_info(machine_id, reg_code)
            QMessageBox.information(self, '成功', '注册成功！')
            dialog.accept()
            self.is_registered = True
        else:
            QMessageBox.critical(
                self, '错误', '注册码无效，请检查后重试'
            )

    def verify_registration_code(self, machine_id, reg_code):
        """验证注册码（委托到编译模块）"""
        return _verify_registration_code(machine_id, reg_code)

    def save_registration_info(self, machine_id, reg_code):
        """保存注册信息"""
        config_dir = CONFIG_DIR
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        reg_info = {
            'machine_id': machine_id,
            'reg_code': reg_code,
            'registered': True
        }

        with open(REGISTRATION_FILE, 'w') as f:
            json.dump(reg_info, f)

    def check_registration(self):
        """检查注册状态（通过编译模块验证注册码）"""
        reg_file = REGISTRATION_FILE
        if not os.path.exists(reg_file):
            self.is_registered = False
            return False

        try:
            with open(reg_file, 'r') as f:
                reg_info = json.load(f)

            if reg_info.get('registered', False):
                machine_id = self.get_machine_id()
                reg_code = reg_info.get('reg_code', '')
                if (reg_info.get('machine_id') == machine_id
                        and self.verify_registration_code(
                            machine_id, reg_code
                        )):
                    self.is_registered = True
                    return True

            self.is_registered = False
            return False
        except Exception:
            self.is_registered = False
            return False
