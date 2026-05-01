"""QThread工作线程模块 - MatPreML"""

import os
import json
import time
import requests
from PyQt6.QtCore import QThread, pyqtSignal

from .config import DEEPSEEK_CONFIG_FILE, DEFAULT_DEEPSEEK_CONFIG


class TrainingThread(QThread):
    """模型训练线程"""
    training_started = pyqtSignal()
    training_finished = pyqtSignal(object, object, object, str, str)
    training_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_analyzer = parent
        self.cpu_limit = None

    def set_cpu_limit(self, cpu_limit):
        self.cpu_limit = cpu_limit

    def run(self):
        try:
            self.training_started.emit()
            if self.cpu_limit is not None:
                os.environ['OMP_NUM_THREADS'] = str(self.cpu_limit)
                os.environ['MKL_NUM_THREADS'] = str(self.cpu_limit)
                os.environ['NUMEXPR_NUM_THREADS'] = str(self.cpu_limit)

            if (hasattr(self.data_analyzer, 'training_all_models')
                    and self.data_analyzer.training_all_models):
                result = self.data_analyzer._perform_all_models_training()
                self.training_finished.emit(
                    result['model_results'], None, None,
                    result['ranking_text'], result['ranking_filename']
                )
                self.data_analyzer.training_all_models = False
            else:
                result = self.data_analyzer._perform_training()
                self.training_finished.emit(
                    result['model'], result['y_test'], result['y_pred'],
                    result['metrics_text'], result['best_params_str']
                )
        except Exception as e:
            model_name = self.data_analyzer.model_combo.currentText() if hasattr(self.data_analyzer, 'model_combo') else '未知'
            self.training_error.emit(f"[{model_name}] {str(e)}")


class DeepSeekAPIThread(QThread):
    """DeepSeek API调用线程"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, user_message, learning_materials, parent=None):
        super().__init__(parent)
        self.user_message = user_message
        self.learning_materials = learning_materials

    def run(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"""
你是DeepSeek AI助手，具有访问互联网的能力。请基于以下学习资料并结合互联网知识回答用户问题：

学习资料：
{self.learning_materials}

用户问题：
{self.user_message}

请提供准确、简洁的回答。你的回答应该：
1. 首先参考提供的学习资料
2. 如果问题涉及最新信息、实时数据或需要网络验证的内容，请主动访问互联网获取最新信息
3. 结合学习资料和互联网信息提供最全面和准确的答案
4. 如果使用了互联网信息，请在回答中说明信息来源

请注意：对于需要最新数据、实时信息或事实验证的问题，必须使用互联网获取最新信息。
"""
                config = self._load_config()
                api_url = config.get("api_url", DEFAULT_DEEPSEEK_CONFIG["api_url"])
                api_key = config.get("api_key", DEFAULT_DEEPSEEK_CONFIG["api_key"])

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一个专业的材料科学和机器学习助手，具有互联网访问能力。你可以搜索和获取最新的信息来回答问题。当需要最新信息时，请主动获取网络信息。"},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "top_p": 0.6
                }

                response = requests.post(api_url, headers=headers, json=data, timeout=20)

                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        ai_response = result['choices'][0]['message']['content']
                        if ai_response and ai_response.strip():
                            self.response_ready.emit(ai_response)
                            return
                        else:
                            if attempt < max_retries - 1:
                                continue
                            self.error_occurred.emit("API返回了空响应")
                    else:
                        if attempt < max_retries - 1:
                            continue
                        self.error_occurred.emit("API响应格式不正确")
                else:
                    error_msg = f"API调用失败，状态码: {response.status_code}"
                    if response.status_code == 401:
                        error_msg += "，请检查API密钥是否正确"
                    elif response.status_code == 429:
                        error_msg += "，API调用次数超限"
                    elif response.status_code == 500:
                        error_msg += "，服务器内部错误"
                    elif response.status_code == 400:
                        error_msg += "，请求参数错误"

                    if response.status_code in [408, 502, 503, 504] and attempt < max_retries - 1:
                        continue
                    self.error_occurred.emit(error_msg)
                    return

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit("API调用超时，请检查网络连接")
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit("网络连接错误，请检查网络设置")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit(f"网络请求错误: {str(e)}")
            except ValueError as e:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit(f"API响应解析错误: {str(e)}")
            except KeyError as e:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit(f"API响应结构错误: 缺少字段 {str(e)}")
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                self.error_occurred.emit(f"发生未知错误: {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(1)

        self.error_occurred.emit("网络连接失败，已尝试多次连接但均未成功，请检查网络设置")

    def _load_config(self):
        if os.path.exists(DEEPSEEK_CONFIG_FILE):
            try:
                with open(DEEPSEEK_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return dict(DEFAULT_DEEPSEEK_CONFIG)
        return dict(DEFAULT_DEEPSEEK_CONFIG)


class ConnectionTestThread(QThread):
    """DeepSeek API连接测试线程"""
    connection_result = pyqtSignal(bool, str)

    def run(self):
        config_file = DEEPSEEK_CONFIG_FILE
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                api_url = config.get("api_url", DEFAULT_DEEPSEEK_CONFIG["api_url"])
                api_key = config.get("api_key", DEFAULT_DEEPSEEK_CONFIG["api_key"])
            except Exception:
                api_url = DEFAULT_DEEPSEEK_CONFIG["api_url"]
                api_key = DEFAULT_DEEPSEEK_CONFIG["api_key"]
        else:
            api_url = DEFAULT_DEEPSEEK_CONFIG["api_url"]
            api_key = DEFAULT_DEEPSEEK_CONFIG["api_key"]

        test_url = (api_url.replace("/chat/completions", "/models")
                    if "/chat/completions" in api_url
                    else api_url + "/models")

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code == 200:
                self.connection_result.emit(True, "连接成功")
            else:
                self.connection_result.emit(False, f"HTTP状态码: {response.status_code}")
        except requests.exceptions.Timeout:
            self.connection_result.emit(False, "连接超时")
        except requests.exceptions.ConnectionError:
            self.connection_result.emit(False, "网络连接错误")
        except Exception as e:
            self.connection_result.emit(False, f"未知错误: {str(e)}")


class FeatureEngineeringThread(QThread):
    """特征工程线程"""
    finished = pyqtSignal(object, str, list)
    error = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent, selected_featurizers, is_learning_data=True):
        super().__init__(parent)
        self.parent_analyzer = parent
        self.selected_featurizers = selected_featurizers
        self.is_learning_data = is_learning_data

    def run(self):
        try:
            self.parent_analyzer._perform_feature_engineering_internal(
                self.selected_featurizers, self.is_learning_data
            )
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
            self.error.emit(error_msg)
        finally:
            self.quit()
