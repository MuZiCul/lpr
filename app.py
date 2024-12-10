from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
from aip import AipOcr
from flask_sqlalchemy import SQLAlchemy
import hashlib
import os.path
from flask_migrate import Migrate
import sqlite3
import sys
import requests
import base64

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///license_plate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# 数据库模型
class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_hash = db.Column(db.String(64), unique=True, nullable=False)
    timestamp = db.Column(db.String(20), nullable=False)
    final_result = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    processed_images = db.Column(db.JSON)
    recognition_results = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.now)
    baidu_result = db.Column(db.JSON)  # 存储百度API的识别结果


# 添加权重设置模型
class WeightSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    step = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    @staticmethod
    def get_default_weights():
        return {
            1: {"weight": 1.0, "description": "完整清晰的原图识别结果最可靠"},
            2: {"weight": 0.7, "description": "颜色特征提取可能受光照影响"},
            3: {"weight": 0.9, "description": "保留了主要特征"},
            4: {"weight": 0.9, "description": "增强对比度，改善识别效果"},
            5: {"weight": 1.0, "description": "去噪但可能损失细节"},
            6: {"weight": 0.5, "description": "边缘特征有用但不完整"},
            7: {"weight": 0.2, "description": "二值化可能丢失信息"},
            8: {"weight": 0.9, "description": "CLAHE算法效果好"},
            9: {"weight": 0.8, "description": "增强边缘但可能引入噪声"},
            10: {"weight": 0.2, "description": "主要用于定位"}
        }


# 添加配置型
class APIConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.String(50), nullable=False)
    api_key = db.Column(db.String(100), nullable=False)
    secret_key = db.Column(db.String(100), nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    @staticmethod
    def get_default_config():
        return {
            'app_id': '',
            'api_key': '',
            'secret_key': ''
        }


def calculate_image_hash(img):
    """
    计算图片的特征哈希值
    """
    # 调整图片大小为8x8
    small = cv2.resize(img, (8, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # 计算平均值
    avg = gray.mean()
    # 计算hash值
    diff = gray > avg
    # 将bool值转换为字符串
    hash_str = ''.join(['1' if b else '0' for b in diff.flatten()])
    # 计算MD5
    return hashlib.md5(hash_str.encode()).hexdigest()


def get_cached_result(img):
    """
    检查是否存在相同图片的缓存结果
    """
    image_hash = calculate_image_hash(img)
    record = ImageRecord.query.filter_by(image_hash=image_hash).first()
    if record:
        print(f'Found cached result for {image_hash}')
        
        # 如果缓存记录中没有百度API结果，获取新的结果并更新缓存
        if not record.baidu_result:
            try:
                # 保存临时文件用于百度API识别
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{image_hash}.jpg")
                cv2.imwrite(temp_path, img)
                
                # 获取百度API识别结果
                baidu_recognizer = PlateRecognizer()
                baidu_number, baidu_confidence = baidu_recognizer.recognize(temp_path)
                
                # 更新缓存记录
                record.baidu_result = {
                    'number': baidu_number,
                    'confidence': baidu_confidence
                }
                db.session.commit()
                
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"获取百度API结果失败: {str(e)}")
                record.baidu_result = {
                    'number': None,
                    'confidence': 0
                }
                db.session.commit()
    
    return record


def save_record(img, timestamp, final_result, confidence, processed_images, recognition_results, baidu_result=None):
    """
    保存识别记录到数据库
    """
    image_hash = calculate_image_hash(img)
    record = ImageRecord(
        image_hash=image_hash,
        timestamp=timestamp,
        final_result=final_result,
        confidence=confidence,
        processed_images=processed_images,
        recognition_results=recognition_results,
        baidu_result=baidu_result  # 添加百度API结果
    )
    db.session.add(record)
    db.session.commit()


def ensure_folders():
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)


def process_and_recognize(img, timestamp):
    """
    处理图片并在每个步骤后进行识别
    """
    steps = []

    # 图片大小标准化
    height, width = img.shape[:2]
    standard_width = 800
    scale = standard_width / width
    standard_size = (standard_width, int(height * scale))
    img = cv2.resize(img, standard_size)

    # 1. 原始图片
    original_filename = f'1_original_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], original_filename), img)
    steps.append({
        'image': img,
        'filename': original_filename,
        'type': '原始图片'
    })

    # 2. 颜色特征分析
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 70, 70])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(blue_mask, yellow_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    color_filename = f'2_color_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], color_filename), color_mask)
    steps.append({
        'image': color_mask,
        'filename': color_filename,
        'type': '颜色分析'
    })

    # 3. 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filename = f'3_gray_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], gray_filename), gray)
    steps.append({
        'image': gray,
        'filename': gray_filename,
        'type': '灰度处理'
    })

    # 4. 直方图均衡化
    equ = cv2.equalizeHist(gray)
    equ_filename = f'4_equalize_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], equ_filename), equ)
    steps.append({
        'image': equ,
        'filename': equ_filename,
        'type': '直方图均衡化'
    })

    # 5. 高斯模糊
    blur = cv2.GaussianBlur(equ, (5, 5), 0)
    blur_filename = f'5_blur_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], blur_filename), blur)
    steps.append({
        'image': blur,
        'filename': blur_filename,
        'type': '高斯模糊'
    })

    # 6. Sobel边缘检测
    gradX = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gradY = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    sobel_filename = f'6_sobel_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], sobel_filename), gradient)
    steps.append({
        'image': gradient,
        'filename': sobel_filename,
        'type': 'Sobel边缘检测'
    })

    # 7. 自适应阈值处理
    thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh_filename = f'7_thresh_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], thresh_filename), thresh)
    steps.append({
        'image': thresh,
        'filename': thresh_filename,
        'type': '自适应阈值'
    })

    # 8. 对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced_filename = f'8_enhanced_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], enhanced_filename), enhanced)
    steps.append({
        'image': enhanced,
        'filename': enhanced_filename,
        'type': '对比度增强'
    })

    # 9. 锐化处理
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    sharp_filename = f'9_sharp_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], sharp_filename), sharpened)
    steps.append({
        'image': sharpened,
        'filename': sharp_filename,
        'type': '锐化处理'
    })

    # 10. 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    morph_filename = f'10_morph_{timestamp}.jpg'
    cv2.imwrite(os.path.join(app.config['PROCESSED_FOLDER'], morph_filename), morph)
    steps.append({
        'image': morph,
        'filename': morph_filename,
        'type': '形态学处理'
    })

    return steps


def get_api_config_from_db():
    """从数据库获取API配置，如果不存在则返回None"""
    config = APIConfig.query.first()
    if config and config.app_id and config.api_key and config.secret_key:
        return config
    return None


def get_ocr_result(image_bytes):
    """调用百度OCR API"""
    config = get_api_config_from_db()
    if not config:
        return None, "请先配置百度OCR API"
    
    ocr_client = AipOcr(config.app_id, config.api_key, config.secret_key)
    options = {
        "detect_direction": "true",
        "plate_detect": "true"
    }
    try:
        result = ocr_client.licensePlate(image_bytes, options)
        if 'words_result' in result:
            return result['words_result']['number'], None
        return None, "无法识别车牌"
    except Exception as e:
        return None, str(e)


def recognize_plate(plate_img):
    """
    识别车牌号码返回所有处理版本的识别结果
    """
    results = []

    # 原始图片
    _, buffer = cv2.imencode('.jpg', plate_img)
    number, error = get_ocr_result(buffer.tobytes())
    results.append({"type": "原始图片", "number": number, "error": error})

    # 灰度图
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, buffer = cv2.imencode('.jpg', gray)
    number, error = get_ocr_result(buffer.tobytes())
    results.append({"type": "灰度处理", "number": number, "error": error})

    # 增强对比度
    lab = cv2.cvtColor(plate_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    _, buffer = cv2.imencode('.jpg', enhanced)
    number, error = get_ocr_result(buffer.tobytes())
    results.append({"type": "对比度增强", "number": number, "error": error})

    # 二值化
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, buffer = cv2.imencode('.jpg', binary)
    number, error = get_ocr_result(buffer.tobytes())
    results.append({"type": "二值化处理", "number": number, "error": error})

    # 计算最终结果和可信度
    valid_results = [r["number"] for r in results if r["number"]]
    if valid_results:
        most_common = max(set(valid_results), key=valid_results.count)
        confidence = valid_results.count(most_common) / len(results)
        return most_common, confidence, results
    return None, 0, results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_step', methods=['POST'])
def process_step():
    """
    处理单个步骤并返回结果
    """
    data = request.get_json()
    step_number = data.get('step')
    image_path = os.path.join(app.config['PROCESSED_FOLDER'], data.get('filename'))

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'error': '图片读取失败'})

    # 进行OCR识别
    _, buffer = cv2.imencode('.jpg', img)
    number, error = get_ocr_result(buffer.tobytes())

    return jsonify({
        'step': step_number,
        'number': number,
        'error': error
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    ensure_folders()
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})

    if file:
        # 读取图片
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': '图片读取失败'})

        # 检查缓存
        cached_record = get_cached_result(img)
        if cached_record:
            return jsonify({
                'message': '处理成功（缓存）',
                'steps': cached_record.processed_images,
                'cached': True,
                'final_result': cached_record.final_result,
                'confidence': cached_record.confidence,
                'recognition_results': cached_record.recognition_results,
                'image_hash': cached_record.image_hash,
                'baidu_result': cached_record.baidu_result
            })
        print('No cache found, processing...')
        # 如果没有缓存，进行常处理
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"original_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(file_bytes)

        # 获取所有处理步骤
        steps = process_and_recognize(img, timestamp)

        # 准备保存的数据
        processed_images = []
        recognition_results = []

        # 处理每个步骤并返回结果
        for i, step in enumerate(steps, 1):
            processed_images.append({
                'filename': step['filename'],
                'type': step['type']
            })

            # 进行OCR识别
            _, buffer = cv2.imencode('.jpg', step['image'])
            number, error = get_ocr_result(buffer.tobytes())
            recognition_results.append({
                'step': i,
                'type': step['type'],
                'number': number,
                'error': error
            })

        # 计算最终结果
        valid_results = [r for r in recognition_results if r['number']]
        if valid_results:
            result_counts = {}
            for r in valid_results:
                result_counts[r['number']] = result_counts.get(r['number'], 0) + 1

            final_result = max(result_counts.items(), key=lambda x: x[1])[0]
            confidence = (result_counts[final_result] / len(recognition_results)) * 100
        else:
            final_result = None
            confidence = 0

        # 获取百度API识别结果
        baidu_recognizer = PlateRecognizer()
        baidu_number, baidu_confidence = baidu_recognizer.recognize(filepath)
        baidu_result = {
            'number': baidu_number,
            'confidence': baidu_confidence
        }
        
        # 保存记录时包含百度API结果
        save_record(
            img, 
            timestamp, 
            final_result, 
            confidence, 
            processed_images, 
            recognition_results,
            baidu_result  # 添加百度API结果
        )

        image_hash = calculate_image_hash(img)
        # 返回结果时包含百度API的识别结果
        return jsonify({
            'message': '处理成功',
            'steps': processed_images,
            'recognition_results': recognition_results,
            'final_result': final_result,
            'confidence': confidence,
            'image_hash': image_hash,
            'baidu_result': baidu_result
        })


@app.route('/check_progress', methods=['POST'])
def check_progress():
    """
    检查处理进度
    """
    data = request.get_json()
    step = data.get('step')
    timestamp = data.get('timestamp')

    # 检查该步骤的处理结果
    result_file = os.path.join(app.config['PROCESSED_FOLDER'], f'{step}_{timestamp}.jpg')
    if os.path.exists(result_file):
        # 获取处理结果
        img = cv2.imread(result_file)
        _, buffer = cv2.imencode('.jpg', img)
        number, error = get_ocr_result(buffer.tobytes())

        return jsonify({
            'step': step,
            'completed': True,
            'filename': f'{step}_{timestamp}.jpg',
            'number': number,
            'error': error
        })

    return jsonify({
        'step': step,
        'completed': False
    })


@app.route('/processed/<path:filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def check_db_connection():
    """
    检查数据库连接是否正常
    """
    try:
        conn = sqlite3.connect('license_plate.db')
        conn.close()
        return True
    except sqlite3.Error:
        return False


def init_db():
    """
    初始化数据库
    """
    db_path = 'license_plate.db'
    db_exists = os.path.exists(db_path)

    if not db_exists:
        print("数据库不存在，正在创建...")
        try:
            with app.app_context():
                db.create_all()
            print("数据库创建成功！")
        except Exception as e:
            print(f"数据库创建失败：{str(e)}")
            sys.exit(1)
    else:
        print("检查数据库连接...")
        if not check_db_connection():
            print("数据库连接失败，尝试重新创建...")
            try:
                os.remove(db_path)
                with app.app_context():
                    db.create_all()
                print("数据库重新创建成功！")
            except Exception as e:
                print(f"数据库重新创建失败：{str(e)}")
                sys.exit(1)
        else:
            print("数据库连接正常")
            # 检查表结构
            try:
                with app.app_context():
                    db.create_all()
                print("数据库表结构已更新")
            except Exception as e:
                print(f"数据库表结构更新失败：{str(e)}")
                sys.exit(1)


def cleanup_old_records():
    """
    清理旧记录和相关文件
    """
    try:
        with app.app_context():
            # 获取30天前的时间
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            # 查找旧记录
            old_records = ImageRecord.query.filter(ImageRecord.created_at < thirty_days_ago).all()

            for record in old_records:
                # 删除相关图片文件
                for image in record.processed_images:
                    filepath = os.path.join(app.config['PROCESSED_FOLDER'], image['filename'])
                    if os.path.exists(filepath):
                        os.remove(filepath)
                # 删除记录
                db.session.delete(record)

            db.session.commit()
            print(f"已清理 {len(old_records)} 条旧记录")
    except Exception as e:
        print(f"清理旧记录失败：{str(e)}")


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """
    清除指定图片的缓存记录
    """
    try:
        data = request.get_json()
        image_hash = data.get('image_hash')

        if not image_hash:
            return jsonify({'error': '未提供图片哈希值'})

        with app.app_context():
            # 查找记录
            record = ImageRecord.query.filter_by(image_hash=image_hash).first()
            if record:
                # 删除相关图片文件
                for image in record.processed_images:
                    filepath = os.path.join(app.config['PROCESSED_FOLDER'], image['filename'])
                    if os.path.exists(filepath):
                        os.remove(filepath)
                # 删除记录
                db.session.delete(record)
                db.session.commit()
                return jsonify({'message': '缓存已清除'})
            else:
                return jsonify({'error': '未找到缓存记录'})
    except Exception as e:
        return jsonify({'error': f'清除缓存失败：{str(e)}'})


@app.route('/clear_all_cache', methods=['POST'])
def clear_all_cache():
    """清除所有缓存记录"""
    try:
        with app.app_context():
            # 获取所有记录
            records = ImageRecord.query.all()
            
            # 删除所有相关文件
            for record in records:
                if record.processed_images:
                    for image in record.processed_images:
                        filepath = os.path.join(app.config['PROCESSED_FOLDER'], image['filename'])
                        if os.path.exists(filepath):
                            os.remove(filepath)
                # 删除记录
                db.session.delete(record)
            
            db.session.commit()
            return jsonify({'message': '所有缓存已清除'})
    except Exception as e:
        return jsonify({'error': f'清除缓存失败：{str(e)}'})


# 添加获取和更新权重的路由
@app.route('/get_weights', methods=['GET'])
def get_weights():
    weights = WeightSetting.query.all()
    if not weights:
        # 如果没有设置，用默认值
        default_weights = WeightSetting.get_default_weights()
        for step, data in default_weights.items():
            weight = WeightSetting(
                step=step,
                weight=data['weight'],
                description=data['description']
            )
            db.session.add(weight)
        db.session.commit()
        weights = WeightSetting.query.all()
    
    return jsonify({
        'weights': [{
            'step': w.step,
            'weight': w.weight,
            'description': w.description
        } for w in weights]
    })

@app.route('/update_weights', methods=['POST'])
def update_weights():
    """
    更新权重设置
    """
    try:
        data = request.get_json()
        if not data or 'weights' not in data:
            return jsonify({'error': '无效的请求数据'})

        weights = data['weights']
        
        with app.app_context():
            # 开始事务
            db.session.begin()
            try:
                # 更新每个权重
                for weight_data in weights:
                    step = weight_data.get('step')
                    weight_value = weight_data.get('weight')
                    description = weight_data.get('description')

                    if step is None or weight_value is None:
                        continue

                    # 查找或创建权重记录
                    weight_record = WeightSetting.query.filter_by(step=step).first()
                    if weight_record:
                        weight_record.weight = weight_value
                        if description:
                            weight_record.description = description
                    else:
                        new_weight = WeightSetting(
                            step=step,
                            weight=weight_value,
                            description=description
                        )
                        db.session.add(new_weight)

                # 提交事务
                db.session.commit()
                return jsonify({'message': '权重更新成功'})
            except Exception as e:
                # 如果出错，回滚事务
                db.session.rollback()
                raise e
    except Exception as e:
        return jsonify({'error': f'更新失败：{str(e)}'})


# 添加 PlateRecognizer 类
class PlateRecognizer:
    def __init__(self):
        self.request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
        self.headers = {'content-type': 'application/x-www-form-urlencoded'}
    
    def _get_access_token(self):
        """获取百度API access_token"""
        config = get_api_config_from_db()
        if not config:
            return None
            
        host = 'https://aip.baidubce.com/oauth/2.0/token'
        params = {
            'grant_type': 'client_credentials',
            'client_id': config.api_key,
            'client_secret': config.secret_key
        }
        response = requests.post(host, params=params)
        if response:
            return response.json().get('access_token')
        return None

    def recognize(self, image_path):
        try:
            config = get_api_config_from_db()
            if not config:
                return None, "请先配置百度OCR API"

            # 获取access_token
            access_token = self._get_access_token()
            if not access_token:
                return None, "获取access_token失败"

            # 二进制方式打开图片文件
            with open(image_path, 'rb') as f:
                img = base64.b64encode(f.read())

            # 组装请求参数
            params = {"image": img}
            request_url = f"{self.request_url}?access_token={access_token}"

            # 发送请求
            response = requests.post(request_url, data=params, headers=self.headers)
            
            if not response:
                return None, 0

            result = response.json()
            
            if 'error_code' in result:
                print(f"百度API调用失败: {result['error_msg']}")
                return None, 0
                
            if 'words_result' not in result or not result['words_result']:
                print(f"百度API未识别到车牌: {image_path}")
                return None, 0
            
            # 获取识别结果
            plate_info = result['words_result']
            plate_number = plate_info['number']
            # 计算平均置信度
            probabilities = plate_info.get('probability', [1.0])  # 果没有置信度信息，默认为1.0
            if isinstance(probabilities, list):
                confidence = sum(float(p) for p in probabilities) / len(probabilities) * 100
            else:
                confidence = float(probabilities) * 100

            return plate_number, confidence

        except Exception as e:
            print(f"百度API识别失败: {str(e)}")
            return None, 0


@app.route('/get_api_config', methods=['GET'])
def get_api_config():
    config = APIConfig.query.first()
    if not config:
        # 如果没有配置，使用默认值
        default_config = APIConfig.get_default_config()
        config = APIConfig(**default_config)
        db.session.add(config)
        db.session.commit()
    
    return jsonify({
        'app_id': config.app_id,
        'api_key': config.api_key,
        'secret_key': config.secret_key
    })

@app.route('/update_api_config', methods=['POST'])
def update_api_config():
    try:
        data = request.get_json()
        if not all(key in data for key in ['app_id', 'api_key', 'secret_key']):
            return jsonify({'error': '缺少必要的配置参数'})

        config = APIConfig.query.first()
        if config:
            config.app_id = data['app_id']
            config.api_key = data['api_key']
            config.secret_key = data['secret_key']
        else:
            config = APIConfig(**data)
            db.session.add(config)
        
        db.session.commit()
        return jsonify({'message': 'API配置已更新'})
    except Exception as e:
        return jsonify({'error': f'更新失败：{str(e)}'})


def init_api_config():
    """初始化API配置"""
    try:
        with app.app_context():
            config = APIConfig.query.first()
            if not config:
                default_config = APIConfig.get_default_config()
                config = APIConfig(**default_config)
                db.session.add(config)
                db.session.commit()
                print("已创建空的API配置，请在使用前配置")
            else:
                if not (config.app_id and config.api_key and config.secret_key):
                    print("API配置不完整，请在使用前完成配置")
                else:
                    print("API配置已存在")
    except Exception as e:
        print(f"API配置初始化失败：{str(e)}")
        sys.exit(1)


@app.route('/delete_api_config', methods=['POST'])
def delete_api_config():
    """删除API配置"""
    try:
        config = APIConfig.query.first()
        if config:
            # 清空配置值而是删除记录
            config.app_id = ''
            config.api_key = ''
            config.secret_key = ''
            db.session.commit()
            return jsonify({'message': 'API配置已删除'})
        return jsonify({'message': '没有找到API配置'})
    except Exception as e:
        return jsonify({'error': f'删除失败：{str(e)}'})


@app.route('/cache_list')
def cache_list():
    # 获取所有缓存记录
    records = ImageRecord.query.order_by(ImageRecord.created_at.desc()).all()
    return render_template('cache_list.html', records=records)

@app.route('/get_cache_records')
def get_cache_records():
    records = ImageRecord.query.order_by(ImageRecord.created_at.desc()).all()
    return jsonify({
        'records': [{
            'image_hash': record.image_hash,
            'timestamp': record.timestamp,
            'final_result': record.final_result,
            'confidence': record.confidence,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'baidu_result': record.baidu_result,
            'original_image': record.processed_images[0]['filename'] if record.processed_images else None  # 添加原图路径
        } for record in records]
    })

@app.route('/cache_detail/<image_hash>')
def cache_detail(image_hash):
    """显示缓存详情页面"""
    record = ImageRecord.query.filter_by(image_hash=image_hash).first()
    if not record:
        return redirect('/')
    return render_template('detail.html', record=record)


@app.route('/weight_settings')
def weight_settings():
    """显示权重设置页面"""
    return render_template('weight_settings.html')


if __name__ == '__main__':
    try:
        # 确保文件夹存在
        ensure_folders()

        # 初始化数据库
        init_db()

        # 初始化API配置
        init_api_config()

        # 清理旧记录
        cleanup_old_records()

        # 启动应用
        app.run(debug=True)
    except Exception as e:
        print(f"应用启动失败：{str(e)}")
        sys.exit(1)
