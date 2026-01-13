"""
Host2Play 自动续期脚本 - YOLO 图像识别 本地版本
- 使用 Playwright + Camoufox 过 Cloudflare
- 使用 YOLO 模型进行 reCAPTCHA 图像识别
- 适合在本地环境中运行，带图形界面

主要特性：
1. 使用 YOLO 模型识别 reCAPTCHA 图像
2. 支持 3x3 和 4x4 网格验证
3. 支持动态验证和一次性选择验证
4. 本地浏览器窗口可见，方便调试
"""
import asyncio
import time
import logging
import random
import os
import shutil
from typing import Optional, List, Tuple
from datetime import datetime
import requests
import cv2
import numpy as np
from PIL import Image

from playwright.async_api import async_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeoutError
from camoufox.async_api import AsyncCamoufox
from browserforge.fingerprints import Screen

# YOLO 模型（可选）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ YOLO 未安装，图形验证将被跳过")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
MODEL_PATH = "model.onnx"
MODEL_DOWNLOAD_URLS = [
    "https://media.githubusercontent.com/media/DannyLuna17/RecaptchaV2-IA-Solver/main/model.onnx",
    "https://github.com/DannyLuna17/RecaptchaV2-IA-Solver/raw/main/model.onnx",
]
RENEW_URL = os.environ.get('RENEW_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
VERBOSE = True


def send_telegram_message(message: str, photo_path: str = None) -> bool:
    """发送Telegram消息"""
    bot_token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    
    if not bot_token or not chat_id:
        logger.warning("⚠️ 未设置 Telegram 配置，跳过消息推送")
        return False
    
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, files=files, data=data, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ Telegram 消息发送成功")
            return True
        else:
            logger.warning(f"⚠️ Telegram 消息发送失败: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Telegram 消息发送出错: {str(e)}")
        return False


def download_yolo_model():
    """下载 YOLO 模型文件（如果不存在）"""
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 1000000:
            logger.info(f"✅ 模型文件已存在: {MODEL_PATH} ({file_size / (1024*1024):.2f} MB)")
            return True
        else:
            logger.warning(f"⚠️ 模型文件大小异常 ({file_size} bytes)，将重新下载")
            os.remove(MODEL_PATH)
    
    logger.info("📥 模型文件不存在，开始下载...")
    
    for i, url in enumerate(MODEL_DOWNLOAD_URLS, 1):
        try:
            logger.info(f"🔄 尝试方法 {i}/{len(MODEL_DOWNLOAD_URLS)}: {url[:80]}...")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            temp_path = MODEL_PATH + ".tmp"
            with open(temp_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if downloaded % (10 * 1024 * 1024) == 0:
                            logger.info(f"   已下载: {downloaded / (1024*1024):.1f} MB")
            
            file_size = os.path.getsize(temp_path)
            if file_size < 1000000:
                logger.warning(f"⚠️ 下载的文件大小异常 ({file_size} bytes)")
                os.remove(temp_path)
                continue
            
            os.rename(temp_path, MODEL_PATH)
            logger.info(f"✅ 模型下载成功！文件大小: {file_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 方法 {i} 失败: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    logger.error("❌ 所有下载方法均失败！")
    return False


def random_delay(mu=0.3, sigma=0.1):
    """随机延迟模拟人类行为"""
    delay = np.random.normal(mu, sigma)
    delay = max(0.1, delay)
    time.sleep(delay)


async def human_like_delay(min_time=0.5, max_time=1.5):
    """更自然的随机延迟"""
    await asyncio.sleep(random.uniform(min_time, max_time))


def download_img(name, url, max_retries=3):
    """下载图片（带重试机制）"""
    for attempt in range(max_retries):
        try:
            # 增加超时时间，添加重试逻辑
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                with open(f'{name}.png', 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                del response
                return True
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ 图片下载超时 {name}，重试 {attempt + 2}/{max_retries}...")
                time.sleep(1)
                continue
            else:
                print(f"  ✗ 图片下载失败 {name}: 超时 (已重试{max_retries}次)")
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ 图片下载失败 {name}，重试 {attempt + 2}/{max_retries}...")
                time.sleep(0.5)
                continue
            else:
                print(f"  ✗ 图片下载失败 {name}: {e}")
                return False
    return False


async def get_target_num(page: Page) -> int:
    """获取验证目标类别编号"""
    target_mappings = {
        "bicycle": 1,
        "bus": 5,
        "boat": 8,
        "car": 2,
        "hydrant": 10,
        "motorcycle": 3,
        "traffic": 9
    }
    
    try:
        # 在挑战 iframe 中查找目标文本
        challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
        target_element = challenge_frame.locator('#rc-imageselect strong').first
        target_text = await target_element.text_content(timeout=10000)
        
        for term, value in target_mappings.items():
            if term in target_text.lower():
                return value
        
        return 1000
    except Exception as e:
        logger.error(f"获取目标类型失败: {e}")
        return 1000


def dynamic_and_selection_solver(target_num, verbose, model):
    """解决 3x3 网格验证（动态和一次性选择）"""
    try:
        if not os.path.exists("0.png"):
            if verbose: print("  ✗ 图片文件不存在: 0.png")
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        # 使用默认参数，像参考项目一样
        result = model.predict(image, task="detect", verbose=False)
        
        # 获取目标索引
        target_index = []
        count = 0
        for num in result[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1
        
        if verbose and len(target_index) > 0:
            print(f"    检测到 {len(target_index)} 个目标物体")
        
        # 计算答案位置 - 简单的中心点算法，不做过多过滤
        answers = []
        boxes = result[0].boxes.data
        for i in target_index:
            target_box = boxes[i]
            x1, y1 = int(target_box[0]), int(target_box[1])
            x2, y2 = int(target_box[2]), int(target_box[3])
            
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            
            row = yc // 100
            col = xc // 100
            answer = int(row * 3 + col + 1)
            answers.append(answer)
        
        return list(set(answers))
    except Exception as e:
        if verbose: print(f"  ✗ 图片识别失败: {e}")
        return []


def get_occupied_cells(vertices):
    """获取被占用的单元格（4x4 网格）"""
    occupied_cells = set()
    rows, cols = zip(*[((v-1)//4, (v-1) % 4) for v in vertices])
    
    for i in range(min(rows), max(rows)+1):
        for j in range(min(cols), max(cols)+1):
            occupied_cells.add(4*i + j + 1)
    
    return sorted(list(occupied_cells))


def square_solver(target_num, verbose, model):
    """解决 4x4 方格验证"""
    try:
        if not os.path.exists("0.png"):
            if verbose: print("  ✗ 图片文件不存在: 0.png")
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        # 使用默认参数，像参考项目一样
        result = model.predict(image, task="detect", verbose=False)
        boxes = result[0].boxes.data
        
        # 获取目标索引
        target_index = []
        count = 0
        for num in result[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1
        
        if verbose and len(target_index) > 0:
            print(f"    检测到 {len(target_index)} 个目标物体")
        
        answers = []
        for i in target_index:
            target_box = boxes[i]
            x1, y1 = int(target_box[0]), int(target_box[1])
            x4, y4 = int(target_box[2]), int(target_box[3])
            x2, y2 = x4, y1
            x3, y3 = x1, y4
            xys = [x1, y1, x2, y2, x3, y3, x4, y4]
            
            four_cells = []
            for j in range(4):
                x = xys[j*2]
                y = xys[(j*2)+1]
                
                # 4x4 网格坐标映射
                if x < 112.5 and y < 112.5: four_cells.append(1)
                if 112.5 < x < 225 and y < 112.5: four_cells.append(2)
                if 225 < x < 337.5 and y < 112.5: four_cells.append(3)
                if 337.5 < x <= 450 and y < 112.5: four_cells.append(4)
                
                if x < 112.5 and 112.5 < y < 225: four_cells.append(5)
                if 112.5 < x < 225 and 112.5 < y < 225: four_cells.append(6)
                if 225 < x < 337.5 and 112.5 < y < 225: four_cells.append(7)
                if 337.5 < x <= 450 and 112.5 < y < 225: four_cells.append(8)
                
                if x < 112.5 and 225 < y < 337.5: four_cells.append(9)
                if 112.5 < x < 225 and 225 < y < 337.5: four_cells.append(10)
                if 225 < x < 337.5 and 225 < y < 337.5: four_cells.append(11)
                if 337.5 < x <= 450 and 225 < y < 337.5: four_cells.append(12)
                
                if x < 112.5 and 337.5 < y <= 450: four_cells.append(13)
                if 112.5 < x < 225 and 337.5 < y <= 450: four_cells.append(14)
                if 225 < x < 337.5 and 337.5 < y <= 450: four_cells.append(15)
                if 337.5 < x <= 450 and 337.5 < y <= 450: four_cells.append(16)
            
            answer = get_occupied_cells(four_cells)
            for ans in answer:
                answers.append(ans)
        
        return sorted(list(set(answers)))
    except Exception as e:
        if verbose: print(f"  ✗ 图片识别失败: {e}")
        return []


async def get_all_captcha_img_urls(page: Page) -> List[str]:
    """获取所有验证码图片 URL"""
    try:
        challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
        images = challenge_frame.locator('#rc-imageselect-target img')
        
        count = await images.count()
        img_urls = []
        for i in range(count):
            img = images.nth(i)
            url = await img.get_attribute("src")
            img_urls.append(url)
        
        return img_urls
    except Exception as e:
        logger.error(f"获取图片 URL 失败: {e}")
        return []


async def get_all_new_dynamic_captcha_img_urls(answers: List[int], before_img_urls: List[str], page: Page) -> Tuple[bool, List[str]]:
    """获取动态验证码的新图片 URL"""
    try:
        challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
        images = challenge_frame.locator('#rc-imageselect-target img')
        
        count = await images.count()
        img_urls = []
        for i in range(count):
            img = images.nth(i)
            url = await img.get_attribute("src")
            img_urls.append(url)
        
        # 检查是否有新图片
        index_common = []
        for answer in answers:
            if img_urls[answer-1] == before_img_urls[answer-1]:
                index_common.append(answer)
        
        if len(index_common) >= 1:
            return False, img_urls
        else:
            return True, img_urls
    except Exception as e:
        logger.error(f"获取新图片 URL 失败: {e}")
        return False, []


def paste_new_img_on_main_img(main, new, loc):
    """将新图片粘贴到主图片上"""
    paste = np.copy(main)
    
    row = (loc - 1) // 3
    col = (loc - 1) % 3
    
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    
    paste[start_row:end_row, start_col:end_col] = new
    
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


async def solve_recaptcha_yolo(page: Page, verbose=True, max_attempts=8) -> bool:
    """使用 YOLO 模型解决 reCAPTCHA"""
    
    # 检查 YOLO 可用性
    if not YOLO_AVAILABLE:
        logger.error("❌ YOLO 不可用")
        return False
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        logger.error(f"✗ 模型文件不存在: {MODEL_PATH}")
        if not download_yolo_model():
            return False
    
    logger.info(f"\n✓ 加载 YOLO 模型: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        logger.error(f"❌ YOLO 模型加载失败: {e}")
        return False
    
    try:
        # 步骤 1: 查找并点击 reCAPTCHA checkbox
        await human_like_delay(1.0, 2.0)  # 大幅减少初始等待
        
        checkbox_frame = page.frame_locator('iframe[title="reCAPTCHA"]').first
        
        logger.info("✓ 点击 reCAPTCHA checkbox...")
        try:
            checkbox = checkbox_frame.locator('div.recaptcha-checkbox-border').first
            await human_like_delay(0.2, 0.5)  # 减少等待
            await checkbox.click(timeout=10000)
            logger.info("  ✅ Checkbox 已点击")
        except Exception as e:
            logger.error(f"  ❌ 点击 checkbox 失败: {e}")
            return False
        
        # 步骤 2: 等待挑战 iframe 出现或验证通过
        await human_like_delay(2.0, 3.0)  # 大幅减少等待
        
        # 检查是否直接通过
        try:
            checked = await checkbox_frame.locator('span[aria-checked="true"]').first.is_visible(timeout=5000)
            if checked:
                logger.info("✅ 无需挑战，checkbox 直接通过！")
                return True
        except:
            pass
        
        # 查找挑战 iframe
        try:
            challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
            await challenge_frame.locator('#recaptcha-reload-button, #rc-imageselect').first.wait_for(timeout=10000)
        except:
            logger.info("✅ 无需挑战，验证已通过！")
            return True
        
        logger.info("✓ 开始识别验证码...")
        
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            if verbose: print(f"\n  尝试 {attempt}/{max_attempts}...")
            
            try:
                reload_attempts = 0
                max_reload_attempts = 2  # 减少重载次数，避免超时
                
                while reload_attempts < max_reload_attempts:
                    reload_attempts += 1
                    
                    try:
                        challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
                        reload_button = challenge_frame.locator('#recaptcha-reload-button').first
                        title_wrapper = challenge_frame.locator('#rc-imageselect').first
                        
                        await reload_button.wait_for(state='visible', timeout=10000)
                        await title_wrapper.wait_for(state='visible', timeout=10000)
                    except Exception as e:
                        if verbose: print(f"  定位元素失败: {e}")
                        await asyncio.sleep(0.5)  # 减少等待
                        continue
                    
                    try:
                        target_num = await get_target_num(page)
                        if verbose:
                            try:
                                target_element = challenge_frame.locator('#rc-imageselect strong').first
                                target_text = await target_element.text_content()
                                print(f"  目标类型: {target_text} (编号: {target_num})")
                            except:
                                print(f"  目标编号: {target_num}")
                    except Exception as e:
                        if verbose: print(f"  获取目标类型失败: {e}")
                        await asyncio.sleep(1)  # 减少等待时间
                        await reload_button.click()
                        await asyncio.sleep(1)
                        continue
                    
                    if target_num == 1000:
                        if verbose: print("  跳过不支持的类型...")
                        await asyncio.sleep(0.3)  # 减少等待时间
                        await reload_button.click()
                        await asyncio.sleep(1)
                    else:
                        title_text = await title_wrapper.text_content()
                        
                        if "squares" in title_text:
                            if verbose: print("  检测到 4x4 方格验证...")
                            try:
                                img_urls = await get_all_captcha_img_urls(page)
                                if not img_urls or not download_img(0, img_urls[0], max_retries=2):
                                    await reload_button.click()
                                    await asyncio.sleep(1)
                                    continue
                            except Exception as e:
                                if verbose: print(f"  获取图片URL失败: {e}")
                                await reload_button.click()
                                await asyncio.sleep(1)
                                continue
                            answers = square_solver(target_num, verbose, model)
                            if len(answers) >= 1 and len(answers) < 16:
                                captcha = "squares"
                                break
                            else:
                                if verbose: print("    检测结果不合理，重新加载...")
                                await reload_button.click()
                                await asyncio.sleep(1)
                        elif "none" in title_text:
                            if verbose: print("  检测到 3x3 动态验证...")
                            try:
                                img_urls = await get_all_captcha_img_urls(page)
                                if not img_urls or not download_img(0, img_urls[0], max_retries=2):
                                    await reload_button.click()
                                    await asyncio.sleep(1)
                                    continue
                            except Exception as e:
                                if verbose: print(f"  获取图片URL失败: {e}")
                                await reload_button.click()
                                await asyncio.sleep(1)
                                continue
                            answers = dynamic_and_selection_solver(target_num, verbose, model)
                            if len(answers) >= 1:
                                captcha = "dynamic"
                                break
                            else:
                                if verbose: print("    未检测到足够的目标，重新加载...")
                                await reload_button.click()
                                await asyncio.sleep(1)
                        else:
                            if verbose: print("  检测到 3x3 一次性选择验证...")
                            try:
                                img_urls = await get_all_captcha_img_urls(page)
                                if not img_urls or not download_img(0, img_urls[0], max_retries=2):
                                    await reload_button.click()
                                    await asyncio.sleep(1)
                                    continue
                            except Exception as e:
                                if verbose: print(f"  获取图片URL失败: {e}")
                                await reload_button.click()
                                await asyncio.sleep(1)
                                continue
                            answers = dynamic_and_selection_solver(target_num, verbose, model)
                            if len(answers) >= 1:
                                captcha = "selection"
                                break
                            else:
                                if verbose: print("    未检测到足够的目标，重新加载...")
                                await reload_button.click()
                                await asyncio.sleep(1)
                    
                    try:
                        first_cell = challenge_frame.locator('#rc-imageselect-target td').first
                        await first_cell.wait_for(state='visible', timeout=10000)
                    except Exception as e:
                        if verbose: print(f"  等待验证码加载失败: {e}")
                        if reload_attempts < max_reload_attempts:
                            continue
                        else:
                            break
                
                if reload_attempts >= max_reload_attempts:
                    if verbose: print("  重载次数过多，跳过此轮...")
                    continue
                
                if verbose: print(f"  ✓ 识别到的答案位置: {answers}")
                if verbose: print(f"  验证类型: {captcha}")
                
                challenge_frame = page.frame_locator('iframe[title*="challenge"]').first
                
                # 处理动态验证码
                if captcha == "dynamic":
                    if verbose: print(f"    点击 {len(answers)} 个目标...")
                    for idx, answer in enumerate(answers):
                        try:
                            cell = challenge_frame.locator(f'#rc-imageselect-target td').nth(answer - 1)
                            # 确保元素可见后再点击
                            await cell.wait_for(state='visible', timeout=3000)
                            # 滚动到元素位置（确保在视口内）
                            await cell.scroll_into_view_if_needed()
                            await asyncio.sleep(0.1)
                            # 使用 force=True 强制点击
                            await cell.click(force=True)
                            if verbose: print(f"      ✓ 已点击格子 {answer} ({idx+1}/{len(answers)})")
                        except Exception as click_error:
                            if verbose: print(f"      ✗ 点击格子 {answer} 失败: {click_error}")
                        # 快速点击，避免过期（动态验证需要速度）
                        await human_like_delay(0.3, 0.6)
                    
                    dynamic_rounds = 0
                    max_dynamic_rounds = 6  # 减少动态验证轮次，避免超时
                    
                    while dynamic_rounds < max_dynamic_rounds:
                        dynamic_rounds += 1
                        if verbose: print(f"    动态验证轮次 {dynamic_rounds}/{max_dynamic_rounds}")
                        
                        before_img_urls = img_urls
                        new_img_wait_count = 0
                        max_new_img_wait = 30
                        
                        while new_img_wait_count < max_new_img_wait:
                            new_img_wait_count += 1
                            await asyncio.sleep(0.2)
                            is_new, img_urls = await get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, page)
                            if is_new:
                                break
                        
                        if new_img_wait_count >= max_new_img_wait:
                            if verbose: print("    等待新图片超时，跳出动态验证")
                            break
                        
                        new_img_index_urls = [answer-1 for answer in answers]
                        
                        for index in new_img_index_urls:
                            if not download_img(index+1, img_urls[index], max_retries=2):
                                if verbose: print("    图片下载失败，跳出动态验证")
                                break
                        
                        for answer in answers:
                            try:
                                main_img = Image.open("0.png")
                                new_img = Image.open(f"{answer}.png")
                                paste_new_img_on_main_img(main_img, new_img, answer)
                            except Exception as e:
                                if verbose: print(f"    图片处理失败: {e}")
                                break
                        
                        answers = dynamic_and_selection_solver(target_num, verbose, model)
                        
                        if len(answers) >= 1:
                            if verbose: print(f"    新一轮检测到 {len(answers)} 个目标")
                            for idx, answer in enumerate(answers):
                                try:
                                    cell = challenge_frame.locator(f'#rc-imageselect-target td').nth(answer - 1)
                                    await cell.wait_for(state='visible', timeout=3000)
                                    await cell.scroll_into_view_if_needed()
                                    await asyncio.sleep(0.1)
                                    await cell.click(force=True)
                                    if verbose: print(f"      ✓ 已点击格子 {answer} ({idx+1}/{len(answers)})")
                                except Exception as click_error:
                                    if verbose: print(f"      ✗ 点击格子 {answer} 失败: {click_error}")
                                # 快速点击，避免过期
                                await human_like_delay(0.3, 0.6)
                        else:
                            if verbose: print("    未识别到更多目标，结束动态验证")
                            break
                
                # 处理一次性选择或方格验证
                elif captcha == "selection" or captcha == "squares":
                    if verbose: print(f"    点击 {len(answers)} 个目标...")
                    for idx, answer in enumerate(answers):
                        try:
                            cell = challenge_frame.locator(f'#rc-imageselect-target td').nth(answer - 1)
                            await cell.wait_for(state='visible', timeout=3000)
                            await cell.scroll_into_view_if_needed()
                            await asyncio.sleep(0.1)
                            await cell.click(force=True)
                            if verbose: print(f"      ✓ 已点击格子 {answer} ({idx+1}/{len(answers)})")
                        except Exception as click_error:
                            if verbose: print(f"      ✗ 点击格子 {answer} 失败: {click_error}")
                        # 适中延迟（一次性选择不会过期）
                        await human_like_delay(0.5, 0.9)
                
                # 点击验证按钮
                await human_like_delay(0.5, 1.0)  # 减少等待
                verify_button = challenge_frame.locator('#recaptcha-verify-button').first
                await human_like_delay(0.3, 0.6)  # 减少等待
                
                # 确保按钮在视口内
                try:
                    await verify_button.scroll_into_view_if_needed(timeout=5000)
                    await asyncio.sleep(0.3)  # 等待滚动完成
                except Exception as scroll_error:
                    if verbose: print(f"    滚动按钮失败（尝试继续）: {scroll_error}")
                
                # 尝试点击，如果失败则使用 force 点击
                try:
                    await verify_button.click(timeout=10000)
                except Exception as click_error:
                    if verbose: print(f"    常规点击失败，尝试强制点击: {click_error}")
                    try:
                        await verify_button.click(force=True, timeout=10000)
                    except Exception as force_error:
                        if verbose: print(f"    强制点击也失败: {force_error}")
                        raise
                
                # 等待验证结果
                await human_like_delay(2.0, 3.0)  # 减少等待
                
                # 检查是否通过
                try:
                    # 方法1: 检查 checkbox 是否被勾选
                    try:
                        checkbox_frame = page.frame_locator('iframe[title="reCAPTCHA"]').first
                        checked = await checkbox_frame.locator('span[aria-checked="true"]').first.is_visible(timeout=3000)
                        if checked:
                            if verbose: print("✓✓✓ reCAPTCHA 验证成功（checkbox已勾选）！")
                            return True
                    except:
                        pass
                    
                    # 方法2: 检查挑战框是否消失或隐藏
                    try:
                        challenge_visible = await page.locator('iframe[title*="challenge"]').first.is_visible(timeout=3000)
                        if not challenge_visible:
                            if verbose: print("✓✓✓ reCAPTCHA 验证成功（挑战框已隐藏）！")
                            return True
                    except:
                        if verbose: print("✓✓✓ reCAPTCHA 验证成功（找不到挑战框）！")
                        return True
                    
                    # 验证未通过，继续下一轮
                    if verbose: print("  验证未通过，重试...")
                    
                except Exception as check_error:
                    if verbose: print(f"  检查验证结果时出错: {check_error}")
            
            except Exception as e:
                if verbose: print(f"  本轮尝试失败: {e}")
                if attempt >= max_attempts:
                    print(f"✗ 达到最大尝试次数 ({max_attempts})，验证失败")
                    return False
                else:
                    if verbose: print("  准备下一轮尝试...")
    
    except Exception as e:
        print(f"✗ reCAPTCHA 解决失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False


async def renew_host2play_server():
    """续期 Host2Play 服务器"""
    
    print("=" * 70)
    print("  🔐 Host2Play 自动续期脚本 (YOLO 版本)")
    print(f"  🌐 续期 URL: {RENEW_URL[:50]}...")
    print("  🤖 模式: Playwright + Camoufox + YOLO")
    print("=" * 70)
    print()
    
    start_time = datetime.now()
    
    # 发送开始通知
    start_message = f"""🚀 *Host2Play 自动续期开始*

🕐 时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🤖 模式: Playwright + Camoufox + YOLO

⏳ 正在处理中..."""
    send_telegram_message(start_message)
    
    # 检测是否在 CI 环境
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if is_ci:
        logger.info("🤖 检测到 CI 环境，使用 headless 模式")
    else:
        logger.info("💻 本地环境，显示浏览器窗口")
    
    # 启动 Camoufox 浏览器
    print("\n启动 Camoufox 浏览器...")
    
    async with AsyncCamoufox(
        headless=is_ci,  # CI 环境使用 headless，本地显示窗口
        humanize=True,   # 启用人性化行为
        locale='en-US',
    ) as browser:
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        # 注入反检测脚本
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            if (!window.chrome) { window.chrome = {}; }
            if (!window.chrome.runtime) { window.chrome.runtime = {}; }
        """)
        
        page = await context.new_page()
        
        try:
            # [1/4] 访问续期页面
            print("\n[1/4] 🌐 访问续期页面...")
            await page.goto(RENEW_URL, wait_until='domcontentloaded', timeout=60000)
            await asyncio.sleep(3)
            
            print(f"✅ 当前 URL: {page.url}")
            
            # [2/4] 检测并处理 Cloudflare（如果需要）
            print("\n[2/4] 🔍 检测 Cloudflare 保护...")
            page_source = await page.content()
            
            if 'cloudflare' in page_source.lower() or 'turnstile' in page_source.lower():
                print("⚠️ 检测到 Cloudflare 保护，等待自动通过...")
                await asyncio.sleep(10)
            else:
                print("✅ 未检测到 Cloudflare 保护")
            
            # [3/4] 查找并点击 Renew 按钮
            print("\n[3/4] 🖱️ 查找并点击 'Renew' 按钮...")
            await asyncio.sleep(2)
            
            renew_button = None
            selectors = [
                "button:has-text('Renew server')",
                "button:has-text('Renew')",
                "a:has-text('Renew server')",
                "a:has-text('Renew')",
                "input[value='Renew server']",
                "input[value='Renew']",
                "button[type='submit']",
            ]
            
            for selector in selectors:
                try:
                    renew_button = page.locator(selector).first
                    if await renew_button.is_visible(timeout=5000):
                        print(f"✅ 找到 Renew 按钮: {selector}")
                        break
                except:
                    continue
            
            if renew_button is None:
                print("❌ 未找到 Renew 按钮")
                await page.screenshot(path='host2play_error_no_button.png')
                print("📸 已保存截图: host2play_error_no_button.png")
                
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: 未找到 Renew 按钮
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"""
                send_telegram_message(error_message, 'host2play_error_no_button.png')
                return
            
            await renew_button.click()
            print("✅ 已点击 Renew 按钮")
            await asyncio.sleep(3)
            
            # [4/4] 处理 reCAPTCHA（YOLO 方式）
            print("\n[4/4] 🔐 处理 reCAPTCHA（YOLO 图像识别）...")
            print("💡 提示：使用 YOLO 模型识别图像")
            print("⏰ 此过程可能需要 10-60 秒，请耐心等待...")
            
            recaptcha_success = await solve_recaptcha_yolo(page, verbose=VERBOSE, max_attempts=8)
            
            if not recaptcha_success:
                print("❌ reCAPTCHA 未通过")
                await page.screenshot(path='host2play_error_recaptcha.png')
                print("📸 已保存截图: host2play_error_recaptcha.png")
                
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: reCAPTCHA 未通过
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"""
                send_telegram_message(error_message, 'host2play_error_recaptcha.png')
                
                if not is_ci:
                    print("\n⚠ 请手动完成验证...")
                    await asyncio.sleep(60)
                return
            
            # 查找并点击弹窗内的确认按钮
            print("\n🖱️ 查找弹窗内的 'Renew' 按钮（排除 'Renew server'）...")
            await asyncio.sleep(1.5)
            
            # 专门查找弹窗内的 Renew 按钮，排除 Renew server
            modal_button_selectors = [
                "div.modal button:has-text('Renew'):not(:has-text('server'))",
                "div.dialog button:has-text('Renew'):not(:has-text('server'))",
                "div.popup button:has-text('Renew'):not(:has-text('server'))",
                "[role='dialog'] button:has-text('Renew'):not(:has-text('server'))",
                "div.swal button:has-text('Renew')",
                "div.swal button:has-text('Confirm')",
                "div.swal2-container button:has-text('Renew')",
                "div.swal2-container button:has-text('Confirm')",
                "div.modal button:has-text('Confirm')",
                "div.modal button[type='submit']",
            ]
            
            modal_button = None
            for selector in modal_button_selectors:
                try:
                    modal_button = page.locator(selector).first
                    if await modal_button.is_visible(timeout=5000):
                        print(f"✅ 找到弹窗内的 Renew 按钮: {selector}")
                        break
                except:
                    continue
            
            if modal_button is None:
                print("⚠️ 标准选择器未找到弹窗按钮，使用 JavaScript 查找...")
                
                # JavaScript 专门在弹窗内查找
                js_code = """
                // 查找弹窗容器
                var modalSelectors = ['.modal', '.dialog', '.popup', '[role="dialog"]', '.swal2-container', '.swal-modal'];
                var modal = null;
                
                for (var i = 0; i < modalSelectors.length; i++) {
                    var modals = document.querySelectorAll(modalSelectors[i]);
                    for (var j = 0; j < modals.length; j++) {
                        if (modals[j].offsetParent !== null) {  // 可见的弹窗
                            modal = modals[j];
                            break;
                        }
                    }
                    if (modal) break;
                }
                
                if (modal) {
                    // 在弹窗内查找按钮，排除 "Renew server"
                    var buttons = modal.querySelectorAll('button, a, input[type="submit"]');
                    for (var i = 0; i < buttons.length; i++) {
                        var text = (buttons[i].textContent || buttons[i].value || '').toLowerCase();
                        // 只匹配 "renew" 但不包含 "server"
                        if (text.includes('renew') && !text.includes('server')) {
                            buttons[i].click();
                            return 'Clicked modal Renew: ' + buttons[i].textContent;
                        }
                        if (text.includes('confirm') || text.includes('yes') || text.includes('ok')) {
                            buttons[i].click();
                            return 'Clicked modal confirm: ' + buttons[i].textContent;
                        }
                    }
                    return 'Modal found but no Renew button (buttons: ' + buttons.length + ')';
                } else {
                    return 'No modal found';
                }
                """
                
                try:
                    result = await page.evaluate(js_code)
                    print(f"  JavaScript 结果: {result}")
                    
                    if 'Clicked' in result:
                        print("✅ 使用 JavaScript 成功点击弹窗内的 Renew 按钮")
                        await asyncio.sleep(2)
                    else:
                        print("❌ 无法找到弹窗内的 Renew 按钮")
                        print("  请手动点击弹窗内的 Renew 按钮...")
                        await asyncio.sleep(30)
                except Exception as js_error:
                    print(f"❌ JavaScript 查找失败: {js_error}")
                    print("  请手动点击弹窗内的 Renew 按钮...")
                    await asyncio.sleep(30)
            else:
                # 找到按钮，点击它
                try:
                    await modal_button.click()
                    print("✅ 已点击弹窗内的 Renew 按钮")
                    await asyncio.sleep(2)
                except Exception as click_error:
                    # 如果普通点击失败，尝试使用 JavaScript 点击
                    print(f"⚠️ 普通点击失败，尝试 JavaScript 点击...")
                    try:
                        await page.evaluate("arguments => arguments[0].click()", await modal_button.element_handle())
                        print("✅ 使用 JavaScript 成功点击")
                        await asyncio.sleep(2)
                    except:
                        print(f"❌ 点击失败: {click_error}")
                        await asyncio.sleep(10)
            
            # 等待页面加载完成
            print("\n等待页面加载完成...")
            await human_like_delay(3, 5)
            
            # 检查结果
            try:
                page_text = await page.locator('body').text_content()
                text_l = page_text.lower()
                
                if ('success' in text_l) or ('renewed' in text_l) or ('续期' in page_text and '成功' in page_text):
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    # 只有成功时才保存截图
                    await page.screenshot(path='host2play_renew_success.png')
                    
                    print("\n" + "="*70)
                    print("  ✅✅✅ 续期成功！")
                    print(f"  🕐 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  🕐 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  ⏱️  耗时: {duration:.1f} 秒")
                    print(f"  📸 截图保存: host2play_renew_success.png")
                    print("="*70)
                    
                    success_message = f"""✅ *Host2Play 续期成功*

🕐 开始: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🕐 结束: `{end_time.strftime('%Y-%m-%d %H:%M:%S')}`
⏱️ 耗时: `{duration:.1f}秒`
🤖 方式: YOLO 图像识别"""
                    send_telegram_message(success_message, 'host2play_renew_success.png')
                else:
                    print("❌ 未检测到成功文案")
                    print("⚠️ 请手动检查续期状态")
                    await page.screenshot(path='host2play_renew_unknown.png')
                    print("📸 已保存截图: host2play_renew_unknown.png")
                    
                    warning_message = f"""⚠️ *Host2Play 续期状态未知*

❗ 未检测到成功文案
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
💡 请手动检查续期状态"""
                    send_telegram_message(warning_message, 'host2play_renew_unknown.png')
            except:
                print("⚠️ 无法检查续期结果，请手动确认")
            
            if not is_ci:
                print("\n浏览器将保持打开 10 秒...")
                await asyncio.sleep(10)
            
        except Exception as e:
            print(f"❌ 执行过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                await page.screenshot(path='host2play_error.png')
                print("📸 已保存错误截图: host2play_error.png")
                
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: `{str(e)}`
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"""
                send_telegram_message(error_message, 'host2play_error.png')
            except:
                pass
        finally:
            # 清理临时图片（GitHub Actions 需要保留 0.png 用于调试）
            keep_captcha_images = os.environ.get('KEEP_CAPTCHA_IMAGES', 'false').strip().lower() in ('1', 'true', 'yes')
            if not keep_captcha_images:
                try:
                    os.remove('0.png')
                except:
                    pass
            # 清理单格截图
            for i in range(1, 17):
                try:
                    os.remove(f"{i}.png")
                except:
                    pass


if __name__ == "__main__":
    try:
        asyncio.run(renew_host2play_server())
        print("\n✓ 脚本执行完成")
    except Exception as e:
        print(f"\n✗ 脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
