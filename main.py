import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from math import sqrt

## settings index
INPUT_IMAGE_PATH = input("請輸入圖片路徑：") + ".jpg"
OUTPUT_IMAGE_PATH = "output/s_overlay_result.png"
NUM_COLORS = 5
PALETTE_WIDTH_RATIO = 0.42
PALETTE_HEIGHT_RATIO = 0.12 
PALETTE_X_PADDING_RATIO = 0.1
color_df = pd.read_csv("colors.csv", header=None, names=["id", "name", "hex", "R", "G", "B"])

# RGB 2 CMYK
def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100
    c = 1 - r / 255
    m = 1 - g / 255
    y = 1 - b / 255
    k = min(c, m, y)
    c = (c - k) / (1 - k) * 100
    m = (m - k) / (1 - k) * 100
    y = (y - k) / (1 - k) * 100
    k *= 100
    return round(c), round(m), round(y), round(k)

# hl4sua 82j6操你媽kmeans有夠好用的啦
def extract_colors(image, n_colors=5):
    img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# 主程式
def overlay_palette(image_path, output_path):
    def closest_color_name(r, g, b):
        try:
            rgb = np.array([r, g, b])
            palette = color_df[['R', 'G', 'B']].to_numpy()
            # 計算每一列與 (r,g,b) 的歐式距離
            distances = np.linalg.norm(palette - rgb, axis=1)
            closest_index = np.argmin(distances)
            name = color_df.iloc[closest_index]["name"]
            print(f"比對中: R={r}, G={g}, B={b} → 最近色彩名稱為: {name}")
            return name
        except Exception as e:
            print(f"比對錯誤: R={r}, G={g}, B={b}, error: {e}")
            return "Unknown"
    original = cv2.imread(image_path)
    if original is None:
        print("XXX 無法讀取圖片")
        return
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    colors = extract_colors(img_rgb, NUM_COLORS)

    pil_img = Image.fromarray(img_rgb)
    width, height = pil_img.size
    dynamic_font_size = max(1, min(int(width * 0.020), 160))  ###### 文字大小

    # 疊加一層透明黑色使背景變暗
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 85))
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(pil_img)

    palette_width = int(width * PALETTE_WIDTH_RATIO)
    palette_height = int(height * PALETTE_HEIGHT_RATIO)
    x_padding = int(width * PALETTE_X_PADDING_RATIO)
    padding_y = int(height * 0.02)
    total_palette_height = NUM_COLORS * palette_height + (NUM_COLORS - 1) * padding_y
    start_y = (height - total_palette_height) // 2

    # 載入字體
    try:
        # 優先使用系統中的襯線字體，若無則使用預設
        font = ImageFont.truetype("Times New Roman.ttf", dynamic_font_size)
    except:
        try:
            font = ImageFont.truetype("Times.ttf", dynamic_font_size)
        except:
            font = ImageFont.load_default()
            dynamic_font_size = 17

    try:
        font_title = ImageFont.truetype("Times New Roman.ttf", dynamic_font_size + 16)
    except:
        try:
            font_title = ImageFont.truetype("Times.ttf", dynamic_font_size + 16)
        except:
            font_title = ImageFont.load_default()

    for i, color in enumerate(colors):
        r, g, b = color
        name_text = closest_color_name(r, g, b)
        hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)
        cmyk = rgb_to_cmyk(r, g, b)
        hex_text = f"{hex_code}"
        rgb_text = f"RGB: {r},{g},{b}"
        cmyk_text = f"CMYK: {cmyk[0]}%, {cmyk[1]}%, {cmyk[2]}%, {cmyk[3]}%"
        combined_lines = [(name_text, font_title), (hex_text, font), (rgb_text, font), (cmyk_text, font)]

        y_top = start_y + i * (palette_height + padding_y)
        y_bottom = y_top + palette_height

        x_start = (width - palette_width) // 2
        # judge text's color based on brighness (YIQ)
        brightness = r * 0.299 + g * 0.587 + b * 0.114
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

        # 計算每行文字高度與最大寬度
        line_heights = []
        max_line_width = 0
        for line, line_font in combined_lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=line_font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
            except AttributeError:
                line_width, line_height = draw.textsize(line, font=line_font)
            line_heights.append(line_height)
            if line_width > max_line_width:
                max_line_width = line_width

        # 保證色塊高度至少比文字總高多出 40px（上下各 20）
        min_palette_height = sum(line_heights) + (len(combined_lines) - 1) * 20 + 40
        if palette_height < min_palette_height:
            palette_height = min_palette_height
            y_bottom = y_top + palette_height

        total_text_height = sum(line_heights) + (len(combined_lines) - 1) * 20  # 再加大行距
        text_start_y = y_top + (palette_height - total_text_height) // 2

        # 畫色塊
        draw.rounded_rectangle(
            [x_start, y_top, x_start + palette_width, y_bottom],
            radius=15,
            fill=(r, g, b)
        )
        # 添加輕微陰影邊框（使用半透明深灰色）
        draw.rounded_rectangle(
            [x_start, y_top, x_start + palette_width, y_bottom],
            radius=15,
            outline=(0, 0, 0, 20),  # 深灰色半透明
            width=0
        )

        for idx, (line, line_font) in enumerate(combined_lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=line_font)
                line_width = bbox[2] - bbox[0]
            except AttributeError:
                line_width, _ = draw.textsize(line, font=line_font)

            text_y = text_start_y + sum(line_heights[:idx]) + idx * 20  # 同步加大行距

            if idx == 0:
                # 左對齊
                text_x = x_start + int(palette_width * 0.05)
            else:
                # 右對齊
                text_x = x_start + palette_width - line_width - int(palette_width * 0.05)

            draw.text((text_x, text_y), line, fill=text_color, font=line_font)

    pil_img.save(output_path)
    print(f"======完成輸出：{output_path}")

if __name__ == "__main__":
    overlay_palette(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)