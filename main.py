import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont

# 設定
INPUT_IMAGE_PATH = "input/IMG_0479.jpg"
OUTPUT_IMAGE_PATH = "output/palette_overlay_result.png"
NUM_COLORS = 5
PALETTE_WIDTH_RATIO = 0.3
PALETTE_HEIGHT_RATIO = 0.15  # 調整高度為原本的 0.15（原為 0.12）
PALETTE_X_PADDING_RATIO = 0.1
FONT_SIZE = 150

# RGB → CMYK
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

# 萃取主色
def extract_colors(image, n_colors=5):
    img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# 主程式
def overlay_palette(image_path, output_path):
    original = cv2.imread(image_path)
    if original is None:
        print("❌ 無法讀取圖片")
        return
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    colors = extract_colors(img_rgb, NUM_COLORS)

    pil_img = Image.fromarray(img_rgb)
    # 疊加一層透明黑色使背景變暗
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 120))
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(pil_img)
    width, height = pil_img.size

    palette_width = int(width * PALETTE_WIDTH_RATIO)
    palette_height = int(height * PALETTE_HEIGHT_RATIO)
    x_padding = int(width * PALETTE_X_PADDING_RATIO)
    padding_y = int(height * 0.02)
    total_palette_height = NUM_COLORS * palette_height + (NUM_COLORS - 1) * padding_y
    start_y = (height - total_palette_height) // 2

    # 載入字體
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
        FONT_SIZE = 20  # Approximate size for default bitmap font

    for i, color in enumerate(colors):
        r, g, b = color
        hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)
        cmyk = rgb_to_cmyk(r, g, b)
        rgb_text = f"RGB: {r},{g},{b}"
        cmyk_text = f"CMYK: {cmyk}"
        combined_text = f"{hex_code}   {rgb_text}   {cmyk_text}"

        y_top = start_y + i * (palette_height + padding_y)
        y_bottom = y_top + palette_height

        x_start = (width - palette_width) // 2
        # 畫色塊
        draw.rounded_rectangle(
            [x_start, y_top, x_start + palette_width, y_top + palette_height],
            radius=15,
            fill=(r, g, b)
        )

        # 判斷文字顏色（亮度法則）
        brightness = r * 0.299 + g * 0.587 + b * 0.114
        text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

        # 計算文字位置（置中對齊）
        try:
            # Pillow >= 10.0
            bbox = draw.textbbox((0, 0), combined_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Pillow < 10.0
            text_width, text_height = draw.textsize(combined_text, font=font)

        text_x = x_start + (palette_width - text_width) // 2
        text_y = y_top + (palette_height - text_height) // 2
        draw.text((text_x, text_y), combined_text, fill=text_color, font=font)

    pil_img.save(output_path)
    print(f"✅ 完成輸出：{output_path}")

# 執行
if __name__ == "__main__":
    overlay_palette(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)