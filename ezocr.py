import cv2
import easyocr
import matplotlib.pyplot as plt

# Step 1: 讀圖與前處理
image = cv2.imread(r'C:\Users\hti07022\Desktop\compare_labels\design_images\test1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 2: 二值化來提取文字區域
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: 尋找輪廓區塊（可能含文字的區域）
contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

reader = easyocr.Reader(['en'])  

results = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 20:  # 過濾過小區域
        roi = image[y:y+h, x:x+w]
        ocr_result = reader.readtext(roi)
        for (bbox, text, conf) in ocr_result:
            results.append(((x, y), text, conf))
            print(f"{text} ({conf:.2f})")

# Step 4: 可視化
for ((x, y), text, _) in results:
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.rectangle(image, (x, y), (x + 200, y + 20), (0, 255, 0), 1)

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

"""

"""