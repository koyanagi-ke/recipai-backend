import functions_framework
import google.generativeai as genai
import json
import os
import re
from google.cloud import translate_v2 as translate

# APIキーの設定（環境変数から取得）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
translate_client = translate.Client()


def translate_to_japanese(text):
    """Translation AI を使って英語を日本語に翻訳"""
    result = translate_client.translate(text, target_language="ja")
    return result["translatedText"]


@functions_framework.http
def detect_ingredients(request):
    """Detect ingredients using Gemini Vision API"""
    request_json = request.get_json()

    # 入力チェック
    if "image" not in request_json:
        return {"error": "Image is required"}, 400

    encoded_image = request_json["image"]

    try:
        # Gemini Vision API で画像解析
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [
                {
                    "parts": [
                        {
                            "text": "Identify the food ingredients present in this image. Return the list in JSON format."
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded_image,
                            }
                        },
                    ]
                }
            ]
        )

        # レスポンスの raw text
        raw_text = response.text

        # 正規表現でJSON部分を抽出
        match = re.search(r"```json\s*({.*?})\s*```", raw_text, re.DOTALL)

        if match:
            json_data = match.group(1)  # マッチしたJSON部分を取得
            ingredients = json.loads(json_data)  # JSONに変換
            ingredients = {
                "ingredients": [
                    translate_to_japanese(ingredient)
                    for ingredient in ingredients.get("ingredients", [])
                ]
            }
        else:
            # もしJSONフォーマットでない場合、行ごとに解析
            ingredients_list = [
                line.strip().strip('"') for line in raw_text.split("\n") if line.strip()
            ]
            ingredients = {"ingredients": ingredients_list}

        return json.dumps(ingredients, indent=2, ensure_ascii=False), 200

    except Exception as e:
        return json.dumps({"error": str(e)}), 500
