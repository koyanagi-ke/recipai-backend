import functions_framework
import google.generativeai as genai
import json
import os
import re
from google.cloud import translate_v2 as translate
import logging

# APIキーの設定（環境変数から取得）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
translate_client = translate.Client()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def translate_to_japanese(text):
    """Translation AI を使って英語を日本語に翻訳"""
    result = translate_client.translate(
        text, target_language="ja", source_language="en"
    )
    return result["translatedText"]


def parse_ingredients(response_text):
    # 正規表現でリスト部分を抽出
    match = re.search(r"\[(.*?)\]", response_text, re.DOTALL)
    if not match:
        return {"ingredients": []}  # マッチしなければ空リストを返す

    # リストの中身を取得し、各アイテムをクリーンアップ
    raw_ingredients = match.group(1).split("\n")
    cleaned_ingredients = [
        item.strip().strip('"') for item in raw_ingredients if item.strip()
    ]

    return {"ingredients": cleaned_ingredients}


@functions_framework.http
def detect_ingredients(request):
    """Detect ingredients using Gemini Vision API"""
    request_json = request.get_json()

    # 入力チェック
    if "image" not in request_json:
        return {"error": "Image is required"}, 400
    if "mime_type" not in request_json:
        return {"error": "Mime_type is required"}, 400

    encoded_image = request_json["image"]
    mime_type = request_json["mime_type"]
    pronpt = """
        Identify the food ingredients present in this image. Return the result strictly as a JSON object in the following format:

        {
            "ingredients": ["ingredient1", "ingredient2", "ingredient3"]
        }

        Do not include any extra text or explanations. Ensure the output is valid JSON.
    """

    try:
        # Gemini Vision API で画像解析
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [
                {
                    "parts": [
                        {"text": pronpt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_image,
                            }
                        },
                    ]
                }
            ]
        )

        # レスポンスの raw text
        raw_text = response.text
        logger.info(raw_text)

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
            ingredients = parse_ingredients(raw_text)

        return json.dumps(ingredients, indent=2, ensure_ascii=False), 200

    except Exception as e:
        return json.dumps({"error": str(e)}), 500
