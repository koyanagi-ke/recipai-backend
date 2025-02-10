import functions_framework
import google.generativeai as genai
import json
import os
import re
import base64
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import translate_v2 as translate

# APIキーの設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Google Cloud プロジェクト情報
PROJECT_ID = "resshipy"
LOCATION = "asia-northeast1"  # 東京リージョン

# Vertex AI の初期化
vertexai.init(project=PROJECT_ID, location=LOCATION)

translate_client = translate.Client()


def translate_to_japanese(text):
    """Translation AI を使って英語を日本語に翻訳"""
    result = translate_client.translate(
        text, target_language="ja", source_language="en"
    )
    return result["translatedText"]


@functions_framework.http
def generate_recipe(request):
    """Generate a recipe using Gemini and create an image with Vertex AI"""
    request_json = request.get_json()

    # 入力チェック
    if "ingredients" not in request_json or "feeling" not in request_json:
        return {"error": "Missing required fields: 'ingredients' and 'feeling'"}, 400

    ingredients = request_json["ingredients"]
    feeling = request_json["feeling"]

    try:
        # ✅ **レシピ生成**
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            [
                f"Create a recipe using only the following ingredients: {', '.join(ingredients)}.",
                "Do not add any ingredients that are not listed, except for seasonings.",
                f"The dish should feel {feeling}.",
                "Return the result as a JSON object with 'title' and 'steps' (list of step-by-step instructions).",
                "The JSON format should match this example:",
                """{
                "title": "Simple Vegetable Stir-Fry",
                "steps": [
                    "Heat a pan and add oil.",
                    "Add chopped carrots and onions, and stir-fry for 5 minutes.",
                    "Season with salt and pepper, then serve."
                ]
            }""",
            ]
        )

        # **レスポンスの解析（JSON抽出）**
        recipe_text = response.text
        match = re.search(r"```json\s*({.*?})\s*```", recipe_text, re.DOTALL)

        if match:
            recipe = json.loads(match.group(1))  # JSON部分を抽出 & 変換
        else:
            recipe = {
                "title": "Unknown Recipe",
                "steps": recipe_text.strip().split("\n"),
            }

        # ✅ **画像生成 (Vertex AI ImageGenerationModel)**
        image_base64 = generate_recipe_image(recipe["title"])

        # **最終レスポンス**
        return (
            json.dumps(
                {
                    "title": translate_to_japanese(recipe["title"]),
                    "steps": [
                        translate_to_japanese(recip) for recip in recipe["steps"]
                    ],
                    "image_base64": image_base64,  # Base64 形式の画像データ
                },
                indent=2,
                ensure_ascii=False,
            ),
            200,
        )

    except Exception as e:
        return json.dumps({"error": str(e)}), 500


def generate_recipe_image(title):
    """Vertex AI Image Generation (Imagen) で料理画像を生成し、Base64 で返す"""
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    images = model.generate_images(
        prompt=f"A high-quality food photograph of {title}. Delicious and well-plated.",
        number_of_images=1,
        language="en",
        aspect_ratio="1:1",
    )

    if images:
        # 画像をBase64エンコード
        img_bytes = images[0]._image_bytes
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

    return None
