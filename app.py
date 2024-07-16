from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import os
import requests
import time
import mindsdb_sdk
import logging

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

def generate_image(prompt, scribble_data_url):
    url = "https://api.replicate.com/v1/predictions"
    payload = {
        "version": "72c05df2daf615fb5cc07c28b662a2a58feb6a4d0a652e67e5a9959d914a9ed2",
        "input": {
            "cfg": 3.5,
            "image": scribble_data_url,
            "prompt": prompt,
            "aspect_ratio": "3:2",
            "output_format": "webp",
            "output_quality": 90,
            "negative_prompt": "",
            "prompt_strength": 0.85
        }
    }
    headers = {
        "Authorization": f"Token {os.getenv('REPLICATE_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 201:
        prediction_id = response.json()['id']
        image_url = check_replicate_prediction(prediction_id)
        return image_url
    else:
        logging.error(f"Failed to generate image: {response.status_code} - {response.text}")
        return None

def check_replicate_prediction(prediction_id):
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    headers = {
        "Authorization": f"Token {os.getenv('REPLICATE_API_KEY')}"
    }
    
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            status = response.json()['status']
            if status == 'succeeded':
                return response.json()['output'][0]
            elif status == 'failed':
                logging.error(f"Prediction failed: {response.json()}")
                return None
        else:
            logging.error(f"Failed to check prediction status: {response.status_code} - {response.text}")
            return None
        time.sleep(5)

def connect_to_mindsdb():
    try:
        server = mindsdb_sdk.connect(url=os.getenv('MINDSDB_URL', 'http://localhost:47334'))
        return server
    except Exception as e:
        logging.error(f"Failed to connect to MindsDB: {e}")
        return None

server = connect_to_mindsdb()

def generate_story(prompt):
    if server:
        try:
            model = server.get_model("story_generator")
            prediction = model.predict({'prompt': prompt})
            generated_content = prediction[0]['story_content']
            return generated_content
        except Exception as e:
            logging.error(f"Failed to generate story: {e}")
            return None
    else:
        logging.error("MindsDB server not available.")
        return None

# Function to generate blog using MindsDB model
def generate_blog(prompt):
    if server:
        try:
            model = server.get_model("blog_generator")
            prediction = model.predict({'prompt': prompt})
            generated_content = prediction[0]['blog_content']
            return generated_content
        except Exception as e:
            logging.error(f"Failed to generate blog: {e}")
            return None
    else:
        logging.error("MindsDB server not available.")
        return None

# Function to generate film script using MindsDB model
def generate_film_script(prompt):
    if server:
        try:
            model = server.get_model("film_script_generator")
            prediction = model.predict({'prompt': prompt})
            generated_content = prediction[0]['script_content']
            return generated_content
        except Exception as e:
            logging.error(f"Failed to generate film script: {e}")
            return None
    else:
        logging.error("MindsDB server not available.")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image_from_scribble():
    scribble_data_url = request.form['scribbleData']
    prompt = request.form['prompt']
    image_url = generate_image(prompt, scribble_data_url)

    if image_url:
        return jsonify({'image_url': image_url})
    else:
        return jsonify({'error': 'Failed to generate image'}), 500

@app.route('/generate-story')
def generate_story_page():
    return render_template('generate_story.html')

# Route to handle form submission for generating story content
@app.route('/generate-story-content', methods=['POST'])
def generate_story_content():
    prompt = request.form['prompt']
    generated_content = generate_story(prompt)
    
    if generated_content:
        return jsonify({'story_content': generated_content})
    else:
        return jsonify({'error': 'Failed to generate story content.'}), 500

# Route to render the generate blog page
@app.route('/generate-blog')
def generate_blog_page():
    return render_template('generate_blog.html')

# Route to handle form submission for generating blog content
@app.route('/generate-blog-content', methods=['POST'])
def generate_blog_content():
    prompt = request.form['prompt']
    generated_content = generate_blog(prompt)
    
    if generated_content:
        return jsonify({'blog_content': generated_content})
    else:
        return jsonify({'error': 'Failed to generate blog content.'}), 500

# Route to render the generate film script page
@app.route('/generate-script')
def generate_script_page():
    return render_template('generate_script.html')

# Route to handle form submission for generating film script content
@app.route('/generate-script-content', methods=['POST'])
def generate_script_content():
    prompt = request.form['prompt']
    generated_content = generate_film_script(prompt)
    
    if generated_content:
        return jsonify({'script_content': generated_content})
    else:
        return jsonify({'error': 'Failed to generate film script content.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
