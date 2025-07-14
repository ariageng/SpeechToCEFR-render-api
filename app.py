from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from speaking_cefr_predictor import get_speaking_top2_CEFR

app = Flask(__name__)
CORS(app)

@app.route('/predict-cefr', methods=['POST'])
def predict_cefr():
    try:
        data = request.get_json()
        
        accuracy_score = data.get('accuracy_score', 0)
        completeness_score = data.get('completeness_score', 0)
        confidence_score = data.get('confidence_score', 0)
        fluency_score = data.get('fluency_score', 0)
        new_content = data.get('new_content', 0)
        new_delivery = data.get('new_delivery', 0)
        
        cefr_levels = get_speaking_top2_CEFR(
            accuracy_score,
            completeness_score,
            confidence_score,
            fluency_score,
            new_content,
            new_delivery
        )
        
        return jsonify({
            'top2_cefr_levels': cefr_levels,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'top2_cefr_levels': ['Unable to determine'],
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)