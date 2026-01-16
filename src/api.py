"""
Flask API for Document Enhancement
"""

from flask import Flask, request, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
import logging
from dotenv import load_dotenv
import shutil

from enhancer import DocumentEnhancer

# Load environment
load_dotenv()

# Setup logging
log_folder = os.getenv('LOG_FOLDER', 'logs')
os.makedirs(log_folder, exist_ok=True)

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_folder, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__, static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 52428800))

# Initialize Enhancer
enhancer = DocumentEnhancer(target_dpi=int(os.getenv('DEFAULT_DPI', 300)))

# Folders - usa path assoluto dalla root del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'pdf,jpg,jpeg,png,tiff,bmp').split(','))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve web interface"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'document-enhancer',
        'version': '1.0.0'
    })


@app.route('/enhance', methods=['POST'])
def enhance_document():
    """
    Enhance single document
    
    Form data:
        file: Document file (PDF, JPG, PNG, TIFF)
        aggressive: true/false (default: true)
        auto_crop: true/false (default: true)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Parameters
    aggressive = request.form.get('aggressive', 'true').lower() == 'true'
    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Save input
    input_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    file.save(input_path)
    
    logger.info(f"Processing file: {filename} (job_id: {job_id})")
    
    try:
        # Process based on type
        if file_ext == '.pdf':
            output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_enhanced.pdf")
            enhancer.process_pdf(input_path, output_path, aggressive, auto_crop)
        else:
            output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_enhanced.jpg")
            enhancer.process_single_image(input_path, output_path, aggressive, auto_crop)
        
        # Return file
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"enhanced_{filename}"
        )
    
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup input
        if os.path.exists(input_path):
            os.remove(input_path)


@app.route('/enhance/batch', methods=['POST'])
def enhance_batch():
    """
    Batch enhancement
    
    Form data:
        files: Multiple document files
        aggressive: true/false
        auto_crop: true/false
    
    Returns:
        ZIP file with all enhanced documents
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400
    
    aggressive = request.form.get('aggressive', 'true').lower() == 'true'
    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
    
    batch_id = str(uuid.uuid4())
    batch_folder = os.path.join(OUTPUT_FOLDER, f"batch_{batch_id}")
    os.makedirs(batch_folder, exist_ok=True)
    
    logger.info(f"Batch processing {len(files)} files (batch_id: {batch_id})")
    
    results = []
    
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue
        
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        input_path = os.path.join(UPLOAD_FOLDER, f"{batch_id}_{filename}")
        file.save(input_path)
        
        try:
            if file_ext == '.pdf':
                output_path = os.path.join(batch_folder, f"{os.path.splitext(filename)[0]}_enhanced.pdf")
                enhancer.process_pdf(input_path, output_path, aggressive, auto_crop)
            else:
                output_path = os.path.join(batch_folder, f"{os.path.splitext(filename)[0]}_enhanced.jpg")
                enhancer.process_single_image(input_path, output_path, aggressive, auto_crop)
            
            results.append(output_path)
            logger.info(f"Successfully processed: {filename}")
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
        
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
    
    if not results:
        shutil.rmtree(batch_folder)
        return jsonify({'error': 'No files were successfully processed'}), 500
    
    # Create ZIP
    zip_path = os.path.join(OUTPUT_FOLDER, f"{batch_id}_enhanced")
    shutil.make_archive(zip_path, 'zip', batch_folder)
    
    # Cleanup
    shutil.rmtree(batch_folder)
    
    return send_file(
        f"{zip_path}.zip",
        as_attachment=True,
        download_name="enhanced_documents.zip"
    )


def main():
    """Entry point"""
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Document Enhancer on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
