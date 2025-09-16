from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
import re
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Global variable to store current dataset
current_data = None
current_filename = None
cleaning_operations = []  # Track all cleaning operations performed

# Caching variables for data preview
preview_cache = None
preview_cache_hash = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_data_hash(data):
    """Generate a hash for the current data to check for changes"""
    if data is None:
        return None
    # Use shape, columns, and a sample of the data to create a hash
    hash_components = [
        str(data.shape),
        str(data.columns.tolist()),
        str(data.head(2).to_dict()) if len(data) > 0 else "empty"
    ]
    return hash(''.join(hash_components))

def invalidate_preview_cache():
    """Clear the preview cache when data changes"""
    global preview_cache, preview_cache_hash
    preview_cache = None
    preview_cache_hash = None

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string with optimized settings"""
    img_buffer = io.BytesIO()
    # Reduced DPI for faster generation, optimized format
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, 
                facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def sample_data_for_plotting(data, max_points=10000):
    """Sample data for faster plotting while preserving patterns"""
    if len(data) <= max_points:
        return data
    
    # For large datasets, take a random sample to speed up plotting
    return data.sample(n=max_points, random_state=42)

def prepare_data_for_plotting(df):
    """Prepare data for plotting by converting date objects to datetime for matplotlib"""
    plot_df = df.copy()
    
    for column in plot_df.columns:
        if plot_df[column].dtype == 'object':
            # Check if it contains date objects
            sample = plot_df[column].dropna().head(5)
            if len(sample) > 0:
                first_val = sample.iloc[0]
                if hasattr(first_val, 'year') and hasattr(first_val, 'month') and hasattr(first_val, 'day'):
                    # Convert date objects to datetime for matplotlib
                    plot_df[column] = pd.to_datetime(plot_df[column], errors='coerce')
    
    return plot_df

def prepare_data_for_plotting(df):
    """Prepare data for plotting by converting date objects back to datetime for proper plotting"""
    plot_df = df.copy()
    
    for column in plot_df.columns:
        col_data = plot_df[column]
        
        # Check if column contains date objects and convert them to datetime for plotting
        if col_data.dtype == 'object':
            sample = col_data.dropna().head(5)
            if len(sample) > 0:
                first_val = sample.iloc[0]
                if hasattr(first_val, 'strftime'):
                    # Convert date objects back to datetime for plotting
                    plot_df[column] = pd.to_datetime(col_data, errors='coerce')
    
    return plot_df

def format_date_columns_for_display(df):
    """Format date columns for display in DD/MM/YYYY format"""
    display_df = df.copy()
    
    for column in display_df.columns:
        col_data = display_df[column]
        
        # Check if column contains date objects
        if col_data.dtype == 'object':
            sample = col_data.dropna().head(5)
            if len(sample) > 0:
                first_val = sample.iloc[0]
                if hasattr(first_val, 'strftime'):
                    # Format date objects as DD/MM/YYYY
                    display_df[column] = col_data.apply(
                        lambda x: x.strftime('%d/%m/%Y') if hasattr(x, 'strftime') else x
                    )
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            # Format datetime64 columns as DD/MM/YYYY (remove time)
            display_df[column] = col_data.dt.strftime('%d/%m/%Y')
    
    return display_df

def detect_datetime_columns(df):
    """
    Enhanced comprehensive datetime detection for DataFrame columns
    """
    import re
    
    # Enhanced datetime patterns to match more formats
    datetime_patterns = [
        # ISO formats
        r'^\d{4}-\d{1,2}-\d{1,2}$',  # YYYY-MM-DD
        r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$',  # YYYY-MM-DD HH:MM:SS
        r'^\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{1,2}:\d{1,2}$',  # YYYY-MM-DDTHH:MM:SS
        r'^\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{1,2}:\d{1,2}Z$',  # ISO with Z
        r'^\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{1,2}:\d{1,2}\.\d+$',  # ISO with microseconds
        
        # US formats
        r'^\d{1,2}/\d{1,2}/\d{4}$',  # MM/DD/YYYY
        r'^\d{1,2}/\d{1,2}/\d{2}$',  # MM/DD/YY
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}$',  # MM/DD/YYYY HH:MM:SS
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}$',  # MM/DD/YYYY HH:MM
        
        # European formats
        r'^\d{1,2}\.\d{1,2}\.\d{4}$',  # DD.MM.YYYY
        r'^\d{1,2}-\d{1,2}-\d{4}$',  # DD-MM-YYYY
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}$',  # DD.MM.YYYY HH:MM:SS
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}$',  # DD-MM-YYYY HH:MM:SS
        
        # Other formats
        r'^\d{8}$',  # YYYYMMDD
        r'^\d{4}/\d{1,2}/\d{1,2}$',  # YYYY/MM/DD
        r'^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2}$',  # YYYY/MM/DD HH:MM:SS
        
        # Month names (English)
        r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$',  # Month DD, YYYY
        r'^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$',  # DD Month YYYY
        r'^[A-Za-z]{3,9}\s+\d{4}$',  # Month YYYY
        
        # Timestamps (Unix)
        r'^\d{10}$',  # Unix timestamp (10 digits)
        r'^\d{13}$',  # Unix timestamp milliseconds (13 digits)
        r'^\d{10}\.\d+$',  # Unix timestamp with decimal
        
        # Compact formats
        r'^\d{6}$',  # YYMMDD or DDMMYY
        
        # Excel serial numbers (potential dates)
        r'^[1-9]\d{4,5}$',  # 5-6 digit numbers (Excel date serials)
        
        # Time only formats (for potential time series)
        r'^\d{1,2}:\d{1,2}:\d{1,2}$',  # HH:MM:SS
        r'^\d{1,2}:\d{1,2}$',  # HH:MM
        
        # 12-hour format with AM/PM
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}\s+[AaPp][Mm]$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{1,2}:\d{1,2}\s+[AaPp][Mm]$',
        r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{1,2}\s+[AaPp][Mm]$',
        r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{1,2}\s+[AaPp][Mm]$',
    ]
    
    # Expanded datetime format strings
    datetime_formats = [
        # Basic date formats
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', 
        '%m-%d-%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d', '%m.%d.%Y',
        
        # Short year formats
        '%y-%m-%d', '%m/%d/%y', '%d/%m/%y', '%y/%m/%d',
        '%m-%d-%y', '%d-%m-%y', '%d.%m.%y', '%y.%m.%d', '%m.%d.%y',
        
        # With full time
        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S', '%m-%d-%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S',
        '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S', '%m.%d.%Y %H:%M:%S',
        
        # With hours and minutes only
        '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
        '%Y/%m/%d %H:%M', '%m-%d-%Y %H:%M', '%d-%m-%Y %H:%M',
        '%d.%m.%Y %H:%M', '%Y.%m.%d %H:%M', '%m.%d.%Y %H:%M',
        
        # Month names (full and abbreviated)
        '%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
        '%B %d %Y', '%b %d %Y', '%B %Y', '%b %Y',
        '%B %d, %Y %H:%M:%S', '%b %d, %Y %H:%M:%S',
        '%d %B %Y %H:%M:%S', '%d %b %Y %H:%M:%S',
        
        # Compact formats
        '%Y%m%d', '%d%m%Y', '%m%d%Y', '%y%m%d', '%d%m%y', '%m%d%y',
        '%Y%m%d%H%M%S', '%y%m%d%H%M%S',
        
        # ISO formats
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%fZ',
        
        # 12-hour formats with AM/PM
        '%m/%d/%Y %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p',
        '%Y-%m-%d %I:%M:%S %p', '%m-%d-%Y %I:%M:%S %p',
        '%d-%m-%Y %I:%M:%S %p', '%Y/%m/%d %I:%M:%S %p',
        '%m/%d/%Y %I:%M %p', '%d/%m/%Y %I:%M %p',
        '%Y-%m-%d %I:%M %p', '%m-%d-%Y %I:%M %p',
        '%d-%m-%Y %I:%M %p', '%Y/%m/%d %I:%M %p',
        
        # Time only
        '%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p',
    ]
    
    converted_columns = []
    
    for column in df.columns:
        # Skip if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            logger.info(f"Column '{column}' is already datetime")
            continue
            
        # Only check object/string columns and numeric columns (for timestamps)
        if df[column].dtype not in ['object', 'int64', 'float64']:
            continue
            
        # Get a sample of non-null values
        sample = df[column].dropna()
        if len(sample) == 0:
            continue
            
        # Convert to string for pattern matching
        sample_str = sample.astype(str).head(100)
        
        # Special handling for potential Unix timestamps (numeric columns)
        if df[column].dtype in ['int64', 'float64']:
            try:
                # Check if values are in Unix timestamp range
                sample_numeric = sample.head(50)
                timestamp_candidates = 0
                
                for val in sample_numeric:
                    # Unix timestamp range check (1970-2050)
                    if 946684800 <= val <= 2524608000:  # seconds range
                        timestamp_candidates += 1
                    elif 946684800000 <= val <= 2524608000000:  # milliseconds range
                        timestamp_candidates += 1
                
                if timestamp_candidates / len(sample_numeric) >= 0.75:
                    # Try to convert as Unix timestamp
                    test_val = float(sample_numeric.iloc[0])
                    if test_val > 1e10:  # Likely milliseconds
                        converted = pd.to_datetime(df[column], unit='ms', errors='coerce')
                    else:  # Likely seconds
                        converted = pd.to_datetime(df[column], unit='s', errors='coerce')
                    
                    success_rate = converted.notna().sum() / df[column].notna().sum()
                    if success_rate >= 0.75:
                        # Convert to date-only format (remove time component)
                        df[column] = converted.dt.date
                        converted_columns.append(column)
                        logger.info(f"✅ Converted '{column}' from Unix timestamp to date-only format ({success_rate:.2%} success)")
                        continue
            except Exception as e:
                logger.debug(f"Unix timestamp check failed for '{column}': {e}")
        
        # Check if string values match datetime patterns
        datetime_likelihood = 0
        matching_pattern = None
        
        for pattern in datetime_patterns:
            matches = sample_str.str.match(pattern, na=False).sum()
            if matches > 0:
                likelihood = matches / len(sample_str)
                if likelihood > datetime_likelihood:
                    datetime_likelihood = likelihood
                    matching_pattern = pattern
        
        # If at least 75% of values look like dates, try to convert
        if datetime_likelihood >= 0.75:
            logger.info(f"Column '{column}' has {datetime_likelihood:.2%} datetime-like values (pattern: {matching_pattern})")
            
            # Try pandas automatic parsing first
            try:
                converted = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
                success_rate = converted.notna().sum() / df[column].notna().sum()
                if success_rate >= 0.75:
                    # Convert to date-only format (remove time component)
                    df[column] = converted.dt.date
                    converted_columns.append(column)
                    logger.info(f"✅ Converted '{column}' to date-only format using automatic parsing ({success_rate:.2%} success)")
                    continue
            except Exception as e:
                logger.debug(f"Auto parsing failed for '{column}': {e}")
            
            # Try specific formats
            best_format = None
            best_success_rate = 0
            
            for fmt in datetime_formats:
                try:
                    # Test on a subset first for efficiency
                    test_sample = sample.head(20)
                    test_converted = pd.to_datetime(test_sample, format=fmt, errors='coerce')
                    test_success_rate = test_converted.notna().sum() / len(test_sample)
                    
                    if test_success_rate >= 0.75:  # If test is promising, try full column
                        converted = pd.to_datetime(df[column], format=fmt, errors='coerce')
                        success_rate = converted.notna().sum() / df[column].notna().sum()
                        if success_rate > best_success_rate:
                            best_success_rate = success_rate
                            best_format = fmt
                except Exception as e:
                    continue
            
            # Apply the best format if success rate is good enough
            if best_format and best_success_rate >= 0.75:
                try:
                    converted = pd.to_datetime(df[column], format=best_format, errors='coerce')
                    # Convert to date-only format (remove time component)
                    df[column] = converted.dt.date
                    converted_columns.append(column)
                    logger.info(f"✅ Converted '{column}' to date-only format using format {best_format} ({best_success_rate:.2%} success)")
                except Exception as e:
                    logger.warning(f"Failed to convert '{column}' despite good format match: {e}")
            else:
                logger.info(f"❌ Could not convert '{column}' - best success rate was {best_success_rate:.2%}")
    
    if converted_columns:
        logger.info(f"🎉 Successfully converted {len(converted_columns)} columns to datetime: {converted_columns}")
    else:
        logger.info("ℹ️ No datetime columns detected or converted")
    
    return converted_columns

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global current_data, current_filename
    
    logger.info("=== FILE UPLOAD REQUEST RECEIVED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request content type: {request.content_type}")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    logger.info(f"File object received: {file}")
    logger.info(f"Original filename: {file.filename}")
    
    if file.filename == '':
        logger.error("No file selected (empty filename)")
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    logger.info(f"File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Secure filename: {filename}")
        logger.info(f"Upload path: {filepath}")
        
        try:
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            logger.info(f"Upload directory ensured: {app.config['UPLOAD_FOLDER']}")
            
            # Save the file
            logger.info("Starting file save...")
            file.save(filepath)
            logger.info(f"✅ FILE SAVED SUCCESSFULLY: {filepath}")
            
            # Verify file was saved
            if os.path.exists(filepath):
                saved_size = os.path.getsize(filepath)
                logger.info(f"✅ File verified on disk. Size: {saved_size} bytes")
            else:
                logger.error(f"❌ File not found after save: {filepath}")
                return jsonify({'error': 'File save verification failed'}), 500
            
            # Read the file based on extension
            file_ext = filename.rsplit('.', 1)[1].lower()
            logger.info(f"Reading file with extension: {file_ext}")
            
            if file_ext == 'csv':
                logger.info("Reading as CSV file...")
                current_data = pd.read_csv(filepath)
            elif file_ext in ['xlsx', 'xls']:
                logger.info("Reading as Excel file...")
                current_data = pd.read_excel(filepath)
            
            # Apply datetime detection
            logger.info("Starting datetime detection...")
            datetime_columns = detect_datetime_columns(current_data)
            if datetime_columns:
                logger.info(f"✅ Detected and converted {len(datetime_columns)} datetime columns: {datetime_columns}")
            else:
                logger.info("No datetime columns detected")
            
            current_filename = filename
            
            # Invalidate preview cache since new data is loaded
            invalidate_preview_cache()
            
            # Track the initial data upload operation
            track_operation(
                'data_upload',
                f'Dataset uploaded: {filename}',
                {
                    'original_shape': current_data.shape,
                    'columns': current_data.columns.tolist(),
                    'file_size_mb': round(file_size / (1024*1024), 2),
                    'file_type': file_ext
                }
            )
            
            logger.info(f"✅ DATA LOADED SUCCESSFULLY:")
            logger.info(f"   - Dataset shape: {current_data.shape}")
            logger.info(f"   - Columns: {list(current_data.columns)}")
            logger.info(f"   - Memory usage: {current_data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
            logger.info("=== UPLOAD PROCESS COMPLETED SUCCESSFULLY ===")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'shape': current_data.shape,
                'columns': current_data.columns.tolist()
            }), 200
            
        except Exception as e:
            logger.error(f"❌ ERROR DURING FILE PROCESSING: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
    
    else:
        logger.error(f"❌ INVALID FILE TYPE: {file.filename}")
        logger.error(f"Allowed extensions: {ALLOWED_EXTENSIONS}")
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/preview', methods=['GET'])
def preview_data():
    global current_data, preview_cache, preview_cache_hash
    
    logger.info("Data preview requested")
    logger.info(f"Current data is: {current_data}")
    
    if current_data is None:
        logger.warning("Preview requested but no data uploaded")
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Generate hash for current data
        current_hash = generate_data_hash(current_data)
        
        # Check if we have cached preview and data hasn't changed
        if preview_cache is not None and preview_cache_hash == current_hash:
            logger.info("✅ Returning cached preview data")
            return jsonify(preview_cache), 200
        
        # Generate new preview data
        try:
            # Convert to records and handle any serialization issues
            preview = current_data.head(5).copy()
            
            # Handle different data types for JSON serialization
            for col in preview.columns:
                if pd.api.types.is_datetime64_any_dtype(preview[col]):
                    # Convert datetime to DD/MM/YYYY format (remove time)
                    preview[col] = preview[col].dt.strftime('%d/%m/%Y').fillna('')
                elif preview[col].dtype == 'object':
                    # Check if it contains date objects
                    sample = preview[col].dropna().head(3)
                    if len(sample) > 0:
                        first_val = sample.iloc[0]
                        if hasattr(first_val, 'strftime'):
                            # Format date objects as DD/MM/YYYY
                            preview[col] = preview[col].apply(
                                lambda x: x.strftime('%d/%m/%Y') if hasattr(x, 'strftime') else str(x)
                            )
                        else:
                            # Convert other objects to string and handle NaN
                            preview[col] = preview[col].fillna('').astype(str)
                    else:
                        preview[col] = preview[col].fillna('').astype(str)
                elif pd.api.types.is_numeric_dtype(preview[col]):
                    # Handle NaN values in numeric columns
                    preview[col] = preview[col].fillna(0)
                else:
                    # Convert everything else to string and handle NaN
                    preview[col] = preview[col].fillna('').astype(str)
            
            preview_data = preview.to_dict('records')
            columns = current_data.columns.tolist()
            
            # Ensure we have valid data
            if not preview_data or not columns:
                raise ValueError("Empty preview data or columns")
            
            # Cache the preview data with correct structure for frontend
            preview_cache = {
                'data': preview_data,
                'columns': columns,
                'total_rows': len(current_data),
                'preview_rows': len(preview_data)
            }
            preview_cache_hash = current_hash
            
            logger.info(f"✅ Preview generated and cached: {len(preview_data)} rows, {len(columns)} columns")
            logger.info(f"Preview response structure: data={len(preview_data) if preview_data else 'None'}, columns={len(columns) if columns else 'None'}")
            
            return jsonify(preview_cache), 200
            
        except Exception as preview_error:
            logger.error(f"Error in preview generation: {str(preview_error)}")
            # Fallback: simple conversion with date formatting
            try:
                fallback_data = current_data.head().copy()
                # Format dates in fallback too
                for col in fallback_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(fallback_data[col]):
                        fallback_data[col] = fallback_data[col].dt.strftime('%d/%m/%Y').fillna('')
                    elif fallback_data[col].dtype == 'object':
                        sample = fallback_data[col].dropna().head(3)
                        if len(sample) > 0:
                            first_val = sample.iloc[0]
                            if hasattr(first_val, 'strftime'):
                                fallback_data[col] = fallback_data[col].apply(
                                    lambda x: x.strftime('%d/%m/%Y') if hasattr(x, 'strftime') else str(x)
                                )
                            else:
                                fallback_data[col] = fallback_data[col].fillna('').astype(str)
                        else:
                            fallback_data[col] = fallback_data[col].fillna('').astype(str)
                    else:
                        fallback_data[col] = fallback_data[col].fillna('').astype(str)
                
                preview_data = fallback_data.to_dict('records')
            except:
                # Ultimate fallback
                preview_data = current_data.head().fillna('').astype(str).to_dict('records')
            
            columns = current_data.columns.tolist()
            
            preview_cache = {
                'data': preview_data,
                'columns': columns,
                'total_rows': len(current_data),
                'preview_rows': len(preview_data)
            }
            preview_cache_hash = current_hash
            
            logger.info(f"✅ Fallback preview generated: {len(preview_data)} rows, {len(columns)} columns")
            logger.info(f"Fallback response structure: data={len(preview_data) if preview_data else 'None'}, columns={len(columns) if columns else 'None'}")
            
            return jsonify(preview_cache), 200
        
    except Exception as e:
        logger.error(f"❌ Error generating preview: {str(e)}")
        return jsonify({'error': f'Error generating preview: {str(e)}'}), 400

@app.route('/api/test-preview', methods=['GET'])
def test_preview():
    """Test endpoint to debug preview issues"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded', 'has_data': False}), 400
    
    try:
        response = {
            'has_data': True,
            'shape': current_data.shape,
            'columns': current_data.columns.tolist(),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'first_row': current_data.iloc[0].fillna('NULL').astype(str).to_dict() if len(current_data) > 0 else {}
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}', 'has_data': True}), 400

@app.route('/api/data', methods=['GET'])
def get_all_data():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Format data for display (convert dates to DD/MM/YYYY format)
        display_data = format_date_columns_for_display(current_data)
        
        # Convert to records for JSON serialization
        data = display_data.to_dict('records')
        columns = current_data.columns.tolist()
        
        return jsonify({
            'data': data,
            'columns': columns,
            'total_rows': len(current_data)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error getting data: {str(e)}'}), 400

@app.route('/api/info', methods=['GET'])
def get_data_info():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Basic info
        shape = current_data.shape
        
        # Missing values (only show columns with missing values)
        missing_values = current_data.isnull().sum()
        missing_values = missing_values[missing_values > 0].to_dict()
        
        # Data types categorization
        dtypes = {}
        numeric_cols = []
        datetime_cols = []
        categorical_cols = []
        
        for column in current_data.columns:
            if pd.api.types.is_datetime64_any_dtype(current_data[column]):
                dtypes[column] = 'datetime64[ns]'
                datetime_cols.append(column)
            elif current_data[column].dtype == 'object':
                # Check if it contains date objects
                sample = current_data[column].dropna().head(10)
                if len(sample) > 0:
                    first_val = sample.iloc[0]
                    if hasattr(first_val, 'year') and hasattr(first_val, 'month') and hasattr(first_val, 'day'):
                        # This is a date object column
                        dtypes[column] = 'date'
                        datetime_cols.append(column)
                    else:
                        dtypes[column] = str(current_data[column].dtype)
                        categorical_cols.append(column)
                else:
                    dtypes[column] = str(current_data[column].dtype)
                    categorical_cols.append(column)
            elif pd.api.types.is_numeric_dtype(current_data[column]):
                dtypes[column] = str(current_data[column].dtype)
                numeric_cols.append(column)
            else:
                dtypes[column] = str(current_data[column].dtype)
                categorical_cols.append(column)
        
        logger.info(f"Info endpoint - Numeric columns found: {numeric_cols}")
        logger.info(f"Info endpoint - Datetime columns found: {datetime_cols}")
        logger.info(f"Info endpoint - Categorical columns found: {categorical_cols}")
        
        # Statistics
        stats = {}
        
        # Numeric statistics
        if numeric_cols:
            stats['numeric'] = current_data[numeric_cols].describe().to_dict()
        
        # Datetime statistics
        datetime_stats = {}
        if datetime_cols:
            for col in datetime_cols:
                try:
                    col_data = current_data[col].dropna()
                    if len(col_data) > 0:
                        # Handle both datetime64 and date objects
                        if pd.api.types.is_datetime64_any_dtype(current_data[col]):
                            # Standard datetime column
                            datetime_stats[col] = {
                                'min_date': col_data.min().strftime('%d/%m/%Y'),
                                'max_date': col_data.max().strftime('%d/%m/%Y'),
                                'date_range_days': (col_data.max() - col_data.min()).days,
                                'unique_dates': int(col_data.nunique()),
                                'null_count': int(current_data[col].isnull().sum()),
                                'sample_values': [d.strftime('%d/%m/%Y') for d in col_data.head(3)]
                            }
                        else:
                            # Date object column
                            first_val = col_data.iloc[0]
                            if hasattr(first_val, 'strftime'):
                                # Convert date objects to strings for calculations
                                min_date = min(col_data)
                                max_date = max(col_data)
                                date_range = (max_date - min_date).days
                                
                                datetime_stats[col] = {
                                    'min_date': min_date.strftime('%d/%m/%Y'),
                                    'max_date': max_date.strftime('%d/%m/%Y'),
                                    'date_range_days': date_range,
                                    'unique_dates': int(col_data.nunique()),
                                    'null_count': int(current_data[col].isnull().sum()),
                                    'sample_values': [d.strftime('%d/%m/%Y') for d in col_data.head(3)]
                                }
                            else:
                                # Fallback for other object types
                                datetime_stats[col] = {
                                    'min_date': str(min(col_data)),
                                    'max_date': str(max(col_data)),
                                    'unique_dates': int(col_data.nunique()),
                                    'null_count': int(current_data[col].isnull().sum()),
                                    'sample_values': [str(d) for d in col_data.head(3)]
                                }
                except Exception as e:
                    logger.warning(f"Error calculating datetime stats for {col}: {e}")
                    datetime_stats[col] = {'error': str(e)}
        
        if datetime_stats:
            stats['datetime'] = datetime_stats
        
        return jsonify({
            'shape': shape,
            'missing_values': missing_values,
            'data_types': dtypes,
            'numeric_columns': numeric_cols,  # Frontend expects this key
            'datetime_columns': datetime_cols,  # Frontend might expect this too
            'categorical_columns': categorical_cols,  # Frontend might expect this too
            'column_categories': {
                'numeric': numeric_cols,
                'datetime': datetime_cols,
                'categorical': categorical_cols
            },
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error getting data info: {str(e)}'}), 400

@app.route('/api/debug-columns', methods=['GET'])
def debug_columns():
    """Debug endpoint to check column types"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        debug_info = {}
        for column in current_data.columns:
            debug_info[column] = {
                'dtype': str(current_data[column].dtype),
                'is_numeric': bool(pd.api.types.is_numeric_dtype(current_data[column])),
                'is_datetime': bool(pd.api.types.is_datetime64_any_dtype(current_data[column])),
                'sample_values': current_data[column].head(3).astype(str).tolist()
            }
        
        return jsonify({
            'total_columns': len(current_data.columns),
            'column_details': debug_info
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Debug failed: {str(e)}'}), 400

@app.route('/api/test-datetime', methods=['GET'])
def test_datetime_detection():
    """Test endpoint to check datetime detection"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        result = {
            'total_columns': len(current_data.columns),
            'column_analysis': {}
        }
        
        for column in current_data.columns:
            col_info = {
                'dtype': str(current_data[column].dtype),
                'is_datetime': bool(pd.api.types.is_datetime64_any_dtype(current_data[column])),
                'sample_values': current_data[column].head(3).astype(str).tolist(),
                'null_count': int(current_data[column].isnull().sum())
            }
            
            if pd.api.types.is_datetime64_any_dtype(current_data[column]):
                try:
                    non_null_data = current_data[column].dropna()
                    if len(non_null_data) > 0:
                        col_info['datetime_info'] = {
                            'min': non_null_data.min().strftime('%Y-%m-%d %H:%M:%S'),
                            'max': non_null_data.max().strftime('%Y-%m-%d %H:%M:%S'),
                            'unique_count': int(non_null_data.nunique())
                        }
                except:
                    col_info['datetime_info'] = 'Error getting datetime info'
            
            result['column_analysis'][column] = col_info
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 400

@app.route('/api/plot', methods=['POST'])
def generate_plot():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        start_time = time.time()
        
        data = request.json
        x_col = data.get('x_axis')
        y_col = data.get('y_axis')
        plot_type = data.get('plot_type')
        
        logger.info(f"Starting plot generation: {plot_type} for {x_col} vs {y_col}")
        
        # Start with optimized matplotlib settings
        try:
            plt.style.use('fast')  # Use fast plotting style if available
        except:
            pass  # Fall back to default style if 'fast' is not available
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Sample data for better performance with large datasets
        plot_data = sample_data_for_plotting(current_data)
        
        # Prepare data for plotting (convert date objects to datetime)
        plot_data = prepare_data_for_plotting(plot_data)
        
        logger.info(f"Plotting with {len(plot_data)} data points (sampled from {len(current_data)})")
        
        sampling_time = time.time()
        logger.info(f"Data sampling took {sampling_time - start_time:.2f} seconds")
        
        # Get column types for enhanced plotting
        x_type = get_column_type(current_data[x_col]) if x_col else None
        y_type = get_column_type(current_data[y_col]) if y_col else None
        
        logger.info(f"Column types: X='{x_col}' ({x_type}), Y='{y_col}' ({y_type})")
        
        if plot_type == 'scatter':
            # Remove NaN values for cleaner plots
            clean_data = plot_data[[x_col, y_col]].dropna()
            if len(clean_data) > 0:
                # Use smaller marker size and rasterization for better performance
                scatter = ax.scatter(clean_data[x_col], clean_data[y_col], 
                                   alpha=0.6, s=20, rasterized=True, edgecolors='none')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                
                # Special formatting for datetime axes
                if x_type == 'datetime':
                    ax.tick_params(axis='x', rotation=45)
                    fig.autofmt_xdate()  # Better datetime formatting
                if y_type == 'datetime':
                    ax.tick_params(axis='y', rotation=45)
            
        elif plot_type == 'line':
            if y_col:
                # Sort data for line plots
                clean_data = plot_data[[x_col, y_col]].dropna().sort_values(x_col)
                if len(clean_data) > 0:
                    line = ax.plot(clean_data[x_col], clean_data[y_col], 
                                  linewidth=1, rasterized=True, alpha=0.8)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'Line Plot: {x_col} vs {y_col}')
                    
                    # Special formatting for datetime axes
                    if x_type == 'datetime':
                        ax.tick_params(axis='x', rotation=45)
                        fig.autofmt_xdate()  # Better datetime formatting
                        # Add grid for time series
                        ax.grid(True, alpha=0.3)
                    if y_type == 'datetime':
                        ax.tick_params(axis='y', rotation=45)
            else:
                # Single datetime column - create timeline plot
                if x_type == 'datetime':
                    clean_data = plot_data[x_col].dropna().sort_values()
                    if len(clean_data) > 0:
                        # Create a simple timeline showing data density over time
                        # Group by time periods and count occurrences
                        time_counts = clean_data.dt.floor('D').value_counts().sort_index()  # Daily grouping
                        
                        line = ax.plot(time_counts.index, time_counts.values, 
                                      linewidth=2, marker='o', markersize=3, alpha=0.8)
                        ax.set_xlabel(f'{x_col} (Date)')
                        ax.set_ylabel('Count')
                        ax.set_title(f'Timeline: Data Count by {x_col}')
                        ax.tick_params(axis='x', rotation=45)
                        fig.autofmt_xdate()
                        ax.grid(True, alpha=0.3)
                else:
                    return jsonify({'error': 'Line plot with single axis requires datetime column'}), 400
            
        elif plot_type == 'bar':
            if plot_data[x_col].dtype == 'object':
                # Limit to top 20 categories for performance
                value_counts = plot_data[x_col].value_counts().head(20)
                if len(value_counts) > 0:
                    ax.bar(range(len(value_counts)), value_counts.values, color='steelblue')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel('Count')
                    ax.set_title(f'Bar Plot: {x_col} (Top 20)')
            else:
                # Use automatic binning for numeric data
                clean_data = plot_data[x_col].dropna()
                if len(clean_data) > 0:
                    # Optimize bin count based on data size
                    bin_count = min(30, max(10, int(np.sqrt(len(clean_data)))))
                    n, bins, patches = ax.hist(clean_data, bins=bin_count, alpha=0.7, 
                                             color='steelblue', edgecolor='none', 
                                             rasterized=True)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Histogram: {x_col}')
                
        elif plot_type == 'histogram':
            clean_data = plot_data[x_col].dropna()
            if len(clean_data) > 0:
                # Optimize bin count based on data size
                bin_count = min(30, max(10, int(np.sqrt(len(clean_data)))))
                n, bins, patches = ax.hist(clean_data, bins=bin_count, alpha=0.7, 
                                         color='steelblue', edgecolor='none',
                                         rasterized=True)
                ax.set_xlabel(x_col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram: {x_col}')
            
        elif plot_type == 'box':
            if y_col:
                # For categorical x and numeric y
                clean_data = plot_data[[x_col, y_col]].dropna()
                if len(clean_data) > 0:
                    # Limit categories for performance
                    top_categories = clean_data[x_col].value_counts().head(10).index
                    filtered_data = clean_data[clean_data[x_col].isin(top_categories)]
                    
                    if len(filtered_data) > 0:
                        box_data = [filtered_data[filtered_data[x_col] == cat][y_col].values 
                                   for cat in top_categories if len(filtered_data[filtered_data[x_col] == cat]) > 0]
                        if box_data:
                            ax.boxplot(box_data, labels=top_categories[:len(box_data)])
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.set_title(f'Box Plot: {y_col} by {x_col} (Top 10 Categories)')
                            plt.xticks(rotation=45)
            else:
                # Single variable box plot
                clean_data = plot_data[x_col].dropna()
                if len(clean_data) > 0:
                    ax.boxplot(clean_data)
                    ax.set_ylabel(x_col)
                    ax.set_title(f'Box Plot: {x_col}')
        
        # Optimize layout and rendering
        plt.tight_layout()
        
        render_time = time.time()
        logger.info(f"Plot rendering took {render_time - sampling_time:.2f} seconds")
        
        # Add data info to plot
        if len(plot_data) < len(current_data):
            ax.text(0.02, 0.98, f'📊 Showing {len(plot_data):,} of {len(current_data):,} points', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plot_url = create_plot_base64(fig)
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total plot generation took {total_time:.2f} seconds")
        
        return jsonify({'plot': plot_url}), 200
        
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        return jsonify({'error': f'Error generating plot: {str(e)}'}), 400

@app.route('/api/correlation', methods=['GET'])
def get_correlation():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Get numeric columns only
        numeric_data = current_data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return jsonify({'error': 'No numeric columns found for correlation analysis'}), 400
        
        # Remove columns that are all NaN or have no variance
        numeric_data = numeric_data.dropna(axis=1, how='all')  # Remove all-NaN columns
        
        # Remove columns with zero variance (constant values)
        variance_check = numeric_data.var()
        valid_columns = variance_check[variance_check > 0].index
        numeric_data = numeric_data[valid_columns]
        
        if numeric_data.empty:
            return jsonify({'error': 'No valid numeric columns found for correlation analysis'}), 400
        
        logger.info(f"Computing correlation for {len(numeric_data.columns)} numeric columns: {list(numeric_data.columns)}")
        
        # Sample data for faster correlation calculation if dataset is large
        if len(numeric_data) > 5000:
            sampled_data = numeric_data.sample(n=5000, random_state=42)
            logger.info(f"Using sampled data ({len(sampled_data)} rows) for correlation calculation")
        else:
            sampled_data = numeric_data
            
        # Calculate correlation matrix
        corr_matrix = sampled_data.corr()
        
        # Handle any NaN values in correlation matrix
        corr_matrix = corr_matrix.fillna(0)  # Replace NaN with 0 for invalid correlations
        
        # Create optimized heatmap
        try:
            plt.style.use('fast')  # Use fast plotting style if available
        except:
            pass  # Fall back to default style if 'fast' is not available
            
        fig, ax = plt.subplots(figsize=(min(12, len(corr_matrix.columns) * 0.8), 
                                       min(10, len(corr_matrix.columns) * 0.6)))
        
        # Use optimized heatmap settings
        heatmap = sns.heatmap(corr_matrix, 
                   annot=len(corr_matrix.columns) <= 10,  # Only annotate if not too many columns
                   cmap='RdBu_r',  # Faster colormap
                   center=0, 
                   square=True, 
                   linewidths=0.1,  # Thinner lines for speed
                   cbar_kws={'shrink': 0.8},
                   ax=ax,
                   rasterized=True,  # Rasterize for faster rendering
                   fmt='.2f' if len(corr_matrix.columns) <= 10 else None)  # Format numbers
        
        ax.set_title('Correlation Matrix Heatmap')
        plt.tight_layout()
        
        heatmap_url = create_plot_base64(fig)
        
        # Convert correlation matrix to dict with proper handling of NaN values
        corr_dict = {}
        for col in corr_matrix.columns:
            corr_dict[col] = {}
            for row in corr_matrix.index:
                value = corr_matrix.loc[row, col]
                # Ensure we don't have any NaN, inf, or invalid values
                if pd.isna(value) or np.isinf(value):
                    corr_dict[col][row] = 0.0
                else:
                    corr_dict[col][row] = float(value)
        
        # Add sampling info if data was sampled
        correlation_info = {
            'heatmap': heatmap_url
        }
        
        if len(numeric_data) > 5000:
            correlation_info['sampling_info'] = {
                'original_rows': len(numeric_data),
                'sampled_rows': len(sampled_data),
                'message': f'Correlation calculated using {len(sampled_data):,} sampled rows from {len(numeric_data):,} total rows'
            }
        
        return jsonify(correlation_info), 200
        
    except Exception as e:
        logger.error(f"Error generating correlation: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return jsonify({'error': f'Error generating correlation: {str(e)}'}), 400

@app.route('/api/test-correlation', methods=['GET'])
def test_correlation():
    """Debug endpoint to test correlation calculation"""
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Get numeric columns only
        numeric_data = current_data.select_dtypes(include=[np.number])
        
        debug_info = {
            'total_columns': len(current_data.columns),
            'numeric_columns': list(numeric_data.columns),
            'numeric_column_count': len(numeric_data.columns),
            'data_shape': numeric_data.shape,
        }
        
        if not numeric_data.empty:
            # Check for columns with all NaN
            nan_columns = numeric_data.columns[numeric_data.isna().all()].tolist()
            debug_info['all_nan_columns'] = nan_columns
            
            # Check for columns with zero variance
            variance_check = numeric_data.var()
            zero_var_columns = variance_check[variance_check == 0].index.tolist()
            debug_info['zero_variance_columns'] = zero_var_columns
            
            # Sample correlation calculation
            try:
                clean_data = numeric_data.dropna(axis=1, how='all')
                variance_check = clean_data.var()
                valid_columns = variance_check[variance_check > 0].index
                final_data = clean_data[valid_columns]
                
                if len(final_data.columns) >= 2:
                    sample_corr = final_data.iloc[:100].corr()  # Small sample
                    debug_info['sample_correlation_shape'] = sample_corr.shape
                    debug_info['sample_correlation_columns'] = list(sample_corr.columns)
                    debug_info['has_nan_in_correlation'] = sample_corr.isna().any().any()
                else:
                    debug_info['error'] = 'Not enough valid columns for correlation'
            except Exception as corr_error:
                debug_info['correlation_error'] = str(corr_error)
        
        return jsonify(debug_info), 200
        
    except Exception as e:
        return jsonify({'error': f'Test correlation failed: {str(e)}'}), 400

@app.route('/api/plot-options', methods=['POST'])
def get_plot_options():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        x_col = data.get('x_axis')
        y_col = data.get('y_axis')
        
        options = []
        
        if x_col and y_col:
            # Both axes selected
            x_type = get_column_type(current_data[x_col])
            y_type = get_column_type(current_data[y_col])
            
            logger.info(f"Plot options: X-axis '{x_col}' is {x_type}, Y-axis '{y_col}' is {y_type}")
            
            if x_type == 'datetime' and y_type == 'numeric':
                options = ['line', 'scatter']  # Time series plots
            elif x_type == 'numeric' and y_type == 'datetime':
                options = ['scatter']  # Less common but possible
            elif x_type == 'datetime' and y_type == 'categorical':
                options = ['scatter']  # Timeline of events
            elif x_type == 'categorical' and y_type == 'datetime':
                options = ['box']  # Time distributions by category
            elif x_type == 'datetime' and y_type == 'datetime':
                options = ['scatter']  # Compare two time series
            elif x_type == 'numeric' and y_type == 'numeric':
                options = ['scatter', 'line']
            elif x_type == 'categorical' and y_type == 'numeric':
                options = ['bar', 'box']
            elif x_type == 'numeric' and y_type == 'categorical':
                options = ['bar']
                
        elif x_col:
            # Only x-axis selected
            x_type = get_column_type(current_data[x_col])
            
            logger.info(f"Single axis plot options: X-axis '{x_col}' is {x_type}")
            
            if x_type == 'datetime':
                options = ['line']  # Timeline/trend plot
            elif x_type == 'numeric':
                options = ['histogram', 'box']
            else:
                options = ['bar']
        
        logger.info(f"Available plot options: {options}")
        return jsonify({'options': options}), 200
        
    except Exception as e:
        logger.error(f"Error getting plot options: {str(e)}")
        return jsonify({'error': f'Error getting plot options: {str(e)}'}), 400

def get_column_type(column):
    """Determine the type of a column for plotting purposes."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return 'datetime'
    elif column.dtype == 'object':
        # Check if it contains date objects
        sample = column.dropna().head(10)
        if len(sample) > 0:
            # Check if sample values are date objects
            first_val = sample.iloc[0]
            if hasattr(first_val, 'year') and hasattr(first_val, 'month') and hasattr(first_val, 'day'):
                # This is likely a date object
                return 'datetime'
        return 'categorical'
    elif pd.api.types.is_numeric_dtype(column):
        return 'numeric'
    else:
        return 'categorical'

@app.route('/api/valid-y-columns', methods=['POST'])
def get_valid_y_columns():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        x_col = data.get('x_axis')
        
        if not x_col or x_col not in current_data.columns:
            # Return all columns if no valid X-axis selected
            return jsonify({'valid_columns': current_data.columns.tolist()}), 200
        
        # Determine X-axis type using enhanced type detection
        x_type = get_column_type(current_data[x_col])
        
        valid_columns = []
        
        for col in current_data.columns:
            if col == x_col:  # Skip the same column
                continue
                
            # Determine column type using enhanced type detection
            col_type = get_column_type(current_data[col])
            
            # Define valid combinations based on enhanced plotting logic
            if x_type == 'datetime':
                # Datetime X-axis works well with numeric (time series) and categorical data
                if col_type in ['numeric', 'categorical']:
                    valid_columns.append(col)
            elif x_type == 'numeric':
                # Numeric X-axis can pair with any type
                valid_columns.append(col)
            elif x_type == 'categorical':
                # Categorical X-axis works best with numeric and datetime Y-axis
                if col_type in ['numeric', 'datetime']:
                    valid_columns.append(col)
        
        logger.info(f"Valid Y columns for X-axis '{x_col}' ({x_type}): {valid_columns}")
        return jsonify({'valid_columns': valid_columns}), 200
        
    except Exception as e:
        logger.error(f"Error getting valid Y columns: {str(e)}")
        return jsonify({'error': f'Error getting valid Y columns: {str(e)}'}), 400

@app.route('/api/column-analysis', methods=['POST'])
def analyze_column():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        column = data.get('column')
        
        if column not in current_data.columns:
            return jsonify({'error': f'Column {column} not found'}), 400
        
        col_data = current_data[column]
        
        # Basic info
        analysis = {
            'column': column,
            'data_type': str(col_data.dtype),
            'non_null_count': int(col_data.count()),
            'null_count': int(col_data.isnull().sum()),
            'is_numeric': pd.api.types.is_numeric_dtype(col_data)
        }
        
        # Value counts (top 10 most frequent values - highest to lowest)
        value_counts = col_data.value_counts().head(10)
        top_frequent_values = []
        for value, count in value_counts.items():
            percentage = (count / len(col_data)) * 100
            top_frequent_values.append({
                'value': str(value),
                'count': int(count),
                'percentage': round(percentage, 2)
            })
        analysis['top_frequent_values'] = top_frequent_values
        
        # Sample values (10 random non-null values for variety)
        available_values = col_data.dropna()
        if len(available_values) > 10:
            sample_values = available_values.sample(10).tolist()
        else:
            sample_values = available_values.tolist()
        analysis['sample_values'] = sample_values
        
        # Numeric statistics if applicable
        if analysis['is_numeric']:
            analysis.update({
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'variance': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            })
        
        return jsonify(analysis), 200
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing column: {str(e)}'}), 400

@app.route('/api/drop-columns', methods=['POST'])
def drop_columns():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        columns_to_drop = data.get('columns', [])
        
        if not columns_to_drop:
            return jsonify({'error': 'No columns specified for dropping'}), 400
        
        # Validate columns exist
        invalid_columns = [col for col in columns_to_drop if col not in current_data.columns]
        if invalid_columns:
            return jsonify({'error': f'Columns not found: {invalid_columns}'}), 400
        
        # Drop columns
        current_data = current_data.drop(columns=columns_to_drop)
        
        # Invalidate preview cache since data structure changed
        invalidate_preview_cache()
        
        # Track the column dropping operation
        track_operation(
            'drop_columns',
            f'Dropped {len(columns_to_drop)} columns',
            {
                'dropped_columns': columns_to_drop,
                'original_shape': [current_data.shape[0], current_data.shape[1] + len(columns_to_drop)],
                'new_shape': current_data.shape,
                'columns_removed': len(columns_to_drop)
            }
        )
        
        return jsonify({
            'message': f'Successfully dropped {len(columns_to_drop)} columns',
            'dropped_columns': columns_to_drop,
            'shape': current_data.shape,
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error dropping columns: {str(e)}'}), 400

@app.route('/api/impute-missing', methods=['POST'])
def impute_missing_values():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        rules = data.get('rules', [])
        
        if not rules:
            return jsonify({'error': 'No imputation rules provided'}), 400
        
        applied_rules = []
        
        for rule in rules:
            column = rule.get('column')
            method = rule.get('method')
            custom_value = rule.get('customValue')
            
            if column not in current_data.columns:
                continue
                
            if method == 'mean' and pd.api.types.is_numeric_dtype(current_data[column]):
                fill_value = current_data[column].mean()
                current_data[column].fillna(fill_value, inplace=True)
                applied_rules.append(f'{column}: filled with mean ({fill_value:.2f})')
                
            elif method == 'median' and pd.api.types.is_numeric_dtype(current_data[column]):
                fill_value = current_data[column].median()
                current_data[column].fillna(fill_value, inplace=True)
                applied_rules.append(f'{column}: filled with median ({fill_value:.2f})')
                
            elif method == 'mode':
                fill_value = current_data[column].mode()
                if not fill_value.empty:
                    current_data[column].fillna(fill_value.iloc[0], inplace=True)
                    applied_rules.append(f'{column}: filled with mode ({fill_value.iloc[0]})')
                    
            elif method == 'forward_fill':
                current_data[column].fillna(method='ffill', inplace=True)
                applied_rules.append(f'{column}: forward filled')
                
            elif method == 'backward_fill':
                current_data[column].fillna(method='bfill', inplace=True)
                applied_rules.append(f'{column}: backward filled')
                
            elif method == 'custom' and custom_value is not None:
                # Try to convert custom value to appropriate type
                try:
                    if pd.api.types.is_numeric_dtype(current_data[column]):
                        fill_value = float(custom_value)
                    else:
                        fill_value = str(custom_value)
                    current_data[column].fillna(fill_value, inplace=True)
                    applied_rules.append(f'{column}: filled with custom value ({fill_value})')
                except ValueError:
                    applied_rules.append(f'{column}: error with custom value')
        
        # Invalidate preview cache since data values changed
        invalidate_preview_cache()
        
        # Track the imputation operation
        track_operation(
            'missing_value_imputation',
            f'Imputed missing values using {len(applied_rules)} rules',
            {
                'rules_applied': applied_rules,
                'affected_columns': len(set(rule.split(':')[0] for rule in applied_rules)),
                'total_rules': len(applied_rules),
                'dataset_shape_after': current_data.shape
            }
        )
        
        return jsonify({
            'message': f'Applied {len(applied_rules)} imputation rules',
            'applied_rules': applied_rules,
            'shape': current_data.shape,
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error applying imputation: {str(e)}'}), 400

@app.route('/api/detect-outliers', methods=['POST'])
def detect_outliers():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        columns = data.get('columns', [])
        
        if not columns:
            # Use all numeric columns if none specified
            columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for col in columns:
            if col not in current_data.columns or not pd.api.types.is_numeric_dtype(current_data[col]):
                continue
                
            col_data = current_data[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Z-score method
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            zscore_outliers = len(z_scores[z_scores > 3])
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = len(col_data[(col_data < lower_bound) | (col_data > upper_bound)])
            
            outlier_info[col] = {
                'total_values': int(len(col_data)),
                'zscore_outliers': int(zscore_outliers),
                'iqr_outliers': int(iqr_outliers)
            }
        
        return jsonify(outlier_info), 200
        
    except Exception as e:
        return jsonify({'error': f'Error detecting outliers: {str(e)}'}), 400

@app.route('/api/remove-outliers', methods=['POST'])
def remove_outliers():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        rules = data.get('rules', [])
        
        if not rules:
            return jsonify({'error': 'No outlier removal rules provided'}), 400
        
        applied_rules = []
        original_shape = current_data.shape
        
        for rule in rules:
            column = rule.get('column')
            method = rule.get('method')
            action = rule.get('action')
            threshold = rule.get('threshold')
            
            if column not in current_data.columns or not pd.api.types.is_numeric_dtype(current_data[column]):
                continue
            
            col_data = current_data[column]
            outlier_mask = pd.Series([False] * len(current_data))
            
            if method == 'zscore':
                threshold_val = float(threshold) if threshold else 3.0
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outlier_mask = z_scores > threshold_val
                
            elif method == 'modified_zscore':
                threshold_val = float(threshold) if threshold else 3.5
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * (col_data - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold_val
                
            elif method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                
            elif method == 'percentile':
                if threshold:
                    try:
                        lower_p, upper_p = map(float, threshold.split(','))
                        lower_bound = col_data.quantile(lower_p/100)
                        upper_bound = col_data.quantile(upper_p/100)
                        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    except:
                        continue
                        
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outlier_mask = outlier_pred == -1
                except ImportError:
                    applied_rules.append(f'{column}: Isolation Forest not available (sklearn required)')
                    continue
            
            outlier_count = int(outlier_mask.sum())
            
            if action == 'remove':
                current_data = current_data[~outlier_mask]
                applied_rules.append(f'{column}: Removed {outlier_count} outliers using {method}')
                
            elif action == 'cap':
                if method in ['zscore', 'modified_zscore']:
                    # Cap at threshold * std
                    threshold_val = float(threshold) if threshold else 3.0
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    lower_cap = mean_val - threshold_val * std_val
                    upper_cap = mean_val + threshold_val * std_val
                elif method == 'iqr':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_cap = Q1 - 1.5 * IQR
                    upper_cap = Q3 + 1.5 * IQR
                else:
                    continue
                    
                current_data.loc[col_data < lower_cap, column] = lower_cap
                current_data.loc[col_data > upper_cap, column] = upper_cap
                applied_rules.append(f'{column}: Capped {outlier_count} outliers using {method}')
                
            elif action == 'transform':
                # Log transformation (add 1 to handle zeros)
                min_val = col_data.min()
                if min_val <= 0:
                    current_data[column] = np.log1p(col_data - min_val + 1)
                else:
                    current_data[column] = np.log1p(col_data)
                applied_rules.append(f'{column}: Applied log transformation')
        
        # Invalidate preview cache since data changed
        invalidate_preview_cache()
        
        # Track the operation
        track_operation('outlier_removal', 
                       f'Applied {len(applied_rules)} outlier removal rules',
                       {
            'applied_rules': applied_rules,
            'rules_count': len(applied_rules),
            'original_shape': list(original_shape),
            'new_shape': list(current_data.shape),
            'rows_removed': int(original_shape[0] - current_data.shape[0]),
            'methods_used': [rule.get('method') for rule in rules],
            'columns_processed': [rule.get('column') for rule in rules]
        })
        
        return jsonify({
            'message': f'Applied {len(applied_rules)} outlier removal rules',
            'applied_rules': applied_rules,
            'original_shape': list(original_shape),
            'new_shape': list(current_data.shape),
            'rows_removed': int(original_shape[0] - current_data.shape[0]),
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error removing outliers: {str(e)}'}), 400

@app.route('/api/standardize-columns', methods=['POST'])
def standardize_columns():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        column_mapping = data.get('column_mapping', {})
        data_standardization = data.get('data_standardization', {})
        
        if not column_mapping and not data_standardization:
            return jsonify({'error': 'No column mapping or data standardization provided'}), 400
        
        operations_performed = []
        
        # Handle column name standardization
        if column_mapping:
            # Validate that all original columns exist
            invalid_columns = [col for col in column_mapping.keys() if col not in current_data.columns]
            if invalid_columns:
                return jsonify({'error': f'Columns not found: {invalid_columns}'}), 400
            
            # Check for duplicate new names
            new_names = list(column_mapping.values())
            duplicates = [name for name in new_names if new_names.count(name) > 1]
            if duplicates:
                return jsonify({'error': f'Duplicate new column names: {list(set(duplicates))}'}), 400
            
            # Rename columns
            current_data = current_data.rename(columns=column_mapping)
            operations_performed.append(f'Renamed {len(column_mapping)} columns')
        
        # Handle data standardization
        if data_standardization:
            for column, standardization_type in data_standardization.items():
                if column not in current_data.columns:
                    continue
                
                if standardization_type == 'lowercase':
                    if current_data[column].dtype == 'object':
                        current_data[column] = current_data[column].astype(str).str.lower()
                        operations_performed.append(f'{column}: Converted to lowercase')
                
                elif standardization_type == 'uppercase':
                    if current_data[column].dtype == 'object':
                        current_data[column] = current_data[column].astype(str).str.upper()
                        operations_performed.append(f'{column}: Converted to uppercase')
                
                elif standardization_type == 'title_case':
                    if current_data[column].dtype == 'object':
                        current_data[column] = current_data[column].astype(str).str.title()
                        operations_performed.append(f'{column}: Converted to title case')
                
                elif standardization_type == 'trim_whitespace':
                    if current_data[column].dtype == 'object':
                        current_data[column] = current_data[column].astype(str).str.strip()
                        operations_performed.append(f'{column}: Trimmed whitespace')
                
                elif standardization_type == 'remove_special_chars':
                    if current_data[column].dtype == 'object':
                        import re
                        current_data[column] = current_data[column].astype(str).apply(
                            lambda x: re.sub(r'[^\w\s]', '', x) if pd.notna(x) else x
                        )
                        operations_performed.append(f'{column}: Removed special characters')
                
                elif standardization_type == 'normalize_spaces':
                    if current_data[column].dtype == 'object':
                        import re
                        current_data[column] = current_data[column].astype(str).apply(
                            lambda x: re.sub(r'\s+', ' ', x).strip() if pd.notna(x) else x
                        )
                        operations_performed.append(f'{column}: Normalized spaces')
                
                elif standardization_type == 'z_score':
                    if pd.api.types.is_numeric_dtype(current_data[column]):
                        from scipy import stats
                        current_data[column] = stats.zscore(current_data[column], nan_policy='omit')
                        operations_performed.append(f'{column}: Applied Z-score standardization')
                
                elif standardization_type == 'min_max':
                    if pd.api.types.is_numeric_dtype(current_data[column]):
                        min_val = current_data[column].min()
                        max_val = current_data[column].max()
                        if max_val != min_val:
                            current_data[column] = (current_data[column] - min_val) / (max_val - min_val)
                            operations_performed.append(f'{column}: Applied Min-Max scaling')
        
        # Invalidate preview cache since data may have changed
        invalidate_preview_cache()
        
        # Track the operation
        track_operation('column_standardization',
                       f'Applied {len(operations_performed)} standardization operations',
                       {
            'renamed_columns': column_mapping if column_mapping else {},
            'data_standardization': data_standardization if data_standardization else {},
            'operations_performed': operations_performed,
            'total_operations': len(operations_performed),
            'total_columns': len(current_data.columns)
        })
        
        return jsonify({
            'message': f'Successfully applied {len(operations_performed)} standardization operations',
            'operations_performed': operations_performed,
            'renamed_columns': column_mapping if column_mapping else {},
            'data_standardization': data_standardization if data_standardization else {},
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error standardizing columns: {str(e)}'}), 400

@app.route('/api/check-duplicates', methods=['POST'])
def check_duplicates():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        # Count total rows
        total_rows = len(current_data)
        
        # Count unique rows
        unique_count = len(current_data.drop_duplicates())
        
        # Count duplicate rows
        duplicate_count = total_rows - unique_count
        
        # Calculate percentage
        duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Find duplicate examples (first 5 groups)
        duplicate_examples = []
        if duplicate_count > 0:
            # Get duplicate rows with their indices
            duplicated_mask = current_data.duplicated(keep=False)
            duplicate_rows = current_data[duplicated_mask]
            
            # Group by actual values to find duplicate groups
            if len(duplicate_rows) > 0:
                # Create a string representation for grouping
                duplicate_rows_str = duplicate_rows.astype(str).apply(lambda x: '||'.join(x), axis=1)
                value_counts = duplicate_rows_str.value_counts()
                
                for i, (value, count) in enumerate(value_counts.head(5).items()):
                    # Find indices where this duplicate occurs
                    mask = duplicate_rows_str == value
                    indices = duplicate_rows[mask].index.tolist()
                    duplicate_examples.append({
                        'indices': indices,
                        'count': int(count)
                    })
        
        return jsonify({
            'total_rows': int(total_rows),
            'unique_count': int(unique_count),
            'duplicate_count': int(duplicate_count),
            'duplicate_percentage': float(duplicate_percentage),
            'duplicate_examples': duplicate_examples
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error checking duplicates: {str(e)}'}), 400

@app.route('/api/remove-duplicates', methods=['POST'])
def remove_duplicates():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        keep = data.get('keep', 'first')  # 'first' or 'last'
        
        # Store original shape
        original_shape = current_data.shape
        
        # Remove duplicates
        current_data = current_data.drop_duplicates(keep=keep)
        
        # Invalidate preview cache since rows were removed
        invalidate_preview_cache()
        
        # Calculate removed count
        removed_count = int(original_shape[0] - current_data.shape[0])
        
        # Track the operation
        track_operation('duplicate_removal',
                       f'Successfully removed {removed_count} duplicate rows',
                       {
            'original_shape': list(original_shape),
            'new_shape': list(current_data.shape),
            'rows_removed': removed_count,
            'keep_strategy': keep,
            'removal_percentage': (removed_count / original_shape[0] * 100) if original_shape[0] > 0 else 0
        })
        
        return jsonify({
            'message': f'Successfully removed {removed_count} duplicate rows',
            'original_shape': list(original_shape),
            'new_shape': list(current_data.shape),
            'rows_removed': removed_count,
            'keep_strategy': keep,
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error removing duplicates: {str(e)}'}), 400

@app.route('/api/analyze-skewness', methods=['POST'])
def analyze_skewness():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        from scipy import stats
        
        # Get numeric columns only
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            return jsonify({
                'total_numeric_columns': 0,
                'skewed_columns': 0,
                'normal_columns': 0,
                'columns': []
            }), 200
        
        column_analysis = []
        skewed_count = 0
        normal_count = 0
        
        for col in numeric_columns:
            # Remove NaN values for analysis
            col_data = current_data[col].dropna()
            
            if len(col_data) < 3:  # Need at least 3 values for skewness
                continue
                
            # Calculate skewness and kurtosis
            skewness = stats.skew(col_data)
            kurtosis = stats.kurtosis(col_data)
            
            # Determine if column is skewed (threshold: |skewness| >= 0.5)
            is_skewed = abs(skewness) >= 0.5
            
            if is_skewed:
                skewed_count += 1
            else:
                normal_count += 1
            
            # Recommend transformation based on skewness
            recommended_transformation = 'none'
            if is_skewed:
                if skewness > 0.5:  # Right-skewed
                    if skewness > 2:
                        recommended_transformation = 'boxcox'
                    elif skewness > 1:
                        recommended_transformation = 'log'
                    else:
                        recommended_transformation = 'sqrt'
                elif skewness < -0.5:  # Left-skewed
                    if skewness < -1:
                        recommended_transformation = 'square'
                    else:
                        recommended_transformation = 'yeojohnson'
            
            # Generate KDE plot with histogram for this column
            kde_plot = None
            try:
                if len(col_data) > 10:  # Need sufficient data for KDE
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create histogram
                    ax.hist(col_data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                    
                    # Create KDE overlay
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(col_data)
                    x_range = np.linspace(col_data.min(), col_data.max(), 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
                    
                    # Add vertical line for mean
                    mean_val = col_data.mean()
                    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                    
                    # Add vertical line for median
                    median_val = col_data.median()
                    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                    
                    ax.set_title(f'Distribution of {col} (Skewness: {skewness:.3f})')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    kde_plot = create_plot_base64(fig)
            except Exception as plot_error:
                logger.warning(f"Could not generate KDE plot for {col}: {plot_error}")
            
            column_analysis.append({
                'column': col,
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'is_skewed': bool(is_skewed),
                'recommended_transformation': recommended_transformation,
                'data_points': int(len(col_data)),
                'kde_plot': kde_plot
            })
        
        return jsonify({
            'total_numeric_columns': len(numeric_columns),
            'skewed_columns': skewed_count,
            'normal_columns': normal_count,
            'columns': column_analysis
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing skewness: {str(e)}'}), 400

@app.route('/api/apply-transformations', methods=['POST'])
def apply_transformations():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        from scipy import stats
        from scipy.stats import boxcox, yeojohnson
        
        data = request.json
        transformations = data.get('transformations', {})
        
        if not transformations:
            return jsonify({'error': 'No transformations specified'}), 400
        
        applied_transformations = []
        
        for column, transformation in transformations.items():
            if column not in current_data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(current_data[column]):
                continue
            
            col_data = current_data[column].copy()
            original_skewness = stats.skew(col_data.dropna())
            
            try:
                if transformation == 'log':
                    # Handle negative/zero values
                    min_val = col_data.min()
                    if min_val <= 0:
                        # Shift data to make all values positive
                        current_data[column] = np.log1p(col_data - min_val + 1)
                    else:
                        current_data[column] = np.log(col_data)
                    
                elif transformation == 'sqrt':
                    # Handle negative values
                    min_val = col_data.min()
                    if min_val < 0:
                        # Shift data to make all values non-negative
                        current_data[column] = np.sqrt(col_data - min_val)
                    else:
                        current_data[column] = np.sqrt(col_data)
                    
                elif transformation == 'reciprocal':
                    # Handle zero values
                    col_data_safe = col_data.replace(0, np.nan)
                    current_data[column] = 1 / col_data_safe
                    
                elif transformation == 'square':
                    current_data[column] = col_data ** 2
                    
                elif transformation == 'boxcox':
                    # Box-Cox requires positive data
                    if col_data.min() <= 0:
                        # Shift data to make it positive
                        shift = abs(col_data.min()) + 1
                        col_data_shifted = col_data + shift
                        transformed, lambda_param = boxcox(col_data_shifted.dropna())
                        # Apply transformation to all data (including NaN)
                        mask = ~col_data.isna()
                        current_data.loc[mask, column] = boxcox(col_data_shifted[mask], lmbda=lambda_param)
                    else:
                        transformed, lambda_param = boxcox(col_data.dropna())
                        mask = ~col_data.isna()
                        current_data.loc[mask, column] = boxcox(col_data[mask], lmbda=lambda_param)
                        
                elif transformation == 'yeojohnson':
                    # Yeo-Johnson can handle negative values and zeros
                    transformed, lambda_param = yeojohnson(col_data.dropna())
                    mask = ~col_data.isna()
                    current_data.loc[mask, column] = yeojohnson(col_data[mask], lmbda=lambda_param)
                
                # Calculate new skewness
                new_skewness = stats.skew(current_data[column].dropna())
                
                applied_transformations.append({
                    'column': column,
                    'transformation': transformation,
                    'original_skewness': float(original_skewness),
                    'new_skewness': float(new_skewness),
                    'improvement': float(abs(original_skewness) - abs(new_skewness))
                })
                
            except Exception as transform_error:
                applied_transformations.append({
                    'column': column,
                    'transformation': transformation,
                    'error': str(transform_error)
                })
        
        # Invalidate preview cache since data values changed
        invalidate_preview_cache()
        
        # Track the operation
        successful_transformations = [t for t in applied_transformations if "error" not in t]
        track_operation('skewness_transformation',
                       f'Applied {len(successful_transformations)} skewness transformations',
                       {
            'transformations_applied': len(successful_transformations),
            'total_requested': len(transformations),
            'transformation_details': successful_transformations,
            'columns_transformed': [t['column'] for t in successful_transformations],
            'methods_used': list(set([t['transformation'] for t in successful_transformations]))
        })
        
        return jsonify({
            'message': f'Applied {len([t for t in applied_transformations if "error" not in t])} transformations',
            'applied_transformations': applied_transformations,
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error applying transformations: {str(e)}'}), 400

@app.route('/api/analyze-encoding', methods=['POST'])
def analyze_encoding():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        result = {
            'categorical_columns': [],
            'text_columns': [],
            'datetime_columns': [],
            'numeric_columns': []
        }
        
        for column in current_data.columns:
            col_data = current_data[column].dropna()
            if len(col_data) == 0:
                continue
            
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(current_data[column]):
                result['datetime_columns'].append({
                    'column': column,
                    'format': 'datetime',
                    'date_range': f"{col_data.min()} to {col_data.max()}"
                })
            # Check if numeric
            elif pd.api.types.is_numeric_dtype(current_data[column]):
                result['numeric_columns'].append(column)
            # Check if text (long strings)
            elif col_data.dtype == 'object':
                # Calculate average string length
                str_lengths = col_data.astype(str).str.len()
                avg_length = str_lengths.mean()
                max_length = str_lengths.max()
                
                if avg_length > 20:  # Likely text data
                    result['text_columns'].append({
                        'column': column,
                        'avg_length': float(avg_length),
                        'max_length': int(max_length),
                        'sample_values': col_data.head(3).tolist()
                    })
                else:  # Likely categorical
                    unique_values = col_data.unique()
                    value_counts = col_data.value_counts()
                    
                    result['categorical_columns'].append({
                        'column': column,
                        'unique_count': int(len(unique_values)),
                        'sample_values': unique_values[:10].tolist(),
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing encoding options: {str(e)}'}), 400

@app.route('/api/apply-encoding', methods=['POST'])
def apply_encoding():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        from sklearn.preprocessing import LabelEncoder 
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        import hashlib
        
        data = request.json
        operations = data.get('operations', [])
        
        if not operations:
            return jsonify({'error': 'No encoding operations specified'}), 400
        
        applied_operations = []
        
        for operation in operations:
            column = operation.get('column')
            method = operation.get('method')
            
            if column not in current_data.columns:
                continue
            
            col_data = current_data[column].copy()
            
            try:
                if method == 'label':
                    # Label encoding
                    le = LabelEncoder()
                    # Handle NaN values
                    mask = col_data.notna()
                    current_data.loc[mask, column] = le.fit_transform(col_data[mask])
                    applied_operations.append(f'{column}: Label encoded with {len(le.classes_)} classes')
                    
                elif method == 'onehot':
                    # One-hot encoding
                    encoded_df = pd.get_dummies(col_data, prefix=column, dummy_na=True)
                    # Remove original column and add encoded columns
                    current_data = current_data.drop(columns=[column])
                    current_data = pd.concat([current_data, encoded_df], axis=1)
                    applied_operations.append(f'{column}: One-hot encoded into {len(encoded_df.columns)} columns')
                    
                elif method == 'ordinal':
                    # Ordinal encoding (assumes natural order)
                    unique_vals = sorted(col_data.dropna().unique())
                    ordinal_map = {val: idx for idx, val in enumerate(unique_vals)}
                    current_data[column] = col_data.map(ordinal_map)
                    applied_operations.append(f'{column}: Ordinal encoded with {len(unique_vals)} levels')
                    
                elif method == 'binary':
                    # Binary encoding
                    unique_vals = col_data.dropna().unique()
                    # Convert to binary representation
                    import math
                    n_bits = math.ceil(math.log2(len(unique_vals))) if len(unique_vals) > 1 else 1
                    
                    val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
                    
                    # Create binary columns
                    for bit in range(n_bits):
                        col_name = f'{column}_bit_{bit}'
                        current_data[col_name] = col_data.map(val_to_int).apply(
                            lambda x: (x >> bit) & 1 if pd.notna(x) else np.nan
                        )
                    
                    # Remove original column
                    current_data = current_data.drop(columns=[column])
                    applied_operations.append(f'{column}: Binary encoded into {n_bits} bit columns')
                    
                elif method == 'tfidf':
                    # TF-IDF vectorization (for text columns)
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                    text_data = col_data.fillna('').astype(str)
                    
                    tfidf_matrix = vectorizer.fit_transform(text_data)
                    feature_names = [f'{column}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                    
                    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                                          columns=feature_names, 
                                          index=current_data.index)
                    
                    # Remove original column and add TF-IDF features
                    current_data = current_data.drop(columns=[column])
                    current_data = pd.concat([current_data, tfidf_df], axis=1)
                    applied_operations.append(f'{column}: TF-IDF vectorized into {len(feature_names)} features')
                    
                elif method == 'countvec':
                    # Count vectorization
                    vectorizer = CountVectorizer(max_features=50, stop_words='english')
                    text_data = col_data.fillna('').astype(str)
                    
                    count_matrix = vectorizer.fit_transform(text_data)
                    feature_names = [f'{column}_count_{i}' for i in range(count_matrix.shape[1])]
                    
                    count_df = pd.DataFrame(count_matrix.toarray(), 
                                          columns=feature_names, 
                                          index=current_data.index)
                    
                    # Remove original column and add count features
                    current_data = current_data.drop(columns=[column])
                    current_data = pd.concat([current_data, count_df], axis=1)
                    applied_operations.append(f'{column}: Count vectorized into {len(feature_names)} features')
                    
                elif method == 'hash':
                    # Hash encoding
                    hash_values = col_data.apply(
                        lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 1000000 
                        if pd.notna(x) else np.nan
                    )
                    current_data[column] = hash_values
                    applied_operations.append(f'{column}: Hash encoded')
                    
                elif method == 'datetime_features':
                    # Extract datetime features
                    dt_col = pd.to_datetime(col_data, errors='coerce')
                    
                    current_data[f'{column}_year'] = dt_col.dt.year
                    current_data[f'{column}_month'] = dt_col.dt.month
                    current_data[f'{column}_day'] = dt_col.dt.day
                    current_data[f'{column}_weekday'] = dt_col.dt.weekday
                    current_data[f'{column}_hour'] = dt_col.dt.hour
                    
                    # Remove original column
                    current_data = current_data.drop(columns=[column])
                    applied_operations.append(f'{column}: Extracted 5 datetime features')
                    
                elif method == 'timestamp':
                    # Convert to timestamp
                    dt_col = pd.to_datetime(col_data, errors='coerce')
                    current_data[column] = dt_col.astype('int64') // 10**9  # Unix timestamp
                    applied_operations.append(f'{column}: Converted to Unix timestamp')
                    
            except Exception as op_error:
                applied_operations.append(f'{column} ({method}): Error - {str(op_error)}')
        
        # Invalidate preview cache since data structure changed
        invalidate_preview_cache()
        
        # Track the operation
        successful_operations = [op for op in applied_operations if "Error" not in op]
        track_operation('data_encoding',
                       f'Applied {len(successful_operations)} encoding operations',
                       {
            'operations_applied': len(successful_operations),
            'total_requested': len(operations),
            'encoding_details': applied_operations,
            'columns_encoded': [op.get('column') for op in operations],
            'methods_used': list(set([op.get('method') for op in operations])),
            'shape_after_encoding': list(current_data.shape)
        })
        
        return jsonify({
            'message': f'Applied {len([op for op in applied_operations if "Error" not in op])} encoding operations',
            'applied_operations': applied_operations,
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error applying encoding: {str(e)}'}), 400

@app.route('/api/analyze-data-integrity', methods=['POST'])
def analyze_data_integrity():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        import re
        
        issues = []
        total_columns = len(current_data.columns)
        columns_with_issues = 0
        total_issues = 0
        
        for column in current_data.columns:
            col_data = current_data[column].dropna()
            if len(col_data) == 0:
                continue
            
            column_issues = []
            
            # Convert to string for pattern analysis
            str_data = col_data.astype(str)
            
            # Check for binary pattern violations
            if len(col_data.unique()) <= 10:  # Likely categorical
                value_counts = col_data.value_counts()
                
                # Check for binary-like patterns
                top_values = value_counts.head(2)
                if len(top_values) >= 2:
                    # Check if it looks like yes/no, true/false, 1/0 pattern
                    v1, v2 = top_values.index[:2]
                    binary_patterns = [
                        ('yes', 'no'), ('true', 'false'), ('1', '0'),
                        ('y', 'n'), ('t', 'f'), ('male', 'female'),
                        ('m', 'f'), ('active', 'inactive'), ('on', 'off')
                    ]
                    
                    is_binary_like = any(
                        (str(v1).lower(), str(v2).lower()) == pattern or 
                        (str(v2).lower(), str(v1).lower()) == pattern
                        for pattern in binary_patterns
                    )
                    
                    if is_binary_like and len(value_counts) > 2:
                        # Find outlier values
                        outlier_values = []
                        for val, count in value_counts.iloc[2:].items():
                            outlier_values.append({'value': str(val), 'count': int(count)})
                        
                        if outlier_values:
                            column_issues.append({
                                'pattern': 'binary',
                                'severity': 'high',
                                'expected_pattern': f'Binary values like "{v1}" and "{v2}"',
                                'problematic_values': outlier_values,
                                'total_affected_rows': int(sum(count for val, count in value_counts.iloc[2:].items())),
                                'suggestions': [
                                    f'Replace outliers with "{v1}" or "{v2}"',
                                    'Remove rows with inconsistent values',
                                    'Create a mapping for outlier values'
                                ]
                            })
            
            # Check for case inconsistency in text data
            if col_data.dtype == 'object':
                unique_lower = set(str_data.str.lower().unique())
                unique_original = set(str_data.unique())
                
                if len(unique_lower) < len(unique_original):
                    # Case inconsistency detected
                    case_issues = []
                    for lower_val in unique_lower:
                        matching_vals = [val for val in unique_original 
                                       if str(val).lower() == lower_val]
                        if len(matching_vals) > 1:
                            # Find the most common case version
                            case_counts = {val: (str_data == val).sum() for val in matching_vals}
                            sorted_cases = sorted(case_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            # Mark less frequent cases as issues
                            for val, count in sorted_cases[1:]:
                                case_issues.append({'value': str(val), 'count': int(count)})
                    
                    if case_issues:
                        column_issues.append({
                            'pattern': 'case_inconsistency',
                            'severity': 'medium',
                            'expected_pattern': 'Consistent letter casing',
                            'problematic_values': case_issues,
                            'total_affected_rows': int(sum(issue['count'] for issue in case_issues)),
                            'suggestions': [
                                'Standardize to lowercase',
                                'Standardize to title case',
                                'Use the most frequent case version'
                            ]
                        })
            
            # Check for whitespace issues
            if col_data.dtype == 'object':
                whitespace_issues = []
                for val in str_data.unique():
                    if str(val) != str(val).strip():
                        count = int((str_data == val).sum())
                        whitespace_issues.append({'value': repr(str(val)), 'count': count})
                
                if whitespace_issues:
                    column_issues.append({
                        'pattern': 'whitespace_issues',
                        'severity': 'low',
                        'expected_pattern': 'Values without leading/trailing whitespace',
                        'problematic_values': whitespace_issues,
                        'total_affected_rows': int(sum(issue['count'] for issue in whitespace_issues)),
                        'suggestions': [
                            'Strip leading and trailing whitespace',
                            'Normalize internal spacing'
                        ]
                    })
            
            # Check for numeric values in text columns
            if col_data.dtype == 'object':
                numeric_in_text = []
                for val in str_data.unique():
                    try:
                        float(str(val))
                        # If it doesn't raise an exception, it's numeric
                        if not re.match(r'^-?\d*\.?\d+$', str(val).strip()):
                            continue  # Skip if it's not a clean number
                        count = int((str_data == val).sum())
                        numeric_in_text.append({'value': str(val), 'count': count})
                    except (ValueError, TypeError):
                        continue
                
                # Only flag if there are both numeric and non-numeric values
                if numeric_in_text and len(numeric_in_text) < len(str_data.unique()):
                    non_numeric_count = len(str_data.unique()) - len(numeric_in_text)
                    if non_numeric_count > 0:
                        column_issues.append({
                            'pattern': 'numeric_in_text',
                            'severity': 'medium',
                            'expected_pattern': 'Consistent data type (all text or all numeric)',
                            'problematic_values': numeric_in_text,
                            'total_affected_rows': int(sum(issue['count'] for issue in numeric_in_text)),
                            'suggestions': [
                                'Convert column to numeric type',
                                'Replace numeric values with text equivalents',
                                'Split into separate columns'
                            ]
                        })
            
            # Add column issues to overall issues
            for issue in column_issues:
                issues.append({
                    'column': column,
                    **issue
                })
                total_issues += 1
            
            if column_issues:
                columns_with_issues += 1
        
        return jsonify({
            'total_columns': total_columns,
            'columns_with_issues': columns_with_issues,
            'clean_columns': total_columns - columns_with_issues,
            'total_issues': total_issues,
            'issues': issues
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing data integrity: {str(e)}'}), 400

@app.route('/api/fix-data-integrity', methods=['POST'])
def fix_data_integrity():
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:
        data = request.json
        column = data.get('column')
        action = data.get('action')
        replacement_value = data.get('replacement_value')
        
        if column not in current_data.columns:
            return jsonify({'error': f'Column {column} not found'}), 400
        
        original_shape = current_data.shape
        
        if action == 'replace':
            if replacement_value is None:
                return jsonify({'error': 'Replacement value is required'}), 400
            
            # Re-analyze the column to find problematic values
            col_data = current_data[column].dropna()
            str_data = col_data.astype(str)
            
            # Find problematic values (simplified logic for replacement)
            problematic_mask = pd.Series([False] * len(current_data))
            
            # For binary patterns, find non-binary values
            if len(col_data.unique()) > 2:
                value_counts = col_data.value_counts()
                top_2_values = value_counts.head(2).index.tolist()
                
                # Mark values not in top 2 as problematic
                problematic_mask = ~current_data[column].isin(top_2_values + [np.nan])
            
            # Replace problematic values
            current_data.loc[problematic_mask, column] = replacement_value
            affected_rows = int(problematic_mask.sum())
            
            message = f'Replaced {affected_rows} problematic values in column "{column}" with "{replacement_value}"'
            
        elif action == 'remove':
            # Remove rows with problematic values
            col_data = current_data[column].dropna()
            
            if len(col_data.unique()) > 2:
                value_counts = col_data.value_counts()
                top_2_values = value_counts.head(2).index.tolist()
                
                # Keep only rows with top 2 values or NaN
                keep_mask = current_data[column].isin(top_2_values) | current_data[column].isna()
                current_data = current_data[keep_mask]
            
            affected_rows = original_shape[0] - current_data.shape[0]
            message = f'Removed {affected_rows} rows with problematic values in column "{column}"'
            
        else:
            return jsonify({'error': 'Invalid action. Use "replace" or "remove"'}), 400
        
        return jsonify({
            'message': message,
            'original_shape': list(original_shape),
            'new_shape': list(current_data.shape),
            'affected_rows': int(affected_rows),
            'shape': list(current_data.shape),
            'columns': current_data.columns.tolist(),
            'filename': current_filename
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error fixing data integrity: {str(e)}")
        return jsonify({'error': f'Error fixing data integrity: {str(e)}'}), 400

@app.route('/api/save-changes', methods=['POST'])
def save_changes():
    global current_data, current_filename
    
    logger.info("=== SAVE CHANGES REQUEST RECEIVED ===")
    
    if current_data is None:
        logger.error("Save changes requested but no data available")
        return jsonify({'error': 'No data to save'}), 400
    
    try:
        # Get request data
        data = request.json or {}
        save_filename = data.get('filename')
        
        # Use original filename if no new filename provided
        if not save_filename:
            save_filename = current_filename or 'cleaned_data.csv'
        
        # Ensure filename has proper extension
        if not save_filename.endswith(('.csv', '.xlsx')):
            save_filename += '.csv'
        
        # Create cleaned data directory if it doesn't exist
        cleaned_dir = os.path.join(UPLOAD_FOLDER, 'cleaned')
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Save path
        save_path = os.path.join(cleaned_dir, save_filename)
        
        logger.info(f"Saving cleaned data to: {save_path}")
        logger.info(f"Data shape: {current_data.shape}")
        logger.info(f"Columns: {list(current_data.columns)}")
        
        # Save the data
        if save_filename.endswith('.csv'):
            current_data.to_csv(save_path, index=False)
        elif save_filename.endswith('.xlsx'):
            current_data.to_excel(save_path, index=False)
        
        # Verify file was saved
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            logger.info(f"✅ FILE SAVED SUCCESSFULLY: {save_path}")
            logger.info(f"Saved file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        else:
            logger.error(f"❌ File not found after save: {save_path}")
            return jsonify({'error': 'File save verification failed'}), 500
        
        # Generate summary of changes
        summary = {
            'original_filename': current_filename,
            'saved_filename': save_filename,
            'saved_path': save_path,
            'final_shape': current_data.shape,
            'final_columns': current_data.columns.tolist(),
            'file_size_mb': round(file_size / (1024*1024), 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("=== SAVE CHANGES COMPLETED SUCCESSFULLY ===")
        
        return jsonify({
            'message': 'Changes saved successfully',
            'summary': summary,
            'filename': current_filename,
            'shape': current_data.shape,
            'columns': current_data.columns.tolist()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error saving changes: {str(e)}")
        return jsonify({'error': f'Error saving changes: {str(e)}'}), 400

@app.route('/api/final-preview', methods=['GET'])
def final_preview():
    global current_data, current_filename
    
    logger.info("Final preview requested")
    
    if current_data is None:
        logger.warning("Final preview requested but no data available")
        return jsonify({'error': 'No data to preview'}), 400
    
    try:
        # Format data for display (convert dates to DD/MM/YYYY format)
        display_data = format_date_columns_for_display(current_data)
        
        # Get comprehensive dataset information
        # Handle NaN values for JSON serialization
        preview_data = display_data.head(10).fillna('').to_dict('records')
        
        # Calculate statistics
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = current_data.select_dtypes(include=['object']).columns.tolist()
        
        # Check for date columns
        date_columns = []
        for col in current_data.columns:
            if get_column_type(current_data[col]) == 'datetime':
                date_columns.append(col)
        
        # Missing values analysis
        missing_values = {}
        for col in current_data.columns:
            missing_count = current_data[col].isnull().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
        
        # Data quality metrics
        total_cells = current_data.shape[0] * current_data.shape[1]
        missing_cells = current_data.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Memory usage
        memory_usage = current_data.memory_usage(deep=True).sum() / (1024*1024)  # MB
        
        summary = {
            'filename': current_filename,
            'shape': current_data.shape,
            'columns': current_data.columns.tolist(),
            'data_types': {
                'numeric': len(numeric_columns),
                'categorical': len(categorical_columns),
                'date': len(date_columns),
                'total': len(current_data.columns)
            },
            'missing_values': missing_values,
            'quality_metrics': {
                'completeness_percentage': round(completeness, 2),
                'missing_cells': int(missing_cells),
                'total_cells': int(total_cells)
            },
            'memory_usage_mb': round(memory_usage, 2),
            'column_details': {
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'date_columns': date_columns
            }
        }
        
        logger.info(f"✅ Final preview generated:")
        logger.info(f"   - Shape: {current_data.shape}")
        logger.info(f"   - Completeness: {completeness:.2f}%")
        logger.info(f"   - Memory usage: {memory_usage:.2f} MB")
        
        return jsonify({
            'preview_data': preview_data,
            'summary': summary
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error generating final preview: {str(e)}")
        return jsonify({'error': f'Error generating final preview: {str(e)}'}), 400

@app.route('/api/generate-report', methods=['GET'])
def generate_report():
    global current_data, current_filename, cleaning_operations
    
    logger.info("Cleaning report generation requested")
    
    if current_data is None:
        logger.warning("Report requested but no data available")
        return jsonify({'error': 'No data to generate report for'}), 400
    
    try:
        # Calculate comprehensive statistics
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = current_data.select_dtypes(include=['object']).columns.tolist()
        
        # Missing values analysis
        missing_values = {}
        for col in current_data.columns:
            missing_count = current_data[col].isnull().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)
        
        # Data quality metrics
        total_cells = current_data.shape[0] * current_data.shape[1]
        missing_cells = current_data.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Memory usage
        memory_usage = current_data.memory_usage(deep=True).sum() / (1024*1024)  # MB
        
        # Generate insights based on data characteristics
        insights = []
        
        # Data completeness insight
        if completeness >= 95:
            insights.append({
                'type': 'success',
                'title': 'Excellent Data Quality',
                'description': f'Your dataset has {completeness:.1f}% completeness, indicating high data quality.',
                'recommendation': 'The dataset is ready for analysis and modeling.'
            })
        elif completeness >= 80:
            insights.append({
                'type': 'warning',
                'title': 'Good Data Quality',
                'description': f'Your dataset has {completeness:.1f}% completeness, which is acceptable for most analyses.',
                'recommendation': 'Consider addressing remaining missing values for better results.'
            })
        else:
            insights.append({
                'type': 'error',
                'title': 'Data Quality Concerns',
                'description': f'Your dataset has only {completeness:.1f}% completeness.',
                'recommendation': 'Significant data cleaning or collection may be needed before analysis.'
            })
        
        # Column distribution insight
        if len(numeric_columns) > len(categorical_columns):
            insights.append({
                'type': 'info',
                'title': 'Numeric-Heavy Dataset',
                'description': f'Your dataset contains {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns.',
                'recommendation': 'Well-suited for statistical analysis, machine learning, and quantitative modeling.'
            })
        elif len(categorical_columns) > len(numeric_columns):
            insights.append({
                'type': 'info',
                'title': 'Categorical-Heavy Dataset',
                'description': f'Your dataset contains {len(categorical_columns)} categorical and {len(numeric_columns)} numeric columns.',
                'recommendation': 'Consider encoding techniques for machine learning or focus on categorical analysis.'
            })
        else:
            insights.append({
                'type': 'info',
                'title': 'Balanced Dataset',
                'description': f'Your dataset has an equal mix of numeric ({len(numeric_columns)}) and categorical ({len(categorical_columns)}) columns.',
                'recommendation': 'Balanced structure allows for diverse analytical approaches.'
            })
        
        # Size insight
        total_rows = current_data.shape[0]
        if total_rows > 100000:
            insights.append({
                'type': 'info',
                'title': 'Large Dataset',
                'description': f'Your dataset contains {total_rows:,} rows.',
                'recommendation': 'Consider sampling techniques for faster processing or use distributed computing for analysis.'
            })
        elif total_rows < 1000:
            insights.append({
                'type': 'warning',
                'title': 'Small Dataset',
                'description': f'Your dataset contains only {total_rows:,} rows.',
                'recommendation': 'Small sample size may limit statistical power. Consider collecting more data if possible.'
            })
        
        # Generate summary of cleaning operations performed
        operation_summary = {
            'total_operations': len(cleaning_operations),
            'operations_by_type': {},
            'detailed_operations': cleaning_operations.copy()
        }
        
        # Count operations by type
        for op in cleaning_operations:
            op_type = op.get('type', 'unknown')
            operation_summary['operations_by_type'][op_type] = operation_summary['operations_by_type'].get(op_type, 0) + 1
        
        report = {
            'dataset_info': {
                'filename': current_filename,
                'shape': current_data.shape,
                'columns': current_data.columns.tolist(),
                'data_types': {
                    'numeric': len(numeric_columns),
                    'categorical': len(categorical_columns),
                    'total': len(current_data.columns)
                },
                'memory_usage_mb': round(memory_usage, 2)
            },
            'quality_metrics': {
                'completeness_percentage': round(completeness, 2),
                'missing_cells': int(missing_cells),
                'total_cells': int(total_cells),
                'missing_values_by_column': missing_values
            },
            'cleaning_summary': operation_summary,
            'insights': insights,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"✅ Cleaning report generated:")
        logger.info(f"   - Total operations: {len(cleaning_operations)}")
        logger.info(f"   - Data completeness: {completeness:.2f}%")
        logger.info(f"   - Insights generated: {len(insights)}")
        
        return jsonify(report), 200
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {str(e)}")
        return jsonify({'error': f'Error generating report: {str(e)}'}), 400

@app.route('/api/download-csv', methods=['GET'])
def download_csv():
    global current_data, current_filename
    
    logger.info("CSV download requested")
    
    if current_data is None:
        logger.warning("Download requested but no data available")
        return jsonify({'error': 'No data to download'}), 400
    
    try:
        # Create cleaned data directory if it doesn't exist
        cleaned_dir = os.path.join(UPLOAD_FOLDER, 'cleaned')
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Generate filename
        base_name = current_filename.rsplit('.', 1)[0] if current_filename else 'cleaned_data'
        download_filename = f"{base_name}_cleaned.csv"
        file_path = os.path.join(cleaned_dir, download_filename)
        
        # Save CSV file
        current_data.to_csv(file_path, index=False)
        
        logger.info(f"✅ CSV file prepared for download: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Return file for download
        return send_from_directory(
            cleaned_dir,
            download_filename,
            as_attachment=True,
            download_name=download_filename
        )
        
    except Exception as e:
        logger.error(f"❌ Error preparing CSV download: {str(e)}")
        return jsonify({'error': f'Error preparing CSV download: {str(e)}'}), 400

# Helper function to track cleaning operations
def track_operation(operation_type, description, details=None):
    global cleaning_operations
    
    operation = {
        'type': operation_type,
        'description': description,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'details': details or {}
    }
    
    cleaning_operations.append(operation)
    logger.info(f"📝 Tracked operation: {operation_type} - {description}")

if __name__ == '__main__':
    # Setup logging for server startup
    logger.info("=== STARTING DATA CLEANING APPLICATION SERVER ===")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB")
    logger.info(f"Allowed file extensions: {ALLOWED_EXTENSIONS}")
    
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload directory created/verified: {os.path.abspath(UPLOAD_FOLDER)}")
    
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    logger.info("Server ready to accept file uploads!")
    logger.info("=" * 50)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)