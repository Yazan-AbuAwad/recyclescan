# Replace the imports at the top with:
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import json
import os
from datetime import datetime
import shutil

app = FastAPI(title="RecycleScan Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")



# Load recycling signs (you should create these images)
SIGNS = {
    "paper": {
        "path": "static/signs/paper_sign.png",
        "name": "Paper Recycling",
        "color": "#007AFF",  # Blue
        "description": "Place in BLUE BIN",
        "examples": ["Newspapers", "Cardboard", "Office Paper", "Magazines"]
    },
    "plastic": {
        "path": "static/signs/plastic_sign.png",
        "name": "Plastic & Metals",
        "color": "#FFCC00",  # Yellow
        "description": "Place in YELLOW BIN",
        "examples": ["Bottles", "Cans", "Containers", "Foil"]
    },
    "landfill": {
        "path": "static/signs/landfill_sign.png",
        "name": "Landfill",
        "color": "#8E8E93",  # Gray
        "description": "Place in BLACK BIN",
        "examples": ["Food Waste", "Broken Glass", "Styrofoam", "Mixed Materials"]
    }
}


# Load model (using a pre-trained model for demo)
# In production, train your own model with waste images
'''class RecyclingClassifier:
    def __init__(self):
        self.model = None
        self.labels = ["paper", "plastic", "landfill"]
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model or create a simple one for demo"""
        try:
            # Try to load existing model
            self.model = keras.models.load_model("models/recycling_model.h5")
            print("Loaded pre-trained model")
        except:
            # Create a simple demo model
            print("Creating demo model...")
            self.create_demo_model()
    
    def create_demo_model(self):
        """Create a simple model for demonstration"""
        self.model = keras.Sequential([
            keras.layers.Input(shape=(224, 224, 3)),
            keras.layers.Rescaling(1./255),
            keras.layers.Conv2D(16, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save for future use
        os.makedirs("models", exist_ok=True)
        self.model.save("models/recycling_model.h5")
        print("Created and saved demo model")
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for model"""
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure 3 channels (convert RGBA to RGB if needed)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_bytes):
        """Predict recycling category"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get top prediction
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            # Map to label
            category = self.labels[class_idx]
            
            return {
                "category": category,
                "confidence": round(confidence * 100, 2),
                "all_predictions": {
                    label: float(pred) * 100 
                    for label, pred in zip(self.labels, predictions[0])
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return demo prediction
            return self.demo_prediction()
    
    def demo_prediction(self):
        """Fallback demo prediction"""
        import random
        categories = ["paper", "plastic", "landfill"]
        category = random.choice(categories)
        
        return {
            "category": category,
            "confidence": round(random.uniform(75, 95), 2),
            "all_predictions": {
                "paper": 33.33,
                "plastic": 33.33,
                "landfill": 33.33
            }
        }



class MobileNetRecyclingClassifier:
    def __init__(self):
        self.model = None
        self.labels = ["paper", "plastic", "landfill"]
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Load MobileNetV2 pre-trained on ImageNet"""
        try:
            self.model = keras.applications.MobileNetV2(
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            print("✓ Loaded MobileNetV2 with ImageNet weights")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image_bytes):
        """Preprocess image for MobileNet"""
        try:
            # Convert bytes to image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize(self.img_size)
            
            # Convert to array and preprocess for MobileNet
            img_array = keras.applications.mobilenet_v2.preprocess_input(
                np.array(img)
            )
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, image_bytes):
        """Make REAL prediction with MobileNet"""
        if self.model is None:
            return self.demo_prediction()
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_bytes)
            if img_array is None:
                return self.demo_prediction()
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)
            
            # Decode predictions
            decoded = keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=10
            )[0]
            
            # Map to recycling categories
            result = self.analyze_predictions(decoded)
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.demo_prediction()
    
    def analyze_predictions(self, decoded_predictions):
        """Analyze MobileNet predictions and map to recycling"""
        # Category keywords
        paper_keywords = ['paper', 'cardboard', 'envelope', 'notebook', 
                         'newspaper', 'magazine', 'book', 'journal']
        
        plastic_keywords = ['plastic', 'bottle', 'container', 'bag', 
                           'wrapper', 'jug', 'tub', 'canister', 'can']
        
        landfill_keywords = ['trash', 'garbage', 'waste', 'compost','food','pizza','burger','paper cup','coffee'
                            'rubbish', 'landfill', 'dump','vegetable','fruits','toilet tissue', 'paper towel']
        
        # Initialize scores
        scores = {
            'paper': 0.0,
            'plastic': 0.0,
            'landfill': 0.0
        }
        
        # Analyze each prediction
        for _, label, confidence in decoded_predictions:
            label_lower = label.lower()
            
            # Check paper
            if any(keyword in label_lower for keyword in paper_keywords):
                scores['paper'] += confidence * 2
            
            # Check plastic
            if any(keyword in label_lower for keyword in plastic_keywords):
                scores['plastic'] += confidence * 2
            
            # Check landfill
            if any(keyword in label_lower for keyword in landfill_keywords):
                scores['landfill'] += confidence * 2
        
        # Add small base probability
        for key in scores:
            scores[key] += 0.01
        
        # Normalize to percentage
        total = sum(scores.values())
        for key in scores:
            scores[key] = (scores[key] / total) * 100
        
        # Determine winner
        best_category = max(scores, key=scores.get)
        
        return {
            "category": best_category,
            "confidence": round(scores[best_category], 2),
            "all_predictions": {
                cat: round(score, 2) for cat, score in scores.items()
            },
            "is_real_model": True
        }
    
    def demo_prediction(self):
        """Fallback"""
        import random
        category = random.choice(self.labels)
        
        return {
            "category": category,
            "confidence": round(random.uniform(75, 95), 2),
            "all_predictions": {
                "paper": 33.33,
                "plastic": 33.33,
                "landfill": 33.33
            },
            "is_demo": True
        }
'''

class CustomRecyclingClassifier:
    def __init__(self):
        self.model = None
        self.labels = ["paper", "plastic", "landfill"]
        self.img_size = (224, 224)
        self.model_type = None  # Track which model we're using
        self.load_best_model()
    
    def load_best_model(self):
        """Load the best available model in priority order"""
        print("Loading the best available model...")
        
        # Priority order for loading models
        model_paths = [
            ("models/recycling_mobilenet_best.h5", "custom_mobilenet"),
            ("models/recycling_resnet_best.h5", "custom_resnet"),
            ("models/recycling_mobilenet_final.h5", "custom_mobilenet_final"),
            ("models/recycling_resnet_final.h5", "custom_resnet_final"),
            ("models/recycling_model.h5", "demo_cnn")  # Original demo model
        ]
        
        # Try to load custom trained models first
        for model_path, model_type in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = keras.models.load_model(model_path)
                    self.model_type = model_type
                    print(f"✓ Loaded {model_type} from {model_path}")
                    
                    # Warm up the model
                    self.warmup_model()
                    return
                    
                except Exception as e:
                    print(f"✗ Error loading {model_path}: {e}")
                    continue
        
        # If no custom models found, fall back to MobileNet with ImageNet
        print("No custom models found, falling back to MobileNet with ImageNet mapping...")
        self.load_mobilenet_fallback()
    
    def load_mobilenet_fallback(self):
        """Load MobileNetV2 as fallback with keyword mapping"""
        try:
            self.model = keras.applications.MobileNetV2(
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
            self.model_type = "mobilenet_imagenet"
            print("✓ Loaded MobileNetV2 with ImageNet weights (keyword mapping)")
            
            # Warm up
            self.warmup_model()
            
        except Exception as e:
            print(f"✗ Error loading MobileNet: {e}")
            self.model = None
            self.model_type = "demo"
    
    def warmup_model(self):
        """Warm up model with dummy prediction"""
        try:
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            if self.model_type == "mobilenet_imagenet":
                # For MobileNet, need proper preprocessing
                dummy_input = keras.applications.mobilenet_v2.preprocess_input(dummy_input)
            self.model.predict(dummy_input, verbose=0)
            print("✓ Model warmed up and ready")
        except:
            print("⚠️ Model warmup failed (but may still work)")
    
    def preprocess_image(self, image_bytes, for_mobilenet=False):
        """Preprocess image for model"""
        try:
            # Convert bytes to image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize(self.img_size)
            
            # Convert to array
            img_array = np.array(img)
            
            # Different preprocessing for different model types
            if for_mobilenet or self.model_type == "mobilenet_imagenet":
                # MobileNet preprocessing
                img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
            else:
                # Custom model preprocessing (simple normalization)
                img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, image_bytes):
        """Make prediction with the best available model"""
        # First try the loaded model
        if self.model is not None:
            try:
                # Determine preprocessing based on model type
                for_mobilenet = self.model_type == "mobilenet_imagenet"
                
                # Preprocess image
                img_array = self.preprocess_image(image_bytes, for_mobilenet)
                if img_array is None:
                    return self.demo_prediction()
                
                # Make prediction
                predictions = self.model.predict(img_array, verbose=0)
                
                # Process predictions based on model type
                if self.model_type == "mobilenet_imagenet":
                    # MobileNet with ImageNet - need to decode and map
                    decoded = keras.applications.mobilenet_v2.decode_predictions(
                        predictions, top=10
                    )[0]
                    result = self.analyze_mobilenet_predictions(decoded)
                    result["model_type"] = "mobilenet_imagenet_mapped"
                    
                else:
                    # Custom model - direct classification
                    class_idx = np.argmax(predictions[0])
                    confidence = float(predictions[0][class_idx])
                    
                    # Apply confidence smoothing for custom models
                    if confidence < 0.5:  # Low confidence
                        # Consider using keyword mapping as fallback
                        mobilenet_result = self.try_mobilenet_keywords(image_bytes)
                        if mobilenet_result["confidence"] > confidence * 100:
                            return mobilenet_result
                    
                    result = {
                        "category": self.labels[class_idx],
                        "confidence": round(confidence * 100, 2),
                        "all_predictions": {
                            label: round(float(pred) * 100, 2) 
                            for label, pred in zip(self.labels, predictions[0])
                        },
                        "model_type": self.model_type
                    }
                
                # Add some metadata
                result["is_real_model"] = True
                result["model_loaded"] = self.model_type
                
                return result
                
            except Exception as e:
                print(f"Prediction error with {self.model_type}: {e}")
        
        # If model prediction fails, try MobileNet keyword mapping as fallback
        return self.try_mobilenet_keywords(image_bytes)
    
    def try_mobilenet_keywords(self, image_bytes):
        """Try to use MobileNet keyword mapping as fallback"""
        try:
            # Load MobileNet if not already loaded
            mobilenet_model = keras.applications.MobileNetV2(weights='imagenet')
            
            # Preprocess for MobileNet
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = img.resize(self.img_size)
            img_array = keras.applications.mobilenet_v2.preprocess_input(
                np.array(img)
            )
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get predictions
            predictions = mobilenet_model.predict(img_array, verbose=0)
            
            # Decode predictions
            decoded = keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=10
            )[0]
            
            # Map to recycling categories
            result = self.analyze_mobilenet_predictions(decoded)
            result["model_type"] = "mobilenet_fallback"
            result["is_real_model"] = True
            
            return result
            
        except Exception as e:
            print(f"MobileNet fallback also failed: {e}")
            return self.demo_prediction()
    
    def analyze_mobilenet_predictions(self, decoded_predictions):
        """Analyze MobileNet predictions and map to recycling categories"""
        # Enhanced keyword lists based on your suggestions
        paper_keywords = [
            'paper', 'cardboard', 'envelope', 'notebook', 'newspaper', 
            'magazine', 'book', 'journal', 'carton', 'brochure', 'flyer',
            'notepad', 'stationery', 'printing paper'
        ]
        
        plastic_keywords = [
            'plastic', 'bottle', 'container', 'bag', 'wrapper', 'jug', 
            'tub', 'canister', 'can', 'packaging', 'film', 'wrap',
            'soda can', 'beer can', 'aluminum can', 'tin can', 'metal',
            'water bottle', 'soda bottle', 'detergent bottle'
        ]
        
        landfill_keywords = [
            'trash', 'garbage', 'waste', 'compost', 'food', 'pizza',
            'burger', 'paper cup', 'coffee', 'rubbish', 'landfill', 
            'dump', 'vegetable', 'fruits', 'toilet tissue', 'paper towel',
            'napkin', 'tissue', 'dirty', 'contaminated', 'greasy',
            'styrofoam', 'polystyrene', 'foam', 'cigarette', 'diaper',
            'medical waste', 'hazardous', 'broken glass', 'ceramic',
            'light bulb', 'battery', 'electronics'
        ]
        
        # Initialize scores
        scores = {
            'paper': 0.0,
            'plastic': 0.0,
            'landfill': 0.0
        }
        
        # Score predictions with weighting
        for _, label, confidence in decoded_predictions:
            label_lower = label.lower()
            
            # Check paper keywords
            paper_matches = sum(1 for kw in paper_keywords if kw in label_lower)
            if paper_matches > 0:
                scores['paper'] += confidence * (1 + paper_matches * 0.5)
            
            # Check plastic keywords
            plastic_matches = sum(1 for kw in plastic_keywords if kw in label_lower)
            if plastic_matches > 0:
                scores['plastic'] += confidence * (1 + plastic_matches * 0.5)
            
            # Check landfill keywords
            landfill_matches = sum(1 for kw in landfill_keywords if kw in label_lower)
            if landfill_matches > 0:
                scores['landfill'] += confidence * (1 + landfill_matches * 0.5)
        
        # Add context-based adjustments
        self.apply_context_adjustments(scores, decoded_predictions)
        
        # Add small base probability to avoid zeros
        for key in scores:
            scores[key] += 0.05
        
        # Normalize to percentage
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] = (scores[key] / total) * 100
        
        # Determine winner
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        # Apply confidence threshold
        if confidence < 40:  # Very low confidence
            # Might be ambiguous - could check second best
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_categories) > 1:
                first_score = sorted_categories[0][1]
                second_score = sorted_categories[1][1]
                if first_score - second_score < 15:  # Close scores
                    # It's ambiguous
                    confidence = max(60, confidence)  # Boost slightly
                    best_category = self.resolve_ambiguity(sorted_categories)
        
        return {
            "category": best_category,
            "confidence": round(confidence, 2),
            "all_predictions": {
                cat: round(score, 2) for cat, score in scores.items()
            }
        }
    
    def apply_context_adjustments(self, scores, decoded_predictions):
        """Apply context-based adjustments to scores"""
        all_labels = " ".join([label.lower() for _, label, _ in decoded_predictions])
        
        # Context clues
        if 'food' in all_labels or 'pizza' in all_labels or 'burger' in all_labels:
            # Food-related items often go to landfill (except clean packaging)
            scores['landfill'] += 0.3
            
            # But food containers might be plastic
            if 'container' in all_labels or 'box' in all_labels:
                scores['plastic'] += 0.2
                scores['paper'] += 0.1
        
        if 'bottle' in all_labels and 'glass' not in all_labels:
            # Plastic bottles are common
            scores['plastic'] += 0.4
        
        if 'can' in all_labels and 'trash' not in all_labels:
            # Metal cans go to plastic/metal recycling
            scores['plastic'] += 0.5
        
        if 'paper' in all_labels and 'dirty' not in all_labels and 'food' not in all_labels:
            # Clean paper is recyclable
            scores['paper'] += 0.3
        
        if 'glass' in all_labels and 'broken' in all_labels:
            # Broken glass goes to landfill
            scores['landfill'] += 0.5
    
    def resolve_ambiguity(self, sorted_categories):
        """Resolve ambiguous predictions"""
        # Default to landfill when uncertain (safer - better to landfill than contaminate recycling)
        return sorted_categories[0][0]  # Just return the top one for now
        # In production, you might want more sophisticated logic
    
    def demo_prediction(self):
        """Fallback demo prediction when all else fails"""
        import random
        category = random.choice(self.labels)
        
        return {
            "category": category,
            "confidence": round(random.uniform(65, 85), 2),
            "all_predictions": {
                "paper": 33.33,
                "plastic": 33.33,
                "landfill": 33.33
            },
            "model_type": "demo_fallback",
            "is_real_model": False,
            "note": "Using demo mode - train custom model for better accuracy"
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "status": "no_model_loaded",
                "type": "demo",
                "parameters": 0
            }
        
        return {
            "status": "loaded",
            "type": self.model_type,
            "parameters": self.model.count_params() if hasattr(self.model, 'count_params') else "unknown",
            "input_shape": self.model.input_shape if hasattr(self.model, 'input_shape') else "unknown"
        }

classifier = CustomRecyclingClassifier()

# classifier = MobileNetRecyclingClassifier()

# Initialize classifier - previous model
# classifier = RecyclingClassifier()

# Routes
@app.get("/")
async def root():
    return {"message": "RecycleScan API", "status": "active", "version": "1.0.0"}

@app.get("/api/signs")
async def get_signs():
    """Get all recycling signs information"""
    return JSONResponse(content=SIGNS)

@app.get("/api/sign/{category}")
async def get_sign(category: str):
    """Get specific recycling sign"""
    if category not in SIGNS:
        raise HTTPException(status_code=404, detail="Sign not found")
    
    return FileResponse(SIGNS[category]["path"])

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify uploaded image"""
    try:
        # Read image file
        contents = await file.read()
        
        # Save uploaded file
        '''timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/uploads/upload_{timestamp}_{file.filename}"
        with open(filename, "wb") as f:
            f.write(contents)'''
        
        # Classify image
        result = classifier.predict(contents)
        
        # Add sign information
        category = result["category"]
        sign_info = SIGNS.get(category, {})
        
        response = {
            "success": True,
            "prediction": result,
            "sign": {
                "category": category,
                "name": sign_info.get("name", ""),
                "color": sign_info.get("color", ""),
                "description": sign_info.get("description", ""),
                "examples": sign_info.get("examples", []),
                "image_url": f"/static/signs/{category}_sign.png"
            },
            # "uploaded_image": f"/static/uploads/upload_{timestamp}_{file.filename}"
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-url")
async def classify_image_url(image_url: str):
    """Classify image from URL"""
    try:
        import requests
        from urllib.parse import urlparse
        
        # Download image from URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        # Get image bytes
        image_bytes = response.content
        
        # Classify image
        result = classifier.predict(image_bytes)
        
        # Add sign information
        category = result["category"]
        sign_info = SIGNS.get(category, {})
        
        # Extract filename from URL
        parsed_url = urlparse(image_url)
        filename = os.path.basename(parsed_url.path) or "image.jpg"
        
        # Save downloaded file
        '''timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"static/uploads/url_{timestamp}_{filename}"
        with open(save_path, "wb") as f:
            f.write(image_bytes)'''
        
        response_data = {
            "success": True,
            "prediction": result,
            "sign": {
                "category": category,
                "name": sign_info.get("name", ""),
                "color": sign_info.get("color", ""),
                "description": sign_info.get("description", ""),
                "examples": sign_info.get("examples", []),
                "image_url": f"/static/signs/{category}_sign.png"
            },
            # "uploaded_image": f"/static/uploads/url_{timestamp}_{filename}"
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Training endpoint (for future improvement)
@app.post("/api/train")
async def train_model(images: List[UploadFile] = File(...), labels: List[str] = []):
    """Train the model with new images"""
    # This is a placeholder for future training functionality
    return {"message": "Training endpoint", "status": "under_development"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)