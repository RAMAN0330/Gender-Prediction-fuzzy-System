import numpy as np
import skfuzzy as fuzz
from skimage import io, transform
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess image
def preprocess_image(image_path):
    try:
        img = io.imread(image_path)
        img_resized = transform.resize(img, (224, 224))
        return img_resized, img
    except Exception as e:
        logger.error(f"Error occurred during image preprocessing: {e}")
        return None, None

# Feature extraction using pre-trained CNN (VGG16)
def extract_features(image):
    try:
        # Mock gender classifier (returns random prediction)
        gender_prediction = np.random.choice(['male', 'female'])
        return gender_prediction
    except Exception as e:
        logger.error(f"Error occurred during feature extraction: {e}")
        return None


# Define fuzzy sets and rules for gender classification
def define_fuzzy_sets_and_rules():
    try:
        # Low, Medium, High membership for gender classification
        feature_range = np.arange(0, 1.01, 0.01)
        low = fuzz.trimf(feature_range, [0, 0, 0.5])
        medium = fuzz.trimf(feature_range, [0.25, 0.5, 0.75])
        high = fuzz.trimf(feature_range, [0.5, 1, 1])

        fuzzy_sets = {'low': low, 'medium': medium, 'high': high}

        rules = [
            (low, 'male'),
            (medium, 'undetermined'),
            (high, 'female')
        ]

        return fuzzy_sets, rules
    except Exception as e:
        logger.error(f"Error occurred during fuzzy sets and rules definition: {e}")
        return None, None


# Fuzzification
def fuzzification(features, fuzzy_sets):
    try:
        # Define mapping from string values to numeric values
        value_mapping = {'male': 0.0, 'female': 1.0}

        # Fuzzify the features based on fuzzy sets
        membership_degrees = {}
        for feature_name, feature_value in features.items():
            print("Feature Name:", feature_name)
            print("Feature Value:", feature_value)
            membership_degrees[feature_name] = {}
            # Map string values to numeric values
            if isinstance(feature_value, str):
                feature_value = value_mapping.get(feature_value, 0.5)  # Default to 0.5 if value not found
            for set_name, membership_func in fuzzy_sets.items():
                membership_degrees[feature_name][set_name] = fuzz.interp_membership(np.arange(0, 1.01, 0.01), membership_func, feature_value)
        return membership_degrees
    except Exception as e:
        logger.error(f"Error occurred during fuzzification: {e}")
        return None


# Inference
def fuzzy_inference(membership_degrees, rules):
    try:
        # Perform fuzzy inference using the defined rules and membership degrees
        activation_levels = {}
        for set_name in membership_degrees.keys():
            activation_levels[set_name] = 0
        for feature_name, membership in membership_degrees.items():
            for i, (mf, label) in enumerate(rules):
                if membership[feature_name][label] > activation_levels[label]:
                    activation_levels[label] = membership[feature_name][label]
        return activation_levels
    except Exception as e:
        logger.error(f"Error occurred during fuzzy inference: {e}")
        return None


# Defuzzification
def defuzzification(inference_result):
    try:
        # Defuzzify the inference result to obtain crisp output
        return max(inference_result, key=inference_result.get)
    except Exception as e:
        logger.error(f"Error occurred during desertification: {e}")
        return None


# Save model
def save_model(model, filename):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully as {filename}")
    except Exception as e:
        logger.error(f"Error occurred during model saving: {e}")


# Load model
def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error occurred during model loading: {e}")
        return None


# Main function
def main():
    try:
        # Example image path
        image_path = r"C:\Users\Satoshi\Downloads\download.jpeg"

        # Load and preprocess image
        image, original_image = preprocess_image(image_path)
        if image is None:
            return

        # Extract features (mock gender prediction)
        gender_prediction = extract_features(image)
        if gender_prediction is None:
            return

        print("Gender prediction:", gender_prediction)

        # Define fuzzy sets and rules
        fuzzy_sets, rules = define_fuzzy_sets_and_rules()
        if fuzzy_sets is None or rules is None:
            return

        # Fuzzification
        membership_degrees = fuzzification({'gender': gender_prediction}, fuzzy_sets)
        if membership_degrees is None:
            return

        # Debugging: Print membership degrees
        print("Membership degrees:", membership_degrees)

        # Fuzzy inference
        inference_result = fuzzy_inference(membership_degrees, rules)
        if inference_result is None:
            return

        # De-fuzzification
        classification_result = defuzzification(inference_result)
        if classification_result is None:
            return

        print("Gender classification result:", classification_result)


    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
