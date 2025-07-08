from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input

def build_model(num_classes):
    base = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    
    # Food classification
    out_class = Dense(num_classes, activation='softmax', name='food_class')(x)
    
    # Calorie regression
    out_calorie = Dense(1, activation='linear', name='calories')(x)
    
    model = Model(inputs=base.input, outputs=[out_class, out_calorie])
    return model
