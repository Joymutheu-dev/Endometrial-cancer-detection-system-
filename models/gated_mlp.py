import tensorflow as tf
from tensorflow.keras import layers, models

def build_gated_mlp(input_shape, num_classes):
    """Build a gated MLP model for endometrial cancer classification."""
    inputs = layers.Input(shape=input_shape)
    
    # Flatten input (e.g., image features)
    x = layers.Flatten()(inputs)
    
    # Gated MLP block
    def gated_block(x, units):
        dense = layers.Dense(units, activation='relu')(x)
        gate = layers.Dense(units, activation='sigmoid')(x)
        return layers.Multiply()([dense, gate])
    
    # Multiple gated blocks
    x = gated_block(x, 512)
    x = layers.Dropout(0.3)(x)
    x = gated_block(x, 256)
    x = layers.Dropout(0.3)(x)
    x = gated_block(x, 128)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    model = build_gated_mlp(input_shape=(224, 224, 3), num_classes=3)
    model.summary()