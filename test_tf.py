try:
    import os
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    path = r"d:\CODING PROJECTS\forest fire prediction\models\wildfire_efficientnetb3_final.keras"
    model = tf.keras.models.load_model(path)
    print("LOADED OK!")
except Exception as e:
    print(f"FAILED WITH EXCEPTION TYPE: {type(e)}")
    print(f"EXCEPTION: {e}")
