import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# إعدادات التدريب
DATASET_PATH = "/home/eng/Downloads/archive(1)/chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
MODEL_SAVE_PATH = "pneu_d_models2.h5"

def analyze_data_distribution():
    """تحليل توزيع البيانات لفهم عدم التوازن"""
    train_normal = len(os.listdir(os.path.join(DATASET_PATH, '/home/eng/Downloads/archive(1)/chest_xray/train/NORMAL')))
    train_pneumonia = len(os.listdir(os.path.join(DATASET_PATH, '/home/eng/Downloads/archive(1)/chest_xray/train/PNEUMONIA')))
    test_normal = len(os.listdir(os.path.join(DATASET_PATH, '/home/eng/Downloads/archive(1)/chest_xray/test/NORMAL')))
    test_pneumonia = len(os.listdir(os.path.join(DATASET_PATH, '/home/eng/Downloads/archive(1)/chest_xray/test/PNEUMONIA')))
    
    print("=" * 50)
    print("تحليل توزيع البيانات:")
    print("=" * 50)
    print(f"صور التدريب - طبيعي: {train_normal}")
    print(f"صور التدريب - التهاب رئوي: {train_pneumonia}")
    print(f"صور الاختبار - طبيعي: {test_normal}")
    print(f"صور الاختبار - التهاب رئوي: {test_pneumonia}")
    
    total_train = train_normal + train_pneumonia
    total_test = test_normal + test_pneumonia
    
    print(f"\nنسب التدريب: طبيعي {train_normal/total_train*100:.1f}%، التهاب رئوي {train_pneumonia/total_train*100:.1f}%")
    print(f"نسب الاختبار: طبيعي {test_normal/total_test*100:.1f}%، التهاب رئوي {test_pneumonia/total_test*100:.1f}%")
    print("=" * 50)
    
    return train_normal, train_pneumonia, test_normal, test_pneumonia

# تحميل البيانات وتجهيزها
def load_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

# بناء النموذج باستخدام Transfer Learning
def build_model():
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # تجميد الطبقات الأساسية في البداية
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        Dropout(0.6),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# حساب أوزان الفئات لمعالجة عدم التوازن
def get_class_weights(generator):
    classes = generator.classes
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(class_weights))

# تحليل مفصل للأداء
def detailed_analysis(model, test_gen):
    print("=" * 50)
    print("تحليل مفصل لأداء النموذج:")
    print("=" * 50)
    
    # تقييم النموذج
    test_results = model.evaluate(test_gen)
    print(f"النتائج النهائية:")
    print(f"الخسارة: {test_results[0]:.4f}")
    print(f"الدقة: {test_results[1]*100:.2f}%")
    print(f"الدقة (Precision): {test_results[2]*100:.2f}%")
    print(f"الاستدعاء (Recall): {test_results[3]*100:.2f}%")
    if len(test_results) > 4:
        print(f"منطقة تحت المنحنى (AUC): {test_results[4]*100:.2f}%")
    
    # إنشاء تقرير مفصل
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int)
    
    print("\nتقرير التصنيف المفصل:")
    print(classification_report(test_gen.classes, predicted_classes, 
                               target_names=['NORMAL', 'PNEUMONIA']))
    
    # رسم مصفوفة الارتباك
    cm = confusion_matrix(test_gen.classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.ylabel('التصنيف الفعلي')
    plt.xlabel('التصنيف المتوقع')
    plt.title('مصفوفة الارتباك')
    plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # حساب F1-score
    from sklearn.metrics import f1_score
    f1 = f1_score(test_gen.classes, predicted_classes)
    print(f"F1-Score: {f1:.4f}")
    
    return test_results

# تدريب النموذج
def train_model():
    # تحليل توزيع البيانات أولاً
    analyze_data_distribution()
    
    print("جاري تحميل البيانات...")
    train_gen, val_gen, test_gen = load_data()
    
    print("جاري بناء النموذج...")
    model = build_model()
    
    # حساب أوزان الفئات
    class_weights = get_class_weights(train_gen)
    print(f"أوزان الفئات: {class_weights}")
    
    # تجميع النموذج
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC()]
    )
    
    # إعداد callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    print("بدء التدريب الأولي...")
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # إزالة التجميد من بعض الطبقات للتدريب الدقيق
    base_model = model.layers[0]
    base_model.trainable = True
    
    # تجميد أول 150 طبقة
    for layer in base_model.layers[:150]:
        layer.trainable = False
    
    # إعادة تجميع النموذج بمعدل تعلم أقل
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC()]
    )
    
    # استمرار التدريب
    print("بدء التدريب الدقيق...")
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=15,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # تحليل مفصل للأداء
    test_results = detailed_analysis(model, test_gen)
    
    # حفظ النموذج النهائي
    model.save(MODEL_SAVE_PATH)
    print(f"تم حفظ النموذج في: {MODEL_SAVE_PATH}")
    
    return model, history, test_results

if __name__ == "__main__":
    # تدريب النموذج
    model, history, test_results = train_model()