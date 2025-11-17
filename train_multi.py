# train_multi.py  - Huấn luyện MobileNetV2 cho bộ nhiều lớp trái cây (2 giai đoạn)
# Hỗ trợ: --use_class_weight, --augment {none,light,strong}, --resume, --unfreeze_from
import argparse, json
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


def build_datasets(data_root: Path, img_size=224, batch_size=32, seed=123):
    """
    Đọc dữ liệu từ fruits_multi/{train,val,test}.
    Nếu thư mục test không có hoặc không có ảnh -> bỏ qua, không báo lỗi.
    """
    data_root = Path(data_root)

    def make(split):
        d = data_root / split
        if not d.exists():
            return None, []

        try:
            ds = tf.keras.utils.image_dataset_from_directory(
                d,
                labels="inferred",
                label_mode="int",
                image_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            )
        except ValueError as e:
            # Trường hợp thư mục tồn tại nhưng không có ảnh
            if "No images found in directory" in str(e):
                print(f"⚠ Thư mục {d} không có ảnh, bỏ qua split '{split}'.")
                return None, []
            else:
                raise

        return ds, ds.class_names

    tr, trn = make("train")
    va, van = make("val")
    te, ten = make("test")

    ref = trn or van or ten or []
    for ns in [trn, van, ten]:
        if ns and ns != ref:
            raise RuntimeError(
                f"Tên lớp không khớp.\ntrain:{trn}\nval:{van}\ntest:{ten}"
            )

    AUTOTUNE = tf.data.AUTOTUNE
    tune = (
        lambda ds: ds.cache().prefetch(AUTOTUNE)
        if ds is not None
        else None
    )
    return tune(tr), tune(va), tune(te), ref


def compute_class_weights(train_ds, num_classes):
    c = Counter()
    for _, y in train_ds.unbatch():
        c[int(y.numpy())] += 1
    total = sum(c.values())
    w = {i: total / (num_classes * max(1, c.get(i, 1))) for i in range(num_classes)}
    return w, c


def make_augmenter(mode):
    mode = (mode or "none").lower()
    if mode == "none":
        return tf.keras.Sequential([], name="aug_none")
    if mode == "light":
        return tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
            ],
            name="aug_light",
        )
    if mode == "strong":
        return tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2),
            ],
            name="aug_strong",
        )
    return tf.keras.Sequential([], name="aug_default")


def build_model(
    num_classes,
    img_size=224,
    augment="light",
    unfreeze_from=None,
    lr=1e-3,
    finetune=False,
):
    """
    Xây dựng model MobileNetV2:
    - input -> augmenter -> preprocess_input -> MobileNetV2 backbone -> GAP -> Dense softmax
    """
    inp = layers.Input((img_size, img_size, 3), name="input_layer_1")
    x = make_augmenter(augment)(inp)

    # Tiền xử lý chuẩn của MobileNetV2 (scale [-1,1])
    # Dùng trực tiếp preprocess_input đã import để dễ đăng ký custom_objects
    x = layers.Lambda(preprocess_input, name="mobilenetv2_preproc")(x)

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    # Stage 1: backbone đóng băng
    base.trainable = False
    x = base(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dense")(x)
    model = models.Model(inp, out, name="fruit_mobilenetv2")

    # Nếu finetune=True và có unfreeze_from -> mở một phần backbone
    if finetune and unfreeze_from is not None:
        base.trainable = True
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= unfreeze_from

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_class_indices(out_dir: Path, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp = {str(i): cls for i, cls in enumerate(class_names)}
    (out_dir / "class_indices.json").write_text(
        json.dumps(mp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(a):
    base = Path.cwd()
    out_dir = base / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _, classes = build_datasets(
        base / a.data_root,
        img_size=a.img_size,
        batch_size=a.batch_size,
    )
    if train_ds is None or val_ds is None:
        raise RuntimeError("Thiếu fruits_multi/{train,val} hoặc không có ảnh.")

    num_classes = len(classes)
    print("Lớp:", classes)

    # Class weights (nếu bật)
    class_weight = None
    if a.use_class_weight:
        class_weight, counts = compute_class_weights(train_ds, num_classes)
        print("Class counts:", dict(counts))
        print("Class weights:", class_weight)

    # Lưu map class -> index
    save_class_indices(out_dir, classes)

    ckpt_best = callbacks.ModelCheckpoint(
        filepath=str(out_dir / "fruit_model.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    early = callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
    )

    # ===== Stage 1: train head (backbone frozen) =====
    if a.epochs_stage1 > 0 and not a.resume:
        print("== Stage 1: training head (backbone frozen) ==")
        model = build_model(
            num_classes,
            img_size=a.img_size,
            augment=a.augment,
            lr=a.lr,
            finetune=False,
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=a.epochs_stage1,
            class_weight=class_weight,
            callbacks=[ckpt_best, early],
            verbose=1,
        )
    else:
        print("Bỏ qua Stage 1.")

    # ===== Load best model sau Stage 1 / khi resume =====
    if (out_dir / "fruit_model.keras").exists():
        # Đăng ký custom_objects cho Lambda(preprocess_input)
        model = tf.keras.models.load_model(
            out_dir / "fruit_model.keras",
            custom_objects={"preprocess_input": preprocess_input},
        )
    else:
        model = build_model(
            num_classes,
            img_size=a.img_size,
            augment=a.augment,
            lr=a.lr,
            finetune=False,
        )

    # ===== Stage 2: fine-tune backbone =====
    if a.epochs_stage2 > 0:
        print("== Stage 2: fine-tune backbone ==")

        # Tìm MobileNetV2 backbone bên trong model
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and layer.name.startswith(
                "mobilenetv2"
            ):
                base_model = layer
                break

        if base_model is not None:
            base_model.trainable = True
            if a.unfreeze_from is not None:
                for i, layer in enumerate(base_model.layers):
                    layer.trainable = i >= a.unfreeze_from

        model.compile(
            optimizer=optimizers.Adam(a.lr_finetune),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=a.epochs_stage2,
            class_weight=class_weight,
            callbacks=[ckpt_best, early],
            verbose=1,
        )
    else:
        print("Bỏ qua Stage 2.")

    print(" Model tốt nhất:", out_dir / "fruit_model.keras")
    print(" Class map:", out_dir / "class_indices.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="fruits_multi")
    ap.add_argument("--out_dir", default="outputs_multi")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR cho Stage 1")
    ap.add_argument("--lr_finetune", type=float, default=3e-4, help="LR cho Stage 2")
    ap.add_argument("--epochs_stage1", type=int, default=8)
    ap.add_argument("--epochs_stage2", type=int, default=12)
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--augment", choices=["none", "light", "strong"], default="light")
    ap.add_argument("--unfreeze_from", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())

