# Crop and Segment Tools

インスタンスセグメンテーションデータセットの処理とYOLO形式データセット分割のための総合ツールセットです。

## 📚 目次

- [概要](#概要)
- [必要なライブラリ](#必要なライブラリ)
- [クイックスタート](#クイックスタート)
- [ツール1: Image Tile Slicer](#ツール1-image-tile-slicer)
- [ツール2: Crop and Segment Tool](#ツール2-crop-and-segment-tool)
- [ツール3: YOLO Dataset Splitter](#ツール3-yolo-dataset-splitter)
- [完全なワークフロー](#完全なワークフロー)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

このプロジェクトには3つの主要なツールが含まれています：

### 🔹 slice_pic.py
X-AnyLabeling形式の画像をタイル分割します。オーバーラップとパディングに対応し、ラベルも正確に変換します。連番出力で上書きを防止し、視覚化モード（--vis）でタイル分割をプレビューできます。

### 🔹 crop_and_segment.py
X-AnyLabeling形式のインスタンスセグメンテーションラベルから各インスタンスをクロップし、クロップ画像の座標系で新しいラベルを作成します。

### 🔹 yolo_split.py
YOLO形式のデータセットをtrain/val/testに分割し、Ultralytics（YOLOv8）用の`data.yaml`を自動生成します。

---

## 必要なライブラリ

```bash
# 仮想環境を作成（例：venv）
python -m venv venv

# 仮想環境を有効化
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows Command Prompt:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# ライブラリをインストール
pip install -r requirements.txt
```

**必要なライブラリ：**
- `numpy>=1.21.0`: 座標計算・配列操作用
- `opencv-python>=4.5.0`: 画像の読み込み・保存・クロップ処理用
- `shapely>=2.0.0`: ポリゴン演算用（タイル分割時のクリッピング）
- `tqdm>=4.62.0`: 進捗表示用
- `scikit-learn>=1.0.0`: データセット分割用（train_test_split）
- `pyyaml>=6.0`: data.yaml生成用

---

## クイックスタート

### 🚀 1分で始める

```bash
# 1. ライブラリをインストール
pip install -r requirements.txt

# 2. （オプション）画像をタイル分割（slice_pic.py）
# --vis で視覚化、--clear_pad 0.5 でパディング50%以上のタイルを削除
python slice_pic.py -i PodSegDataset/train02 -o PodSegDataset/sliced --vis --clear_pad 0.5

# 3. インスタンスをクロップ（crop_and_segment.py）
python crop_and_segment.py -i PodSegDataset/train -o PodSegDataset/crop

# 4. （別途）YOLO形式に変換
# convert_to_yolo.py などを使用

# 5. データセットを分割（yolo_split.py）
python yolo_split.py -i Yolo_crop -o Yolo_split --class-names pod

# 6. YOLOv8でトレーニング
cd Yolo_split
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

---

# ツール1: Image Tile Slicer

X-AnyLabeling形式の画像をタイル分割します。オーバーラップとパディングに対応し、ラベルも正確に変換します。

## 使い方

### 基本的な使い方

```bash
# デフォルト設定で実行（640x640、オーバーラップ50px）
python slice_pic.py

# 入力と出力を指定
python slice_pic.py -i ./dataset/train -o ./dataset/sliced

# タイルサイズとオーバーラップを指定
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --tile-size 1024 --overlap 100

# パディング面積が50%以上のタイルを削除
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --clear_pad 0.5

# パディング面積が75%以上のタイルを削除（視覚化付き）
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --clear_pad 0.75 --vis
```

### 視覚化モード

`--vis`オプションを付けると、通常のタイル分割に加えて視覚化画像も生成されます：

```bash
# 視覚化モード（タイル分割 + プレビュー画像を生成）
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --vis
```

**出力構造:**
```
dataset/
└── sliced/
    ├── xxx_tile_r0_c0.jpg      # 通常のタイル画像
    ├── xxx_tile_r0_c0.json     # ラベル
    ├── xxx_tile_r0_c1.jpg
    ├── ...
    └── visualization/          # 視覚化画像
        ├── xxx_visualization.jpg
        └── yyy_visualization.jpg
```

視覚化画像では以下の情報が表示されます：
- **青色の矩形**: タイルの境界
- **緑色の領域**: オーバーラップ領域（半透明）
- **赤色の領域**: パディング領域（半透明）
- **グレー背景**: パディング色（114,114,114）で埋められた領域
- **オレンジ色のタイル + 白色Xマーク + "SKIP"**: `--clear_pad`オプションで削除されるタイル

パディング領域も含めて正確に可視化されるため、画像サイズがタイルサイズの倍数でない場合でも、どのようにパディングされるか確認できます。

`--clear_pad`オプションと`--vis`オプションを併用すると、削除されるタイルが**オレンジ色の半透明オーバーレイ**、**白色の太いXマーク**、**白色の"SKIP"テキスト**で明確に表示されます。削除対象のタイルは通常のタイル（青色の枠線）と一目で区別できます。

### パディング面積によるタイル削除

`--clear_pad`オプションで、パディング面積の割合が一定以上のタイルを自動的に削除できます。

```bash
# パディング面積が50%以上のタイルを削除
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --clear_pad 0.5

# パディング面積が75%以上のタイルを削除
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --clear_pad 0.75
```

**指定方法:**
- 値は `0.0`~`1.0` の範囲で指定
- `0.5` = 50%、`0.75` = 75%、`1.0` = 100%
- 指定した割合以上のパディング面積を持つタイルが削除される

**パディング面積の計算:**
- タイル全体の面積 = タイル幅 × タイル高さ
- 有効領域の面積 = 元画像が実際に存在する領域
- パディング面積 = タイル全体 - 有効領域
- パディング割合 = パディング面積 ÷ タイル全体の面積

**使用例:**
- `--clear_pad 0.3`: パディングが30%以上のタイルを削除（緩い）
- `--clear_pad 0.5`: パディングが50%以上のタイルを削除（標準）
- `--clear_pad 0.7`: パディングが70%以上のタイルを削除（厳しい）

このオプションは、学習に不要なパディング領域の多いデータを削除し、データセットの品質を向上させます。

**視覚化と併用:**
```bash
# 視覚化と併用して削除されるタイルを確認
python slice_pic.py -i ./dataset/train -o ./dataset/sliced --clear_pad 0.5 --vis
```

視覚化画像では、削除されるタイルに**オレンジ色の半透明オーバーレイ**、**白色の太い枠線**、**中央に白色の大きなXマーク**、**白色の"SKIP"テキスト**が表示されます。削除対象のタイルは背景色が変わるため、通常のタイルと一目で区別でき、視覚的に非常にわかりやすくなっています。

### コマンドラインオプション

- `-i, --input`: 入力ディレクトリ（デフォルト: `PodSegDataset/train02`）
- `-o, --output`: 出力ディレクトリ（デフォルト: `PodSegDataset/sliced`）
- `-t, --tile-size`: タイルサイズ（例: 640 または 640x640）（デフォルト: 640）
- `-ov, --overlap`: オーバーラップのピクセル数（デフォルト: 50）
- `--vis`: 視覚化モード（タイル分割 + 視覚化画像を生成）
- `--clear_pad RATIO`: パディング面積の閾値（0.0~1.0）。この割合以上のパディングを含むタイルを削除（例: 0.5=50%, 0.75=75%）

## 特徴

- **連番ディレクトリ出力**: 実行するたびに自動で連番ディレクトリを作成（sliced → sliced02 → sliced03）し、上書きを防止
- **オーバーラップ対応**: 境界のオブジェクトを確実にキャプチャ
- **自動パディング**: 端数部分は自動でパディング（YOLO標準の背景色: 114,114,114）
- **ラベル変換**: ポリゴン座標を自動変換（Shapely使用）
- **空タイルスキップ**: オブジェクトが含まれないタイルは自動スキップ
- **パディング面積によるフィルタリング**: --clear_pad RATIOオプションでパディング面積が指定割合以上のタイルを削除（0.0~1.0で指定）
- **視覚化モード**: --visオプションでタイル分割と視覚化画像を同時生成（visualizationサブフォルダーに保存）
- **見やすい視覚化**: 削除対象タイルはオレンジ色の背景＋白色のXマークで一目で識別可能

## 出力形式

### 通常モード

実行するたびに連番ディレクトリが作成されます：

```
PodSegDataset/
├── sliced/                 # 1回目の実行
│   ├── VID_xxx_tile_r0_c0.jpg
│   ├── VID_xxx_tile_r0_c0.json
│   ├── VID_xxx_tile_r0_c1.jpg
│   └── ...
├── sliced02/               # 2回目の実行
│   ├── VID_yyy_tile_r0_c0.jpg
│   └── ...
└── sliced03/               # 3回目の実行
    └── ...
```

### 視覚化モード（--vis）

視覚化画像はvisualizationサブフォルダーに保存され、通常のタイル分割も同時に実行されます：

```
PodSegDataset/
└── sliced/
    ├── VID_xxx_tile_r0_c0.jpg      # タイル画像
    ├── VID_xxx_tile_r0_c0.json     # ラベル
    ├── VID_xxx_tile_r0_c1.jpg
    ├── VID_xxx_tile_r0_c1.json
    ├── ...
    └── visualization/              # 視覚化画像
        ├── VID_xxx_visualization.jpg
        ├── VID_yyy_visualization.jpg
        └── ...
```

---

# ツール2: Crop and Segment Tool

インスタンスセグメンテーションラベルから各インスタンスをクロップし、クロップ画像の座標系で新しいラベルを作成するツールです。

## 使い方

### 基本的な使い方

```bash
# デフォルト設定で実行
# 入力: PodSegDataset/train
# 出力: PodSegDataset/crop
python crop_and_segment.py
```

### オプション指定

```bash
# 入力と出力を指定
python crop_and_segment.py -i ./labels -o ./output

# 画像ディレクトリを別に指定（ラベルと画像が別の場所にある場合）
python crop_and_segment.py -i ./labels -img ./images -o ./output

# パディングを追加（バウンディングボックスの周りに余白を追加）
python crop_and_segment.py -i ./labels -o ./output -p 10

# すべてのオプションを指定
python crop_and_segment.py -i ./PodSegDataset/train -img ./images -o ./output/crops -p 5

# 背景を透明化（PNG形式で保存）
python crop_and_segment.py -i ./labels -o ./output -m --mask-buffer 5
```

### コマンドラインオプション

- `-i, --input`: 入力ラベルファイルのディレクトリまたはファイルパス（デフォルト: `PodSegDataset/train`）
- `-o, --output`: 出力ディレクトリ（デフォルト: `PodSegDataset/crop`）
- `-img, --image-dir`: 画像ファイルのディレクトリ（ラベルと別の場所にある場合）
- `-p, --padding`: クロップ時のパディング（ピクセル）（デフォルト: 0）
- `-m, --mask-background`: 背景をマスクして透明化する（PNG形式で保存）
- `-mb, --mask-buffer`: マスクのバッファーサイズ（ピクセル）- ポリゴンの周りに余白を追加（デフォルト: 5）
- `-h, --help`: ヘルプを表示

## 入力形式

### ディレクトリ構造（ラベルと画像が同じ場所）

```
PodSegDataset/
└── train/
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    └── image2.json
```

### ディレクトリ構造（ラベルと画像が別の場所）

```
project/
├── labels/
│   ├── image1.json
│   └── image2.json
└── images/
    ├── image1.jpg
    └── image2.jpg
```

この場合、以下のように実行：
```bash
python crop_and_segment.py -i ./labels -img ./images -o ./output
```

## 出力形式

各インスタンスがクロップされ、連番が付きます：

```
PodSegDataset/
└── crop/
    ├── VID_20220226_0924511_30_0.jpg      # 1つ目のインスタンス
    ├── VID_20220226_0924511_30_0.json
    ├── VID_20220226_0924511_30_1.jpg      # 2つ目のインスタンス
    ├── VID_20220226_0924511_30_1.json
    └── ...
```

各JSONファイルには、クロップ画像の座標系に変換された1つのインスタンスのラベルが含まれます。

## ラベルファイル形式

X-AnyLabeling形式のJSONファイルに対応しています：

```json
{
  "version": "2.4.4",
  "flags": {},
  "shapes": [
    {
      "label": "pod",
      "shape_type": "polygon",
      "flags": {},
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "description": null,
      "difficult": false,
      "attributes": {}
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 1280,
  "imageWidth": 720
}
```

## 背景マスク機能

`-m`または`--mask-background`オプションを使用すると、セグメンテーション領域以外の背景を透明化できます。

### 特徴

- **透明化**: ポリゴン領域の外側を透明にします
- **PNG形式**: 透明度をサポートするため、PNG形式で保存されます
- **バッファー**: `--mask-buffer`でポリゴンの周りに余白を追加できます

### マスクバッファーの効果

マスクバッファーは、ポリゴンを少し膨張させて、境界をソフトにします：

- **0px**: ポリゴンの境界ぴったりでカット（シャープな境界）
- **5px（デフォルト）**: ポリゴンから5ピクセル外側まで含める（適度な余白）
- **10px**: より大きな余白（柔らかい境界）

### 使用例

```bash
# 背景を透明化（デフォルトバッファー5px）
python crop_and_segment.py -i ./labels -o ./output -m

# バッファーサイズをカスタマイズ
python crop_and_segment.py -i ./labels -o ./output -m --mask-buffer 10

# パディングと組み合わせる
python crop_and_segment.py -i ./labels -o ./output -p 20 -m --mask-buffer 8
```

### 出力形式の違い

**背景マスクなし（デフォルト）**:
- 形式: JPG
- 背景: 元画像の背景がそのまま含まれる
- ファイル名例: `image_0.jpg`

**背景マスクあり**:
- 形式: PNG（透明度をサポート）
- 背景: 透明
- ファイル名例: `image_0.png`

---

# ツール3: YOLO Dataset Splitter

YOLO形式のデータセットをtrain/val/testに分割し、**Ultralytics用のdata.yamlを自動生成**します。scikit-learnの`train_test_split`を使用して効率的に分割します。

## 使い方

### 基本的な使い方

```bash
# デフォルト設定で実行
# 入力: Yolo_crop
# 出力: Yolo_split
# 比率: train=70%, val=20%, test=10%
python yolo_split.py
```

### オプション指定

```bash
# 入力と出力を指定
python yolo_split.py -i ./yolo_dataset -o ./yolo_split

# 分割比率をカスタマイズ（80/10/10）
python yolo_split.py -i ./yolo_dataset -o ./yolo_split --train 0.8 --val 0.1 --test 0.1

# testなし（train/valのみ）
python yolo_split.py -i ./yolo_dataset -o ./yolo_split --train 0.8 --val 0.2 --test 0

# クラス名を指定してdata.yamlを生成
python yolo_split.py -i ./yolo_dataset -o ./yolo_split --class-names pod flower leaf

# ランダムシードを指定（再現性のため）
python yolo_split.py -i ./yolo_dataset -o ./yolo_split --seed 123
```

### コマンドラインオプション

- `-i, --input`: 入力ディレクトリ（YOLO形式のデータセット）（デフォルト: `Yolo_crop`）
- `-o, --output`: 出力ディレクトリ（デフォルト: `Yolo_split`）
- `--train`: トレーニングセットの比率（デフォルト: 0.7）
- `--val`: 検証セットの比率（デフォルト: 0.2）
- `--test`: テストセットの比率（デフォルト: 0.1）
- `--seed`: ランダムシード（デフォルト: 42）
- `--class-names`: クラス名のリスト（例: `--class-names pod flower leaf`）省略時は自動生成
- `-h, --help`: ヘルプを表示

## 入力形式

YOLO形式のデータセット：

```
yolo_dataset/
├── image1.jpg
├── image1.txt  # ラベルファイル（YOLO形式）
├── image2.jpg
├── image2.txt
└── ...
```

### YOLO形式のラベルファイル

```
0 0.5 0.5 0.3 0.4
0 0.7 0.3 0.2 0.2
```

各行：`<class_id> <x_center> <y_center> <width> <height>`（すべて0-1で正規化）

## 出力形式

```
Yolo_split/
├── data.yaml          # ← 自動生成されるUltralytics用設定ファイル
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**重要**: `data.yaml`が自動的に生成されます！

## data.yaml の自動生成

`yolo_split.py`は、分割完了後に**自動的に`data.yaml`を生成**します。

### 自動生成される`data.yaml`の例

```yaml
path: D:/prog_py/crop_and_seg/Yolo_split
train: train/images
val: val/images
test: test/images
nc: 1
names:
- class0
```

### クラス名をカスタマイズする場合

```bash
# クラス名を指定
python yolo_split.py -i Yolo_crop -o Yolo_split --class-names pod
```

生成される`data.yaml`:
```yaml
path: D:/prog_py/crop_and_seg/Yolo_split
train: train/images
val: val/images
test: test/images
nc: 1
names:
- pod
```

複数クラスの場合:
```bash
python yolo_split.py -i Yolo_crop -o Yolo_split --class-names pod flower leaf
```

## YOLOv8との統合（Ultralytics）

生成された`data.yaml`を使ってそのままトレーニング可能：

```bash
# コマンドラインから
cd Yolo_split
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

または、Pythonスクリプトから:

```python
from ultralytics import YOLO

# モデルをロード
model = YOLO('yolov8n.pt')

# トレーニング（自動生成されたdata.yamlを使用）
model.train(
    data='Yolo_split/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

---

## 完全なワークフロー

### エンドツーエンドの処理例

```bash
# 1. 画像をタイル分割（大きな画像を小さいタイルに分割）
# --vis で視覚化、--clear_pad 0.5 でパディング50%以上のタイルを削除
python slice_pic.py -i PodSegDataset/train02 -o PodSegDataset/sliced --tile-size 640 --overlap 50 --vis --clear_pad 0.5

# 2. インスタンスをクロップ
python crop_and_segment.py -i PodSegDataset/sliced -o PodSegDataset/crop

# 3. （別途）YOLO形式に変換
# convert_to_yolo.py などを使用

# 4. データセットを分割（data.yaml も自動生成）
python yolo_split.py -i Yolo_crop -o Yolo_split --train 0.7 --val 0.2 --test 0.1 --class-names pod

# 5. YOLOv8でトレーニング（自動生成されたdata.yamlを使用）
cd Yolo_split
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

**ポイント**: 
- ステップ1で--visオプションを使えば、タイル分割と視覚化画像を同時に生成できます
- ステップ1で--clear_pad 0.5を指定すれば、パディング50%以上のタイルを自動削除できます
- ステップ4で`data.yaml`が自動生成されるので、手動で作成する必要はありません

---

## トラブルシューティング

### slice_pic.py 関連

#### ❌ ラベルファイル（.json）が見つかりません

入力ディレクトリにX-AnyLabeling形式のJSONファイルが存在することを確認してください。

#### ❌ 画像が見つかりません

ラベルファイル（JSON）と画像ファイル（JPG/PNG）が同じディレクトリにあることを確認してください。

#### 💡 視覚化モードで確認

タイル分割を実行する前に、`--vis`オプションで視覚化画像を生成し、タイルサイズとオーバーラップが適切か確認することをお勧めします。

### crop_and_segment.py 関連

#### ❌ 画像が見つからない場合

ラベルファイル（JSON）と画像ファイル（JPG/PNG）が同じディレクトリにあることを確認してください。

別のディレクトリにある場合：
```bash
python crop_and_segment.py -i ./labels -img ./images -o ./output
```

#### ❌ OpenCV エラー: !_img.empty()

バウンディングボックスのサイズが0、またはクロップ結果が空の画像です。このエラーは自動的にスキップされ、処理が継続されます。

### yolo_split.py 関連

#### ❌ エラー: 入力ディレクトリが存在しません

`-i`オプションで指定したパスを確認してください。

#### ⚠️ 有効な画像とラベルのペアが見つかりません

→ 以下を確認:
- 画像ファイルの拡張子が `.jpg`, `.jpeg`, `.png` のいずれか
- 各画像に対応する `.txt` ファイルが存在する
- ファイル名（拡張子を除く）が一致している
  - ✅ OK: `image1.jpg` と `image1.txt`
  - ❌ NG: `image1.jpg` と `image2.txt`

#### ⚠️ 警告: 分割比率の合計が1.0ではありません

`--train`, `--val`, `--test`の合計が1.0になるように調整してください。

### 一般的な問題

#### ライブラリのインポートエラー

仮想環境を有効化していることを確認し、再度インストール：
```bash
pip install -r requirements.txt
```

#### メモリ不足

大量のファイルや大きな画像を処理する場合、メモリ不足になる可能性があります。
その場合は、少量ずつ処理してください：

```bash
# 特定のファイルだけ処理
python crop_and_segment.py -i ./PodSegDataset/train/file1.json -o ./output
```

---

## 注意事項

### crop_and_segment.py
- 現在はポリゴン（polygon）形状のみに対応しています
- 画像形式はJPG、PNG、JPEGに対応しています
- ラベルファイルのエンコーディングはUTF-8です

### yolo_split.py
- **元データは変更されません**: すべてコピーで処理されます
- **ディスク容量**: 元データと同じ容量が必要です
- **分割比率**: 合計が1.0になるようにしてください
- **上書き注意**: 出力ディレクトリが既に存在する場合、内容が上書きされます

---

## プロジェクト構造

```
easy_yolo_converter/
├── slice_pic.py              # タイル分割ツール
├── crop_and_segment.py       # インスタンスクロップツール
├── yolo_split.py             # YOLOデータセット分割ツール
├── requirements.txt          # 必要なライブラリ
├── README.md                 # このファイル
├── PodSegDataset/           # サンプルデータセット
│   ├── train02/             # 学習データ（X-AnyLabeling形式）
│   ├── sliced/              # タイル分割結果
│   ├── val/                 # 検証データ
│   ├── test/                # テストデータ
│   └── crop/                # クロップ結果の出力先
└── Yolo_split/              # YOLO分割結果の出力先
    ├── data.yaml            # 自動生成される設定ファイル
    ├── train/
    ├── val/
    └── test/
```

---

## ライセンス

このプロジェクトは自由に使用できます。

## 参考リンク

- [Ultralytics YOLOv8公式ドキュメント](https://docs.ultralytics.com/)
- [YOLO形式について](https://docs.ultralytics.com/datasets/detect/)
- [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
