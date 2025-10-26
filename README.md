# Crop and Segment Tools

インスタンスセグメンテーションデータセットの処理とYOLO形式データセット分割のための総合ツールセットです。

## 📚 目次

- [概要](#概要)
- [必要なライブラリ](#必要なライブラリ)
- [クイックスタート](#クイックスタート)
- [ツール1: Crop and Segment Tool](#ツール1-crop-and-segment-tool)
- [ツール2: YOLO Dataset Splitter](#ツール2-yolo-dataset-splitter)
- [完全なワークフロー](#完全なワークフロー)
- [トラブルシューティング](#トラブルシューティング)

---

## 概要

このプロジェクトには2つの主要なツールが含まれています：

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
- `tqdm>=4.62.0`: プログレスバー表示用
- `scikit-learn>=1.0.0`: データセット分割用（train_test_split）
- `pyyaml>=6.0`: data.yaml生成用

---

## クイックスタート

### 🚀 1分で始める

```bash
# 1. ライブラリをインストール
pip install -r requirements.txt

# 2. インスタンスをクロップ（crop_and_segment.py）
python crop_and_segment.py -i PodSegDataset/train -o PodSegDataset/crop

# 3. （別途）YOLO形式に変換
# convert_to_yolo.py などを使用

# 4. データセットを分割（yolo_split.py）
python yolo_split.py -i Yolo_crop -o Yolo_split --class-names pod

# 5. YOLOv8でトレーニング
cd Yolo_split
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 📊 動作確認

サンプルファイルで動作確認：

```bash
# 1つのファイルだけ処理してみる
python crop_and_segment.py -i PodSegDataset/train/VID_20220226_0924511_30.json -o test_output
```

実行中はプログレスバーが表示されます：

```
ファイル処理: 100%|██████████| 1/1 [00:02<00:00] VID_20220226_0924511_30.json -> 49個
```

---

# ツール1: Crop and Segment Tool

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

## エラー処理

スクリプトは以下の問題を検出し、自動的にスキップして処理を継続します：

- **無効なバウンディングボックス**: サイズが0のバウンディングボックス
- **ポリゴンのポイント不足**: 3点未満のポリゴン
- **空のクロップ画像**: クロップ結果が空の画像
- **画像保存の失敗**: ファイル書き込みエラー
- **その他の予期しないエラー**: 各インスタンスで発生したエラー

処理完了後、スキップされたインスタンスの総数が表示されます：

```
全処理完了: 合計 19845個のインスタンスをクロップしました
スキップ: 23個のインスタンス（無効なデータ、エラーなど）
```

---

# ツール2: YOLO Dataset Splitter

YOLO形式のデータセットをtrain/val/testに分割し、**Ultralytics用のdata.yamlを自動生成**します。scikit-learnの`train_test_split`を使用して効率的に分割します。

## ✨ 特徴

- ✅ **scikit-learn**: 効率的で信頼性の高い`train_test_split`を使用
- ✅ **data.yaml自動生成**: Ultralytics（YOLOv8）用の設定ファイルを自動作成
- ✅ **CLI対応**: コマンドラインから簡単に実行
- ✅ **デフォルト設定**: すぐに使える便利なデフォルト値
- ✅ **確実**: ファイルペアの検証機能付き
- ✅ **プログレスバー**: 進捗状況を視覚的に確認
- ✅ **再現性**: ランダムシード設定で同じ分割を再現可能
- ✅ **安全**: 元データは変更せず、コピーのみ

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

## 実行例

```bash
$ python yolo_split.py -i Yolo_crop -o Yolo_split --class-names pod

============================================================
YOLO Dataset Splitter
============================================================
入力: Yolo_crop
出力: Yolo_split
分割比率: Train=0.7, Val=0.2, Test=0.1
ランダムシード: 42
------------------------------------------------------------
データセットを確認しています...

📊 データセット情報:
   正常なペア: 1000組
------------------------------------------------------------

データセットを分割しています...

分割結果:
  Train:  700組 ( 70.0%)
  Val:    200組 ( 20.0%)
  Test:   100組 ( 10.0%)
  合計:  1000組
------------------------------------------------------------

ファイルを Yolo_split にコピーしています...

train: 100%|████████████████████| 700/700 [00:05<00:00, 132.15it/s]
val:   100%|████████████████████| 200/200 [00:01<00:00, 135.21it/s]
test:  100%|████████████████████| 100/100 [00:00<00:00, 138.45it/s]

------------------------------------------------------------

出力ディレクトリの確認:
  📁 train: 画像= 700枚, ラベル= 700個
  📁 val:   画像= 200枚, ラベル= 200個
  📁 test:  画像= 100枚, ラベル= 100個

------------------------------------------------------------

data.yaml を生成しています...
   検出されたクラスID: [0]
   使用するクラス名: ['pod']
   ✅ data.yaml を作成しました: Yolo_split\data.yaml

📄 data.yaml の内容:
   path: D:/prog_py/crop_and_seg/Yolo_split
   train: train/images
   val: val/images
   test: test/images
   nc: 1
   names:
   - pod

============================================================
✅ 完了！
   出力先: Yolo_split
   data.yaml: Yolo_split\data.yaml
============================================================
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
# 1. インスタンスをクロップ
python crop_and_segment.py -i PodSegDataset/train -o PodSegDataset/crop

# 2. （別途）YOLO形式に変換
# convert_to_yolo.py などを使用

# 3. データセットを分割（data.yaml も自動生成）
python yolo_split.py -i Yolo_crop -o Yolo_split --train 0.7 --val 0.2 --test 0.1 --class-names pod

# 4. YOLOv8でトレーニング（自動生成されたdata.yamlを使用）
cd Yolo_split
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

**ポイント**: ステップ3で`data.yaml`が自動生成されるので、手動で作成する必要はありません！

---

## トラブルシューティング

### crop_and_segment.py 関連

#### ❌ 画像が見つからない場合

ラベルファイル（JSON）と画像ファイル（JPG/PNG）が同じディレクトリにあることを確認してください。

別のディレクトリにある場合：
```bash
python crop_and_segment.py -i ./labels -img ./images -o ./output
```

#### ❌ OpenCV エラー: !_img.empty()

バウンディングボックスのサイズが0、またはクロップ結果が空の画像です。このエラーは自動的にスキップされ、処理が継続されます。

#### ⚠️ プログレスバーの表示が崩れる

すべての出力は`tqdm.write()`を使用しているため、プログレスバーとの表示崩れは発生しません。

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
crop_and_seg/
├── crop_and_segment.py      # インスタンスクロップツール
├── yolo_split.py             # YOLOデータセット分割ツール
├── requirements.txt          # 必要なライブラリ
├── README.md                 # このファイル
├── PodSegDataset/           # サンプルデータセット
│   ├── train/               # 学習データ（X-AnyLabeling形式）
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
- [tqdm プログレスバーの使い方](https://qiita.com/kuroitu/items/f18acf87269f4267e8c1)
