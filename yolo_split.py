#!/usr/bin/env python3
"""
YOLO Dataset Splitter

YOLO形式のデータセットをtrain/val/testに分割します。
scikit-learnのtrain_test_splitを使用して効率的に分割し、
Ultralytics用のdata.yamlを自動生成します。

Reference: https://docs.ultralytics.com/ja/datasets/segment/coco128-seg/#dataset-yaml
"""

import argparse
import shutil
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# 定数
DEFAULT_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
DEFAULT_INPUT_DIR = 'Yolo_crop'
DEFAULT_OUTPUT_DIR = 'Yolo_split'
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42


@dataclass
class ImageLabelPair:
    """画像とラベルのペア"""
    image_path: Path
    label_path: Path
    name: str


@dataclass
class SplitConfig:
    """データセット分割の設定"""
    input_dir: Path
    output_dir: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    class_names: Optional[List[str]] = None
    image_extensions: List[str] = None
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = DEFAULT_IMAGE_EXTENSIONS
    
    def validate(self) -> bool:
        """設定の妥当性をチェック"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            tqdm.write(f"⚠️ 警告: 分割比率の合計が {total_ratio:.3f} です（1.0であるべき）")
            return False
        return True


class YoloDatasetSplitter:
    """YOLO形式のデータセットをtrain/val/testに分割するクラス"""
    
    def __init__(self, config: SplitConfig):
        """
        初期化
        
        Args:
            config: 分割設定
        """
        self.config = config
        self.pairs: List[ImageLabelPair] = []
        self.train_pairs: List[ImageLabelPair] = []
        self.val_pairs: List[ImageLabelPair] = []
        self.test_pairs: List[ImageLabelPair] = []
        self.class_ids: Set[int] = set()
    
    def find_image_label_pairs(self) -> Tuple[List[str], List[str]]:
        """
        画像とラベルのペアを探す
        
        Returns:
            (ラベルがない画像リスト, 画像がないラベルリスト)
        """
        input_dir = self.config.input_dir
        
        # 画像ファイルを収集
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
        
        # 画像とラベルのペアを作成
        missing_labels = []
        for img_file in image_files:
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                self.pairs.append(ImageLabelPair(
                    image_path=img_file,
                    label_path=label_file,
                    name=img_file.stem
                ))
            else:
                missing_labels.append(img_file.name)
        
        # ラベルファイルのみ存在するケースをチェック
        all_labels = list(input_dir.glob("*.txt"))
        missing_images = []
        for label_file in all_labels:
            has_image = any(
                label_file.with_suffix(ext).exists() 
                for ext in self.config.image_extensions
            )
            if not has_image:
                missing_images.append(label_file.name)
        
        return missing_labels, missing_images
    
    def split_dataset(self) -> None:
        """データセットをtrain/val/testに分割"""
        if not self.pairs:
            return
        
        # まずtrain+valとtestに分割
        if self.config.test_ratio > 0:
            train_val_pairs, self.test_pairs = train_test_split(
                self.pairs,
                test_size=self.config.test_ratio,
                random_state=self.config.random_seed
            )
        else:
            train_val_pairs = self.pairs
            self.test_pairs = []
        
        # train+valをtrainとvalに分割
        if self.config.val_ratio > 0 and train_val_pairs:
            val_ratio_adjusted = self.config.val_ratio / (
                self.config.train_ratio + self.config.val_ratio
            )
            self.train_pairs, self.val_pairs = train_test_split(
                train_val_pairs,
                test_size=val_ratio_adjusted,
                random_state=self.config.random_seed
            )
        else:
            self.train_pairs = train_val_pairs
            self.val_pairs = []
    
    def copy_pairs(
        self, 
        pairs: List[ImageLabelPair], 
        split_name: str
    ) -> int:
        """
        画像とラベルのペアをコピー
        
        Args:
            pairs: コピーするペアのリスト
            split_name: 分割名（train/val/test）
        
        Returns:
            コピーした数
        """
        if not pairs:
            return 0
        
        output_dir = self.config.output_dir
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for pair in tqdm(pairs, desc=f"{split_name:5s}", ncols=100):
            shutil.copy2(pair.image_path, images_dir / pair.image_path.name)
            shutil.copy2(pair.label_path, labels_dir / pair.label_path.name)
        
        return len(pairs)
    
    def extract_class_ids(self) -> None:
        """すべてのラベルファイルからクラスIDを抽出"""
        all_pairs = self.train_pairs + self.val_pairs + self.test_pairs
        
        for pair in all_pairs:
            try:
                with open(pair.label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                self.class_ids.add(class_id)
            except Exception as e:
                tqdm.write(f"⚠️ ラベルファイル読み込みエラー: {pair.label_path.name} - {e}")
    
    def create_data_yaml(self) -> Optional[Path]:
        """
        Ultralytics用のdata.yamlファイルを作成
        
        Reference: https://docs.ultralytics.com/ja/datasets/segment/coco128-seg/#dataset-yaml
        
        Returns:
            作成したdata.yamlのパス（失敗時はNone）
        """
        if not self.class_ids:
            tqdm.write("⚠️ 警告: クラスIDが見つかりませんでした。data.yamlの生成をスキップします。")
            return None
        
        yaml_path = self.config.output_dir / 'data.yaml'
        sorted_class_ids = sorted(self.class_ids)
        
        # クラス名の辞書を作成
        if self.config.class_names and len(self.config.class_names) == len(sorted_class_ids):
            names_dict = {cls_id: name for cls_id, name in zip(sorted_class_ids, self.config.class_names)}
        else:
            if self.config.class_names:
                tqdm.write(f"⚠️ 警告: クラス名の数({len(self.config.class_names)})とクラスIDの数({len(sorted_class_ids)})が一致しません")
                tqdm.write(f"   自動生成されたクラス名を使用します")
            names_dict = {cls_id: f'class{cls_id}' for cls_id in sorted_class_ids}
        
        # Ultralytics形式のdata.yamlを構築
        # Reference: https://docs.ultralytics.com/ja/datasets/segment/coco128-seg/#dataset-yaml
        data = {
            'path': str(self.config.output_dir.absolute()).replace('\\', '/'),
            'train': 'train/images',
            'val': 'val/images',
            'names': names_dict  # 辞書形式（0: class_name）
        }
        
        # testセットがある場合は追加
        if self.test_pairs:
            data['test'] = 'test/images'
        
        # YAMLファイルに書き込み
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return yaml_path
    
    def print_statistics(self) -> None:
        """統計情報を表示"""
        total = len(self.pairs)
        if total == 0:
            return
        
        tqdm.write("\n分割結果:")
        tqdm.write(f"  Train: {len(self.train_pairs):4d}組 ({len(self.train_pairs)/total*100:5.1f}%)")
        tqdm.write(f"  Val:   {len(self.val_pairs):4d}組 ({len(self.val_pairs)/total*100:5.1f}%)")
        tqdm.write(f"  Test:  {len(self.test_pairs):4d}組 ({len(self.test_pairs)/total*100:5.1f}%)")
        tqdm.write(f"  合計:  {total:4d}組")
    
    def print_output_summary(self) -> None:
        """出力ディレクトリの概要を表示"""
        tqdm.write("\n出力ディレクトリの確認:")
        for split_name in ["train", "val", "test"]:
            images_dir = self.config.output_dir / split_name / "images"
            labels_dir = self.config.output_dir / split_name / "labels"
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob("*")))
                label_count = len(list(labels_dir.glob("*.txt")))
                tqdm.write(f"  📁 {split_name:5s}: 画像={image_count:4d}枚, ラベル={label_count:4d}個")
    
    def run(self) -> bool:
        """
        データセット分割の実行
        
        Returns:
            成功した場合True、失敗した場合False
        """
        # 設定の検証
        self.config.validate()
        
        # ヘッダー表示
        tqdm.write("=" * 60)
        tqdm.write("YOLO Dataset Splitter")
        tqdm.write("=" * 60)
        tqdm.write(f"入力: {self.config.input_dir}")
        tqdm.write(f"出力: {self.config.output_dir}")
        tqdm.write(f"分割比率: Train={self.config.train_ratio}, Val={self.config.val_ratio}, Test={self.config.test_ratio}")
        tqdm.write(f"ランダムシード: {self.config.random_seed}")
        tqdm.write("-" * 60)
        
        # ペアの検索
        tqdm.write("データセットを確認しています...")
        missing_labels, missing_images = self.find_image_label_pairs()
        
        # 結果表示
        tqdm.write(f"\n📊 データセット情報:")
        tqdm.write(f"   正常なペア: {len(self.pairs)}組")
        
        if missing_labels:
            tqdm.write(f"   ⚠️ ラベルがない画像: {len(missing_labels)}個")
            for name in missing_labels[:5]:
                tqdm.write(f"      - {name}")
            if len(missing_labels) > 5:
                tqdm.write(f"      （他 {len(missing_labels) - 5}個）")
        
        if missing_images:
            tqdm.write(f"   ⚠️ 画像がないラベル: {len(missing_images)}個")
            for name in missing_images[:5]:
                tqdm.write(f"      - {name}")
            if len(missing_images) > 5:
                tqdm.write(f"      （他 {len(missing_images) - 5}個）")
        
        if not self.pairs:
            tqdm.write("\n❌ エラー: 有効な画像とラベルのペアが見つかりません")
            return False
        
        tqdm.write("-" * 60)
        
        # データセット分割
        tqdm.write("\nデータセットを分割しています...")
        self.split_dataset()
        self.print_statistics()
        tqdm.write("-" * 60)
        
        # ファイルコピー
        tqdm.write(f"\nファイルを {self.config.output_dir} にコピーしています...\n")
        self.copy_pairs(self.train_pairs, "train")
        self.copy_pairs(self.val_pairs, "val")
        self.copy_pairs(self.test_pairs, "test")
        
        tqdm.write("\n" + "-" * 60)
        
        # 出力確認
        self.print_output_summary()
        tqdm.write("\n" + "-" * 60)
        
        # data.yaml生成
        tqdm.write("\ndata.yaml を生成しています...")
        self.extract_class_ids()
        
        if self.class_ids:
            tqdm.write(f"   検出されたクラスID: {sorted(self.class_ids)}")
            if self.config.class_names:
                tqdm.write(f"   使用するクラス名: {self.config.class_names}")
            
            yaml_path = self.create_data_yaml()
            if yaml_path:
                tqdm.write(f"   ✅ data.yaml を作成しました: {yaml_path}")
                
                # data.yamlの内容を表示
                tqdm.write("\n📄 data.yaml の内容:")
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        tqdm.write(f"   {line.rstrip()}")
        
        # 完了メッセージ
        tqdm.write("\n" + "=" * 60)
        tqdm.write("✅ 完了！")
        tqdm.write(f"   出力先: {self.config.output_dir}")
        if self.class_ids:
            tqdm.write(f"   data.yaml: {self.config.output_dir / 'data.yaml'}")
        tqdm.write("=" * 60)
        
        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='YOLO形式のデータセットをtrain/val/testに分割',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行（data.yamlも自動生成）
  python yolo_split.py
  
  # 入力と出力を指定
  python yolo_split.py -i ./yolo_dataset -o ./yolo_split
  
  # 分割比率をカスタマイズ（80/10/10）
  python yolo_split.py -i ./yolo_dataset -o ./yolo_split --train 0.8 --val 0.1 --test 0.1
  
  # testなし（train/valのみ）
  python yolo_split.py -i ./yolo_dataset -o ./yolo_split --train 0.8 --val 0.2 --test 0
  
  # クラス名を指定してdata.yamlを生成
  python yolo_split.py -i ./yolo_dataset -o ./yolo_split --class-names pod flower leaf
  
  # ランダムシードを指定
  python yolo_split.py -i ./yolo_dataset -o ./yolo_split --seed 123

参考:
  Ultralytics YOLO Dataset Format: https://docs.ultralytics.com/ja/datasets/segment/coco128-seg/#dataset-yaml
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f'入力ディレクトリ（YOLO形式のデータセット） (デフォルト: {DEFAULT_INPUT_DIR})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'出力ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f'トレーニングセットの比率 (デフォルト: {DEFAULT_TRAIN_RATIO})'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f'検証セットの比率 (デフォルト: {DEFAULT_VAL_RATIO})'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f'テストセットの比率 (デフォルト: {DEFAULT_TEST_RATIO})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'ランダムシード（再現性のため） (デフォルト: {DEFAULT_RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=None,
        help='クラス名のリスト（例: --class-names pod flower leaf）省略時は自動生成'
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの確認
    if not args.input.exists():
        tqdm.write(f"❌ エラー: 入力ディレクトリが存在しません: {args.input}")
        return
    
    # 設定を作成
    config = SplitConfig(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed,
        class_names=args.class_names
    )
    
    # スプリッターを実行
    splitter = YoloDatasetSplitter(config)
    splitter.run()


if __name__ == '__main__':
    main()
