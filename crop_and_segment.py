#!/usr/bin/env python3
"""
インスタンスセグメンテーションラベルから各インスタンスをクロップし、
クロップ画像の座標系で新しいラベルを作成するツール
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def load_label_file(label_path: Path) -> Dict:
    """ラベルファイル（JSON）を読み込む"""
    with open(label_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_bounding_box(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """ポリゴンの座標からバウンディングボックスを計算
    
    Returns:
        (x_min, y_min, x_max, y_max)
    """
    points_array = np.array(points)
    x_min = int(np.floor(points_array[:, 0].min()))
    y_min = int(np.floor(points_array[:, 1].min()))
    x_max = int(np.ceil(points_array[:, 0].max()))
    y_max = int(np.ceil(points_array[:, 1].max()))
    
    return x_min, y_min, x_max, y_max


def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """画像をバウンディングボックスでクロップ"""
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max].copy()


def create_polygon_mask(
    image_shape: Tuple[int, int],
    points: List[List[float]],
    buffer_size: int = 0
) -> np.ndarray:
    """ポリゴンからマスクを作成
    
    Args:
        image_shape: (height, width)
        points: ポリゴンの座標リスト
        buffer_size: マスクの膨張サイズ（ピクセル）
    
    Returns:
        マスク画像（0または255）
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # ポリゴンを描画
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    
    # バッファーサイズが指定されている場合、マスクを膨張
    if buffer_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_size * 2 + 1, buffer_size * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """マスクを画像に適用して背景を透明化
    
    Args:
        image: 入力画像（BGR）
        mask: マスク画像（0または255）
    
    Returns:
        マスク適用後の画像（BGRA、背景は透明）
    """
    # BGRからBGRAに変換
    if image.shape[2] == 3:
        image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    else:
        image_with_alpha = image.copy()
    
    # アルファチャンネルにマスクを設定
    image_with_alpha[:, :, 3] = mask
    
    return image_with_alpha


def transform_points(points: List[List[float]], x_offset: int, y_offset: int) -> List[List[float]]:
    """座標をオフセット分移動（クロップ画像の座標系に変換）"""
    return [[x - x_offset, y - y_offset] for x, y in points]


def create_cropped_label(
    original_label: Dict,
    shape: Dict,
    crop_width: int,
    crop_height: int,
    crop_image_name: str,
    x_offset: int,
    y_offset: int
) -> Dict:
    """クロップ画像用の新しいラベルファイルを作成"""
    
    # 座標を変換
    transformed_points = transform_points(shape['points'], x_offset, y_offset)
    
    # 新しいshapeオブジェクトを作成
    new_shape = {
        "label": shape['label'],
        "shape_type": shape['shape_type'],
        "flags": shape.get('flags', {}),
        "points": transformed_points,
        "group_id": shape.get('group_id'),
        "description": shape.get('description'),
        "difficult": shape.get('difficult', False),
        "attributes": shape.get('attributes', {})
    }
    
    # 新しいラベルファイルを作成
    new_label = {
        "version": original_label.get('version', '2.4.4'),
        "flags": {},
        "shapes": [new_shape],
        "imagePath": crop_image_name,
        "imageData": None,
        "imageHeight": crop_height,
        "imageWidth": crop_width
    }
    
    return new_label


def find_image_path(label_path: Path, image_dir: Optional[Path] = None) -> Path:
    """ラベルファイルに対応する画像ファイルを探す"""
    # ラベルファイルから画像名を取得
    label_data = load_label_file(label_path)
    image_name = label_data.get('imagePath', '')
    
    if image_dir:
        # 画像ディレクトリが指定されている場合
        image_path = image_dir / image_name
    else:
        # ラベルと同じディレクトリを探す
        image_path = label_path.parent / image_name
    
    if not image_path.exists():
        # .jpgや.pngなどの拡張子を試す
        base_name = label_path.stem
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
            test_path = label_path.parent / (base_name + ext) if not image_dir else image_dir / (base_name + ext)
            if test_path.exists():
                return test_path
    
    return image_path


def process_single_file(
    label_path: Path,
    output_dir: Path,
    image_dir: Optional[Path] = None,
    padding: int = 0,
    show_progress: bool = True,
    mask_background: bool = False,
    mask_buffer: int = 5
) -> Tuple[int, int]:
    """1つのラベルファイルを処理
    
    Args:
        label_path: ラベルファイルのパス
        output_dir: 出力ディレクトリ
        image_dir: 画像ディレクトリ（オプション）
        padding: クロップのパディング
        show_progress: プログレスバーを表示するか
        mask_background: 背景をマスクするか
        mask_buffer: マスクのバッファーサイズ（ピクセル）
    
    Returns:
        (成功したクロップ数, スキップした数)
    """
    
    # ラベルを読み込む
    label_data = load_label_file(label_path)
    
    # 画像を読み込む
    image_path = find_image_path(label_path, image_dir)
    if not image_path.exists():
        if show_progress:
            tqdm.write(f"警告: 画像が見つかりません: {image_path}")
        return 0, 0
    
    image = cv2.imread(str(image_path))
    if image is None:
        if show_progress:
            tqdm.write(f"エラー: 画像を読み込めません: {image_path}")
        return 0, 0
    
    img_height, img_width = image.shape[:2]
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各shapeについて処理
    base_name = label_path.stem
    crop_count = 0
    skip_count = 0
    shapes = label_data.get('shapes', [])
    
    # プログレスバーを表示（shapeが多い場合）
    if show_progress and len(shapes) > 10:
        shapes_iter = tqdm(shapes, desc=f"  {label_path.name}", leave=False, ncols=100)
    else:
        shapes_iter = shapes
    
    for idx, shape in enumerate(shapes_iter):
        try:
            if shape['shape_type'] != 'polygon':
                if show_progress:
                    tqdm.write(f"警告: ポリゴン以外の形状はスキップします: {shape['shape_type']}")
                skip_count += 1
                continue
            
            # ポイント数をチェック
            if len(shape.get('points', [])) < 3:
                if show_progress:
                    tqdm.write(f"警告: ポリゴンのポイント数が不足（{len(shape.get('points', []))}点）: {label_path.name} shape#{idx}")
                skip_count += 1
                continue
            
            # バウンディングボックスを取得
            x_min, y_min, x_max, y_max = get_bounding_box(shape['points'])
            
            # パディングを追加
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(img_width, x_max + padding)
            y_max = min(img_height, y_max + padding)
            
            # バウンディングボックスのサイズをチェック
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            if bbox_width <= 0 or bbox_height <= 0:
                if show_progress:
                    tqdm.write(f"警告: 無効なバウンディングボックス（サイズ: {bbox_width}x{bbox_height}）をスキップ: {label_path.name} shape#{idx}")
                skip_count += 1
                continue
            
            # 画像をクロップ
            cropped_image = crop_image(image, (x_min, y_min, x_max, y_max))
            
            # クロップ画像が空でないかチェック
            if cropped_image.size == 0:
                if show_progress:
                    tqdm.write(f"警告: クロップ画像が空です（bbox: {x_min},{y_min},{x_max},{y_max}）をスキップ: {label_path.name} shape#{idx}")
                skip_count += 1
                continue
            
            crop_height, crop_width = cropped_image.shape[:2]
            
            # 背景をマスクする場合
            if mask_background:
                # クロップ画像の座標系に変換したポリゴン座標
                transformed_points = transform_points(shape['points'], x_min, y_min)
                
                # マスクを作成
                mask = create_polygon_mask(
                    (crop_height, crop_width),
                    transformed_points,
                    mask_buffer
                )
                
                # マスクを適用
                cropped_image = apply_mask_to_image(cropped_image, mask)
                
                # PNG形式で保存（透明度をサポート）
                crop_image_name = f"{base_name}_{idx}.png"
                crop_label_name = f"{base_name}_{idx}.json"
            else:
                # JPG形式で保存
                crop_image_name = f"{base_name}_{idx}.jpg"
                crop_label_name = f"{base_name}_{idx}.json"
            
            # クロップ画像を保存
            crop_image_path = output_dir / crop_image_name
            success = cv2.imwrite(str(crop_image_path), cropped_image)
            
            if not success:
                if show_progress:
                    tqdm.write(f"警告: 画像の保存に失敗: {crop_image_path}")
                skip_count += 1
                continue
            
            # 新しいラベルを作成
            new_label = create_cropped_label(
                label_data,
                shape,
                crop_width,
                crop_height,
                crop_image_name,
                x_min,
                y_min
            )
            
            # ラベルを保存
            crop_label_path = output_dir / crop_label_name
            with open(crop_label_path, 'w', encoding='utf-8') as f:
                json.dump(new_label, f, indent=2, ensure_ascii=False)
            
            crop_count += 1
            
        except Exception as e:
            if show_progress:
                tqdm.write(f"エラー: {label_path.name} shape#{idx}の処理中にエラーが発生: {str(e)}")
            skip_count += 1
            continue
    
    return crop_count, skip_count


def main():
    parser = argparse.ArgumentParser(
        description='インスタンスセグメンテーションラベルから各インスタンスをクロップ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python crop_and_segment.py
  
  # 入力と出力を指定
  python crop_and_segment.py -i ./labels -o ./output
  
  # 画像ディレクトリを別に指定
  python crop_and_segment.py -i ./labels -img ./images -o ./output
  
  # パディングを追加
  python crop_and_segment.py -i ./labels -o ./output -p 10
  
  # 背景を透明化（PNG形式で保存）
  python crop_and_segment.py -i ./labels -o ./output -m
  
  # 背景マスクとバッファーをカスタマイズ
  python crop_and_segment.py -i ./labels -o ./output -m --mask-buffer 10
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path('PodSegDataset/train'),
        help='入力ラベルファイルのディレクトリまたはファイルパス (デフォルト: PodSegDataset/train)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('PodSegDataset/crop'),
        help='出力ディレクトリ (デフォルト: PodSegDataset/crop)'
    )
    
    parser.add_argument(
        '-img', '--image-dir',
        type=Path,
        default=None,
        help='画像ファイルのディレクトリ（ラベルと別の場所にある場合）'
    )
    
    parser.add_argument(
        '-p', '--padding',
        type=int,
        default=0,
        help='クロップ時のパディング（ピクセル） (デフォルト: 0)'
    )
    
    parser.add_argument(
        '-m', '--mask-background',
        action='store_true',
        help='背景をマスクして透明化する（PNG形式で保存）'
    )
    
    parser.add_argument(
        '-mb', '--mask-buffer',
        type=int,
        default=5,
        help='マスクのバッファーサイズ（ピクセル）- ポリゴンの周りに余白を追加 (デフォルト: 5)'
    )
    
    args = parser.parse_args()
    
    # 入力パスの確認
    if not args.input.exists():
        tqdm.write(f"エラー: 入力パスが存在しません: {args.input}")
        return
    
    # ラベルファイルのリストを取得
    if args.input.is_file():
        label_files = [args.input]
    else:
        label_files = sorted(args.input.glob('*.json'))
    
    if not label_files:
        tqdm.write(f"警告: ラベルファイルが見つかりません: {args.input}")
        return
    
    tqdm.write(f"処理開始: {len(label_files)}個のラベルファイルを処理します")
    tqdm.write(f"入力: {args.input}")
    tqdm.write(f"出力: {args.output}")
    if args.image_dir:
        tqdm.write(f"画像ディレクトリ: {args.image_dir}")
    if args.padding > 0:
        tqdm.write(f"パディング: {args.padding}px")
    if args.mask_background:
        tqdm.write(f"背景マスク: 有効 (バッファー: {args.mask_buffer}px, PNG形式)")
    tqdm.write("-" * 60)
    
    # 各ファイルを処理（プログレスバー付き）
    total_crops = 0
    total_skips = 0
    with tqdm(label_files, desc="ファイル処理", ncols=100, unit="file") as pbar:
        for label_file in pbar:
            pbar.set_postfix_str(f"{label_file.name}")
            crop_count, skip_count = process_single_file(
                label_file,
                args.output,
                args.image_dir,
                args.padding,
                show_progress=True,
                mask_background=args.mask_background,
                mask_buffer=args.mask_buffer
            )
            total_crops += crop_count
            total_skips += skip_count
            
            if skip_count > 0:
                pbar.set_postfix_str(f"{label_file.name} -> {crop_count}個 (スキップ: {skip_count})")
            else:
                pbar.set_postfix_str(f"{label_file.name} -> {crop_count}個")
    
    tqdm.write("-" * 60)
    tqdm.write(f"全処理完了: 合計 {total_crops}個のインスタンスをクロップしました")
    if total_skips > 0:
        tqdm.write(f"スキップ: {total_skips}個のインスタンス（無効なデータ、エラーなど）")


if __name__ == '__main__':
    main()

