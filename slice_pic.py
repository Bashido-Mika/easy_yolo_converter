#!/usr/bin/env python3
"""
Image Tile Slicer for X-AnyLabeling Format

X-AnyLabeling形式の画像とラベルをタイル分割します。
オーバーラップとパディングに対応し、ラベルも正確に変換します。
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
from shapely.geometry import Polygon, box
from shapely.validation import make_valid


# 定数
DEFAULT_INPUT_DIR = 'PodSegDataset/train02'
DEFAULT_OUTPUT_DIR = 'PodSegDataset/sliced'
DEFAULT_TILE_SIZE = 640
DEFAULT_OVERLAP = 50
DEFAULT_PADDING_COLOR = (114, 114, 114)  # YOLO標準の背景色


@dataclass
class SliceConfig:
    """タイル分割の設定"""
    input_dir: Path
    output_dir: Path
    tile_width: int
    tile_height: int
    overlap: int
    padding_color: Tuple[int, int, int]
    visualize: bool = False
    clear_pad: float = 0.0  # 0.0=無効, 0.0~1.0=パディング面積の閾値
    
    def __post_init__(self):
        """設定の妥当性チェック"""
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError("タイルサイズは正の値である必要があります")
        if self.overlap < 0:
            raise ValueError("オーバーラップは0以上である必要があります")
        if self.overlap >= min(self.tile_width, self.tile_height):
            raise ValueError("オーバーラップはタイルサイズより小さい必要があります")


@dataclass
class TileInfo:
    """タイル情報"""
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    actual_width: int
    actual_height: int
    needs_padding: bool


class ImageTileSlicer:
    """X-AnyLabeling形式の画像をタイル分割するクラス"""
    
    def __init__(self, config: SliceConfig):
        """
        初期化
        
        Args:
            config: タイル分割設定
        """
        self.config = config
        self.total_tiles = 0
        self.total_shapes = 0
        self.skipped_pad_tiles = 0  # パディング領域のみのタイルをスキップした数
        # 出力ディレクトリに連番を付ける
        self.config.output_dir = self._get_numbered_output_dir(config.output_dir)
    
    def _get_numbered_output_dir(self, base_dir: Path) -> Path:
        """
        出力ディレクトリに連番を付ける
        
        Args:
            base_dir: 基本の出力ディレクトリパス
        
        Returns:
            連番付きの出力ディレクトリパス
        """
        # ディレクトリが存在しない場合はそのまま使用
        if not base_dir.exists():
            return base_dir
        
        # 既にファイルがある場合は連番を付ける
        if list(base_dir.iterdir()):
            # 連番を探す
            counter = 2
            while True:
                numbered_dir = Path(f"{base_dir}{counter:02d}")
                if not numbered_dir.exists() or not list(numbered_dir.iterdir()):
                    return numbered_dir
                counter += 1
        
        return base_dir
    
    def load_label_file(self, label_path: Path) -> Optional[Dict]:
        """
        X-AnyLabeling形式のJSONラベルファイルを読み込み
        
        Args:
            label_path: ラベルファイルのパス
        
        Returns:
            ラベルデータ（失敗時はNone）
        """
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            tqdm.write(f"⚠️ ラベルファイル読み込みエラー: {label_path.name} - {e}")
            return None
    
    def find_image_path(self, label_path: Path) -> Optional[Path]:
        """
        ラベルファイルに対応する画像ファイルを探す
        
        Args:
            label_path: ラベルファイルのパス
        
        Returns:
            画像ファイルのパス（見つからない場合はNone）
        """
        label_data = self.load_label_file(label_path)
        if not label_data:
            return None
        
        image_path_str = label_data.get('imagePath', '')
        if not image_path_str:
            return None
        
        # 同じディレクトリ内で画像を探す
        image_path = label_path.parent / image_path_str
        if image_path.exists():
            return image_path
        
        # 拡張子を変えて探す
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = label_path.with_suffix(ext)
            if test_path.exists():
                return test_path
        
        return None
    
    def calculate_tiles(
        self, 
        image_width: int, 
        image_height: int
    ) -> List[TileInfo]:
        """
        タイル位置を計算（オーバーラップ考慮）
        
        Args:
            image_width: 画像の幅
            image_height: 画像の高さ
        
        Returns:
            タイル情報のリスト
        """
        tiles = []
        stride_x = self.config.tile_width - self.config.overlap
        stride_y = self.config.tile_height - self.config.overlap
        
        row = 0
        y_start = 0
        while y_start < image_height:
            col = 0
            x_start = 0
            
            while x_start < image_width:
                # タイルの終了位置を計算
                x_end = min(x_start + self.config.tile_width, image_width)
                y_end = min(y_start + self.config.tile_height, image_height)
                
                # 実際のタイルサイズ
                actual_width = x_end - x_start
                actual_height = y_end - y_start
                
                # パディングが必要か判定
                needs_padding = (
                    actual_width < self.config.tile_width or 
                    actual_height < self.config.tile_height
                )
                
                tiles.append(TileInfo(
                    row=row,
                    col=col,
                    x_start=x_start,
                    y_start=y_start,
                    x_end=x_end,
                    y_end=y_end,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    needs_padding=needs_padding
                ))
                
                x_start += stride_x
                col += 1
                
                # 最後のタイルに到達したら終了
                if x_end >= image_width:
                    break
            
            y_start += stride_y
            row += 1
            
            # 最後のタイルに到達したら終了
            if y_end >= image_height:
                break
        
        return tiles
    
    def extract_tile(
        self, 
        image: np.ndarray, 
        tile: TileInfo
    ) -> np.ndarray:
        """
        画像からタイルを切り出し（必要に応じてパディング）
        
        Args:
            image: 元画像
            tile: タイル情報
        
        Returns:
            タイル画像
        """
        # タイルを切り出し
        tile_image = image[tile.y_start:tile.y_end, tile.x_start:tile.x_end].copy()
        
        # パディングが必要な場合
        if tile.needs_padding:
            padded = np.full(
                (self.config.tile_height, self.config.tile_width, 3),
                self.config.padding_color,
                dtype=np.uint8
            )
            padded[:tile.actual_height, :tile.actual_width] = tile_image
            return padded
        
        return tile_image
    
    def is_polygon_in_tile(
        self, 
        points: List[List[float]], 
        tile: TileInfo
    ) -> bool:
        """
        ポリゴンがタイル内に存在するかチェック
        
        Args:
            points: ポリゴンの座標リスト
            tile: タイル情報
        
        Returns:
            タイル内に存在する場合True
        """
        if not points:
            return False
        
        # ポリゴンのバウンディングボックスを計算
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        poly_x_min, poly_x_max = min(xs), max(xs)
        poly_y_min, poly_y_max = min(ys), max(ys)
        
        # タイルと重なるかチェック
        return not (
            poly_x_max < tile.x_start or 
            poly_x_min > tile.x_end or
            poly_y_max < tile.y_start or 
            poly_y_min > tile.y_end
        )
    
    def transform_polygon_to_tile(
        self, 
        points: List[List[float]], 
        tile: TileInfo
    ) -> Optional[List[List[float]]]:
        """
        ポリゴン座標をタイル座標系に変換（正確なクリッピング）
        
        Args:
            points: 元画像のポリゴン座標
            tile: タイル情報
        
        Returns:
            変換後の座標（タイル外の場合はNone）
        """
        if not self.is_polygon_in_tile(points, tile):
            return None
        
        try:
            # Shapelyポリゴンを作成
            original_polygon = Polygon(points)
            
            # 無効なポリゴンを修正
            if not original_polygon.is_valid:
                original_polygon = make_valid(original_polygon)
            
            # タイルの矩形を作成（元画像座標系）
            tile_box = box(
                tile.x_start, 
                tile.y_start, 
                tile.x_end, 
                tile.y_end
            )
            
            # ポリゴンとタイル矩形の交差部分を計算
            clipped = original_polygon.intersection(tile_box)
            
            # 交差部分が存在しない、または空の場合
            if clipped.is_empty:
                return None
            
            # 交差結果がPolygonでない場合（LineString, Point等）はスキップ
            if clipped.geom_type == 'Polygon':
                clipped_coords = list(clipped.exterior.coords[:-1])  # 最後の点（重複）を除く
            elif clipped.geom_type == 'MultiPolygon':
                # 複数のポリゴンがある場合、最も大きいものを使用
                largest = max(clipped.geoms, key=lambda p: p.area)
                clipped_coords = list(largest.exterior.coords[:-1])
            else:
                # LineString, Point, GeometryCollection等はスキップ
                return None
            
            # タイル座標系に変換
            transformed = []
            for x, y in clipped_coords:
                new_x = x - tile.x_start
                new_y = y - tile.y_start
                transformed.append([new_x, new_y])
            
            # 有効なポリゴンかチェック（3点以上必要）
            if len(transformed) < 3:
                return None
            
            # 面積が極小の場合は除外（ピクセル単位で1.0以下）
            test_polygon = Polygon(transformed)
            if test_polygon.area < 1.0:
                return None
            
            return transformed
            
        except Exception as e:
            # エラーが発生した場合はスキップ
            return None
    
    def is_polygon_only_in_padding(
        self,
        points: List[List[float]],
        tile: TileInfo
    ) -> bool:
        """
        ポリゴンがパディング領域にのみ存在するかチェック
        
        Args:
            points: タイル座標系のポリゴン座標
            tile: タイル情報
        
        Returns:
            パディング領域にのみ存在する場合True
        """
        if not points or not tile.needs_padding:
            return False
        
        # パディングが必要なタイルの場合、元画像の有効範囲を取得
        valid_width = tile.actual_width
        valid_height = tile.actual_height
        
        # すべての頂点がパディング領域にあるかチェック
        # パディング領域 = x >= valid_width または y >= valid_height
        all_in_padding = True
        for x, y in points:
            # 少なくとも一つの頂点が有効領域内にある場合
            if x < valid_width and y < valid_height:
                all_in_padding = False
                break
        
        return all_in_padding
    
    def has_shapes_only_in_padding(
        self,
        tile_label: Dict,
        tile: TileInfo
    ) -> bool:
        """
        タイルラベルのすべてのshapeがパディング領域にのみ存在するかチェック
        
        Args:
            tile_label: タイルのラベルデータ
            tile: タイル情報
        
        Returns:
            すべてのshapeがパディング領域にのみ存在する場合True
        """
        if not tile.needs_padding:
            return False
        
        shapes = tile_label.get("shapes", [])
        if not shapes:
            return False
        
        # すべてのshapeがパディング領域にのみ存在するかチェック
        for shape in shapes:
            points = shape.get("points", [])
            if not self.is_polygon_only_in_padding(points, tile):
                return False
        
        return True
    
    def get_padding_ratio(self, tile: TileInfo) -> float:
        """
        タイルのパディング面積の割合を計算
        
        Args:
            tile: タイル情報
        
        Returns:
            パディング面積の割合（0.0~1.0）
        """
        if not tile.needs_padding:
            return 0.0
        
        # タイル全体の面積
        total_area = self.config.tile_width * self.config.tile_height
        
        # 有効領域の面積
        valid_area = tile.actual_width * tile.actual_height
        
        # パディング領域の面積
        padding_area = total_area - valid_area
        
        # パディング面積の割合
        padding_ratio = padding_area / total_area
        
        return padding_ratio
    
    def should_skip_tile(self, tile_label: Dict, tile: TileInfo) -> bool:
        """
        タイルをスキップすべきかどうかを判定
        
        Args:
            tile_label: タイルのラベルデータ
            tile: タイル情報
        
        Returns:
            スキップすべき場合True
        """
        if self.config.clear_pad <= 0.0:
            return False
        
        if not tile.needs_padding:
            return False
        
        # パディング面積の割合を取得
        padding_ratio = self.get_padding_ratio(tile)
        
        # 閾値以上の場合はスキップ
        if padding_ratio >= self.config.clear_pad:
            # shapesがある場合のみスキップ（空のタイルは既に別の条件でスキップされる）
            if tile_label.get("shapes"):
                return True
        
        return False
    
    def visualize_tiles(
        self,
        image: np.ndarray,
        tiles: List[TileInfo],
        output_path: Path,
        label_data: Optional[Dict] = None
    ) -> None:
        """
        タイル分割を視覚化した画像を生成
        
        Args:
            image: 元画像
            tiles: タイル情報のリスト
            output_path: 出力パス
            label_data: ラベルデータ（clear_pad判定用、オプション）
        """
        height, width = image.shape[:2]
        
        # パディング領域も含めて必要なサイズを計算
        max_x = max(tile.x_start + self.config.tile_width for tile in tiles)
        max_y = max(tile.y_start + self.config.tile_height for tile in tiles)
        
        # 拡張されたキャンバスを作成（パディング色で埋める）
        extended_width = max(width, max_x)
        extended_height = max(height, max_y)
        vis_image = np.full(
            (extended_height, extended_width, 3),
            self.config.padding_color,
            dtype=np.uint8
        )
        overlay = vis_image.copy()
        
        # 元画像を貼り付け
        vis_image[:height, :width] = image
        overlay[:height, :width] = image
        
        # clear_pad判定用のタイルラベルを作成（label_dataがある場合）
        tiles_to_delete = set()
        if label_data:
            for tile in tiles:
                tile_filename = f"temp_tile_r{tile.row}_c{tile.col}.jpg"
                tile_label = self.create_tile_label(label_data, tile, tile_filename)
                
                # スキップ判定（実際の処理ロジックと同じ）
                # 1. shapesが空の場合はスキップ
                if not tile_label.get("shapes"):
                    tiles_to_delete.add((tile.row, tile.col))
                # 2. clear_padオプションが有効で、パディング面積が閾値以上の場合はスキップ
                elif self.should_skip_tile(tile_label, tile):
                    tiles_to_delete.add((tile.row, tile.col))
        
        # 各タイルを描画
        for tile in tiles:
            # タイル境界の色を決定
            tile_x_end = tile.x_start + self.config.tile_width
            tile_y_end = tile.y_start + self.config.tile_height
            
            is_to_delete = (tile.row, tile.col) in tiles_to_delete
            
            # 削除対象のタイル全体に半透明のオーバーレイを追加（黄色/オレンジ）
            if is_to_delete:
                cv2.rectangle(
                    overlay,
                    (tile.x_start, tile.y_start),
                    (tile_x_end, tile_y_end),
                    (0, 165, 255),  # オレンジ色（BGR）
                    -1  # 塗りつぶし
                )
            
            # タイル境界（削除対象は白色の太線、通常は青色）
            border_color = (255, 255, 255) if is_to_delete else (255, 0, 0)  # 白色 or 青色
            border_thickness = 5 if is_to_delete else 2
            
            cv2.rectangle(
                vis_image,
                (tile.x_start, tile.y_start),
                (tile_x_end, tile_y_end),
                border_color,
                border_thickness
            )
            
            # オーバーラップ領域を色付け（緑色の半透明）
            # 削除対象でない場合のみ表示
            if self.config.overlap > 0 and not is_to_delete:
                # 右側のオーバーラップ
                if tile.x_end < width:
                    overlap_right = min(self.config.overlap, tile.x_end - tile.x_start)
                    cv2.rectangle(
                        overlay,
                        (tile.x_end - overlap_right, tile.y_start),
                        (min(tile.x_end, width), min(tile_y_end, height)),
                        (0, 255, 0),  # 緑色
                        -1
                    )
                
                # 下側のオーバーラップ
                if tile.y_end < height:
                    overlap_bottom = min(self.config.overlap, tile.y_end - tile.y_start)
                    cv2.rectangle(
                        overlay,
                        (tile.x_start, tile.y_end - overlap_bottom),
                        (min(tile_x_end, width), min(tile.y_end, height)),
                        (0, 255, 0),  # 緑色
                        -1
                    )
            
            # パディングが必要な領域を色付け（赤色の半透明）
            # 削除対象でない場合のみ表示
            if tile.needs_padding and not is_to_delete:
                # 右側のパディング（元画像の範囲外）
                if tile.actual_width < self.config.tile_width:
                    pad_x_start = tile.x_start + tile.actual_width
                    pad_x_end = tile.x_start + self.config.tile_width
                    cv2.rectangle(
                        overlay,
                        (pad_x_start, tile.y_start),
                        (pad_x_end, tile_y_end),
                        (0, 0, 255),  # 赤色
                        -1
                    )
                
                # 下側のパディング（元画像の範囲外）
                if tile.actual_height < self.config.tile_height:
                    pad_y_start = tile.y_start + tile.actual_height
                    pad_y_end = tile.y_start + self.config.tile_height
                    cv2.rectangle(
                        overlay,
                        (tile.x_start, pad_y_start),
                        (tile_x_end, pad_y_end),
                        (0, 0, 255),  # 赤色
                        -1
                    )
            
            # タイル番号をテキストで表示
            text = f"r{tile.row}c{tile.col}"
            text_pos = (tile.x_start + 5, tile.y_start + 20)
            text_color = (255, 255, 255) if is_to_delete else (255, 255, 0)  # 白色 or シアン色
            text_thickness = 2 if is_to_delete else 2
            
            # 削除対象の場合は背景を付ける
            if is_to_delete:
                # テキストのサイズを取得
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_thickness
                )
                # 黒い背景を描画
                cv2.rectangle(
                    vis_image,
                    (text_pos[0] - 2, text_pos[1] - text_height - 2),
                    (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2),
                    (0, 0, 0),
                    -1
                )
            
            cv2.putText(
                vis_image,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                text_thickness
            )
            
            # 削除対象のタイルにXマークを描画
            if is_to_delete:
                # タイルの中心を計算
                center_x = tile.x_start + self.config.tile_width // 2
                center_y = tile.y_start + self.config.tile_height // 2
                
                # Xマークのサイズ
                mark_size = min(self.config.tile_width, self.config.tile_height) // 4
                
                # 斜め線を描画（左上から右下）
                cv2.line(
                    vis_image,
                    (center_x - mark_size, center_y - mark_size),
                    (center_x + mark_size, center_y + mark_size),
                    (255, 255, 255),  # 白色
                    6
                )
                
                # 斜め線を描画（右上から左下）
                cv2.line(
                    vis_image,
                    (center_x + mark_size, center_y - mark_size),
                    (center_x - mark_size, center_y + mark_size),
                    (255, 255, 255),  # 白色
                    6
                )
                
                # "SKIP"テキストを表示
                skip_text = "SKIP"
                skip_text_size = 0.9
                skip_text_thickness = 3
                (skip_width, skip_height), skip_baseline = cv2.getTextSize(
                    skip_text, cv2.FONT_HERSHEY_SIMPLEX, skip_text_size, skip_text_thickness
                )
                skip_text_pos = (center_x - skip_width // 2, center_y + mark_size + skip_height + 15)
                
                # SKIPテキストの背景（黒）
                cv2.rectangle(
                    vis_image,
                    (skip_text_pos[0] - 5, skip_text_pos[1] - skip_height - 5),
                    (skip_text_pos[0] + skip_width + 5, skip_text_pos[1] + skip_baseline + 5),
                    (0, 0, 0),
                    -1
                )
                
                # SKIPテキスト（白）
                cv2.putText(
                    vis_image,
                    skip_text,
                    skip_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    skip_text_size,
                    (255, 255, 255),  # 白色
                    skip_text_thickness
                )
        
        # 半透明を合成（透明度30%）
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        # 凡例を追加
        legend_height = 120 if (self.config.clear_pad and tiles_to_delete) else 100
        legend = np.zeros((legend_height, extended_width, 3), dtype=np.uint8)
        legend[:] = (50, 50, 50)  # 濃いグレー背景
        
        # 凡例テキスト
        cv2.putText(legend, "Tile Visualization:", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(legend, "Blue: Tile boundary", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(legend, "Green: Overlap region", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(legend, "Red: Padding region", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # clear_padで削除されるタイルがある場合、凡例に追加
        if self.config.clear_pad > 0.0 and tiles_to_delete:
            cv2.putText(legend, f"Orange + White X: Skipped tiles (clear_pad) [{len(tiles_to_delete)} tiles]", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # 画像と凡例を結合
        final_image = np.vstack([vis_image, legend])
        
        # 保存
        cv2.imwrite(str(output_path), final_image)
    
    def create_tile_label(
        self,
        original_label: Dict,
        tile: TileInfo,
        tile_filename: str
    ) -> Dict:
        """
        タイル用の新しいラベルJSONを作成
        
        Args:
            original_label: 元のラベルデータ
            tile: タイル情報
            tile_filename: タイル画像のファイル名
        
        Returns:
            新しいラベルデータ
        """
        new_label = {
            "version": original_label.get("version", "2.4.4"),
            "flags": original_label.get("flags", {}),
            "shapes": [],
            "imagePath": tile_filename,
            "imageData": None,
            "imageHeight": self.config.tile_height,
            "imageWidth": self.config.tile_width
        }
        
        # 元の全てのshapeを処理
        for shape in original_label.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue
            
            points = shape.get("points", [])
            if not points:
                continue
            
            # ポリゴンをタイル座標系に変換
            transformed_points = self.transform_polygon_to_tile(points, tile)
            if not transformed_points:
                continue
            
            # 新しいshapeを追加
            new_shape = {
                "label": shape.get("label", ""),
                "shape_type": "polygon",
                "flags": shape.get("flags", {}),
                "points": transformed_points,
                "group_id": shape.get("group_id"),
                "description": shape.get("description", ""),
                "difficult": shape.get("difficult", False),
                "attributes": shape.get("attributes", {})
            }
            new_label["shapes"].append(new_shape)
            self.total_shapes += 1
        
        return new_label
    
    def process_single_file(self, label_path: Path) -> int:
        """
        単一のラベルファイルを処理
        
        Args:
            label_path: ラベルファイルのパス
        
        Returns:
            生成されたタイル数
        """
        # ラベル読み込み
        label_data = self.load_label_file(label_path)
        if not label_data:
            return 0
        
        # 画像ファイルを探す
        image_path = self.find_image_path(label_path)
        if not image_path:
            tqdm.write(f"⚠️ 画像が見つかりません: {label_path.name}")
            return 0
        
        # 画像読み込み
        image = cv2.imread(str(image_path))
        if image is None:
            tqdm.write(f"⚠️ 画像読み込みエラー: {image_path.name}")
            return 0
        
        height, width = image.shape[:2]
        
        # タイル位置を計算
        tiles = self.calculate_tiles(width, height)
        
        # 視覚化モードの場合、視覚化画像を生成
        if self.config.visualize:
            # visualizationサブディレクトリを作成
            vis_dir = self.config.output_dir / "visualization"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            vis_output_path = vis_dir / f"{label_path.stem}_visualization.jpg"
            self.visualize_tiles(image, tiles, vis_output_path, label_data)
        
        # 各タイルを処理
        tile_count = 0
        base_name = label_path.stem
        
        for tile in tiles:
            # タイル画像を切り出し
            tile_image = self.extract_tile(image, tile)
            
            # ファイル名を生成（元の画像名 + タイル位置）
            tile_filename = f"{base_name}_tile_r{tile.row}_c{tile.col}.jpg"
            tile_label = self.create_tile_label(label_data, tile, tile_filename)
            
            # shapesが空の場合はスキップ（オブジェクトがないタイル）
            if not tile_label["shapes"]:
                continue
            
            # clear_padオプションが有効で、パディング面積が閾値以上の場合はスキップ
            if self.should_skip_tile(tile_label, tile):
                self.skipped_pad_tiles += 1
                continue
            
            # 保存
            output_image_path = self.config.output_dir / tile_filename
            output_label_path = output_image_path.with_suffix('.json')
            
            cv2.imwrite(str(output_image_path), tile_image)
            
            with open(output_label_path, 'w', encoding='utf-8') as f:
                json.dump(tile_label, f, ensure_ascii=False, indent=2)
            
            tile_count += 1
        
        return tile_count
    
    def run(self) -> bool:
        """
        タイル分割の実行
        
        Returns:
            成功した場合True、失敗した場合False
        """
        # ヘッダー表示
        tqdm.write("=" * 70)
        if self.config.visualize:
            tqdm.write("Image Tile Slicer (X-AnyLabeling Format) + Visualization")
        else:
            tqdm.write("Image Tile Slicer (X-AnyLabeling Format)")
        tqdm.write("=" * 70)
        tqdm.write(f"入力: {self.config.input_dir}")
        tqdm.write(f"出力: {self.config.output_dir}")
        tqdm.write(f"タイルサイズ: {self.config.tile_width}x{self.config.tile_height}")
        tqdm.write(f"オーバーラップ: {self.config.overlap}px")
        tqdm.write(f"パディング色: {self.config.padding_color}")
        if self.config.visualize:
            tqdm.write(f"可視化: 有効 (visualization/に保存)")
        if self.config.clear_pad > 0.0:
            tqdm.write(f"パディング削除閾値: {self.config.clear_pad:.2%} 以上のパディング面積でスキップ")
        tqdm.write("-" * 70)
        
        # 出力ディレクトリ作成
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ラベルファイルを収集
        label_files = list(self.config.input_dir.glob("*.json"))
        
        if not label_files:
            tqdm.write("❌ エラー: ラベルファイル（.json）が見つかりません")
            return False
        
        if self.config.visualize:
            desc = "タイル分割+可視化"
        else:
            desc = "タイル分割処理中"
        
        # 各ファイルを処理
        with tqdm(label_files, desc=desc, ncols=100, unit="file") as pbar:
            for label_path in pbar:
                pbar.set_postfix_str(label_path.name[:30])  # ファイル名を30文字に制限
                tile_count = self.process_single_file(label_path)
                self.total_tiles += tile_count
        
        # 完了メッセージ
        tqdm.write("\n" + "=" * 70)
        tqdm.write("完了！")
        tqdm.write(f"   処理ファイル数: {len(label_files)}個")
        tqdm.write(f"   生成タイル数: {self.total_tiles}個")
        tqdm.write(f"   変換shape数: {self.total_shapes}個")
        if self.config.clear_pad > 0.0 and self.skipped_pad_tiles > 0:
            tqdm.write(f"   スキップ（パディング{self.config.clear_pad:.2%}以上）: {self.skipped_pad_tiles}個")
        if self.config.visualize:
            tqdm.write(f"   視覚化画像: {len(label_files)}枚生成")
            tqdm.write(f"   視覚化保存先: {self.config.output_dir / 'visualization'}")
        tqdm.write(f"   出力先: {self.config.output_dir}")
        tqdm.write("=" * 70)
        
        return True


def parse_tile_size(size_str: str) -> Tuple[int, int]:
    """
    タイルサイズ文字列をパース
    
    Args:
        size_str: タイルサイズ文字列（例: "640x640" または "640"）
    
    Returns:
        (幅, 高さ)のタプル
    """
    if 'x' in size_str.lower():
        parts = size_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f"無効なタイルサイズ形式: {size_str}")
        return int(parts[0]), int(parts[1])
    else:
        # 正方形の場合
        size = int(size_str)
        return size, size


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='X-AnyLabeling形式の画像をタイル分割',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行（640x640、オーバーラップ50px）
  python slice_pic.py
  
  # 入力と出力を指定
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced
  
  # タイルサイズを指定（正方形）
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --tile-size 1024
  
  # タイルサイズを指定（矩形）
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --tile-size 1280x720
  
  # オーバーラップを変更
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --overlap 100
  
  # 視覚化モード（タイル分割 + プレビュー画像を生成）
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --vis
  
  # パディング面積が50%以上のタイルを削除
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --clear_pad 0.5
  
  # パディング面積が75%以上のタイルを削除（視覚化付き）
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --clear_pad 0.75 --vis
  
  # すべてのオプションを指定
  python slice_pic.py -i ./dataset/images -o ./dataset/sliced --tile-size 640x640 --overlap 50

特徴:
  - 連番ディレクトリで上書きを防止（sliced → sliced02 → sliced03）
  - オーバーラップ分割で境界のオブジェクトを確実にキャプチャ
  - 端数部分は自動パディング（YOLO標準の背景色: 114,114,114）
  - ラベル座標を自動変換
  - オブジェクトが含まれないタイルは自動スキップ
  - 視覚化モード（--vis）でタイル分割を確認可能（visualizationフォルダーに保存）
  - パディング領域も正確に可視化
  - パディング面積の閾値でタイルを削除（--clear_pad 0.0~1.0）
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f'入力ディレクトリ（X-AnyLabeling形式） (デフォルト: {DEFAULT_INPUT_DIR})'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'出力ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '-t', '--tile-size',
        type=str,
        default=str(DEFAULT_TILE_SIZE),
        help=f'タイルサイズ（例: 640 または 640x640） (デフォルト: {DEFAULT_TILE_SIZE})'
    )
    
    parser.add_argument(
        '-ov', '--overlap',
        type=int,
        default=DEFAULT_OVERLAP,
        help=f'オーバーラップのピクセル数 (デフォルト: {DEFAULT_OVERLAP})'
    )
    
    parser.add_argument(
        '--vis', '--visualize',
        action='store_true',
        dest='visualize',
        help='視覚化モード: タイル分割と視覚化画像を同時に生成（visualizationフォルダーに保存）'
    )
    
    parser.add_argument(
        '--clear_pad', '--clear-pad',
        type=float,
        default=0.0,
        dest='clear_pad',
        metavar='RATIO',
        help='パディング面積の閾値（0.0~1.0）: この割合以上のパディングを含むタイルを削除（例: 0.5=50%%, 0.75=75%%）'
    )
    
    args = parser.parse_args()
    
    # 入力ディレクトリの確認
    if not args.input.exists():
        tqdm.write(f"❌ エラー: 入力ディレクトリが存在しません: {args.input}")
        return
    
    # タイルサイズをパース
    try:
        tile_width, tile_height = parse_tile_size(args.tile_size)
    except ValueError as e:
        tqdm.write(f"❌ エラー: {e}")
        return
    
    # clear_padの値をチェック
    if args.clear_pad < 0.0 or args.clear_pad > 1.0:
        tqdm.write(f"❌ エラー: --clear_padの値は0.0~1.0の範囲で指定してください（指定値: {args.clear_pad}）")
        return
    
    # 設定を作成
    try:
        config = SliceConfig(
            input_dir=args.input,
            output_dir=args.output,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=args.overlap,
            padding_color=DEFAULT_PADDING_COLOR,
            visualize=args.visualize,
            clear_pad=args.clear_pad
        )
    except ValueError as e:
        tqdm.write(f"❌ エラー: {e}")
        return
    
    # スライサーを実行
    slicer = ImageTileSlicer(config)
    slicer.run()


if __name__ == '__main__':
    main()

